// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/CollisionPredicates.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshIterator.h>
using namespace dolfinx_mpc;

std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
          std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>>
dolfinx_mpc::locate_cells_with_dofs(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const std::vector<std::int64_t> global_indices
      = dofmap.index_map->global_indices(false);

  /// Data structures
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_to_dofs;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> cell_to_dofs_offsets;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cells_with_dofs;

  /// Loop over all cells on this process
  std::int64_t offset_index = 0;
  cell_to_dofs_offsets.conservativeResize(1);
  cell_to_dofs_offsets.tail(1) = offset_index;
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  for (int cell_index = 0; cell_index < num_cells; cell_index++)
  {

    auto cell_dofs = dofmap.cell_dofs(cell_index);

    /// Check if dof is in cell
    bool in_cell = false;
    for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
    {
      for (Eigen::Index j = 0; j < dofs.size(); ++j)
      {
        if (global_indices[cell_dofs[i]] == dofs[j])
        {
          cell_to_dofs.conservativeResize(cell_to_dofs.size() + 1);
          cell_to_dofs.tail(1) = dofs[j];
          offset_index++;
          in_cell = true;
        }
      }
    }
    if (in_cell)
    {
      cells_with_dofs.conservativeResize(cells_with_dofs.size() + 1);
      cells_with_dofs.tail(1) = cell_index;
      cell_to_dofs_offsets.conservativeResize(cell_to_dofs_offsets.size() + 1);
      cell_to_dofs_offsets.tail(1) = offset_index;
    }
  }

  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> cell_adj
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
          cell_to_dofs, cell_to_dofs_offsets);

  return std::make_pair(cells_with_dofs, cell_adj);
}
// //-----------------------------------------------------------------------------
void dolfinx_mpc::build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                                         const dolfinx::fem::Form& a)
{
  // Get dof maps
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::cell) > 0)
  {
    dolfinx::fem::SparsityPatternBuilder::cells(pattern, mesh.topology(),
                                                {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::interior_facet)
      > 0)
  {

    mesh.create_entities(mesh.topology().dim() - 1);
    mesh.create_connectivity(mesh.topology().dim() - 1, mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::interior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::exterior_facet)
      > 0)
  {
    mesh.create_entities(mesh.topology().dim() - 1);
    mesh.create_connectivity(mesh.topology().dim() - 1, mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::exterior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
dolfinx_mpc::get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index)
{
  // TODO: This could be easily made more efficient by exploiting points
  // being ordered by the cell to which they belong.

  // Get mesh
  assert(V);
  assert(V->mesh());
  const dolfinx::mesh::Mesh& mesh = *V->mesh();
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& connectivity_g
      = mesh.geometry().dofmap();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate mapping
  std::shared_ptr<const dolfinx::fem::CoordinateElement> cmap
      = mesh.geometry().coord_mapping;
  if (!cmap)
  {
    throw std::runtime_error(
        "dolfinx::fem::CoordinateElement has not been attached to mesh.");
  }

  // Get element
  assert(V->element());
  const dolfinx::fem::FiniteElement& element = *V->element();
  const int reference_value_size = element.reference_value_size();
  const int value_size = element.value_size();
  const int space_dimension = element.space_dimension();

  // Prepare geometry data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
  Eigen::Array<double, Eigen::Dynamic, 1> detJ(1);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(1,
                                                                          tdim);

  // Prepare basis function data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      1, space_dimension, reference_value_size);
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(1, space_dimension,
                                                         value_size);

  // Create work vector for expansion coefficients
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> coefficients(
      element.space_dimension());

  // Get dofmap
  assert(V->dofmap());
  const dolfinx::fem::DofMap& dofmap = *V->dofmap();

  mesh.create_entity_permutations();

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& permutation_info
      = mesh.topology().get_cell_permutation_info();
  // Skip negative cell indices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_array(space_dimension, value_size);
  if (index < 0)
    return basis_array;

  // Get cell geometry (coordinate dofs)
  for (int i = 0; i < num_dofs_g; ++i)
    for (int j = 0; j < gdim; ++j)
      coordinate_dofs(i, j) = x_g(cell_g[pos_g[index] + i], j);

  // Compute reference coordinates X, and J, detJ and K
  cmap->compute_reference_geometry(X, J, detJ, K, x.head(gdim),
                                   coordinate_dofs);

  // Compute basis on reference element
  element.evaluate_reference_basis(basis_reference_values, X);

  // Push basis forward to physical element
  element.transform_reference_basis(basis_values, basis_reference_values, X, J,
                                    detJ, K, permutation_info[index]);

  for (int i = 0; i < space_dimension; ++i)
  {
    for (int j = 0; j < value_size; ++j)
    {
      // TODO: Find an Eigen shortcut for this operation
      basis_array(i, j) = basis_values(0, i, j);
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
Eigen::Array<bool, Eigen::Dynamic, 1> dolfinx_mpc::check_cell_point_collision(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cells,
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Vector3d point)
{
  // FIXME: Should probably add some exception for vector spaces where we cannot
  // recontruct the element.
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  dolfinx::fem::ElementDofLayout layout = *(dofmap->element_dof_layout);
  dolfinx::mesh::CellType celltype = V->mesh()->topology().cell_type();
  std::int32_t num_vertices = dolfinx::mesh::num_cell_vertices(celltype);
  // NOTE: Assume same structure on all vertices
  std::int32_t num_vertex_dofs = layout.num_entity_closure_dofs(0);
  // Find all dofs on vertices and structure them in a Matrix
  Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_dofs(num_vertices, num_vertex_dofs);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = V->tabulate_dof_coordinates();
  Eigen::Array<bool, Eigen::Dynamic, 1> collisions(cells.size());
  for (std::int32_t i = 0; i < num_vertices; ++i)
  {
    vertex_dofs.row(i) = layout.entity_dofs(0, i);
  }
  // Loop through all cells and check collision with point
  for (std::int32_t i = 0; i < cells.size(); ++i)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        vertices(num_vertices, 3);
    for (std::int32_t j = 0; j < num_vertices; ++j)
    {
      Eigen::Array<std::int32_t, Eigen::Dynamic, 1> cell_dofs
          = dofmap->cell_dofs(cells[i]);
      std::int32_t first_dof = vertex_dofs(j, 0);
      vertices.row(j) = x.row(cell_dofs(first_dof));
    }
    collisions[i] = collision_cell_point(vertices, celltype, point);
  }
  return collisions;
}
//-----------------------------------------------------------------------------
bool dolfinx_mpc::collision_cell_point(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        vertices,
    dolfinx::mesh::CellType celltype, Eigen::Vector3d point)
{
  switch (celltype)
  {
  case dolfinx::mesh::CellType::tetrahedron:
  {
    return dolfinx::geometry::CollisionPredicates::
        collides_tetrahedron_point_3d(vertices.row(0), vertices.row(1),
                                      vertices.row(2), vertices.row(3), point);
  }
  case dolfinx::mesh::CellType::hexahedron:
  {
    throw std::runtime_error(
        "Collsion between hexahedron and point not implemented.");
  }
  case dolfinx::mesh::CellType::quadrilateral:
  {
    // FIXME: Should really be a 3D collision detection
    return dolfinx::geometry::CollisionPredicates::collides_quad_point_2d(
        vertices.row(0), vertices.row(1), vertices.row(2), vertices.row(3),
        point);
  }
  case dolfinx::mesh::CellType::triangle:
  {
    return dolfinx::geometry::CollisionPredicates::collides_triangle_point_3d(
        vertices.row(0), vertices.row(1), vertices.row(2), point);
  }
  case dolfinx::mesh::CellType::interval:
  {
    return dolfinx::geometry::CollisionPredicates::collides_segment_point_3d(
        vertices.row(0), vertices.row(1), point);
  }
  default:
  {
    throw std::runtime_error(
        "Collision between cell and point not implemented.");
  }
  };
}

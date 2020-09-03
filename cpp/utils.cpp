// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
using namespace dolfinx_mpc;

//-----------------------------------------------------------------------------
void dolfinx_mpc::build_standard_pattern(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx::fem::Form<PetscScalar>& a)
{
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (Classic)");
  // Get dof maps
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    dolfinx::fem::SparsityPatternBuilder::cells(pattern, mesh.topology(),
                                                {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::interior_facet)
      > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::interior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::exterior_facet)
      > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
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
  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate mapping
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  assert(V->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const int block_size = element->block_size();
  const int reference_value_size = element->reference_value_size() / block_size;
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;

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

  // Get dofmap
  assert(V->dofmap());
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();

  mesh->topology_mutable().create_entity_permutations();

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& permutation_info
      = mesh->topology().get_cell_permutation_info();
  // Skip negative cell indices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_array(space_dimension * block_size, value_size * block_size);
  basis_array.setZero();
  if (index < 0)
    return basis_array;

  // Get cell geometry (coordinate dofs)
  auto x_dofs = x_dofmap.links(index);
  for (int i = 0; i < num_dofs_g; ++i)
    coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

  // Compute reference coordinates X, and J, detJ and K
  cmap.compute_reference_geometry(X, J, detJ, K, x.head(gdim), coordinate_dofs);

  // Compute basis on reference element
  element->evaluate_reference_basis(basis_reference_values, X);

  // Push basis forward to physical element
  element->transform_reference_basis(basis_values, basis_reference_values, X, J,
                                     detJ, K, permutation_info[index]);

  // Get the degrees of freedom for the current cell
  auto dofs = dofmap->cell_dofs(index);

  // Expand basis values for each dof
  for (int block = 0; block < block_size; ++block)
  {
    for (int i = 0; i < space_dimension; ++i)
    {
      for (int j = 0; j < value_size; ++j)
      {
        basis_array(i * block_size + block, j * block_size + block)
            = basis_values(0, i, j);
      }
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<dolfinx::function::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
};
//-----------------------------------------------------------------------------
void dolfinx_mpc::add_pattern_diagonal(
    dolfinx::la::SparsityPattern& pattern,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> blocks,
    std::int32_t block_size)
{
  for (std::int32_t i = 0; i < blocks.rows(); ++i)
  {
    Eigen::Array<PetscInt, Eigen::Dynamic, 1> diag_block(block_size);
    for (std::size_t comp = 0; comp < block_size; comp++)
      diag_block(comp) = block_size * blocks[i] + comp;
    pattern.insert(diag_block, diag_block);
  }
}
//-----------------------------------------------------------------------------
dolfinx::la::PETScMatrix dolfinx_mpc::create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint> mpc)
{

  // Build sparsitypattern
  dolfinx::la::SparsityPattern pattern = mpc->create_sparsity_pattern(a);

  // Finalise communication
  pattern.assemble();

  // Initialize matrix
  dolfinx::la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);

  return A;
}
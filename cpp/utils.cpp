// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
using namespace dolfinx_mpc;

std::vector<std::pair<
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
    std::pair<std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>>>>
dolfinx_mpc::locate_cells_with_dofs(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> dofs)
{
  dolfinx::common::Timer timer(
      "*MPC: Init: Locate slave and master cells given their dofs");

  // Flatten data from dofs (To vector of function space, 1 if marked, else 0)
  std::vector<std::vector<std::int64_t>> flatten_dofs(dofs.size());
  std::vector<std::vector<std::int64_t>> local_dof_indices(dofs.size());
  for (std::size_t c = 0; c < dofs.size(); ++c)
  {
    std::vector<std::int64_t> dofs_c(V->dim(), 0);
    std::vector<std::int64_t> local_dof_indices_c(V->dim(), -1);
    for (std::size_t i = 0; i < dofs[c].size(); ++i)
    {
      dofs_c[dofs[c][i]] = 1;
      local_dof_indices_c[dofs[c][i]] = i;
    }
    flatten_dofs[c] = dofs_c;
    local_dof_indices[c] = local_dof_indices_c;
  }

  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const std::vector<std::int64_t> global_indices
      = dofmap.index_map->global_indices(false);

  /// Data structures
  std::vector<std::vector<std::int64_t>> cell_to_dofs_vec(dofs.size());
  std::vector<std::vector<std::int64_t>> cell_to_index_vec(dofs.size());
  std::vector<std::vector<std::int32_t>> cell_to_dofs_offsets_vec(dofs.size());
  std::vector<std::vector<std::int64_t>> cells_with_dofs_vec(dofs.size());

  std::vector<std::int64_t> offset_index(dofs.size());
  for (std::size_t c = 0; c < dofs.size(); ++c)
  {
    offset_index[c] = 0;
    cell_to_dofs_offsets_vec[c].push_back(0);
  }
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  // Loop through all cells
  for (int cell_index = 0; cell_index < num_cells; cell_index++)
  {
    auto cell_dofs = dofmap.cell_dofs(cell_index);
    std::vector<bool> in_cell(dofs.size(), false);

    for (std::size_t c = 0; c < dofs.size(); ++c)
    {
      for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
      {
        // Check if cell dof is in any list
        if (flatten_dofs[c][global_indices[cell_dofs[i]]] == 1)
        {
          in_cell[c] = true;
          cell_to_dofs_vec[c].push_back(global_indices[cell_dofs[i]]);
          // Map from cell to index in slave list
          cell_to_index_vec[c].push_back(
              local_dof_indices[c][global_indices[cell_dofs[i]]]);
          offset_index[c]++;
        }
      }
    }
    // Append cell to list if it has dofs in any list
    for (std::size_t c = 0; c < dofs.size(); ++c)
    {
      if (in_cell[c])
      {
        cells_with_dofs_vec[c].push_back(cell_index);
        cell_to_dofs_offsets_vec[c].push_back(offset_index[c]);
      }
    }
  }
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> cell_to_dofs_list(
      dofs.size());
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
      cell_to_dofs_offsets_list(dofs.size());
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
      cells_with_dofs_list(dofs.size());
  std::vector<std::pair<
      Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
      std::pair<std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>,
                std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>>>>
      out_data(dofs.size());

  // Create Eigen arrays for the vectors to wrap nicely to adjacency lists.
  for (std::size_t c = 0; c < dofs.size(); ++c)
  {
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
        cell_to_dofs(cell_to_dofs_vec[c].data(), cell_to_dofs_vec[c].size());
    Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
        cell_to_dofs_offsets(cell_to_dofs_offsets_vec[c].data(),
                             cell_to_dofs_offsets_vec[c].size());
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
        cells_with_dofs(cells_with_dofs_vec[c].data(),
                        cells_with_dofs_vec[c].size());
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
        cell_to_index(cell_to_index_vec[c].data(), cell_to_index_vec[c].size());

    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> cell_adj
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
            cell_to_dofs, cell_to_dofs_offsets);
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> cell_adj_idx
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
            cell_to_index, cell_to_dofs_offsets);
    out_data[c] = std::make_pair(cells_with_dofs,
                                 std::make_pair(cell_adj, cell_adj_idx));
  }

  return out_data;
}
// //-----------------------------------------------------------------------------
void dolfinx_mpc::build_standard_pattern(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx::fem::Form<PetscScalar>& a)
{
  dolfinx::common::Timer timer("MPC: Build classic sparsity pattern");
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
  const dolfinx::fem::CoordinateElement& cmap = mesh.geometry().cmap();

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

  mesh.topology_mutable().create_entity_permutations();

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
  cmap.compute_reference_geometry(X, J, detJ, K, x.head(gdim), coordinate_dofs);

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
std::map<std::int32_t, std::set<int>> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<dolfinx::function::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
};
//-----------------------------------------------------------------------------

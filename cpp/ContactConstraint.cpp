// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ContactConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>

using namespace dolfinx_mpc;

ContactConstraint::ContactConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_cells,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets)
    : _V(V), _slaves(slaves), _slave_cells()
{
  _slave_cells = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      slave_cells, offsets);
  auto [slave_to_cells, cell_to_slaves] = create_cell_maps(slaves);
}

std::pair<
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
    std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
ContactConstraint::create_cell_maps(
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(_V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  std::vector<std::int32_t> dofs_to_cell;
  std::vector<std::int32_t> offsets{0};
  std::map<std::int32_t, std::vector<std::int32_t>> cell_to_dof;

  // Loop over all dofs
  for (std::int32_t i = 0; i < dofs.rows(); ++i)
  {
    // Loop over all cells
    for (std::int32_t j = 0; j < num_cells; ++j)
    {
      // Check if dof is in cell
      auto cell_dofs = dofmap.cell_dofs(j);
      if ((cell_dofs.array() == dofs[i]).any())
      {
        dofs_to_cell.push_back(j);
        // Check if cell allready added to unstructured map
        auto vec = cell_to_dof.find(j);
        if (vec == cell_to_dof.end())
        {
          std::vector<std::int32_t> dof_vec{dofs[i]};
          cell_to_dof.insert({j, dof_vec});
        }
        else
          vec->second.push_back(dofs[i]);
      }
      // Add offsets for AdjacencyList
      offsets.push_back(dofs_to_cell.size());
    }
  }
  // Create dof to cells adjacency list
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj_ptr
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          dofs_to_cell, offsets);

  std::vector<std::int32_t> cells_to_dof;
  std::vector<std::int32_t> offsets_2{0};
  std::vector<std::int32_t> unique_cells;

  // Flatten unstructured map to array
  for (auto it = cell_to_dof.begin(); it != cell_to_dof.end(); ++it)
  {
    unique_cells.push_back(it->first);
    auto vec = it->second;
    cells_to_dof.insert(cells_to_dof.end(), vec.begin(), vec.end());
    offsets_2.push_back(cells_to_dof.size());
  }

  // Create cell to dofs adjacency list
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> dof_cells(
      unique_cells.data(), unique_cells.size());
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj2_ptr;
  adj2_ptr = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      cells_to_dof, offsets_2);

  return std::make_pair(adj_ptr, std::make_pair(dof_cells, adj2_ptr));
};
// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cell_maps.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

//-----------------------------------------------------------------------------
std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
dolfinx_mpc::create_cell_to_dofs_map(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const xtl::span<const std::int32_t>& dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  const std::int32_t local_size
      = dofmap.index_map->size_local() + dofmap.index_map->num_ghosts();
  const std::int32_t block_size = dofmap.index_map_bs();

  // Create dof -> cells map where only slave dofs have entries
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> cell_map;
  {
    std::vector<std::int32_t> num_slave_cells(local_size * block_size, 0);

    std::vector<std::int32_t> in_num_cells(local_size * block_size, 0);
    // Loop through all cells and count number of cells a dof occurs in
    for (std::int32_t i = 0; i < num_cells; i++)
    {
      auto blocks = dofmap.cell_dofs(i);
      for (auto block : blocks)
        for (std::int32_t j = 0; j < block_size; j++)
          in_num_cells[block * block_size + j]++;
    }
    // Count only number of slave cells for dofs
    for (auto dof : dofs)
      num_slave_cells[dof] = in_num_cells[dof];

    std::vector<std::int32_t> insert_position(local_size * block_size, 0);
    std::vector<std::int32_t> cell_offsets(local_size * block_size + 1);
    cell_offsets[0] = 0;
    std::inclusive_scan(num_slave_cells.begin(), num_slave_cells.end(),
                        cell_offsets.begin() + 1);
    std::vector<std::int32_t> cell_data(cell_offsets.back());

    // Accumulate those cells whose contains a slave dof
    for (std::int32_t i = 0; i < num_cells; i++)
    {
      auto blocks = dofmap.cell_dofs(i);
      for (auto block : blocks)
        for (std::int32_t j = 0; j < block_size; j++)
        {
          const std::int32_t dof = block * block_size + j;
          if (num_slave_cells[dof] > 0)
            cell_data[cell_offsets[dof] + insert_position[dof]++] = i;
        }
    }
    cell_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        cell_data, cell_offsets);
  }

  // Create inverse map (cells -> slave dofs)
  std::vector<std::int32_t> num_slaves(num_cells, 0);
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
  {
    auto cells = cell_map->links(i);
    for (auto cell : cells)
      num_slaves[cell]++;
  }
  std::vector<std::int32_t> insert_position(num_cells, 0);
  std::vector<std::int32_t> dof_offsets(num_cells + 1);
  dof_offsets[0] = 0;
  std::inclusive_scan(num_slaves.begin(), num_slaves.end(),
                      dof_offsets.begin() + 1);
  std::vector<std::int32_t> dof_data(dof_offsets.back());
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
  {
    auto cells = cell_map->links(i);
    for (auto cell : cells)
      dof_data[dof_offsets[cell] + insert_position[cell]++] = i;
  }
  return std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      dof_data, dof_offsets);
}

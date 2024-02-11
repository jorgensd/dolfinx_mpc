// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx_mpc
{

/// Create a map from cell to a set of dofs
/// @param[in] The degrees of freedom (local to process)
/// @tparam The floating type of the mesh
/// @returns The map from cell index (local to process) to dofs (local to
/// process) in the cell
template <std::floating_point U>
std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
create_cell_to_dofs_map(const dolfinx::fem::FunctionSpace<U>& V,
                        std::span<const std::int32_t> dofs)
{
  const auto& mesh = *(V.mesh());
  const dolfinx::fem::DofMap& dofmap = *(V.dofmap());
  const int tdim = mesh.topology()->dim();
  const int num_cells = mesh.topology()->index_map(tdim)->size_local();

  const std::int32_t local_size
      = dofmap.index_map->size_local() + dofmap.index_map->num_ghosts();
  const std::int32_t block_size = dofmap.index_map_bs();

  // Create dof -> cells map where only slave dofs have entries
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> cell_map;
  {
    std::vector<std::int32_t> num_slave_cells(local_size * block_size, 0);

    std::vector<std::int32_t> in_num_cells(local_size * block_size, 0);
    // Loop through all cells and count number of cells a dof occurs in
    for (std::int32_t i = 0; i < num_cells; i++)
      for (auto block : dofmap.cell_dofs(i))
        for (std::int32_t j = 0; j < block_size; j++)
          in_num_cells[block * block_size + j]++;

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
      for (auto block : dofmap.cell_dofs(i))
      {
        for (std::int32_t j = 0; j < block_size; j++)
        {
          if (const std::int32_t dof = block * block_size + j;
              num_slave_cells[dof] > 0)
          {
            cell_data[cell_offsets[dof] + insert_position[dof]++] = i;
          }
        }
      }
    }
    cell_map
        = std::make_shared<const dolfinx::graph::AdjacencyList<std::int32_t>>(
            cell_data, cell_offsets);
  }

  // Create inverse map (cells -> slave dofs)
  std::vector<std::int32_t> num_slaves(num_cells, 0);
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
    for (auto cell : cell_map->links(i))
      num_slaves[cell]++;

  std::vector<std::int32_t> insert_position(num_cells, 0);
  std::vector<std::int32_t> dof_offsets(num_cells + 1);
  dof_offsets[0] = 0;
  std::inclusive_scan(num_slaves.begin(), num_slaves.end(),
                      dof_offsets.begin() + 1);
  std::vector<std::int32_t> dof_data(dof_offsets.back());
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
    for (auto cell : cell_map->links(i))
      dof_data[dof_offsets[cell] + insert_position[cell]++] = i;

  return std::make_shared<const dolfinx::graph::AdjacencyList<std::int32_t>>(
      dof_data, dof_offsets);
}

/// Given a list of global degrees of freedom, map them to their local index
/// @param[in] V The original function space
/// @param[in] global_dofs The list of dofs (global index)
/// @tparam The floating type of the mesh
/// @returns List of local dofs
template <std::floating_point U>
std::vector<std::int32_t>
map_dofs_global_to_local(const dolfinx::fem::FunctionSpace<U>& V,
                         const std::vector<std::int64_t>& global_dofs)
{
  const std::size_t num_dofs = global_dofs.size();

  const std::int32_t& block_size = V.dofmap()->index_map_bs();
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V.dofmap()->index_map;

  std::vector<std::int64_t> global_blocks;
  global_blocks.reserve(num_dofs);
  std::vector<std::int32_t> remainders;
  remainders.reserve(num_dofs);
  std::for_each(global_dofs.cbegin(), global_dofs.cend(),
                [block_size, &global_blocks, &remainders](const auto slave)
                {
                  global_blocks.push_back(slave / block_size);
                  remainders.push_back(slave % block_size);
                });
  // Compute the new local index of the master blocks
  std::vector<std::int32_t> local_blocks(num_dofs);
  imap->global_to_local(global_blocks, local_blocks);

  // Go from blocks to actual local dof
  for (std::size_t i = 0; i < local_blocks.size(); i++)
    local_blocks[i] = local_blocks[i] * block_size + remainders[i];
  return local_blocks;
}

/// Create an function space with an extended index map, where all input dofs
/// (global index) is added to the local index map as ghosts.
/// @param[in] V The original function space
/// @param[in] global_dofs The list of master dofs (global index)
/// @param[in] owners The owners of the master degrees of freedom
/// @tparam The floating type of the mesh
template <std::floating_point U>
dolfinx::fem::FunctionSpace<U>
create_extended_functionspace(const dolfinx::fem::FunctionSpace<U>& V,
                              const std::vector<std::int64_t>& global_dofs,
                              const std::vector<std::int32_t>& owners)
{
  dolfinx::common::Timer timer(
      "~MPC: Create new index map with additional ghosts");
  MPI_Comm comm = V.mesh()->comm();
  const dolfinx::fem::DofMap& old_dofmap = *(V.dofmap());
  std::shared_ptr<const dolfinx::common::IndexMap> old_index_map
      = old_dofmap.index_map;

  const std::int32_t& block_size = V.dofmap()->index_map_bs();

  // Compute local master block index.
  const std::size_t num_dofs = global_dofs.size();
  std::vector<std::int64_t> global_blocks(num_dofs);
  std::vector<std::int32_t> local_blocks(num_dofs);
  std::transform(global_dofs.cbegin(), global_dofs.cend(),
                 global_blocks.begin(),
                 [block_size](const auto dof) { return dof / block_size; });

  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  std::shared_ptr<const dolfinx::common::IndexMap> new_index_map;
  if (mpi_size == 1)
  {
    new_index_map = old_index_map;
  }
  else
  {
    // Map global master blocks to local blocks
    V.dofmap()->index_map->global_to_local(global_blocks, local_blocks);

    // Check which local masters that are not on the process already
    std::vector<std::int64_t> additional_ghosts;
    additional_ghosts.reserve(num_dofs);
    std::vector<std::int32_t> additional_owners;
    additional_owners.reserve(num_dofs);
    for (std::size_t i = 0; i < num_dofs; i++)
    {
      // Check if master block already has a local index and
      // if has has already been ghosted, which is the case
      // when we have multiple masters from the same block

      if ((local_blocks[i] == -1)
          and (std::find(additional_ghosts.begin(), additional_ghosts.end(),
                         global_blocks[i])
               == additional_ghosts.end()))
      {
        additional_ghosts.push_back(global_blocks[i]);
        additional_owners.push_back(owners[i]);
      }
    }

    // Append new ghosts (and corresponding rank) at the end of the old set of
    // ghosts originating from the old index map
    std::span<const int> ghost_owners = old_index_map->owners();

    std::span<const std::int64_t> ghosts = old_index_map->ghosts();
    const std::int32_t num_ghosts = ghosts.size();
    assert(ghost_owners.size() == ghosts.size());

    std::vector<std::int64_t> all_ghosts(num_ghosts + additional_ghosts.size());
    std::copy(ghosts.begin(), ghosts.end(), all_ghosts.begin());
    std::copy(additional_ghosts.cbegin(), additional_ghosts.cend(),
              all_ghosts.begin() + num_ghosts);

    std::vector<int> all_owners(all_ghosts.size());
    std::copy(ghost_owners.begin(), ghost_owners.end(), all_owners.begin());
    std::copy(additional_owners.cbegin(), additional_owners.cend(),
              all_owners.begin() + num_ghosts);

    // Create new indexmap with ghosts for master blocks added
    new_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, old_index_map->size_local(), all_ghosts, all_owners);
  }

  // Extract information from the old dofmap to create a new one
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      dofmap_adj = old_dofmap.map();
  // Copy dofmap
  std::vector<std::int32_t> flattened_dofmap;
  flattened_dofmap.reserve(dofmap_adj.extent(0) * dofmap_adj.extent(1));
  for (std::size_t i = 0; i < dofmap_adj.extent(0); ++i)
    for (std::size_t j = 0; j < dofmap_adj.extent(1); ++j)
      flattened_dofmap.push_back(dofmap_adj(i, j));

  auto element = V.element();

  // Create the new dofmap based on the extended index map
  auto new_dofmap = std::make_shared<const dolfinx::fem::DofMap>(
      old_dofmap.element_dof_layout(), new_index_map, old_dofmap.bs(),
      std::move(flattened_dofmap), old_dofmap.bs());

  return dolfinx::fem::FunctionSpace(
      V.mesh(), element, new_dofmap,
      dolfinx::fem::compute_value_shape(element, V.mesh()->topology()->dim(),
                                        V.mesh()->geometry().dim()));
}

} // namespace dolfinx_mpc
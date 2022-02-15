// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "mpc_helpers.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/ordering.h>
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

//-----------------------------------------------------------------------------
std::vector<std::int32_t> dolfinx_mpc::map_dofs_global_to_local(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::vector<std::int64_t>& global_dofs)
{
  const std::size_t num_dofs = global_dofs.size();
  const std::int32_t& block_size = V->dofmap()->index_map_bs();
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;

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
//-----------------------------------------------------------------------------
dolfinx::fem::FunctionSpace dolfinx_mpc::create_extended_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::vector<std::int64_t>& global_dofs,
    const std::vector<std::int32_t>& owners)
{
  dolfinx::common::Timer timer(
      "~MPC: Create new index map with additional ghosts");
  MPI_Comm comm = V->mesh()->comm();
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;

  const std::int32_t& block_size = V->dofmap()->index_map_bs();

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
    new_index_map = index_map;
  }
  else
  {
    // Map global master blocks to local blocks
    V->dofmap()->index_map->global_to_local(global_blocks, local_blocks);

    // Check which local masters that are not on the process already
    std::vector<std::int64_t> new_ghosts;
    new_ghosts.reserve(num_dofs);
    std::vector<std::int32_t> new_ranks;
    new_ranks.reserve(num_dofs);
    for (std::size_t i = 0; i < num_dofs; i++)
    {
      // Check if master block already has a local index
      if (local_blocks[i] == -1)
      {
        // Check if master block has already been ghosted, which is the case
        // when we have multiple masters from the same block
        auto already_ghosted
            = std::find(new_ghosts.begin(), new_ghosts.end(), global_blocks[i]);
        if (already_ghosted == new_ghosts.end())
        {
          new_ghosts.push_back(global_blocks[i]);
          new_ranks.push_back(owners[i]);
        }
      }
    }

    // Append new ghosts (and corresponding rank) at the end of the old set of
    // ghosts originating from the old index map
    std::vector<std::int32_t> ranks = index_map->ghost_owner_rank();
    std::vector<std::int64_t> ghosts = index_map->ghosts();
    const std::int32_t num_ghosts = ghosts.size();
    ghosts.resize(num_ghosts + new_ghosts.size());
    ranks.resize(num_ghosts + new_ghosts.size());
    for (std::size_t i = 0; i < new_ghosts.size(); i++)
    {
      ghosts[num_ghosts + i] = new_ghosts[i];
      ranks[num_ghosts + i] = new_ranks[i];
    }

    // Create new indexmap with ghosts for master blocks added
    new_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, index_map->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD, std::set<int>(ranks.begin(), ranks.end())),
        ghosts, ranks);
  }

  // Extract information from the old dofmap to create a new one
  const dolfinx::fem::DofMap* old_dofmap = V->dofmap().get();
  dolfinx::mesh::Topology topology = V->mesh()->topology();
  dolfinx::fem::ElementDofLayout layout = old_dofmap->element_dof_layout();

  // Create new dofmap based on the MPC index map
  // NOTE: This is a workaround to get the Adjacency list and block size
  // used to initialize the original dofmap
  auto [_, bs, dofmap_adj] = dolfinx::fem::build_dofmap_data(
      V->mesh()->comm(), topology, layout,
      [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
      { return dolfinx::graph::reorder_gps(g); });

  // Permute dofmap as done in DOLFINx
  // If the element's DOF transformations are permutations, permute the DOF
  // numbering on each cell
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  if (element->needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    const std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
        unpermute_dofs = element->get_dof_permutation_function(true, true);
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      unpermute_dofs(dofmap_adj.links(cell), cell_info[cell]);
  }

  // Create the new dofmap based on the extended index map
  std::shared_ptr<dolfinx::fem::DofMap> new_dofmap
      = std::make_shared<dolfinx::fem::DofMap>(
          old_dofmap->element_dof_layout(), new_index_map, bs, dofmap_adj, bs);

  return dolfinx::fem::FunctionSpace(V->mesh(), element, new_dofmap);
}
//-----------------------------------------------------------------------------
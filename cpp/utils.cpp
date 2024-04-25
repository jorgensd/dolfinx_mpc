// Copyright (C) 2020 Jorgen S. Dokken and Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx_mpc;

//-----------------------------------------------------------------------------
std::array<MPI_Comm, 2> dolfinx_mpc::create_neighborhood_comms(
    MPI_Comm comm, const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    const bool has_slave, std::int32_t& master_marker)
{
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  std::uint8_t slave_val = has_slave ? 1 : 0;
  std::vector<std::uint8_t> has_slaves(mpi_size, slave_val);
  // Check if entities if master entities are on this processor
  std::vector<std::uint8_t> has_masters(mpi_size, 0);
  if (std::find(meshtags.values().begin(), meshtags.values().end(),
                master_marker)
      != meshtags.values().end())
    std::fill(has_masters.begin(), has_masters.end(), 1);

  // Get received data sizes from each rank
  std::vector<std::uint8_t> procs_with_masters(mpi_size, -1);
  MPI_Alltoall(has_masters.data(), 1, MPI_UINT8_T, procs_with_masters.data(), 1,
               MPI_UINT8_T, comm);
  std::vector<std::uint8_t> procs_with_slaves(mpi_size, -1);
  MPI_Alltoall(has_slaves.data(), 1, MPI_UINT8_T, procs_with_slaves.data(), 1,
               MPI_UINT8_T, comm);

  // Create communicator with edges slaves (sources) -> masters (destinations)
  std::vector<std::int32_t> source_edges;
  std::vector<std::int32_t> dest_edges;
  // If current rank owns masters add all slaves as source edges
  if (procs_with_masters[rank] == 1)
    for (int i = 0; i < mpi_size; ++i)
      if ((i != rank) && (procs_with_slaves[i] == 1))
        source_edges.push_back(i);

  // If current rank owns a slave add all masters as destinations
  if (procs_with_slaves[rank] == 1)
    for (int i = 0; i < mpi_size; ++i)
      if ((i != rank) && (procs_with_masters[i] == 1))
        dest_edges.push_back(i);
  std::array comms{MPI_COMM_NULL, MPI_COMM_NULL};
  // Create communicator with edges slaves (sources) -> masters (destinations)
  {
    std::vector<int> source_weights(source_edges.size(), 1);
    std::vector<int> dest_weights(dest_edges.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, source_edges.size(), source_edges.data(), source_weights.data(),
        dest_edges.size(), dest_edges.data(), dest_weights.data(),
        MPI_INFO_NULL, false, &comms[0]);
  }
  // Create communicator with edges masters (sources) -> slaves (destinations)
  {
    std::vector<int> source_weights(dest_edges.size(), 1);
    std::vector<int> dest_weights(source_edges.size(), 1);

    MPI_Dist_graph_create_adjacent(comm, dest_edges.size(), dest_edges.data(),
                                   source_weights.data(), source_edges.size(),
                                   source_edges.data(), dest_weights.data(),
                                   MPI_INFO_NULL, false, &comms[1]);
  }
  return comms;
}
//-----------------------------------------------------------------------------
MPI_Comm dolfinx_mpc::create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_blocks,
    std::vector<std::int32_t>& ghost_blocks,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map)
{
  // Get data from IndexMap
  std::span<const int> ghost_owners = index_map->owners();
  const std::int32_t size_local = index_map->size_local();
  dolfinx::graph::AdjacencyList<int> shared_indices
      = index_map->index_to_dest_ranks();

  MPI_Comm comm = create_owner_to_ghost_comm(*index_map);

  // Array of processors sending to the ghost_dofs
  std::set<std::int32_t> src_edges;
  // Array of processors the local_dofs are sent to
  std::set<std::int32_t> dst_edges;
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  for (auto block : local_blocks)
    for (auto proc : shared_indices.links(block))
      dst_edges.insert(proc);

  for (auto block : ghost_blocks)
    src_edges.insert(ghost_owners[block - size_local]);

  MPI_Comm comm_loc = MPI_COMM_NULL;
  // Create communicator with edges owners (sources) -> ghosts (destinations)
  std::vector<std::int32_t> source_edges;
  source_edges.assign(src_edges.begin(), src_edges.end());
  std::vector<std::int32_t> dest_edges;
  dest_edges.assign(dst_edges.begin(), dst_edges.end());
  std::vector<int> source_weights(source_edges.size(), 1);
  std::vector<int> dest_weights(dest_edges.size(), 1);
  MPI_Dist_graph_create_adjacent(comm, source_edges.size(), source_edges.data(),
                                 source_weights.data(), dest_edges.size(),
                                 dest_edges.data(), dest_weights.data(),
                                 MPI_INFO_NULL, false, &comm_loc);

  return comm_loc;
}

std::vector<std::int32_t>
dolfinx_mpc::create_block_to_cell_map(const dolfinx::mesh::Topology& topology,
                                      const dolfinx::fem::DofMap& dofmap,
                                      std::span<const std::int32_t> blocks)
{
  std::vector<std::int32_t> cells;
  cells.reserve(blocks.size());
  // Create block -> cells map

  // Compute number of cells each dof is in
  auto imap = dofmap.index_map;
  const int size_local = imap->size_local();
  std::span<const int> ghost_owners = imap->owners();
  std::vector<std::int32_t> num_cells_per_dof(size_local + ghost_owners.size());

  const int tdim = topology.dim();
  auto cell_imap = topology.index_map(tdim);
  const int num_cells_local = cell_imap->size_local();
  const int num_ghost_cells = cell_imap->num_ghosts();
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap.cell_dofs(i);
    for (auto dof : dofs)
      num_cells_per_dof[dof]++;
  }
  std::vector<std::int32_t> cell_dofs_disp(num_cells_per_dof.size() + 1, 0);
  std::partial_sum(num_cells_per_dof.begin(), num_cells_per_dof.end(),
                   cell_dofs_disp.begin() + 1);
  std::vector<std::int32_t> cell_map(cell_dofs_disp.back());
  // Reuse num_cells_per_dof for insertion
  std::fill(num_cells_per_dof.begin(), num_cells_per_dof.end(), 0);

  // Create the block -> cells map
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap.cell_dofs(i);
    for (auto dof : dofs)
      cell_map[cell_dofs_disp[dof] + num_cells_per_dof[dof]++] = i;
  }

  // Populate map from slaves to corresponding cell (choose first cell in map)
  std::for_each(blocks.begin(), blocks.end(),
                [&cell_dofs_disp, &cell_map, &cells](const auto dof)
                { cells.push_back(cell_map[cell_dofs_disp[dof]]); });
  assert(cells.size() == blocks.size());
  return cells;
}

//-----------------------------------------------------------------------------

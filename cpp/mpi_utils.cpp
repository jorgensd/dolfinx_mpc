// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "mpi_utils.h"

MPI_Comm
dolfinx_mpc::create_owner_to_ghost_comm(const dolfinx::common::IndexMap& map)
{
  // Get source (owner of ghosts) and destination (processes that
  // ghost an owned index) ranks
  std::span<const int> src_ranks = map.src();
  std::span<const int> dest_ranks = map.dest();

  // Check that src and dest ranks are unique and sorted
  assert(std::ranges::is_sorted(src_ranks));
  assert(std::ranges::is_sorted(dest_ranks));

  // Create communicators with directed edges owner -> ghost,
  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(map.comm(), (int)src_ranks.size(),
                                 src_ranks.data(), MPI_UNWEIGHTED,
                                 (int)dest_ranks.size(), dest_ranks.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
  return comm;
}
//-------------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfinx_mpc::compute_neighborhood(const MPI_Comm& comm)
{
  int status;
  MPI_Topo_test(comm, &status);
  assert(status != MPI_UNDEFINED);

  // Get list of neighbors
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
  std::vector<int> src_ranks(indegree);
  std::vector<int> dest_ranks(outdegree);
  MPI_Dist_graph_neighbors(comm, indegree, src_ranks.data(), MPI_UNWEIGHTED,
                           outdegree, dest_ranks.data(), MPI_UNWEIGHTED);
  return {src_ranks, dest_ranks};
}
//-------------------------------------------------------------------------------

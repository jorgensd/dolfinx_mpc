// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>

namespace dolfinx_mpc
{
/// @brief Create a MPI-communicator from owners in the index map to the
/// processes with ghosts
///
/// @param[in] map The index map
/// @returns The mpi communicator
MPI_Comm create_owner_to_ghost_comm(const dolfinx::common::IndexMap& map);

std::pair<std::vector<int>, std::vector<int>>
compute_neighborhood(const MPI_Comm& comm);

} // namespace dolfinx_mpc
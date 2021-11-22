// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <xtl/xspan.hpp>

namespace dolfinx_mpc
{

/// Create a map from cell to a set of dofs
/// @param[in] The degrees of freedom (local to process)
/// @returns The map from cell index (local to process) to dofs (local to
/// process) in the cell
std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
create_cell_to_dofs_map(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                        const xtl::span<const std::int32_t>& dofs);

/// Given a list of global degrees of freedom, map them to their local index
/// @param[in] V The original function space
/// @param[in] global_dofs The list of dofs (global index)
/// @returns List of local dofs
std::vector<std::int32_t>
map_dofs_global_to_local(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                         std::vector<std::int64_t>& global_dofs);

/// Create an function space with an extended index map, where all input dofs
/// (global index) is added to the local index map as ghosts.
/// @param[in] V The original function space
/// @param[in] global_dofs The list of master dofs (global index)
/// @param[in] owners The owners of the master degrees of freedom
dolfinx::fem::FunctionSpace create_extended_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::vector<std::int64_t>& global_dofs, std::vector<std::int32_t>& owners);

} // namespace dolfinx_mpc
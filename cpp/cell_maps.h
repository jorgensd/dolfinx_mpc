// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

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

} // namespace dolfinx_mpc
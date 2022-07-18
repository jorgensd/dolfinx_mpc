// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
namespace dolfinx_mpc
{
/// For a set of unrolled dofs (slaves) compute the index (local to the cell
/// dofs)
/// @param[in] slaves List of unrolled dofs
/// @param[in] num_dofs Number of dofs (blocked)
/// @param[in] bs The block size
/// @param[in] cell_dofs The cell dofs
/// @param[in] is_slave Array indicating if any dof (unrolled, local to process)
/// is a slave
/// @returns Map from position in slaves array to dof local to the cell
std::vector<std::int32_t>
compute_local_slave_index(const std::span<const int32_t>& slaves,
                          const std::uint32_t num_dofs, const int bs,
                          const std::span<const int32_t> cell_dofs,
                          const std::vector<std::int8_t>& is_slave);

} // namespace dolfinx_mpc

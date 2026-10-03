// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <iterator>
#include <span>
#include <vector>

namespace dolfinx_mpc
{

/// @brief Copy the coordinates of the geometry dofs of `cell` into
/// `coordinate_dofs`, three per dof, as the kernels expect them.
/// @param[in] x_dofmap Geometry dofmap
/// @param[in] x_g Geometry coordinates, three per node
/// @param[in] cell The cell
/// @param[out] coordinate_dofs Destination, of at least `3 *
/// x_dofmap.extent(1)` entries
template <std::floating_point U>
void gather_cell_coordinates(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x_dofmap,
    std::span<const U> x_g, std::int32_t cell, std::span<U> coordinate_dofs)
{
  auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
  assert(coordinate_dofs.size() >= 3 * x_dofs.size());
  for (std::size_t i = 0; i < x_dofs.size(); ++i)
  {
    std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                        std::next(coordinate_dofs.begin(), 3 * i));
  }
}

/// For a set of unrolled dofs (slaves) compute the index (local to the cell
/// dofs)
/// @param[in] slaves List of unrolled dofs
/// @param[in] num_dofs Number of dofs (blocked)
/// @param[in] bs The block size
/// @param[in] cell_dofs The cell dofs (blocked)
/// @param[in] is_slave Array indicating if any dof (unrolled, local to process)
/// is a slave
/// @returns Map from position in slaves array to dof local to the cell
std::vector<std::int32_t>
compute_local_slave_index(std::span<const std::int32_t> slaves,
                          const std::uint32_t num_dofs, const int bs,
                          std::span<const std::int32_t> cell_dofs,
                          std::span<const std::int8_t> is_slave);

} // namespace dolfinx_mpc

// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_utils.h"
#include <algorithm>
#include <assert.h>
std::vector<std::int32_t> dolfinx_mpc::compute_local_slave_index(
    std::span<const std::int32_t> slaves, const std::uint32_t num_dofs,
    const int bs, std::span<const std::int32_t> cell_dofs,
    std::span<const std::int8_t> is_slave)
{
  std::vector<std::int32_t> local_index(slaves.size());
  for (std::uint32_t i = 0; i < num_dofs; i++)
    for (int j = 0; j < bs; j++)
    {
      const std::int32_t dof = cell_dofs[i] * bs + j;
      assert((std::uint32_t)dof < is_slave.size());
      if (is_slave[dof])
      {
        auto it = std::ranges::find(slaves, dof);
        const auto slave_index = std::distance(slaves.begin(), it);
        local_index[slave_index] = i * bs + j;
      }
    }
  return local_index;
};
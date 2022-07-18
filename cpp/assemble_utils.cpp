// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_utils.h"
#include <algorithm>

std::vector<std::int32_t> dolfinx_mpc::compute_local_slave_index(
    const std::span<const std::int32_t>& slaves, const std::uint32_t num_dofs,
    const int bs, const std::span<const std::int32_t> cell_dofs,
    const std::vector<std::int8_t>& is_slave)
{
  std::vector<std::int32_t> local_index(slaves.size());

  for (std::uint32_t i = 0; i < num_dofs; i++)
    for (int j = 0; j < bs; j++)
    {
      const std::int32_t dof = cell_dofs[i] * bs + j;
      if (is_slave[dof])
      {
        auto it = std::find(slaves.begin(), slaves.end(), dof);
        const auto slave_index = std::distance(slaves.begin(), it);
        local_index[slave_index] = i * bs + j;
      }
    }
  return local_index;
};
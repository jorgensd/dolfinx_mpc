// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN-X MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/function/FunctionSpace.h>


namespace dolfinx_mpc
{

  /// Returning the cell indices for all elements containing to input dofs, and a mapping from these cells to the corresponding dofs
  /// @param[in] V The function space the multi point constraint is applied on
  /// @param[in] A list of the global degrees of freedom for the dofs that should be located.
  std::pair<std::vector<std::int64_t>,std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>
  locate_cells_with_dofs(std::shared_ptr<const dolfinx::function::FunctionSpace> V,
						 std::vector<std::int64_t> slaves);


} // namespace dolfinx_mpc

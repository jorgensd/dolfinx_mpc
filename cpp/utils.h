// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx_mpc
{

/// Returning the cell indices for all elements containing to a list of input
/// dofs, and the mapping from these cells to the corresponding dofs
/// @param[in] V The function space the multi point constraint is applied on
/// @param[in] dofs A nested list of the global degrees of freedom for the dofs
/// that should be located.
std::vector<
    std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>>>
locate_cells_with_dofs(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> dofs);

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form& a);

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index);

} // namespace dolfinx_mpc

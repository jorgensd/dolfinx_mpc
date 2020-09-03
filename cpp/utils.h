// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx_mpc
{

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<PetscScalar>& a);

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index);

/// Given a function space, compute its shared entities
std::map<std::int32_t, std::set<int>>
compute_shared_indices(std::shared_ptr<dolfinx::function::FunctionSpace> V);

/// Append diagonal entries to sparsity pattern
/// @param[in] pattern The sparsity pattern
/// @param[in] dofs The dofs that require diagonal additions
/// @param[in] block size of problem
void add_pattern_diagonal(dolfinx::la::SparsityPattern& pattern,
                          Eigen::Array<std::int32_t, Eigen::Dynamic, 1> blocks,
                          std::int32_t block_size);

dolfinx::la::PETScMatrix
create_matrix(const dolfinx::fem::Form<PetscScalar>& a,
              const std::shared_ptr<dolfinx_mpc::MultiPointConstraint> mpc);

} // namespace dolfinx_mpc

// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "MultiPointConstraint.h"
#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>

namespace dolfinx_mpc
{
template <typename T, std::floating_point U>
class MultiPointConstraint;

/// @brief Add a value on the diagonal of every slave row owned by the process.
///
/// Kept out of `assemble_matrix` because the diagonal belongs to a block, not
/// to a form: a block may carry slaves without having a diagonal bilinear form
/// to assemble, and a block appearing in several forms must still receive
/// exactly one diagonal entry. Callers run this once per diagonal block, after
/// every form has been assembled.
///
/// @param[in] mat_add Function for adding values into the matrix
/// @param[in] mpc Constraint whose slave rows are given a diagonal entry
/// @param[in] diagval Value to add on the diagonal
template <typename T, std::floating_point U>
void insert_slave_diagonal(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>&)>& mat_add,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc, T diagval)
{
  const std::vector<std::int32_t>& slaves = mpc.slaves();
  const std::int32_t num_local_slaves = mpc.num_local_slaves();
  std::array<std::int32_t, 1> dof;
  const std::array<T, 1> value = {diagval};
  for (std::int32_t i = 0; i < num_local_slaves; ++i)
  {
    dof[0] = slaves[i];
    mat_add(dof, dof, value);
  }
}

/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const double>&)>& mat_add_block,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const double>&)>& mat_add,
    const dolfinx::fem::Form<double>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    std::size_t num_threads=1);

//-----------------------------------------------------------------------------
/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<double>>&)>& mat_add_block,
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<double>>&)>& mat_add,
    const dolfinx::fem::Form<std::complex<double>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc1,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    std::size_t num_threads=1);

/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const float>&)>& mat_add_block,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const float>&)>& mat_add,
    const dolfinx::fem::Form<float>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<float>>>&
        bcs,
    std::size_t num_threads=1);

//-----------------------------------------------------------------------------
/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<float>>&)>& mat_add_block,
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<float>>&)>& mat_add,
    const dolfinx::fem::Form<std::complex<float>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc1,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<float>>>>&
        bcs,
    std::size_t num_threads = 1);

} // namespace dolfinx_mpc
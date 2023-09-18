// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "MultiPointConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>

namespace dolfinx_mpc
{
template <typename T, std::floating_point U>
class MultiPointConstraint;

/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] diagval Value to set on diagonal of matrix for slave dofs and
/// Dirichlet BC (default=1)
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
    const double diagval = 1.0);

//-----------------------------------------------------------------------------
/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] diagval Value to set on diagonal of matrix for slave dofs and
/// Dirichlet BC (default=1)
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
    const std::complex<double> diagval = 1.0);

/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] diagval Value to set on diagonal of matrix for slave dofs and
/// Dirichlet BC (default=1)
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
    const float diagval = 1.0);

//-----------------------------------------------------------------------------
/// Assemble bilinear form into a matrix
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
/// @param[in] diagval Value to set on diagonal of matrix for slave dofs and
/// Dirichlet BC (default=1)
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
    const std::complex<float> diagval = 1.0);

} // namespace dolfinx_mpc
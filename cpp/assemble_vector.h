// Copyright (C) 2021 Jorgen S. Dokken, Nathan Sime, and Connor D. Pierce
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "assemble_utils.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint;


/// Assemble a linear form into a vector
/// @param[in] b The vector to be assembled. It will not be zeroed before
/// assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] mpc The multi-point constraint
void assemble_vector(
    std::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc);

/// Assemble a linear form into a vector
/// @param[in] b The vector to be assembled. It will not be zeroed before
/// assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] mpc The multi-point constraint
void assemble_vector(
    std::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc);

} // namespace dolfinx_mpc

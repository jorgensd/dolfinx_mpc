// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "MultiPointConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>
#include <xtensor/xcomplex.hpp>

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
    xtl::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc);

/// Assemble a linear form into a vector
/// @param[in] b The vector to be assembled. It will not be zeroed before
/// assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] mpc The multi-point constraint
void assemble_vector(
    xtl::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc);

} // namespace dolfinx_mpc

// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include "MultiPointConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>
#include <xtensor/xcomplex.hpp>

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint;

/// Assemble bilinear form into a matrix
/// @param[in] b Span of the underlying vector
/// @param[in] L The linear from to assemble
/// @param[in] mpc The multi-point constraint for degrees of freedom in the
/// test space
void assemble_vector(
    xtl::span<double> b,
    const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc);


/// Assemble a complex valued bilinear form into a matrix
/// @param[in] b Span of the underlying vector
/// @param[in] L The linear from to assemble
/// @param[in] mpc The multi-point constraint for degrees of freedom in the
/// test space
void assemble_vector(
    xtl::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc);

} // namespace dolfinx_mpc
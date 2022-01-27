// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "ContactConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace dolfinx_mpc

{
mpc_data create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        indicator,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale);

mpc_data create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        indicator,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale);

mpc_data create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale);

mpc_data create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale);
} // namespace dolfinx_mpc
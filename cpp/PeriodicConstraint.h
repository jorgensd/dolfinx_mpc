// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "ContactConstraint.h"
#include <dolfinx/fem/DirichletBC.h>

namespace dolfinx_mpc

{
mpc_data create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<std::size_t, 3,
                                       std::experimental::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale, bool collapse);

mpc_data create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<std::size_t, 3,
                                       std::experimental::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale, bool collapse);

mpc_data create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale, bool collapse);

mpc_data create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale, bool collapse);
} // namespace dolfinx_mpc
// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "utils.h"
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/MeshTags.h>
#include <petscsys.h>
namespace dolfinx_mpc
{

/// Create a slip condition between two sets of facets
/// @param[in] V The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] nh Function containing the normal at the slave marker interface
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
mpc_data<PetscScalar> create_contact_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker, std::shared_ptr<dolfinx::fem::Function<PetscScalar>> nh,
    const double eps2 = 1e-20);

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
mpc_data<PetscScalar> create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker, const double eps2 = 1e-20);

} // namespace dolfinx_mpc

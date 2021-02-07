// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/MeshTags.h>
#include <petscsys.h>
namespace dolfinx_mpc
{
typedef struct mpc_data mpc_data;

struct mpc_data
{
  std::vector<std::int32_t> local_slaves;
  std::vector<std::int32_t> ghost_slaves;
  std::vector<std::int64_t> local_masters;
  std::vector<std::int64_t> ghost_masters;
  std::vector<PetscScalar> local_coeffs;
  std::vector<PetscScalar> ghost_coeffs;
  std::vector<std::int32_t> local_offsets;
  std::vector<std::int32_t> ghost_offsets;
  std::vector<std::int32_t> local_owners;
  std::vector<std::int32_t> ghost_owners;

  std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>> get_slaves()
  {
    return std::make_pair(local_slaves, ghost_slaves);
  };
  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> get_masters()
  {
    return std::make_pair(local_masters, ghost_masters);
  };
  std::pair<std::vector<PetscScalar>, std::vector<PetscScalar>> get_coeffs()
  {
    return std::make_pair(local_coeffs, ghost_coeffs);
  };
  std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>> get_offsets()
  {
    return std::make_pair(local_offsets, ghost_offsets);
  };
  std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>> get_owners()
  {
    return std::make_pair(local_owners, ghost_owners);
  };
};

/// Create a slip contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] nh Function containing the normal at the slave marker interface
mpc_data create_contact_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>> nh);

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
mpc_data create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker);

} // namespace dolfinx_mpc

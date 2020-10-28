// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/MeshTags.h>
#include <petscsys.h>
namespace dolfinx_mpc
{
typedef struct mpc_data mpc_data;

struct mpc_data
{
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_slaves;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_slaves;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_masters;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghost_masters;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> local_coeffs;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> ghost_coeffs;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_offsets;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_offsets;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_owners;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners;
  std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
            Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  get_slaves()
  {
    return std::make_pair(local_slaves, ghost_slaves);
  };
  std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
            Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
  get_masters()
  {
    return std::make_pair(local_masters, ghost_masters);
  };
  std::pair<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>,
            Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
  get_coeffs()
  {
    return std::make_pair(local_coeffs, ghost_coeffs);
  };
  std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
            Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  get_offsets()
  {
    return std::make_pair(local_offsets, ghost_offsets);
  };
  std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
            Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  get_owners()
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
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> nh);

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
mpc_data create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker);

} // namespace dolfinx_mpc

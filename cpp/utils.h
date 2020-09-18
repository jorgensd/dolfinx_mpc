// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>

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

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<PetscScalar>& a);

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index);

/// Given a function space, compute its shared entities
std::map<std::int32_t, std::set<int>>
compute_shared_indices(std::shared_ptr<dolfinx::function::FunctionSpace> V);

/// Append diagonal entries to sparsity pattern
/// @param[in] pattern The sparsity pattern
/// @param[in] dofs The dofs that require diagonal additions
/// @param[in] block size of problem
void add_pattern_diagonal(dolfinx::la::SparsityPattern& pattern,
                          Eigen::Array<std::int32_t, Eigen::Dynamic, 1> blocks,
                          std::int32_t block_size);

/// Create neighbourhood communicators for a MeshTag with two separate entity
/// tags, slave_marker->master_marker, master_marker->slave_marker
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
std::array<MPI_Comm, 2>
create_neighborhood_comms(dolfinx::mesh::MeshTags<std::int32_t> meshtags,
                          std::int32_t slave_marker,
                          std::int32_t master_marker);

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] nh Function containing the normal at the slave marker interface
//   std::tuple<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
//              Eigen::Array<PetscScalar, Eigen::Dynamic, 1>,
//              Eigen::Array<PetscScalar, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
//              Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
mpc_data create_contact_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> nh);

} // namespace dolfinx_mpc

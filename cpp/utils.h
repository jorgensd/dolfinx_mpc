// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint;

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_basis_functions(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                    const std::array<double, 3>& x, const int index);

/// Given a function space, compute its shared entities
std::map<std::int32_t, std::set<int>>
compute_shared_indices(std::shared_ptr<dolfinx::fem::FunctionSpace> V);

dolfinx::la::PETScMatrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc,
    const std::string& type = std::string());
/// Create neighborhood communicators from every processor with a slave dof on
/// it, to the processors with a set of master facets.
/// @param[in] meshtags The meshtag
/// @param[in] has_slaves Boolean saying if the processor owns slave dofs
/// @param[in] master_marker Tag for the other interface
std::array<MPI_Comm, 2>
create_neighborhood_comms(dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                          const bool has_slave, std::int32_t& master_marker);

/// Create neighbourhood communicators from local_dofs to processors who has
/// this as a ghost.
/// @param[in] local_dofs Vector of local dofs
/// @param[in] ghost_dofs Vector of ghost dofs
/// @param[in] index_map The index map relating procs and ghosts
/// @param[in] block_size block_size of input dofs
MPI_Comm create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_dofs,
    std::vector<std::int32_t>& ghost_dofs,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map, int block_size);

/// Create a map from each dof found on the set of facets topologically, to the
/// connecting facets
std::map<std::int32_t, std::vector<std::int32_t>>
create_dof_to_facet_map(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                        const xtl::span<const std::int32_t>& facets);

/// For a dof, create an average normal over the topological entities it is
/// connected to
Eigen::Vector3d
create_average_normal(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                      std::int32_t dof, std::int32_t dim,
                      const xtl::span<const std::int32_t>& entities);

/// Creates a normal approximation for the dofs in the closure of the attached
/// facets, where the normal is an average if a dof belongs to multiple facets
void create_normal_approximation(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                                 const xtl::span<const std::int32_t>& entities,
                                 xtl::span<PetscScalar> vector);
} // namespace dolfinx_mpc

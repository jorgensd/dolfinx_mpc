// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MultiPointConstraint.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <xtensor/xtensor.hpp>

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint;

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
xt::xtensor<double, 2>
get_basis_functions(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                    const xt::xtensor<double, 2>& x, const int index);

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

/// Create neighbourhood communicators from a set of local indices to process
/// who has these indices as ghosts.
/// @param[in] local_dofs Vector of local blocks
/// @param[in] ghost_dofs Vector of ghost blocks
/// @param[in] index_map The index map relating procs and ghosts
MPI_Comm create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_blocks,
    std::vector<std::int32_t>& ghost_blocks,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map);

/// Creates a normal approximation for the dofs in the closure of the attached
/// facets, where the normal is an average if a dof belongs to multiple facets
dolfinx::fem::Function<PetscScalar>
create_normal_approximation(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                            std::int32_t dim,
                            const xtl::span<const std::int32_t>& entities);

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
template <typename T>
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<T>& a)
{

  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (Classic)");
  // Get dof maps
  std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2>
      dofmaps{*a.function_spaces().at(0)->dofmap(),
              *a.function_spaces().at(1)->dofmap()};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integral_ids(dolfinx::fem::IntegralType::cell).size() > 0)
  {
    dolfinx::fem::sparsitybuild::cells(pattern, mesh.topology(),
                                       {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integral_ids(dolfinx::fem::IntegralType::interior_facet).size() > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::sparsitybuild::interior_facets(pattern, mesh.topology(),
                                                 {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integral_ids(dolfinx::fem::IntegralType::exterior_facet).size() > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::sparsitybuild::exterior_facets(pattern, mesh.topology(),
                                                 {{dofmaps[0], dofmaps[1]}});
  }
}

/// Add sparsity pattern for multi-point constraints to existing
/// sparsity pattern
/// @param[in] a bi-linear form for the current variational problem
/// (The one used to generate input sparsity-pattern)
/// @param[in] mpc The multi point constraint.
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc);

/// Compute the dot product u . vs
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The dot product `u . v`. The type will be the same as value size
/// of u.
template <typename U, typename V>
typename U::value_type dot(const U& u, const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
};

/// Given a list of global degrees of freedom, map them to their local index
/// @param[in] V The original function space
/// @param[in] global_dofs The list of dofs (global index)
/// @returns List of local dofs
std::vector<std::int32_t>
map_dofs_global_to_local(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                         std::vector<std::int64_t>& global_dofs);

/// Create an function space with an extended index map, where all input dofs
/// (global index) is added to the local index map as ghosts.
/// @param[in] V The original function space
/// @param[in] global_dofs The list of master dofs (global index)
/// @param[in] owners The owners of the master degrees of freedom
dolfinx::fem::FunctionSpace create_extended_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::vector<std::int64_t>& global_dofs, std::vector<std::int32_t>& owners);

} // namespace dolfinx_mpc

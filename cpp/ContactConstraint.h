// Copyright (C) 2020-2026 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "PeriodicConstraint.h"
#include "point_basis.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <format>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace dolfinx_mpc
{

/// Create a slip condition between two sets of facets
/// @param[in] V The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] nh Function containing the normal at the slave marker interface
/// @param[in] distance_tol The largest distance from a slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @note Collective. Throws `std::runtime_error` on every process if a slave
/// is in no cell attached to the master facets.
template <typename T, std::floating_point U>
mpc_data<T> create_contact_slip_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    std::int32_t slave_marker, std::int32_t master_marker,
    const dolfinx::fem::Function<T, U>& nh,
    U distance_tol = default_tolerance<U>(),
    U coefficient_tol = default_tolerance<U>(), std::size_t num_threads = 1)
{
  dolfinx::common::Timer timer("~MPC: Create slip constraint");
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = V.mesh();
  MPI_Comm comm = mesh->comm();
  const int rank = dolfinx::MPI::rank(comm);

  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V.dofmap()->index_map;
  assert(mesh->topology() == meshtags.topology());
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V.dofmap()->index_map_bs();
  const std::int32_t size_local = imap->size_local();
  assert(block_size == gdim);

  mesh->topology_mutable()->create_connectivity(fdim, tdim);
  mesh->topology_mutable()->create_connectivity(tdim, tdim);

  // Owned slave blocks
  std::vector<std::int32_t> local_slave_blocks;
  std::ranges::copy_if(locate_tagged_blocks<U>(V, meshtags, slave_marker),
                       std::back_inserter(local_slave_blocks),
                       [size_local](std::int32_t block)
                       { return block < size_local; });

  // The slave of a block is its component with the largest normal component,
  // to avoid dividing by zero in the constraint
  std::span<const T> normal_array = nh.x()->array();
  std::vector<U> normals(3 * local_slave_blocks.size(), 0);
  std::vector<std::int32_t> local_slaves(local_slave_blocks.size());
  std::vector<std::int32_t> local_rems(local_slave_blocks.size());
  for (std::size_t i = 0; i < local_slave_blocks.size(); ++i)
  {
    std::span<U, 3> normal(std::next(normals.begin(), 3 * i), 3);
    for (int j = 0; j < gdim; ++j)
      normal[j]
          = std::real(normal_array[local_slave_blocks[i] * block_size + j]);
    local_rems[i] = std::ranges::distance(
        normal.begin(),
        std::ranges::max_element(normal, {}, [](U n) { return std::abs(n); }));
    local_slaves[i] = local_slave_blocks[i] * block_size + local_rems[i];
  }

  // dot(u, n) on the slave side involves the other components of the slave's
  // block, and on the master side the dofs at the slave's coordinate
  std::vector<std::int64_t> global_slave_blocks(local_slave_blocks.size());
  imap->local_to_global(local_slave_blocks, global_slave_blocks);
  const std::vector<std::int32_t> slave_cells = create_block_to_cell_map(
      *mesh->topology(), *V.dofmap(), local_slave_blocks);
  const std::vector<U> points
      = tabulate_dof_coordinates<U>(V, local_slave_blocks, slave_cells, false,
                                    num_threads)
            .first;
  assert(mesh->topology() == meshtags.topology());
  const std::vector<std::int32_t> master_cells
      = dolfinx::mesh::compute_incident_entities(
          *meshtags.topology(), meshtags.find(master_marker), meshtags.dim(),
          meshtags.topology()->dim());
  const point_basis<U> basis = evaluate_basis_at_points<U>(
      V, master_cells, points, distance_tol, distance_tol * distance_tol, {}, V,
      num_threads);

  // A slave has at most the other components of its block, and every
  // component-matched dof of its master cell, as masters
  const int width = basis.num_dofs * block_size;
  const std::size_t max_masters
      = local_slaves.size() * (block_size - 1 + width);
  std::vector<std::int64_t> masters;
  masters.reserve(max_masters);
  std::vector<T> coeffs;
  coeffs.reserve(max_masters);
  std::vector<std::int32_t> owners;
  owners.reserve(max_masters);
  std::vector<std::int32_t> num_masters(local_slaves.size(), 0);
  std::vector<std::int64_t> row_masters;
  row_masters.reserve(block_size - 1 + width);
  std::vector<T> row_coeffs;
  row_coeffs.reserve(block_size - 1 + width);
  std::vector<std::int32_t> row_owners;
  row_owners.reserve(block_size - 1 + width);
  std::int32_t num_missing = 0;
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    if (!basis.found[i])
    {
      ++num_missing;
      continue;
    }
    row_masters.clear();
    row_coeffs.clear();
    row_owners.clear();
    std::span<const U, 3> normal(std::next(normals.begin(), 3 * i), 3);
    for (int b = 0; b < block_size; ++b)
    {
      if (b != local_rems[i])
      {
        row_masters.push_back(global_slave_blocks[i] * block_size + b);
        row_coeffs.push_back(-normal[b] / normal[local_rems[i]]);
        row_owners.push_back(rank);
      }
    }
    for (int j = 0; j < basis.num_dofs; ++j)
    {
      for (int b = 0; b < block_size; ++b)
      {
        row_masters.push_back(basis.dofs[i * width + j * block_size + b]);
        row_coeffs.push_back(normal[b] / normal[local_rems[i]]
                             * basis.values[i * basis.num_dofs + j]);
        row_owners.push_back(basis.owners[i * width + j * block_size + b]);
      }
    }
    num_masters[i] = append_significant_masters<T, U>(
        row_masters, row_coeffs, row_owners, coefficient_tol, masters, coeffs,
        owners);
  }
  MPI_Allreduce(MPI_IN_PLACE, &num_missing, 1, MPI_INT32_T, MPI_SUM, comm);
  if (num_missing > 0)
  {
    throw std::runtime_error(std::format(
        "No masters found on the contact surface for {} slave(s). Make sure "
        "that the surfaces are in contact, or increase distance_tol.",
        num_missing));
  }
  return add_ghost_rows<T>(std::move(local_slaves), std::move(masters),
                           std::move(coeffs), std::move(owners),
                           std::move(num_masters), imap, block_size);
}

/// Create an inelastic contact condition between two sets of facets: each
/// slave equals the solution at its coordinate on the master side.
///
/// It is the periodic condition with the identity relation, its masters
/// searched among the cells attached to the master facets.
///
/// @param[in] V The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] distance_tol The largest distance from a slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master.
/// @param[in] allow_missing_masters If true, a slave in no cell attached to the
/// master facets is left unconstrained. Else it is an error.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @note Dofs in the closure of both the slave and the master facets are
/// shared by the two sides, hence already continuous, and are not constrained.
/// This ties the fine facets of a hanging-node interface to the coarse facet
/// they subdivide.
/// @note Collective. Throws `std::runtime_error` on every process for a
/// missing master, unless `allow_missing_masters`.
template <typename T, std::floating_point U>
mpc_data<T> create_contact_inelastic_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    std::int32_t slave_marker, std::int32_t master_marker,
    U distance_tol = default_tolerance<U>(),
    U coefficient_tol = default_tolerance<U>(),
    bool allow_missing_masters = false, std::size_t num_threads = 1)
{
  dolfinx::common::Timer timer("~MPC: Inelastic condition");
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = V.mesh();
  const int tdim = mesh->topology()->dim();
  mesh->topology_mutable()->create_entity_permutations(num_threads);
  mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  mesh->topology_mutable()->create_connectivity(tdim, tdim);

  assert(mesh->topology() == meshtags.topology());
  const std::vector<std::int32_t> master_cells
      = dolfinx::mesh::compute_incident_entities(
          *meshtags.topology(), meshtags.find(master_marker), meshtags.dim(),
          meshtags.topology()->dim());

  // A slave on a master facet would be its own master. The located blocks
  // include those of facets marked on other processes only.
  std::vector<std::int32_t> slave_blocks
      = locate_tagged_blocks<U>(V, meshtags, slave_marker);
  std::vector<std::int32_t> master_blocks
      = locate_tagged_blocks<U>(V, meshtags, master_marker);
  std::ranges::sort(slave_blocks);
  std::ranges::sort(master_blocks);
  std::vector<std::int32_t> blocks;
  blocks.reserve(slave_blocks.size());
  std::ranges::set_difference(slave_blocks, master_blocks,
                              std::back_inserter(blocks));

  auto identity
      = [](std::span<const U> x) { return std::vector<U>(x.begin(), x.end()); };
  return impl::_create_periodic_condition<T, U>(
      V, blocks, identity, T(1), {}, V, master_cells, distance_tol,
      coefficient_tol, allow_missing_masters, num_threads);
}
} // namespace dolfinx_mpc

// Copyright (C) 2020-2026 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "point_basis.h"
#include "utils.h"
#include <algorithm>
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
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace impl
{

/// @brief The cells attached to the entities of a mesh tag with a given value.
/// @param[in] mesh The mesh
/// @param[in] meshtags The meshtags for a set of entities
/// @param[in] marker The value in meshtags to extract entities for
template <std::floating_point U>
std::vector<std::int32_t>
incident_cells(const dolfinx::mesh::Mesh<U>& mesh,
               const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
               std::int32_t marker)
{
  assert(mesh.topology() == meshtags.topology());
  return dolfinx::mesh::compute_incident_entities(
      *mesh.topology(), meshtags.find(marker), meshtags.dim(),
      mesh.topology()->dim());
}

/// Find slave dofs topologically
/// @param[in] V The function space
/// @param[in] meshtags The meshtags for the set of entities
/// @param[in] marker The marker values in the mesh tag
/// @returns The degrees of freedom located on all entities of V that are
/// tagged with the marker
template <std::floating_point U>
std::vector<std::int32_t>
locate_slave_dofs(const dolfinx::fem::FunctionSpace<U>& V,
                  const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                  std::int32_t slave_marker)
{
  const std::int32_t edim = meshtags.dim();
  // Extract slave_facets
  std::vector<std::int32_t> slave_facets;
  slave_facets.reserve(meshtags.indices().size());
  for (std::size_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == slave_marker)
      slave_facets.push_back(meshtags.indices()[i]);

  // Find all dofs on slave facets
  if (V.element()->num_sub_elements() == 0)
  {
    std::vector<std::int32_t> slave_dofs
        = dolfinx::fem::locate_dofs_topological(
            *V.mesh()->topology(), *V.dofmap(), edim, std::span(slave_facets));
    return slave_dofs;
  }
  else
  {
    // NOTE: Assumption that we are only working with vector spaces, which is
    // ordered as xyz,xyzgeometry
    auto V_sub = V.sub({0});
    auto [V0, map] = V_sub.collapse();
    auto sub_dofmap = V_sub.dofmap();
    std::array<std::vector<std::int32_t>, 2> slave_dofs
        = dolfinx::fem::locate_dofs_topological(*V.mesh()->topology(),
                                                {*sub_dofmap, *V0.dofmap()},
                                                edim, std::span(slave_facets));
    return slave_dofs[0];
  }
}

/// Compute contributions to slip MPC from slave facet side, i.e. dot(u,
/// n)|_slave_facet
/// @param[in] local_slaves The slave dofs (local index)
/// @param[in] local_slave_blocks The corresponding blocks for each slave
/// @param[in] normals The normal vectors, shape (local_slaves.size(), 3).
/// Storage flattened row major.
/// @param[in] imap The index map
/// @param[in] block_size The block size of the index map
/// @param[in] rank The rank of current process
/// @returns A mpc_data struct with slaves, masters, coeffs and owners
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> compute_block_contributions(
    const std::vector<std::int32_t>& local_slaves,
    const std::vector<std::int32_t>& local_slave_blocks,
    std::span<const U> normals,
    const std::shared_ptr<const dolfinx::common::IndexMap> imap,
    std::int32_t block_size, int rank)
{
  assert(normals.size() % 3 == 0);
  assert(normals.size() / 3 == local_slave_blocks.size());
  std::vector<std::int32_t> dofs(block_size);
  // Count number of masters for each local slave (only contributions from)
  // the same block as the actual slave dof
  std::vector<std::int32_t> num_masters_in_cell(local_slaves.size());
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    std::iota(dofs.begin(), dofs.end(), local_slave_blocks[i] * block_size);
    const std::int32_t local_slave = local_slaves[i];
    for (std::int32_t j = 0; j < block_size; ++j)
      if ((dofs[j] != local_slave) && std::abs(normals[3 * i + j]) > 1e-6)
        num_masters_in_cell[i]++;
  }
  std::vector<std::int32_t> masters_offsets(local_slaves.size() + 1);
  masters_offsets[0] = 0;
  std::inclusive_scan(num_masters_in_cell.begin(), num_masters_in_cell.end(),
                      masters_offsets.begin() + 1);

  // Reuse num masters as fill position array
  std::ranges::fill(num_masters_in_cell, 0);

  // Compute coeffs and owners for local cells
  std::vector<std::int64_t> global_slave_blocks(local_slaves.size());
  imap->local_to_global(local_slave_blocks, global_slave_blocks);
  std::vector<std::int64_t> masters_in_cell(masters_offsets.back());
  std::vector<T> coefficients_in_cell(masters_offsets.back());
  const std::vector<std::int32_t> owners_in_cell(masters_offsets.back(), rank);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    const std::int32_t local_slave = local_slaves[i];
    std::iota(dofs.begin(), dofs.end(), local_slave_blocks[i] * block_size);
    auto local_max = std::ranges::find(dofs, local_slave);
    const auto max_index = std::ranges::distance(dofs.begin(), local_max);
    for (std::int32_t j = 0; j < block_size; j++)
    {
      if ((dofs[j] != local_slave) && std::abs(normals[3 * i + j]) > 1e-6)
      {
        T coeff_j = -normals[3 * i + j] / normals[3 * i + max_index];
        coefficients_in_cell[masters_offsets[i] + num_masters_in_cell[i]]
            = coeff_j;
        masters_in_cell[masters_offsets[i] + num_masters_in_cell[i]]
            = global_slave_blocks[i] * block_size + j;
        num_masters_in_cell[i]++;
      }
    }
  }

  dolfinx_mpc::mpc_data<T> mpc;
  mpc.slaves = local_slaves;
  mpc.masters = masters_in_cell;
  mpc.coeffs = coefficients_in_cell;
  mpc.offsets = masters_offsets;
  mpc.owners = owners_in_cell;
  return mpc;
}

} // namespace impl

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
/// @param[in] num_threads The number of threads to use for certain operations.
/// @note Collective. Throws `std::runtime_error` on every process if a slave
/// is in no cell attached to the master facets.
template <typename T, std::floating_point U>
mpc_data<T> create_contact_slip_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    std::int32_t slave_marker, std::int32_t master_marker,
    const dolfinx::fem::Function<T, U>& nh, const U eps2 = 1e-20,
    std::size_t num_threads = 1)
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
  for (std::int32_t dof : impl::locate_slave_dofs<U>(V, meshtags, slave_marker))
    if (const std::int32_t block = dof / block_size; block < size_local)
      local_slave_blocks.push_back(block);

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

  // dot(u, n) on the slave side: the other components of the slave's block
  const mpc_data<T> in_block = impl::compute_block_contributions<T, U>(
      local_slaves, local_slave_blocks, normals, imap, block_size, rank);

  // and on the master side, at the slave's coordinate
  const std::vector<std::int32_t> slave_cells = create_block_to_cell_map(
      *mesh->topology(), *V.dofmap(), local_slave_blocks);
  const std::vector<U> points
      = tabulate_dof_coordinates<U>(V, local_slave_blocks, slave_cells).first;
  const point_basis<U> basis = evaluate_basis_at_points<U>(
      V, impl::incident_cells(*mesh, meshtags, master_marker), points,
      std::sqrt(eps2), eps2, {}, V, num_threads);

  std::vector<std::int64_t> masters;
  std::vector<T> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> num_masters(local_slaves.size(), 0);
  const int width = basis.num_dofs * block_size;
  std::int32_t num_missing = 0;
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    for (std::int32_t k = in_block.offsets[i]; k < in_block.offsets[i + 1]; ++k)
    {
      masters.push_back(in_block.masters[k]);
      coeffs.push_back(in_block.coeffs[k]);
      owners.push_back(in_block.owners[k]);
      ++num_masters[i];
    }
    if (!basis.found[i])
    {
      ++num_missing;
      continue;
    }
    for (int j = 0; j < basis.num_dofs; ++j)
    {
      for (int b = 0; b < block_size; ++b)
      {
        if (const T val = normals[3 * i + b] / normals[3 * i + local_rems[i]]
                          * basis.values[i * basis.num_dofs + j];
            std::abs(val) > 1e-6)
        {
          masters.push_back(basis.dofs[i * width + j * block_size + b]);
          coeffs.push_back(val);
          owners.push_back(basis.owners[i * width + j * block_size + b]);
          ++num_masters[i];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &num_missing, 1, MPI_INT32_T, MPI_SUM, comm);
  if (num_missing > 0)
  {
    throw std::runtime_error(std::format(
        "No masters found on the contact surface for {} slave(s). Make sure "
        "that the surfaces are in contact, or increase eps2.",
        num_missing));
  }
  return add_ghost_rows<T>(std::move(local_slaves), std::move(masters),
                           std::move(coeffs), std::move(owners),
                           std::move(num_masters), imap, block_size);
}

/// Create a contact condition between two sets of facets
/// @param[in] V The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
/// @param[in] allow_missing_masters If true, a slave in no cell attached to the
/// master facets is left unconstrained. Else it is an error.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @note Collective. Throws `std::runtime_error` on every process for a
/// missing master, unless `allow_missing_masters`.
template <typename T, std::floating_point U>
mpc_data<T> create_contact_inelastic_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    std::int32_t slave_marker, std::int32_t master_marker, const U eps2 = 1e-20,
    bool allow_missing_masters = false, std::size_t num_threads = 1)
{
  dolfinx::common::Timer timer("~MPC: Inelastic condition");
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = V.mesh();
  MPI_Comm comm = mesh->comm();

  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V.dofmap()->index_map;
  const int tdim = mesh->topology()->dim();
  const int fdim = tdim - 1;
  const int block_size = V.dofmap()->index_map_bs();
  const std::int32_t size_local = imap->size_local();

  mesh->topology_mutable()->create_entity_permutations(num_threads);
  mesh->topology_mutable()->create_connectivity(fdim, tdim);
  mesh->topology_mutable()->create_connectivity(tdim, tdim);

  // Owned slave blocks
  std::vector<std::int32_t> local_blocks;
  for (std::int32_t dof : impl::locate_slave_dofs<U>(V, meshtags, slave_marker))
    if (const std::int32_t block = dof / block_size; block < size_local)
      local_blocks.push_back(block);

  // The masters are at the slave's coordinate on the master side
  const std::vector<std::int32_t> slave_cells
      = create_block_to_cell_map(*mesh->topology(), *V.dofmap(), local_blocks);
  const std::vector<U> points
      = tabulate_dof_coordinates<U>(V, local_blocks, slave_cells).first;
  const point_basis<U> basis = evaluate_basis_at_points<U>(
      V, impl::incident_cells(*mesh, meshtags, master_marker), points,
      std::sqrt(eps2), eps2, {}, V, num_threads);

  // Component j of a slave is tied to component j of the masters
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<T> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> num_masters;
  const int width = basis.num_dofs * block_size;
  std::int32_t num_missing = 0;
  for (std::size_t i = 0; i < local_blocks.size(); ++i)
  {
    if (!basis.found[i])
    {
      ++num_missing;
      continue;
    }
    for (int j = 0; j < block_size; ++j)
    {
      slaves.push_back(local_blocks[i] * block_size + j);
      std::int32_t num = 0;
      for (int k = 0; k < basis.num_dofs; ++k)
      {
        if (const T c = basis.values[i * basis.num_dofs + k];
            std::abs(c) > 1e-6)
        {
          masters.push_back(basis.dofs[i * width + k * block_size + j]);
          coeffs.push_back(c);
          owners.push_back(basis.owners[i * width + k * block_size + j]);
          ++num;
        }
      }
      num_masters.push_back(num);
    }
  }
  if (!allow_missing_masters)
  {
    MPI_Allreduce(MPI_IN_PLACE, &num_missing, 1, MPI_INT32_T, MPI_SUM, comm);
    if (num_missing > 0)
    {
      throw std::runtime_error(std::format(
          "No masters found on the contact surface for {} slave block(s). "
          "Make sure that the surfaces are in contact, or increase eps2.",
          num_missing));
    }
  }
  return add_ghost_rows<T>(std::move(slaves), std::move(masters),
                           std::move(coeffs), std::move(owners),
                           std::move(num_masters), imap, block_size);
}
} // namespace dolfinx_mpc

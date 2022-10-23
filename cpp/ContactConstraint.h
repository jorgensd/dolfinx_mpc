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

namespace dolfinx_mpc::impl
{

/// Create bounding box tree (of cells) based on a mesh tag and a given set of
/// markers in the tag. This means that for a given set of facets, we compute
/// the bounding box tree of the cells connected to the facets
/// @param[in] meshtags The meshtags for a set of entities
/// @param[in] dim The entitiy dimension
/// @param[in] marker The value in meshtags to extract entities for
/// @param[in] padding How much to pad the boundingboxtree
/// @returns A bounding box tree of the cells connected to the entities
dolfinx::geometry::BoundingBoxTree
create_boundingbox_tree(const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                        std::int32_t dim, std::int32_t marker, double padding)
{

  auto mesh = meshtags.mesh();
  const std::int32_t tdim = mesh->topology().dim();
  auto entity_to_cell = meshtags.mesh()->topology().connectivity(dim, tdim);
  assert(entity_to_cell);

  // Find all cells connected to master facets for collision detection
  std::int32_t num_local_cells = mesh->topology().index_map(tdim)->size_local();
  std::set<std::int32_t> cells;
  auto facet_values = meshtags.values();
  auto facet_indices = meshtags.indices();
  for (std::size_t i = 0; i < facet_indices.size(); ++i)
  {
    if (facet_values[i] == marker)
    {
      auto cell = entity_to_cell->links(facet_indices[i]);
      assert(cell.size() == 1);
      if (cell[0] < num_local_cells)
        cells.insert(cell[0]);
    }
  }
  // Create bounding box tree of all master dofs
  std::vector<int> cells_vec(cells.begin(), cells.end());
  dolfinx::geometry::BoundingBoxTree bb_tree(*mesh, tdim, cells_vec, padding);
  return bb_tree;
}

/// Compute contributions to slip constrain from master side (local to process)
/// @param[in] local_rems List containing which block each slave dof is in
/// @param[in] local_colliding_cell List with one-to-one correspondes to a cell
/// that the block is colliding with
/// @param[in] normals The normals at each slave dofs
/// @param[in] V the function space
/// @param[in] tabulated_basis_values The basis values tabulated for the given
/// cells at the given coordinates
/// @returns The mpc data (exluding slave indices)
mpc_data compute_master_contributions(
    std::span<const std::int32_t> local_rems,
    std::span<const std::int32_t> local_colliding_cell,
    std::experimental::mdspan<
        double, std::experimental::extents<
                    std::size_t, std::experimental::dynamic_extent, 3>>
        normals,
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::experimental::mdspan<double,
                              std::experimental::dextents<std::size_t, 2>>
        tabulated_basis_values)
{
  const double tol = 1e-6;
  auto mesh = V->mesh();
  const std::int32_t block_size = V->dofmap()->index_map_bs();
  std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int bs = V->dofmap()->index_map_bs();
  const std::int32_t size_local = imap->size_local();
  MPI_Comm comm = mesh->comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);
  // Count number of masters for in local contribution if found, else add to
  // array that is communicated to other processes
  const std::size_t num_slaves_local = local_rems.size();
  std::vector<std::int32_t> num_masters_local(num_slaves_local, 0);

  assert(num_slaves_local == local_colliding_cell.size());

  for (std::size_t i = 0; i < num_slaves_local; ++i)
  {
    if (const std::int32_t cell = local_colliding_cell[i]; cell != -1)
    {
      auto cell_blocks = V->dofmap()->cell_dofs(cell);
      for (std::size_t j = 0; j < cell_blocks.size(); ++j)
      {
        for (int b = 0; b < bs; b++)
        {
          // NOTE: Assuming 0 value size
          if (const PetscScalar val = normals(i, b) / normals(i, local_rems[i])
                                      * tabulated_basis_values(i, j);
              std::abs(val) > tol)
          {
            num_masters_local[i]++;
          }
        }
      }
    }
  }

  std::vector<std::int32_t> masters_offsets(num_slaves_local + 1);
  masters_offsets[0] = 0;
  std::inclusive_scan(num_masters_local.begin(), num_masters_local.end(),
                      masters_offsets.begin() + 1);
  std::vector<std::int64_t> masters_other_side(masters_offsets.back());
  std::vector<PetscScalar> coefficients_other_side(masters_offsets.back());
  std::vector<std::int32_t> owners_other_side(masters_offsets.back());
  const std::vector<int>& ghost_owners = imap->owners();

  // Temporary array holding global indices
  std::vector<std::int64_t> global_blocks;

  // Reuse num_masters_local for insertion
  std::fill(num_masters_local.begin(), num_masters_local.end(), 0);
  for (std::size_t i = 0; i < num_slaves_local; ++i)
  {
    if (const std::int32_t cell = local_colliding_cell[i]; cell != -1)
    {
      auto cell_blocks = V->dofmap()->cell_dofs(cell);
      global_blocks.resize(cell_blocks.size());
      imap->local_to_global(cell_blocks, global_blocks);
      // Compute coefficients for each master
      for (std::size_t j = 0; j < cell_blocks.size(); ++j)
      {
        const std::int32_t cell_block = cell_blocks[j];
        for (int b = 0; b < bs; b++)
        {
          // NOTE: Assuming 0 value size
          if (const PetscScalar val = normals(i, b) / normals(i, local_rems[i])
                                      * tabulated_basis_values(i, j);
              std::abs(val) > tol)
          {
            const std::int32_t m_pos
                = masters_offsets[i] + num_masters_local[i];
            masters_other_side[m_pos] = global_blocks[j] * block_size + b;
            coefficients_other_side[m_pos] = val;
            owners_other_side[m_pos]
                = cell_block < size_local
                      ? rank
                      : ghost_owners[cell_block - size_local];
            num_masters_local[i]++;
          }
        }
      }
    }
  }
  // Do not add in slaves data to mpc_data, as we allready know the slaves
  mpc_data mpc_local;
  mpc_local.masters = masters_other_side;
  mpc_local.coeffs = coefficients_other_side;
  mpc_local.offsets = masters_offsets;
  mpc_local.owners = owners_other_side;
  return mpc_local;
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
mpc_data compute_block_contributions(
    const std::vector<std::int32_t>& local_slaves,
    const std::vector<std::int32_t>& local_slave_blocks,
    std::span<const double> normals,
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
  std::fill(num_masters_in_cell.begin(), num_masters_in_cell.end(), 0);

  // Compute coeffs and owners for local cells
  std::vector<std::int64_t> global_slave_blocks(local_slaves.size());
  imap->local_to_global(local_slave_blocks, global_slave_blocks);
  std::vector<std::int64_t> masters_in_cell(masters_offsets.back());
  std::vector<PetscScalar> coefficients_in_cell(masters_offsets.back());
  const std::vector<std::int32_t> owners_in_cell(masters_offsets.back(), rank);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    const std::int32_t local_slave = local_slaves[i];
    std::iota(dofs.begin(), dofs.end(), local_slave_blocks[i] * block_size);
    auto local_max = std::find(dofs.begin(), dofs.end(), local_slave);
    const auto max_index = std::distance(dofs.begin(), local_max);
    for (std::int32_t j = 0; j < block_size; j++)
    {
      if ((dofs[j] != local_slave) && std::abs(normals[3 * i + j]) > 1e-6)
      {
        PetscScalar coeff_j = -normals[3 * i + j] / normals[3 * i + max_index];
        coefficients_in_cell[masters_offsets[i] + num_masters_in_cell[i]]
            = coeff_j;
        masters_in_cell[masters_offsets[i] + num_masters_in_cell[i]]
            = global_slave_blocks[i] * block_size + j;
        num_masters_in_cell[i]++;
      }
    }
  }

  mpc_data mpc;
  mpc.slaves = local_slaves;
  mpc.masters = masters_in_cell;
  mpc.coeffs = coefficients_in_cell;
  mpc.offsets = masters_offsets;
  mpc.owners = owners_in_cell;
  return mpc;
}

/// Concatatenate to mpc_data structures with same number of offsets
mpc_data concatenate(mpc_data& mpc0, mpc_data& mpc1)
{

  assert(mpc0.offsets.size() == mpc1.offsets.size());
  std::vector<std::int32_t>& offsets0 = mpc0.offsets;
  std::vector<std::int32_t>& offsets1 = mpc1.offsets;
  std::vector<std::int64_t>& masters0 = mpc0.masters;
  std::vector<std::int64_t>& masters1 = mpc1.masters;
  std::vector<PetscScalar>& coeffs0 = mpc0.coeffs;
  std::vector<PetscScalar>& coeffs1 = mpc1.coeffs;
  std::vector<std::int32_t>& owners0 = mpc0.owners;
  std::vector<std::int32_t>& owners1 = mpc1.owners;

  const std::size_t num_slaves = offsets0.size() - 1;
  // Concatenate the two constraints as one
  std::vector<std::int32_t> num_masters_per_slave(num_slaves, 0);
  for (std::size_t i = 0; i < num_slaves; i++)
  {
    num_masters_per_slave[i]
        = offsets0[i + 1] - offsets0[i] + offsets1[i + 1] - offsets1[i];
  }
  std::vector<std::int32_t> masters_offsets(offsets0.size());
  masters_offsets[0] = 0;
  std::inclusive_scan(num_masters_per_slave.begin(),
                      num_masters_per_slave.end(), masters_offsets.begin() + 1);

  // Reuse num_masters_per_slave for indexing
  std::fill(num_masters_per_slave.begin(), num_masters_per_slave.end(), 0);
  std::vector<std::int64_t> masters_out(masters_offsets.back());
  std::vector<PetscScalar> coefficients_out(masters_offsets.back());
  std::vector<std::int32_t> owners_out(masters_offsets.back());
  for (std::size_t i = 0; i < num_slaves; ++i)
  {
    for (std::int32_t j = offsets0[i]; j < offsets0[i + 1]; ++j)
    {
      masters_out[masters_offsets[i] + num_masters_per_slave[i]] = masters0[j];
      coefficients_out[masters_offsets[i] + num_masters_per_slave[i]]
          = coeffs0[j];
      owners_out[masters_offsets[i] + num_masters_per_slave[i]] = owners0[j];
      num_masters_per_slave[i]++;
    }
    for (std::int32_t j = offsets1[i]; j < offsets1[i + 1]; ++j)
    {
      masters_out[masters_offsets[i] + num_masters_per_slave[i]] = masters1[j];
      coefficients_out[masters_offsets[i] + num_masters_per_slave[i]]
          = coeffs1[j];
      owners_out[masters_offsets[i] + num_masters_per_slave[i]] = owners1[j];
      num_masters_per_slave[i]++;
    }
  }

  // Structure storing mpc arrays
  dolfinx_mpc::mpc_data mpc;
  mpc.masters = masters_out;
  mpc.coeffs = coefficients_out;
  mpc.owners = owners_out;
  mpc.offsets = masters_offsets;
  return mpc;
}

/// Find slave dofs topologically
/// @param[in] V The function space
/// @param[in] meshtags The meshtags for the set of entities
/// @param[in] marker The marker values in the mesh tag
/// @returns The degrees of freedom located on all entities of V that are
/// tagged with the marker
std::vector<std::int32_t>
locate_slave_dofs(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                  const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                  std::int32_t slave_marker)
{
  const std::int32_t edim = meshtags.dim();
  // Extract slave_facets
  const std::vector<std::int32_t> slave_facets = meshtags.find(slave_marker);

  // Find all dofs on slave facets
  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  auto [V0, map] = V->sub({0})->collapse();
  std::array<std::vector<std::int32_t>, 2> slave_dofs
      = dolfinx::fem::locate_dofs_topological({*V->sub({0}).get(), V0}, edim,
                                              std::span(slave_facets));
  return slave_dofs[0];
}
} // namespace dolfinx_mpc::impl

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
mpc_data create_contact_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>> nh,
    const double eps2 = 1e-20);

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
mpc_data create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker, const double eps2 = 1e-20);

} // namespace dolfinx_mpc

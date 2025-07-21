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

namespace impl
{

/// Create bounding box tree (of cells) based on a mesh tag and a given set of
/// markers in the tag. This means that for a given set of facets, we compute
/// the bounding box tree of the cells connected to the facets
/// @param[in] mesh The mesh
/// @param[in] meshtags The meshtags for a set of entities
/// @param[in] marker The value in meshtags to extract entities for
/// @param[in] padding How much to pad the boundingboxtree
/// @returns A bounding box tree of the cells connected to the entities
template <std::floating_point U>
dolfinx::geometry::BoundingBoxTree<U>
create_boundingbox_tree(const dolfinx::mesh::Mesh<U>& mesh,
                        const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                        std::int32_t marker, double padding)
{

  assert(mesh.topology() == meshtags.topology());
  const std::int32_t tdim = mesh.topology()->dim();
  int dim = meshtags.dim();
  auto entity_to_cell = mesh.topology()->connectivity(dim, tdim);
  assert(entity_to_cell);

  // Find all cells connected to master facets for collision detection
  std::int32_t num_local_cells = mesh.topology()->index_map(tdim)->size_local();
  const std::vector<std::int32_t> facets = meshtags.find(marker);
  std::vector<std::int32_t> cells = dolfinx::mesh::compute_incident_entities(
      *mesh.topology(), facets, dim, tdim);

  dolfinx::geometry::BoundingBoxTree<U> bb_tree(mesh, tdim, cells, padding);
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
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> compute_master_contributions(
    std::span<const std::int32_t> local_rems,
    std::span<const std::int32_t> local_colliding_cell,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
        normals,
    const dolfinx::fem::FunctionSpace<U>& V,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        tabulated_basis_values)
{
  const double tol = 1e-6;
  auto mesh = V.mesh();
  const std::int32_t block_size = V.dofmap()->index_map_bs();
  std::shared_ptr<const dolfinx::common::IndexMap> imap = V.dofmap()->index_map;
  const int bs = V.dofmap()->index_map_bs();
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
      auto cell_blocks = V.dofmap()->cell_dofs(cell);
      for (std::size_t j = 0; j < cell_blocks.size(); ++j)
      {
        for (int b = 0; b < bs; b++)
        {
          // NOTE: Assuming 0 value size
          if (const T val = normals(i, b) / normals(i, local_rems[i])
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
  std::vector<T> coefficients_other_side(masters_offsets.back());
  std::vector<std::int32_t> owners_other_side(masters_offsets.back());
  std::span<const int> ghost_owners = imap->owners();

  // Temporary array holding global indices
  std::vector<std::int64_t> global_blocks;

  // Reuse num_masters_local for insertion
  std::ranges::fill(num_masters_local, 0);
  for (std::size_t i = 0; i < num_slaves_local; ++i)
  {
    if (const std::int32_t cell = local_colliding_cell[i]; cell != -1)
    {
      auto cell_blocks = V.dofmap()->cell_dofs(cell);
      global_blocks.resize(cell_blocks.size());
      imap->local_to_global(cell_blocks, global_blocks);
      // Compute coefficients for each master
      for (std::size_t j = 0; j < cell_blocks.size(); ++j)
      {
        const std::int32_t cell_block = cell_blocks[j];
        for (int b = 0; b < bs; b++)
        {
          // NOTE: Assuming 0 value size
          if (const T val = normals(i, b) / normals(i, local_rems[i])
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
  dolfinx_mpc::mpc_data<T> mpc_local;
  mpc_local.masters = masters_other_side;
  mpc_local.coeffs = coefficients_other_side;
  mpc_local.offsets = masters_offsets;
  mpc_local.owners = owners_other_side;
  return mpc_local;
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
    const auto max_index = std::distance(dofs.begin(), local_max);
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

/// Concatatenate to mpc_data structures with same number of offsets
template <typename T>
dolfinx_mpc::mpc_data<T> concatenate(dolfinx_mpc::mpc_data<T>& mpc0,
                                     dolfinx_mpc::mpc_data<T>& mpc1)
{

  assert(mpc0.offsets.size() == mpc1.offsets.size());
  std::vector<std::int32_t>& offsets0 = mpc0.offsets;
  std::vector<std::int32_t>& offsets1 = mpc1.offsets;
  std::vector<std::int64_t>& masters0 = mpc0.masters;
  std::vector<std::int64_t>& masters1 = mpc1.masters;
  std::vector<T>& coeffs0 = mpc0.coeffs;
  std::vector<T>& coeffs1 = mpc1.coeffs;
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
  std::ranges::fill(num_masters_per_slave, 0);
  std::vector<std::int64_t> masters_out(masters_offsets.back());
  std::vector<T> coefficients_out(masters_offsets.back());
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
  dolfinx_mpc::mpc_data<T> mpc;
  mpc.masters = masters_out;
  mpc.coeffs = coefficients_out;
  mpc.owners = owners_out;
  mpc.offsets = masters_offsets;
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
template <typename T, std::floating_point U>
mpc_data<T> create_contact_slip_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
    std::int32_t slave_marker, std::int32_t master_marker,
    const dolfinx::fem::Function<T, U>& nh, const U eps2 = 1e-20)
{

  dolfinx::common::Timer timer("~MPC: Create slip constraint");
  std::shared_ptr<const mesh::Mesh<U>> mesh = V.mesh();
  MPI_Comm comm = mesh->comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V.dofmap()->index_map;
  assert(mesh->topology() == meshtags.topology());
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V.dofmap()->index_map_bs();
  std::int32_t size_local = V.dofmap()->index_map->size_local();

  mesh->topology_mutable()->create_connectivity(fdim, tdim);
  mesh->topology_mutable()->create_connectivity(tdim, tdim);
  mesh->topology_mutable()->create_entity_permutations();

  // Find all slave dofs and split them into locally owned and ghosted blocks
  std::vector<std::int32_t> local_slave_blocks;
  {
    std::vector<std::int32_t> slave_dofs
        = impl::locate_slave_dofs<U>(V, meshtags, slave_marker);

    local_slave_blocks.reserve(slave_dofs.size());
    std::ranges::for_each(slave_dofs,
                          [&local_slave_blocks, bs = block_size,
                           sl = size_local](const std::int32_t dof)
                          {
                            std::div_t div = std::div(dof, bs);
                            if (div.quot < sl)
                              local_slave_blocks.push_back(div.quot);
                          });
  }

  // Data structures to hold information about slave data local to process
  std::vector<std::int32_t> local_slaves(local_slave_blocks.size());
  std::vector<std::int32_t> local_rems(local_slave_blocks.size());
  dolfinx_mpc::mpc_data<T> mpc_local;

  // Find all local contributions to MPC, meaning:
  // 1. Degrees of freedom from the same block as the slave
  // 2. Degrees of freedom from the other interface

  // Helper function
  // Determine component of each block has the largest normal value, and use
  // it as slave dofs to avoid zero division in constraint
  // Note that this function casts the normal array from being potentially
  // complex to real valued
  std::vector<std::int32_t> dofs(block_size);
  std::span<const T> normal_array = nh.x()->array();
  const auto largest_normal_component
      = [&dofs, block_size, &normal_array, gdim](const std::int32_t block,
                                                 std::span<U, 3> normal)
  {
    std::iota(dofs.begin(), dofs.end(), block * block_size);
    for (int j = 0; j < gdim; ++j)
      normal[j] = std::real(normal_array[dofs[j]]);
    U norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1]
                       + normal[2] * normal[2]);
    std::ranges::for_each(normal,
                          [norm](auto& n) { return std::abs(n / norm); });
    return std::distance(
        normal.begin(),
        std::ranges::max_element(normal, [](T a, T b)
                                 { return std::norm(a) < std::norm(b); }));
  };

  // Determine which dof in local slave block is the actual slave
  std::vector<U> normals(3 * local_slave_blocks.size(), 0);
  assert(block_size == gdim);
  for (std::size_t i = 0; i < local_slave_blocks.size(); ++i)
  {
    const std::int32_t slave = local_slave_blocks[i];
    const auto block = largest_normal_component(
        slave, std::span<U, 3>(std::next(normals.begin(), 3 * i), 3));
    local_slaves[i] = block_size * slave + block;
    local_rems[i] = block;
  }

  // Compute local contributions to constraint using helper function
  // i.e. compute dot(u, n) on slave side
  mpc_data<T> mpc_in_cell = impl::compute_block_contributions<T, U>(
      local_slaves, local_slave_blocks, normals, imap, block_size, rank);

  dolfinx::geometry::BoundingBoxTree bb_tree = impl::create_boundingbox_tree(
      *mesh, meshtags, master_marker, std::sqrt(eps2));

  // Compute contributions on other side local to process
  mpc_data<T> mpc_master_local;

  // Create map from slave dof blocks to a cell containing them
  std::vector<std::int32_t> slave_cells = dolfinx_mpc::create_block_to_cell_map(
      *mesh->topology(), *V.dofmap(), local_slave_blocks);
  std::vector<U> slave_coordinates;
  std::array<std::size_t, 2> coord_shape;
  {
    std::tie(slave_coordinates, coord_shape)
        = dolfinx_mpc::tabulate_dof_coordinates<U>(V, local_slave_blocks,
                                                   slave_cells);
    std::vector<std::int32_t> local_cell_collisions
        = dolfinx_mpc::find_local_collisions<U>(*mesh, bb_tree,
                                                slave_coordinates, eps2);

    auto [basis, basis_shape] = dolfinx_mpc::evaluate_basis_functions<U>(
        V, slave_coordinates, local_cell_collisions);
    assert(basis_shape.back() == 1);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        basis_span(basis.data(), basis_shape[0], basis_shape[1]);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
        normal_span(normals.data(), local_slave_blocks.size(), 3);
    mpc_master_local = impl::compute_master_contributions<T, U>(
        local_rems, local_cell_collisions, normal_span, V, basis_span);
  }
  // // Find slave indices were contributions are not found on the process
  std::vector<std::int32_t>& l_offsets = mpc_master_local.offsets;
  std::vector<std::int32_t> slave_indices_remote;
  slave_indices_remote.reserve(local_rems.size());
  for (std::size_t i = 0; i < local_rems.size(); i++)
  {
    if (l_offsets[i + 1] - l_offsets[i] == 0)
      slave_indices_remote.push_back((int)i);
  }

  // Structure storing mpc arrays mpc_local
  mpc_local = impl::concatenate(mpc_in_cell, mpc_master_local);

  // If serial, we gather the resulting mpc data as one constraint
  if (int mpi_size = dolfinx::MPI::size(comm); mpi_size == 1)
  {
    if (!slave_indices_remote.empty())
    {
      throw std::runtime_error(
          "No masters found on contact surface (when executed in serial). "
          "Please make sure that the surfaces are in contact, or increase the "
          "tolerance eps2.");
    }
    // Serial assumptions
    mpc_local.slaves = local_slaves;
    return mpc_local;
  }

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = !local_slave_blocks.empty();
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(comm, meshtags, has_slave, master_marker);

  // Get the  slave->master recv from and send to ranks
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);

  // Convert slaves missing master contributions to global index
  // and prepare data (coordinates and normals) to send to other procs
  const std::array<std::size_t, 2> send_shape
      = {slave_indices_remote.size(), 3};
  std::vector<U> coordinates_send(send_shape.front() * send_shape.back());
  std::vector<U> normals_send(send_shape.front() * send_shape.back());
  std::vector<std::int32_t> send_rems(slave_indices_remote.size());
  for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
  {
    const std::int32_t slave_idx = slave_indices_remote[i];
    send_rems[i] = local_rems[slave_idx];
    std::ranges::copy_n(std::next(slave_coordinates.begin(), 3 * slave_idx), 3,
                        std::next(coordinates_send.begin(), 3 * i));
    std::ranges::copy_n(std::next(normals.begin(), 3 * slave_idx), 3,
                        std::next(normals_send.begin(), 3 * i));
  }

  // Figure out how much data to receive from each neighbor
  const std::size_t out_collision_slaves = slave_indices_remote.size();
  std::vector<std::int32_t> num_slaves_recv(indegree + 1);
  MPI_Neighbor_allgather(
      &out_collision_slaves, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_slaves_recv.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[0]);
  num_slaves_recv.pop_back();

  // Compute displacements for data to receive
  std::vector<int> disp(indegree + 1, 0);
  std::partial_sum(num_slaves_recv.begin(), num_slaves_recv.end(),
                   disp.begin() + 1);

  // Send data to neighbors and receive data
  std::vector<std::int32_t> recv_rems(disp.back());
  MPI_Neighbor_allgatherv(send_rems.data(), (int)send_rems.size(),
                          dolfinx::MPI::mpi_type<std::int32_t>(),
                          recv_rems.data(), num_slaves_recv.data(), disp.data(),
                          dolfinx::MPI::mpi_type<std::int32_t>(),
                          neighborhood_comms[0]);

  // Multiply recv size by three to accommodate vector coordinates and
  // function data
  std::vector<std::int32_t> num_slaves_recv3;
  num_slaves_recv3.reserve(indegree);
  std::ranges::transform(num_slaves_recv, std::back_inserter(num_slaves_recv3),
                         [](std::int32_t num_slaves)
                         { return 3 * num_slaves; });
  std::vector<int> disp3(indegree + 1, 0);
  std::partial_sum(num_slaves_recv3.begin(), num_slaves_recv3.end(),
                   disp3.begin() + 1);

  // Send slave normal and coordinate to neighbors
  std::vector<U> recv_coords(disp.back() * 3);
  MPI_Neighbor_allgatherv(coordinates_send.data(), (int)coordinates_send.size(),
                          dolfinx::MPI::mpi_type<U>(), recv_coords.data(),
                          num_slaves_recv3.data(), disp3.data(),
                          dolfinx::MPI::mpi_type<U>(), neighborhood_comms[0]);
  std::vector<U> slave_normals(disp.back() * 3);
  MPI_Neighbor_allgatherv(normals_send.data(), (int)normals_send.size(),
                          dolfinx::MPI::mpi_type<U>(), slave_normals.data(),
                          num_slaves_recv3.data(), disp3.data(),
                          dolfinx::MPI::mpi_type<U>(), neighborhood_comms[0]);

  int err0 = MPI_Comm_free(&neighborhood_comms[0]);
  dolfinx::MPI::check_error(comm, err0);

  // Compute off-process contributions
  mpc_data<T> remote_data;
  {
    std::vector<std::int32_t> remote_cell_collisions
        = dolfinx_mpc::find_local_collisions<U>(*mesh, bb_tree, recv_coords,
                                                eps2);
    auto [recv_basis_values, shape] = dolfinx_mpc::evaluate_basis_functions<U>(
        V, recv_coords, remote_cell_collisions);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        basis_span(recv_basis_values.data(), shape[0], shape[1]);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
        normal_span(slave_normals.data(), disp.back(), 3);
    remote_data = impl::compute_master_contributions<T, U>(
        recv_rems, remote_cell_collisions, normal_span, V, basis_span);
  }

  // Get info about reverse communicator
  auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx_mpc::compute_neighborhood(neighborhood_comms[1]);
  const std::size_t indegree_rev = src_ranks_rev.size();

  // Count number of masters found on the process and convert the offsets
  // to be per process
  std::vector<std::int32_t> num_collision_masters(indegree + 1, 0);
  std::vector<int> num_out_offsets;
  num_out_offsets.reserve(indegree);
  std::ranges::transform(num_slaves_recv, std::back_inserter(num_out_offsets),
                         [](std::int32_t num_slaves)
                         { return num_slaves + 1; });
  const std::int32_t num_offsets
      = std::accumulate(num_out_offsets.begin(), num_out_offsets.end(), 0);
  std::vector<std::int32_t> offsets_remote(num_offsets);
  std::int32_t counter = 0;
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    const std::int32_t first_pos = disp[i];
    const std::int32_t first_offset = remote_data.offsets[first_pos];
    num_collision_masters[i] += remote_data.offsets[disp[i + 1]] - first_offset;
    offsets_remote[first_pos + counter++] = 0;
    for (std::int32_t j = first_pos; j < disp[i + 1]; ++j)
      offsets_remote[j + counter] = remote_data.offsets[j + 1] - first_offset;
  }

  // Communicate number of incoming masters to each process after collision
  // detection
  std::vector<int> inc_num_collision_masters(indegree_rev + 1);
  MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
                        inc_num_collision_masters.data(), 1, MPI_INT,
                        neighborhood_comms[1]);
  inc_num_collision_masters.pop_back();
  num_collision_masters.pop_back();

  // Create displacement vector for masters and coefficients
  std::vector<int> disp_inc_masters(indegree_rev + 1, 0);
  std::partial_sum(inc_num_collision_masters.begin(),
                   inc_num_collision_masters.end(),
                   disp_inc_masters.begin() + 1);

  // Compute send offsets for masters and coefficients
  std::vector<int> send_disp_masters(indegree + 1, 0);
  std::partial_sum(num_collision_masters.begin(), num_collision_masters.end(),
                   send_disp_masters.begin() + 1);

  // Create displacement vector for incoming offsets
  std::vector<int> inc_disp_offsets(indegree_rev + 1);
  std::vector<int> num_inc_offsets(indegree_rev,
                                   (int)slave_indices_remote.size() + 1);
  std::partial_sum(num_inc_offsets.begin(), num_inc_offsets.end(),
                   inc_disp_offsets.begin() + 1);

  // Compute send offsets for master offsets
  std::vector<int> send_disp_offsets(indegree + 1, 0);

  std::partial_sum(num_out_offsets.begin(), num_out_offsets.end(),
                   send_disp_offsets.begin() + 1);

  // Get offsets for master dofs from remote process
  std::vector<MPI_Request> requests(4);
  std::vector<std::int32_t> remote_colliding_offsets(inc_disp_offsets.back());
  MPI_Ineighbor_alltoallv(
      offsets_remote.data(), num_out_offsets.data(), send_disp_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), remote_colliding_offsets.data(),
      num_inc_offsets.data(), inc_disp_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), neighborhood_comms[1],
      &requests[0]);
  // Receive colliding masters and relevant data from other processor
  std::vector<std::int64_t> remote_colliding_masters(disp_inc_masters.back());
  MPI_Ineighbor_alltoallv(
      remote_data.masters.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      remote_colliding_masters.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      neighborhood_comms[1], &requests[1]);
  std::vector<T> remote_colliding_coeffs(disp_inc_masters.back());
  MPI_Ineighbor_alltoallv(
      remote_data.coeffs.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<T>(),
      remote_colliding_coeffs.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<T>(),
      neighborhood_comms[1], &requests[2]);
  std::vector<std::int32_t> remote_colliding_owners(disp_inc_masters.back());
  MPI_Ineighbor_alltoallv(
      remote_data.owners.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      remote_colliding_owners.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1], &requests[3]);

  // Wait for offsets to be sent
  std::vector<MPI_Status> status(4);
  MPI_Wait(&requests[0], &status[0]);

  std::vector<std::int8_t> slave_found(slave_indices_remote.size(), false);
  std::vector<std::int32_t> num_inc_masters(slave_indices_remote.size());
  // Iterate through the processors and find one set of inputs per slave that
  // was sent to the other processes
  for (std::size_t i = 0; i < src_ranks_rev.size(); ++i)
  {
    [[maybe_unused]] const std::int32_t num_offsets_on_proc
        = inc_disp_offsets[i + 1] - inc_disp_offsets[i];
    assert(num_offsets_on_proc
           == std::int32_t(slave_indices_remote.size()) + 1);
    for (std::size_t c = 0; c < slave_indices_remote.size(); c++)

    {
      const std::int32_t slave_min
          = remote_colliding_offsets[inc_disp_offsets[i] + c];
      const std::int32_t slave_max
          = remote_colliding_offsets[inc_disp_offsets[i] + c + 1];
      if (const std::int32_t num_inc = slave_max - slave_min;
          !(slave_found[c]) && (num_inc > 0))
      {
        slave_found[c] = true;
        num_inc_masters[c] = num_inc;
      }
    }
  }
  if (auto not_found = std::ranges::find(slave_found, false);
      not_found != slave_found.end())
  {
    std::runtime_error(
        "Masters not found on contact surface with local search or remote "
        "search. Consider running the code in serial to make sure that one can "
        "detect the contact surface, or increase eps2.");
  }

  /// Wait for all communication to finish
  MPI_Waitall(4, requests.data(), status.data());

  int err1 = MPI_Comm_free(&neighborhood_comms[1]);
  dolfinx::MPI::check_error(comm, err1);

  // Move the masters, coeffs and owners from the input adjacency list
  // to one where each node corresponds to an entry in slave_indices_remote
  std::vector<std::int32_t> offproc_offsets(slave_indices_remote.size() + 1, 0);
  std::partial_sum(num_inc_masters.begin(), num_inc_masters.end(),
                   offproc_offsets.begin() + 1);
  std::vector<std::int64_t> offproc_masters(offproc_offsets.back());
  std::vector<T> offproc_coeffs(offproc_offsets.back());
  std::vector<std::int32_t> offproc_owners(offproc_offsets.back());

  std::ranges::fill(slave_found, false);
  for (std::size_t i = 0; i < src_ranks_rev.size(); ++i)
  {
    const std::int32_t proc_start = disp_inc_masters[i];
    [[maybe_unused]] const std::int32_t num_offsets_on_proc
        = inc_disp_offsets[i + 1] - inc_disp_offsets[i];
    assert(num_offsets_on_proc
           == std::int32_t(slave_indices_remote.size()) + 1);
    for (std::size_t c = 0; c < slave_indices_remote.size(); c++)
    {
      assert(std::int32_t(remote_colliding_offsets.size())
             > std::int32_t(inc_disp_offsets[i] + c));
      assert(std::int32_t(remote_colliding_offsets.size())
             > std::int32_t(inc_disp_offsets[i] + c + 1));
      assert(std::int32_t(inc_disp_offsets[i] + c) < inc_disp_offsets[i + 1]);
      const std::int32_t slave_min
          = remote_colliding_offsets[inc_disp_offsets[i] + c];
      const std::int32_t slave_max
          = remote_colliding_offsets[inc_disp_offsets[i] + c + 1];

      assert(c < slave_found.size());
      if (!(slave_found[c]) && (slave_max - slave_min > 0))
      {
        slave_found[c] = true;
        std::ranges::copy(
            remote_colliding_masters.begin() + proc_start + slave_min,
            remote_colliding_masters.begin() + proc_start + slave_max,
            offproc_masters.begin() + offproc_offsets[c]);

        std::ranges::copy(
            remote_colliding_coeffs.begin() + proc_start + slave_min,
            remote_colliding_coeffs.begin() + proc_start + slave_max,
            offproc_coeffs.begin() + offproc_offsets[c]);

        std::ranges::copy(
            remote_colliding_owners.begin() + proc_start + slave_min,
            remote_colliding_owners.begin() + proc_start + slave_max,
            offproc_owners.begin() + offproc_offsets[c]);
      }
    }
  }
  // Merge local data with incoming data
  // First count number of local masters
  std::vector<std::int32_t>& masters_offsets = mpc_local.offsets;
  std::vector<std::int64_t>& masters_out = mpc_local.masters;
  std::vector<T>& coefficients_out = mpc_local.coeffs;
  std::vector<std::int32_t>& owners_out = mpc_local.owners;

  std::vector<std::int32_t> num_masters_per_slave(local_slaves.size(), 0);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
    num_masters_per_slave[i] += masters_offsets[i + 1] - masters_offsets[i];
  // Then add the remote masters
  for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
    num_masters_per_slave[slave_indices_remote[i]]
        += offproc_offsets[i + 1] - offproc_offsets[i];

  // Create new offset array
  std::vector<std::int32_t> local_offsets(local_slaves.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   local_offsets.begin() + 1);

  // Reuse num_masters_per_slave for input indices
  std::vector<std::int64_t> local_masters(local_offsets.back());
  std::vector<std::int32_t> local_owners(local_offsets.back());
  std::vector<T> local_coeffs(local_offsets.back());

  // Insert local contributions
  {
    std::vector<std::int32_t> loc_pos(local_slaves.size(), 0);
    for (std::size_t i = 0; i < local_slaves.size(); ++i)
    {
      const std::int32_t master_min = masters_offsets[i];
      const std::int32_t master_max = masters_offsets[i + 1];
      std::ranges::copy(masters_out.begin() + master_min,
                        masters_out.begin() + master_max,
                        local_masters.begin() + local_offsets[i] + loc_pos[i]);
      std::ranges::copy(coefficients_out.begin() + master_min,
                        coefficients_out.begin() + master_max,
                        local_coeffs.begin() + local_offsets[i] + loc_pos[i]);
      std::ranges::copy(owners_out.begin() + master_min,
                        owners_out.begin() + master_max,
                        local_owners.begin() + local_offsets[i] + loc_pos[i]);
      loc_pos[i] += master_max - master_min;
    }

    // Insert remote contributions
    for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
    {
      const std::int32_t master_min = offproc_offsets[i];
      const std::int32_t master_max = offproc_offsets[i + 1];
      const std::int32_t slave_index = slave_indices_remote[i];
      std::ranges::copy(offproc_masters.begin() + master_min,
                        offproc_masters.begin() + master_max,
                        local_masters.begin() + local_offsets[slave_index]
                            + loc_pos[slave_index]);
      std::ranges::copy(offproc_coeffs.begin() + master_min,
                        offproc_coeffs.begin() + master_max,
                        local_coeffs.begin() + local_offsets[slave_index]
                            + loc_pos[slave_index]);
      std::ranges::copy(offproc_owners.begin() + master_min,
                        offproc_owners.begin() + master_max,
                        local_owners.begin() + local_offsets[slave_index]
                            + loc_pos[slave_index]);
      loc_pos[slave_index] += master_max - master_min;
    }
  }
  // Distribute ghost data
  dolfinx_mpc::mpc_data ghost_data = dolfinx_mpc::distribute_ghost_data<T>(
      local_slaves, local_masters, local_coeffs, local_owners,
      num_masters_per_slave, *imap, block_size);

  // Add ghost data to existing arrays
  const std::vector<std::int32_t>& ghost_slaves = ghost_data.slaves;
  local_slaves.insert(std::end(local_slaves), std::cbegin(ghost_slaves),
                      std::cend(ghost_slaves));
  const std::vector<std::int64_t>& ghost_masters = ghost_data.masters;
  local_masters.insert(std::end(local_masters), std::cbegin(ghost_masters),
                       std::cend(ghost_masters));
  const std::vector<std::int32_t>& ghost_num = ghost_data.offsets;
  num_masters_per_slave.insert(std::end(num_masters_per_slave),
                               std::cbegin(ghost_num), std::cend(ghost_num));
  const std::vector<T>& ghost_coeffs = ghost_data.coeffs;
  local_coeffs.insert(std::end(local_coeffs), std::cbegin(ghost_coeffs),
                      std::cend(ghost_coeffs));
  const std::vector<std::int32_t>& ghost_owner_ranks = ghost_data.owners;
  local_owners.insert(std::end(local_owners), std::cbegin(ghost_owner_ranks),
                      std::cend(ghost_owner_ranks));

  // Compute offsets
  std::vector<std::int32_t> offsets(num_masters_per_slave.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   offsets.begin() + 1);

  dolfinx_mpc::mpc_data<T> output;
  output.offsets = offsets;
  output.masters = local_masters;
  output.coeffs = local_coeffs;
  output.owners = local_owners;
  output.slaves = local_slaves;
  return output;
}

/// Create a contact condition between two sets of facets
/// @param[in] The mpc function space
/// @param[in] meshtags The meshtag
/// @param[in] slave_marker Tag for the first interface
/// @param[in] master_marker Tag for the other interface
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
/// @param[in] allow_missing_masters If true, the function will not throw an error if
/// a degree of freedom in the closure of the master entities does not have a corresponding set of slave degrees of freedom.
template <typename T, std::floating_point U>
mpc_data<T> create_contact_inelastic_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker, const U eps2 = 1e-20, bool allow_missing_masters = false)
{
  dolfinx::common::Timer timer("~MPC: Inelastic condition");

  MPI_Comm comm = V.mesh()->comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V.dofmap()->index_map;
  const int tdim = V.mesh()->topology()->dim();
  const int fdim = tdim - 1;
  const int block_size = V.dofmap()->index_map_bs();
  std::int32_t size_local = V.dofmap()->index_map->size_local();

  // Create entity permutations needed in evaluate_basis_functions
  V.mesh()->topology_mutable()->create_entity_permutations();
  // Create connectivities needed for evaluate_basis_functions and
  // select_colliding cells
  V.mesh()->topology_mutable()->create_connectivity(fdim, tdim);
  V.mesh()->topology_mutable()->create_connectivity(tdim, tdim);

  std::vector<std::int32_t> slave_blocks
      = impl::locate_slave_dofs<U>(V, meshtags, slave_marker);
  std::ranges::for_each(slave_blocks,
                        [block_size](std::int32_t& d) { d /= block_size; });

  // Vector holding what blocks local to process are slaves
  std::vector<std::int32_t> local_blocks;

  // Array holding ghost slaves blocks (masters,coeffs and offsets will be
  // received)
  std::vector<std::int32_t> ghost_blocks;

  // Map slave blocks to arrays holding local bocks and ghost blocks
  std::ranges::for_each(
      slave_blocks,
      [size_local, &local_blocks, &ghost_blocks](const std::int32_t block)
      {
        if (block < size_local)
          local_blocks.push_back(block);
        else
          ghost_blocks.push_back(block);
      });

  // Map local blocks to global indices
  std::vector<std::int64_t> local_blocks_as_glob(local_blocks.size());
  imap->local_to_global(local_blocks, local_blocks_as_glob);

  // Create slave_dofs->master facets and master->slave dofs neighborhood
  // comms
  const bool has_slave = !local_blocks.empty();
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(comm, meshtags, has_slave, master_marker);

  /// Compute which rank (relative to neighbourhood) to send each ghost to
  std::span<const int> ghost_owners = imap->owners();

  // Create new index-map where there are only ghosts for slaves
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    std::vector<int> slave_ranks(ghost_blocks.size());
    for (std::size_t i = 0; i < ghost_blocks.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_blocks[i] - size_local];
    std::vector<std::int64_t> ghosts_as_global(ghost_blocks.size());
    imap->local_to_global(ghost_blocks, ghosts_as_global);
    slave_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, imap->size_local(), ghosts_as_global, slave_ranks);
  }

  // Create boundingboxtree for master surface
  auto facet_to_cell = V.mesh()->topology()->connectivity(fdim, tdim);
  assert(facet_to_cell);
  dolfinx::geometry::BoundingBoxTree<U> bb_tree = impl::create_boundingbox_tree(
      *V.mesh(), meshtags, master_marker, std::sqrt(eps2));

  // Tabulate slave block coordinates and find colliding cells
  std::vector<std::int32_t> slave_cells = dolfinx_mpc::create_block_to_cell_map(
      *V.mesh()->topology(), *V.dofmap(), local_blocks);
  std::vector<U> slave_coordinates;
  {
    std::array<std::size_t, 2> c_shape;
    std::tie(slave_coordinates, c_shape)
        = dolfinx_mpc::tabulate_dof_coordinates<U>(V, local_blocks,
                                                   slave_cells);
  }

  // Loop through all masters on current processor and check if they
  // collide with a local master facet
  std::map<std::int32_t, std::vector<std::int32_t>> local_owners;
  std::map<std::int32_t, std::vector<std::int64_t>> local_masters;
  std::map<std::int32_t, std::vector<T>> local_coeffs;
  std::vector<std::int64_t> blocks_wo_local_collision;
  std::vector<size_t> collision_to_local;
  {
    std::vector<std::int32_t> colliding_cells
        = dolfinx_mpc::find_local_collisions<U>(*V.mesh(), bb_tree,
                                                slave_coordinates, eps2);
    auto [basis_values, basis_shape] = dolfinx_mpc::evaluate_basis_functions<U>(
        V, slave_coordinates, colliding_cells);
    assert(basis_shape.back() == 1);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        basis(basis_values.data(), basis_shape[0], basis_shape[1]);
    std::vector<std::int64_t> master_block_global;
    std::vector<std::int32_t> l_master;
    std::vector<T> coeff;

    for (std::size_t i = 0; i < local_blocks.size(); ++i)
    {
      if (const auto& cell = colliding_cells[i]; cell != -1)
      {
        auto cell_blocks = V.dofmap()->cell_dofs(cell);
        l_master.reserve(cell_blocks.size());
        coeff.reserve(cell_blocks.size());
        assert(l_master.empty());
        assert(coeff.empty());
        // Store block and non-zero basis value
        for (std::size_t j = 0; j < cell_blocks.size(); ++j)
        {
          if (const T c = basis(i, j); std::abs(c) > 1e-6)
          {
            coeff.push_back(c);
            l_master.push_back(cell_blocks[j]);
          }
        }
        // If no local contributions found add to remote list
        if (l_master.empty())
        {
          blocks_wo_local_collision.push_back(local_blocks_as_glob[i]);
          collision_to_local.push_back(i);
        }
        else
        {

          // Convert local block to global block
          const std::size_t num_masters = l_master.size();
          master_block_global.resize(num_masters);
          imap->local_to_global(l_master, master_block_global);

          // Insert local contributions in each block
          for (int j = 0; j < block_size; ++j)
          {
            const std::int32_t local_slave = local_blocks[i] * block_size + j;
            for (std::size_t k = 0; k < num_masters; k++)
            {
              local_masters[local_slave].push_back(
                  master_block_global[k] * block_size + j);
              local_coeffs[local_slave].push_back(coeff[k]);
              local_owners[local_slave].push_back(
                  l_master[k] < size_local
                      ? rank
                      : ghost_owners[l_master[k] - size_local]);
            }
          }
          l_master.clear();
          coeff.clear();
        }
      }
      else
      {
        blocks_wo_local_collision.push_back(local_blocks_as_glob[i]);
        collision_to_local.push_back(i);
      }
    }
  }
  // Extract coordinates and normals to distribute
  std::vector<U> distribute_coordinates(blocks_wo_local_collision.size() * 3);
  for (std::size_t i = 0; i < collision_to_local.size(); ++i)
  {
    std::ranges::copy_n(
        std::next(slave_coordinates.begin(), 3 * collision_to_local[i]), 3,
        std::next(distribute_coordinates.begin(), 3 * i));
  }
  dolfinx_mpc::mpc_data<T> mpc;

  // If serial, we only have to gather slaves, masters, coeffs in 1D
  // arrays
  if (int mpi_size = dolfinx::MPI::size(comm); mpi_size == 1)
  {

    if ((!blocks_wo_local_collision.empty()) and (!allow_missing_masters))
    {
      throw std::runtime_error(
          "No masters found on contact surface (when executed in serial). "
          "Please make sure that the surfaces are in contact, or increase "
          "the "
          "tolerance eps2.");
    }

    std::vector<std::int64_t> masters_out;
    std::vector<T> coeffs_out;
    std::vector<std::int32_t> offsets_out = {0};
    std::vector<std::int32_t> slaves_out;
    // Flatten the maps to 1D arrays (assuming all slaves are local
    // slaves)
    std::vector<std::int32_t> slaves;
    slaves.reserve(block_size * local_blocks.size());
    // If we allow missing masters, we need to skip those slaves from the MPC
    if ((!blocks_wo_local_collision.empty()))
    {
      int counter = 0;
      std::ranges::for_each(
        local_blocks,
        [block_size, tdim, &masters_out, &local_masters, &coeffs_out,
         &local_coeffs, &offsets_out, &slaves, &counter, &collision_to_local](const std::int32_t block)
        {
          if (std::find(collision_to_local.begin(), collision_to_local.end(), counter++) != collision_to_local.end())
          {
            std::cout << "Skipping block " << block << " without local collision." << counter << std::endl;
            return; // Skip blocks without local collision
          }
          else
          {

          for (std::int32_t j = 0; j < block_size; ++j)
          {
            const std::int32_t slave = block * block_size + j;
            slaves.push_back(slave);
            masters_out.insert(masters_out.end(), local_masters[slave].begin(),
                               local_masters[slave].end());
            coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                              local_coeffs[slave].end());
            offsets_out.push_back((std::int32_t)masters_out.size());
          }
        }
        });

    }
    else
    {
      std::ranges::for_each(
        local_blocks,
        [block_size, tdim, &masters_out, &local_masters, &coeffs_out,
         &local_coeffs, &offsets_out, &slaves](const std::int32_t block)
        {
          for (std::int32_t j = 0; j < block_size; ++j)
          {
            const std::int32_t slave = block * block_size + j;
            slaves.push_back(slave);
            masters_out.insert(masters_out.end(), local_masters[slave].begin(),
                               local_masters[slave].end());
            coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                              local_coeffs[slave].end());
            offsets_out.push_back((std::int32_t)masters_out.size());
          }
        });
      }
    std::vector<std::int32_t> owners(masters_out.size());
    std::ranges::fill(owners, 0);
    mpc.slaves = slaves;
    mpc.masters = masters_out;
    mpc.offsets = offsets_out;
    mpc.owners = owners;
    mpc.coeffs = coeffs_out;
    return mpc;
  }

  // Get the slave->master recv from and send to ranks
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);

  // Figure out how much data to receive from each neighbor
  const auto num_colliding_blocks = (int)blocks_wo_local_collision.size();
  std::vector<std::int32_t> num_slave_blocks(indegree + 1);
  MPI_Neighbor_allgather(
      &num_colliding_blocks, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_slave_blocks.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[0]);
  num_slave_blocks.pop_back();

  // Compute displacements for data to receive
  std::vector<int> disp(indegree + 1, 0);
  std::partial_sum(num_slave_blocks.begin(), num_slave_blocks.end(),
                   disp.begin() + 1);

  // Send data to neighbors and receive data
  std::vector<std::int64_t> remote_slave_blocks(disp.back());
  MPI_Neighbor_allgatherv(
      blocks_wo_local_collision.data(), num_colliding_blocks,
      dolfinx::MPI::mpi_type<std::int64_t>(), remote_slave_blocks.data(),
      num_slave_blocks.data(), disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[0]);

  // Multiply recv size by three to accommodate block coordinates
  std::vector<std::int32_t> num_block_coordinates(indegree);
  for (std::size_t i = 0; i < num_slave_blocks.size(); ++i)
    num_block_coordinates[i] = num_slave_blocks[i] * 3;
  std::vector<int> coordinate_disp(indegree + 1, 0);
  std::partial_sum(num_block_coordinates.begin(), num_block_coordinates.end(),
                   coordinate_disp.begin() + 1);

  // Send slave coordinates to neighbors
  std::vector<U> recv_coords(disp.back() * 3);
  MPI_Neighbor_allgatherv(distribute_coordinates.data(),
                          (int)distribute_coordinates.size(),
                          dolfinx::MPI::mpi_type<U>(), recv_coords.data(),
                          num_block_coordinates.data(), coordinate_disp.data(),
                          dolfinx::MPI::mpi_type<U>(), neighborhood_comms[0]);

  int err0 = MPI_Comm_free(&neighborhood_comms[0]);
  dolfinx::MPI::check_error(comm, err0);

  // Vector for processes with slaves, mapping slaves with
  // collision on this process
  std::vector<std::vector<std::int64_t>> collision_slaves(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int64_t>>>
      collision_masters(indegree);
  std::vector<std::map<std::int64_t, std::vector<T>>> collision_coeffs(
      indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
      collision_owners(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
      collision_block_offsets(indegree);
  {
    std::vector<std::int32_t> remote_cell_collisions
        = dolfinx_mpc::find_local_collisions<U>(*V.mesh(), bb_tree, recv_coords,
                                                eps2);
    auto [basis, basis_shape] = dolfinx_mpc::evaluate_basis_functions<U>(
        V, recv_coords, remote_cell_collisions);
    assert(basis_shape.back() == 1);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        basis_span(basis.data(), basis_shape[0], basis_shape[1]);
    // TODO: Rework this so it is the same as the code on owning process.
    // Preferably get rid of all the std::map's
    // Work arrays for loop
    std::vector<std::int32_t> r_master;
    std::vector<T> r_coeff;
    std::vector<std::int64_t> remote_master_global;
    for (std::int32_t i = 0; i < indegree; ++i)
    {
      for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
      {
        const std::int64_t slave_loc = remote_slave_blocks[j];
        // Initialize number of masters for each incoming slave to 0
        collision_block_offsets[i][remote_slave_blocks[j]]
            = std::vector<std::int32_t>(tdim, 0);
        if (const auto& cell = remote_cell_collisions[j]; cell != -1)
        {
          auto cell_blocks = V.dofmap()->cell_dofs(cell);
          r_master.reserve(cell_blocks.size());
          r_coeff.reserve(cell_blocks.size());
          assert(r_master.empty());
          assert(r_coeff.empty());
          // Store block and non-zero basis values
          for (std::size_t k = 0; k < cell_blocks.size(); ++k)
          {
            if (const T c = basis_span(j, k); std::abs(c) > 1e-6)
            {
              r_coeff.push_back(c);
              r_master.push_back(cell_blocks[k]);
            }
          }
          // If no local contributions do nothing
          if (!r_master.empty())
          {
            const std::size_t num_masters = r_master.size();
            remote_master_global.resize(num_masters);
            imap->local_to_global(r_master, remote_master_global);
            // Insert local contributions in each block
            assert(block_size == tdim);
            for (int l = 0; l < tdim; ++l)
            {
              for (std::size_t k = 0; k < num_masters; k++)
              {
                collision_masters[i][slave_loc].push_back(
                    remote_master_global[k] * block_size + l);
                collision_coeffs[i][slave_loc].push_back(r_coeff[k]);
                collision_owners[i][slave_loc].push_back(
                    r_master[k] < size_local
                        ? rank
                        : ghost_owners[r_master[k] - size_local]);
                collision_block_offsets[i][slave_loc][l]++;
              }
            }
            r_master.clear();
            r_coeff.clear();
            collision_slaves[i].push_back(slave_loc);
          }
        }
      }
    }
  }

  // Flatten data structures before send/recv
  std::vector<std::int32_t> num_found_slave_blocks(indegree + 1);
  for (std::int32_t i = 0; i < indegree; ++i)
    num_found_slave_blocks[i] = (std::int32_t)collision_slaves[i].size();

  // Get info about reverse communicator (masters->slaves)
  auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx_mpc::compute_neighborhood(neighborhood_comms[1]);
  const std::size_t indegree_rev = src_ranks_rev.size();

  // Communicate number of incoming slaves and masters after coll
  // detection
  std::vector<int> inc_num_found_slave_blocks(indegree_rev + 1);
  MPI_Neighbor_alltoall(num_found_slave_blocks.data(), 1, MPI_INT,
                        inc_num_found_slave_blocks.data(), 1, MPI_INT,
                        neighborhood_comms[1]);
  inc_num_found_slave_blocks.pop_back();
  num_found_slave_blocks.pop_back();
  std::vector<std::int32_t> num_collision_masters(indegree + 1);
  std::vector<std::int64_t> found_slave_blocks;
  std::vector<std::int64_t> found_masters;
  std::vector<std::int32_t> offset_for_blocks;
  std::vector<std::int32_t> offsets_in_blocks;
  std::vector<std::int32_t> found_owners;
  std::vector<T> found_coefficients;
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    std::int32_t master_offset = 0;
    std::vector<std::int64_t>& slaves_i = collision_slaves[i];
    found_slave_blocks.insert(found_slave_blocks.end(), slaves_i.begin(),
                              slaves_i.end());
    for (auto slave : slaves_i)
    {
      std::vector<std::int64_t>& masters_ij = collision_masters[i][slave];
      num_collision_masters[i] += masters_ij.size();
      found_masters.insert(found_masters.end(), masters_ij.begin(),
                           masters_ij.end());

      std::vector<T>& coeffs_ij = collision_coeffs[i][slave];
      found_coefficients.insert(found_coefficients.end(), coeffs_ij.begin(),
                                coeffs_ij.end());

      std::vector<std::int32_t>& owners_ij = collision_owners[i][slave];
      found_owners.insert(found_owners.end(), owners_ij.begin(),
                          owners_ij.end());
      std::vector<std::int32_t>& blocks_ij = collision_block_offsets[i][slave];
      offsets_in_blocks.insert(offsets_in_blocks.end(), blocks_ij.begin(),
                               blocks_ij.end());
      master_offset += masters_ij.size();
      offset_for_blocks.push_back(master_offset);
    }
  }

  std::vector<int> num_inc_masters(indegree_rev + 1);
  MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
                        num_inc_masters.data(), 1, MPI_INT,
                        neighborhood_comms[1]);
  num_inc_masters.pop_back();
  num_collision_masters.pop_back();
  // Create displacement vector for slaves and masters
  std::vector<int> disp_inc_slave_blocks(indegree_rev + 1, 0);
  std::partial_sum(inc_num_found_slave_blocks.begin(),
                   inc_num_found_slave_blocks.end(),
                   disp_inc_slave_blocks.begin() + 1);
  std::vector<int> disp_inc_masters(indegree_rev + 1, 0);
  std::partial_sum(num_inc_masters.begin(), num_inc_masters.end(),
                   disp_inc_masters.begin() + 1);

  // Compute send offsets
  std::vector<int> send_disp_slave_blocks(indegree + 1, 0);
  std::partial_sum(num_found_slave_blocks.begin(), num_found_slave_blocks.end(),
                   send_disp_slave_blocks.begin() + 1);
  std::vector<int> send_disp_masters(indegree + 1, 0);
  std::partial_sum(num_collision_masters.begin(), num_collision_masters.end(),
                   send_disp_masters.begin() + 1);

  // Receive colliding blocks from other processor
  std::vector<std::int64_t> remote_colliding_blocks(
      disp_inc_slave_blocks.back());
  MPI_Neighbor_alltoallv(
      found_slave_blocks.data(), num_found_slave_blocks.data(),
      send_disp_slave_blocks.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      remote_colliding_blocks.data(), inc_num_found_slave_blocks.data(),
      disp_inc_slave_blocks.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      neighborhood_comms[1]);
  std::vector<std::int32_t> recv_blocks_as_local(
      remote_colliding_blocks.size());
  imap->global_to_local(remote_colliding_blocks, recv_blocks_as_local);
  std::vector<std::int32_t> remote_colliding_offsets(
      disp_inc_slave_blocks.back());
  MPI_Neighbor_alltoallv(
      offset_for_blocks.data(), num_found_slave_blocks.data(),
      send_disp_slave_blocks.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      remote_colliding_offsets.data(), inc_num_found_slave_blocks.data(),
      disp_inc_slave_blocks.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);
  // Receive colliding masters and relevant data from other processor
  std::vector<std::int64_t> remote_colliding_masters(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      found_masters.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      remote_colliding_masters.data(), num_inc_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      neighborhood_comms[1]);
  std::vector<T> remote_colliding_coeffs(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(found_coefficients.data(),
                         num_collision_masters.data(), send_disp_masters.data(),
                         dolfinx::MPI::mpi_type<T>(),
                         remote_colliding_coeffs.data(), num_inc_masters.data(),
                         disp_inc_masters.data(), dolfinx::MPI::mpi_type<T>(),
                         neighborhood_comms[1]);
  std::vector<std::int32_t> remote_colliding_owners(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      found_owners.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      remote_colliding_owners.data(), num_inc_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);

  // Create receive displacement of data per slave block
  std::vector<std::int32_t> recv_num_found_blocks(indegree_rev);
  for (std::size_t i = 0; i < recv_num_found_blocks.size(); ++i)
    recv_num_found_blocks[i] = inc_num_found_slave_blocks[i] * tdim;
  std::vector<int> inc_block_disp(indegree_rev + 1, 0);
  std::partial_sum(recv_num_found_blocks.begin(), recv_num_found_blocks.end(),
                   inc_block_disp.begin() + 1);
  // Create send displacement of data per slave block
  std::vector<std::int32_t> num_found_blocks(indegree);
  for (std::int32_t i = 0; i < indegree; ++i)
    num_found_blocks[i] = tdim * num_found_slave_blocks[i];
  std::vector<int> send_block_disp(indegree + 1, 0);
  std::partial_sum(num_found_blocks.begin(), num_found_blocks.end(),
                   send_block_disp.begin() + 1);
  // Send the block information to slave processor
  std::vector<std::int32_t> block_dofs_recv(inc_block_disp.back());
  MPI_Neighbor_alltoallv(
      offsets_in_blocks.data(), num_found_blocks.data(), send_block_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), block_dofs_recv.data(),
      recv_num_found_blocks.data(), inc_block_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), neighborhood_comms[1]);

  int err1 = MPI_Comm_free(&neighborhood_comms[1]);
  dolfinx::MPI::check_error(comm, err1);

  // Iterate through the processors
  for (std::size_t i = 0; i < src_ranks_rev.size(); ++i)
  {
    // Find offsets for masters on given proc
    std::vector<std::int32_t> master_offsets = {0};
    const std::int32_t min = disp_inc_slave_blocks[i];
    const std::int32_t max = disp_inc_slave_blocks[i + 1];
    master_offsets.insert(master_offsets.end(),
                          remote_colliding_offsets.begin() + min,
                          remote_colliding_offsets.begin() + max);

    // Extract the number of masters per topological dimensions
    const std::int32_t min_dim = inc_block_disp[i];
    const std::int32_t max_dim = inc_block_disp[i + 1];
    const std::vector<std::int32_t> num_dofs_per_block(
        block_dofs_recv.begin() + min_dim, block_dofs_recv.begin() + max_dim);

    // Loop through slaves and add them if they havent already been added
    for (std::size_t j = 0; j < master_offsets.size() - 1; ++j)
    {
      // Get the slave block
      const std::int32_t local_block = recv_blocks_as_local[min + j];
      std::vector<std::int32_t> block_offsets(tdim + 1, 0);
      std::partial_sum(num_dofs_per_block.begin() + j * tdim,
                       num_dofs_per_block.begin() + (j + 1) * tdim,
                       block_offsets.begin() + 1);

      for (std::int32_t k = 0; k < tdim; ++k)
      {
        std::int32_t local_slave = local_block * block_size + k;

        // Skip if already found on other incoming processor
        if (local_masters[local_slave].empty())
        {
          std::vector<std::int64_t> masters_(
              remote_colliding_masters.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k],
              remote_colliding_masters.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k + 1]);
          local_masters[local_slave].insert(local_masters[local_slave].end(),
                                            masters_.begin(), masters_.end());

          std::vector<T> coeffs_(
              remote_colliding_coeffs.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k],
              remote_colliding_coeffs.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k + 1]);
          local_coeffs[local_slave].insert(local_coeffs[local_slave].end(),
                                           coeffs_.begin(), coeffs_.end());

          std::vector<std::int32_t> owners_(
              remote_colliding_owners.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k],
              remote_colliding_owners.begin() + disp_inc_masters[i]
                  + master_offsets[j] + block_offsets[k + 1]);
          local_owners[local_slave].insert(local_owners[local_slave].end(),
                                           owners_.begin(), owners_.end());
        }
      }
    }
  }
  for (auto block : local_blocks)
  {
    for (int d = 0; d < block_size; d++)
    {
      if ((local_masters[block * block_size + d].empty()) and (!allow_missing_masters))
      {
        throw std::runtime_error(
            "No masters found on contact surface for slave. Please run in "
            "serial "
            "to check that there is contact, or increase eps2.");
      }
    }
  }

  // Distribute data for ghosted slaves (the coeffs, owners and offsets)
  std::vector<std::int32_t> ghost_slaves(tdim * ghost_blocks.size());
  for (std::size_t i = 0; i < ghost_blocks.size(); ++i)
    for (std::int32_t j = 0; j < tdim; ++j)
      ghost_slaves[i * tdim + j] = ghost_blocks[i] * block_size + j;

  // Compute source and dest ranks of communicator
  MPI_Comm slave_to_ghost
      = create_owner_to_ghost_comm(local_blocks, ghost_blocks, imap);
  auto neighbour_ranks = dolfinx_mpc::compute_neighborhood(slave_to_ghost);
  const std::vector<int>& src_ranks_ghost = neighbour_ranks.first;
  const std::vector<int>& dest_ranks_ghost = neighbour_ranks.second;

  // Count number of incoming slaves
  std::vector<std::int32_t> inc_num_slaves(src_ranks_ghost.size(), 0);
  std::ranges::for_each(ghost_slaves,
                        [block_size, size_local, &ghost_owners, &inc_num_slaves,
                         &src_ranks_ghost](std::int32_t slave)
                        {
                          const std::int32_t owner
                              = ghost_owners[slave / block_size - size_local];
                          const auto it
                              = std::ranges::find(src_ranks_ghost, owner);
                          const auto index
                              = std::distance(src_ranks_ghost.begin(), it);
                          inc_num_slaves[index]++;
                        });
  // Count number of outgoing slaves and masters
  dolfinx::graph::AdjacencyList<int> shared_indices
      = slave_index_map->index_to_dest_ranks();

  std::vector<std::int32_t> out_num_slaves(dest_ranks_ghost.size(), 0);
  std::vector<std::int32_t> out_num_masters(dest_ranks_ghost.size() + 1, 0);
  // Create mappings from ghosted process to the data to recv
  // (can include repeats of data)
  std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost;
  std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost_masters;
  std::map<std::int32_t, std::vector<T>> proc_to_ghost_coeffs;
  std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_owners;
  std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_offsets;
  std::vector<std::int32_t> loc_block(1);
  std::vector<std::int64_t> glob_block(1);

  std::ranges::for_each(
      local_blocks,
      [block_size, &local_masters, &local_coeffs, &local_owners,
       &shared_indices, &dest_ranks_ghost, &loc_block, &glob_block,
       &out_num_masters, &out_num_slaves, &imap, &proc_to_ghost,
       &proc_to_ghost_masters, &proc_to_ghost_coeffs, &proc_to_ghost_owners,
       &proc_to_ghost_offsets, tdim](const auto block)
      {
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          const std::int32_t slave = block * block_size + j;
          const std::vector<std::int64_t>& masters_i = local_masters[slave];
          const std::vector<T>& coeffs_i = local_coeffs[slave];
          const std::vector<std::int32_t>& owners_i = local_owners[slave];
          const auto num_masters = (std::int32_t)masters_i.size();
          for (auto proc : shared_indices.links(slave / block_size))
          {
            const auto it = std::ranges::find(dest_ranks_ghost, proc);
            std::int32_t index = std::distance(dest_ranks_ghost.begin(), it);
            out_num_masters[index] += num_masters;
            out_num_slaves[index]++;
            // Map slaves to global dof to be recognized by recv proc
            std::div_t div = std::div(slave, block_size);
            loc_block[0] = div.quot;
            imap->local_to_global(loc_block, glob_block);
            glob_block[0] = glob_block[0] * block_size + div.rem;
            proc_to_ghost[index].push_back(glob_block[0]);
            // Add master data in process-wise fashion
            proc_to_ghost_masters[index].insert(
                proc_to_ghost_masters[index].end(), masters_i.begin(),
                masters_i.end());
            proc_to_ghost_coeffs[index].insert(
                proc_to_ghost_coeffs[index].end(), coeffs_i.begin(),
                coeffs_i.end());
            proc_to_ghost_owners[index].insert(
                proc_to_ghost_owners[index].end(), owners_i.begin(),
                owners_i.end());
            proc_to_ghost_offsets[index].push_back(
                proc_to_ghost_masters[index].size());
          }
        }
      });
  // Flatten map of global slave ghost dofs to use alltoallv
  std::vector<std::int64_t> out_ghost_slaves;
  std::vector<std::int64_t> out_ghost_masters;
  std::vector<T> out_ghost_coeffs;
  std::vector<std::int32_t> out_ghost_owners;
  std::vector<std::int32_t> out_ghost_offsets;
  std::vector<std::int32_t> num_send_slaves(dest_ranks_ghost.size());
  std::vector<std::int32_t> num_send_masters(dest_ranks_ghost.size());
  for (std::size_t i = 0; i < dest_ranks_ghost.size(); ++i)
  {
    num_send_slaves[i] = proc_to_ghost[i].size();
    num_send_masters[i] = proc_to_ghost_masters[i].size();
    out_ghost_slaves.insert(out_ghost_slaves.end(), proc_to_ghost[i].begin(),
                            proc_to_ghost[i].end());
    out_ghost_masters.insert(out_ghost_masters.end(),
                             proc_to_ghost_masters[i].begin(),
                             proc_to_ghost_masters[i].end());
    out_ghost_coeffs.insert(out_ghost_coeffs.end(),
                            proc_to_ghost_coeffs[i].begin(),
                            proc_to_ghost_coeffs[i].end());
    out_ghost_owners.insert(out_ghost_owners.end(),
                            proc_to_ghost_owners[i].begin(),
                            proc_to_ghost_owners[i].end());
    out_ghost_offsets.insert(out_ghost_offsets.end(),
                             proc_to_ghost_offsets[i].begin(),
                             proc_to_ghost_offsets[i].end());
  }

  // Receive global slave dofs for ghosts structured as on src proc
  // Compute displacements for data to send and receive
  std::vector<int> disp_recv_ghost_slaves(src_ranks_ghost.size() + 1, 0);
  std::partial_sum(inc_num_slaves.begin(), inc_num_slaves.end(),
                   disp_recv_ghost_slaves.begin() + 1);

  std::vector<int> disp_send_ghost_slaves(dest_ranks_ghost.size() + 1, 0);
  std::partial_sum(num_send_slaves.begin(), num_send_slaves.end(),
                   disp_send_ghost_slaves.begin() + 1);

  std::vector<std::int64_t> in_ghost_slaves(disp_recv_ghost_slaves.back());
  MPI_Neighbor_alltoallv(
      out_ghost_slaves.data(), num_send_slaves.data(),
      disp_send_ghost_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      in_ghost_slaves.data(), inc_num_slaves.data(),
      disp_recv_ghost_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      slave_to_ghost);
  std::vector<std::int32_t> in_ghost_offsets(disp_recv_ghost_slaves.back());
  MPI_Neighbor_alltoallv(
      out_ghost_offsets.data(), num_send_slaves.data(),
      disp_send_ghost_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      in_ghost_offsets.data(), inc_num_slaves.data(),
      disp_recv_ghost_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_ghost);

  // Communicate size of communication of masters
  std::vector<int> inc_num_masters(src_ranks_ghost.size() + 1);
  MPI_Neighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
                        inc_num_masters.data(), 1, MPI_INT, slave_to_ghost);
  inc_num_masters.pop_back();
  out_num_masters.pop_back();
  // Send and receive the masters (the proc owning the master) and the
  // corresponding coeffs from the processor owning the slave
  std::vector<int> disp_recv_ghost_masters(src_ranks_ghost.size() + 1, 0);
  std::partial_sum(inc_num_masters.begin(), inc_num_masters.end(),
                   disp_recv_ghost_masters.begin() + 1);
  std::vector<int> disp_send_ghost_masters(dest_ranks_ghost.size() + 1, 0);
  std::partial_sum(num_send_masters.begin(), num_send_masters.end(),
                   disp_send_ghost_masters.begin() + 1);
  std::vector<std::int64_t> in_ghost_masters(disp_recv_ghost_masters.back());
  MPI_Neighbor_alltoallv(
      out_ghost_masters.data(), num_send_masters.data(),
      disp_send_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      in_ghost_masters.data(), inc_num_masters.data(),
      disp_recv_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      slave_to_ghost);
  std::vector<T> in_ghost_coeffs(disp_recv_ghost_masters.back());
  MPI_Neighbor_alltoallv(out_ghost_coeffs.data(), num_send_masters.data(),
                         disp_send_ghost_masters.data(),
                         dolfinx::MPI::mpi_type<T>(), in_ghost_coeffs.data(),
                         inc_num_masters.data(), disp_recv_ghost_masters.data(),
                         dolfinx::MPI::mpi_type<T>(), slave_to_ghost);
  std::vector<std::int32_t> in_ghost_owners(disp_recv_ghost_masters.back());
  MPI_Neighbor_alltoallv(
      out_ghost_owners.data(), num_send_masters.data(),
      disp_send_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      in_ghost_owners.data(), inc_num_masters.data(),
      disp_recv_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_ghost);

  int err3 = MPI_Comm_free(&slave_to_ghost);
  dolfinx::MPI::check_error(comm, err3);

  // Accumulate offsets of masters from different processors
  std::vector<std::int32_t> ghost_offsets = {0};
  for (std::size_t i = 0; i < src_ranks_ghost.size(); ++i)
  {
    const std::int32_t min = disp_recv_ghost_slaves[i];
    const std::int32_t max = disp_recv_ghost_slaves[i + 1];
    std::vector<std::int32_t> inc_offset;
    inc_offset.insert(inc_offset.end(), in_ghost_offsets.begin() + min,
                      in_ghost_offsets.begin() + max);
    for (std::int32_t& offset : inc_offset)
      offset += *(ghost_offsets.end() - 1);
    ghost_offsets.insert(ghost_offsets.end(), inc_offset.begin(),
                         inc_offset.end());
  }

  // Flatten local slaves data
  std::vector<std::int32_t> slaves;
  slaves.reserve(tdim * local_blocks.size());
  std::vector<std::int64_t> masters;
  masters.reserve(slaves.size());
  std::vector<T> coeffs_out;
  coeffs_out.reserve(slaves.size());
  std::vector<std::int32_t> owners_out;
  owners_out.reserve(slaves.size());
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(slaves.size() + 1);

  std::ranges::for_each(
      local_blocks,
      [block_size, tdim, &masters, &local_masters, &coeffs_out, &local_coeffs,
       &owners_out, &local_owners, &offsets, &slaves](const std::int32_t block)
      {
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          // Add the slave if it has masters
          const std::int32_t slave = block * block_size + j;
          if (local_masters[slave].empty())
            continue;
          slaves.push_back(slave);
          masters.insert(masters.end(), local_masters[slave].begin(),
                         local_masters[slave].end());
          coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                            local_coeffs[slave].end());
          offsets.push_back((std::int32_t)masters.size());
          owners_out.insert(owners_out.end(), local_owners[slave].begin(),
                            local_owners[slave].end());
        }
      });

  // Extend local data with ghost entries
  const std::int32_t num_loc_slaves = slaves.size();
  const std::int32_t num_local_masters = masters.size();
  const std::int32_t num_ghost_slaves = in_ghost_slaves.size();
  const std::int32_t num_ghost_masters = in_ghost_masters.size();
  masters.reserve(num_local_masters + num_ghost_masters);
  coeffs_out.reserve(num_local_masters + num_ghost_masters);
  owners_out.reserve(num_local_masters + num_ghost_masters);
  offsets.reserve(num_local_masters + num_ghost_masters + 1);
  std::vector<std::int32_t> local_ghosts
      = map_dofs_global_to_local<U>(V, in_ghost_slaves);
  for (std::int32_t i = 0; i < num_ghost_slaves; i++)
  {
    if (ghost_offsets[i+1]  - ghost_offsets[i] == 0)
      continue; // Skip if no masters for ghost slave
    slaves.push_back(local_ghosts[i]);
    for (std::int32_t j = ghost_offsets[i]; j < ghost_offsets[i + 1]; j++)
    {
      masters.push_back(in_ghost_masters[j]);
      owners_out.push_back(in_ghost_owners[j]);
      coeffs_out.push_back(in_ghost_coeffs[j]);
    }
    offsets.push_back((std::int32_t)masters.size());
  }
  mpc.slaves = slaves;
  mpc.masters = masters;
  mpc.offsets = offsets;
  mpc.owners = owners_out;
  mpc.coeffs = coeffs_out;
  timer.stop();
  return mpc;
}
//-----------------------------------------------------------------------------

} // namespace dolfinx_mpc

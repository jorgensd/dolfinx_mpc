// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "ContactConstraint.h"
#include "MultiPointConstraint.h"
#include "dolfinx/fem/DirichletBC.h"
#include "utils.h"
#include <chrono>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
using namespace dolfinx_mpc;

namespace
{

/// Create bounding box tree (of cells) based on a mesh tag and a given set of
/// markers in the tag. This means that for a given set of facets, we compute
/// the bounding box tree of the cells connected to the facets
/// @param[in] meshtags The meshtags for a set of entities
/// @param[in] dim The entitiy dimension
/// @param[in] marker The value in meshtags to extract entities for
/// @returns A bounding box tree of the cells connected to the entities
dolfinx::geometry::BoundingBoxTree
create_boundingbox_tree(const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                        std::int32_t dim, std::int32_t marker)
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
  dolfinx::geometry::BoundingBoxTree bb_tree(*mesh, tdim, cells_vec, 1e-12);
  return bb_tree;
}

/// Compute contributions to slip constrain from master side (local to process)
/// @param[in] local_rems List containing which block each slave dof is in
/// @param[in] local_colliding_cell List with one-to-one correspondes to a cell
/// that the block is colliding with
/// @param[in] normals The normals at each slave dofs
/// @param[in] V the function space
/// @param[in] tabulated_basis_valeus The basis values tabulated for the given
/// cells at the given coordiantes
/// @returns The mpc data (exluding slave indices)
mpc_data compute_master_contributions(
    const tcb::span<const std::int32_t>& local_rems,
    const tcb::span<const std::int32_t>& local_colliding_cell,
    const xt::xtensor<PetscScalar, 2>& normals,
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    xt::xtensor<double, 3> tabulated_basis_values)
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
      auto basis_values
          = xt::view(tabulated_basis_values, i, xt::all(), xt::all());

      auto cell_blocks = V->dofmap()->cell_dofs(cell);
      for (std::size_t j = 0; j < cell_blocks.size(); ++j)
      {
        for (int b = 0; b < bs; b++)
        {
          // NOTE: Assuming 0 value size
          if (const PetscScalar val
              = normals(i, b) / normals(i, local_rems[i]) * basis_values(j, 0);
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
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();

  // Temporary array holding global indices
  std::vector<std::int64_t> global_blocks;

  // Reuse num_masters_local for insertion
  std::fill(num_masters_local.begin(), num_masters_local.end(), 0);
  for (std::size_t i = 0; i < num_slaves_local; ++i)
  {
    if (const std::int32_t cell = local_colliding_cell[i]; cell != -1)
    {
      auto basis_values
          = xt::view(tabulated_basis_values, i, xt::all(), xt::all());

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
          if (const PetscScalar val
              = normals(i, b) / normals(i, local_rems[i]) * basis_values(j, 0);
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
/// @param[in] normals The normal vectors, shape (local_slaves.size(), 3)
/// @param[in] imap The index map
/// @param[in] block_size The block size of the index map
/// @param[in] rank The rank of current process
/// @returns A mpc_data struct with slaves, masters, coeffs and owners
mpc_data compute_block_contributions(
    const std::vector<std::int32_t>& local_slaves,
    const std::vector<std::int32_t>& local_slave_blocks,
    const xt::xtensor<PetscScalar, 2>& normals,
    const std::shared_ptr<const dolfinx::common::IndexMap> imap,
    std::int32_t block_size, int rank)
{
  std::vector<std::int32_t> dofs(block_size);
  // Count number of masters for each local slave (only contributions from)
  // the same block as the actual slave dof
  std::vector<std::int32_t> num_masters_in_cell(local_slaves.size());
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    std::iota(dofs.begin(), dofs.end(), local_slave_blocks[i] * block_size);
    const std::int32_t local_slave = local_slaves[i];
    for (std::int32_t j = 0; j < block_size; ++j)
      if ((dofs[j] != local_slave) && std::abs(normals(i, j)) > 1e-6)
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
      if ((dofs[j] != local_slave) && std::abs(normals(i, j)) > 1e-6)
      {
        PetscScalar coeff_j = -normals(i, j) / normals(i, max_index);
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
/// @returns The degrees of freedom located on all entities of V that are tagged
/// with the marker
std::vector<std::int32_t>
locate_slave_dofs(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
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
  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  auto [V0, map] = V->sub({0})->collapse();
  std::array<std::vector<std::int32_t>, 2> slave_dofs
      = dolfinx::fem::locate_dofs_topological({*V->sub({0}).get(), V0}, edim,
                                              tcb::make_span(slave_facets));
  return slave_dofs[0];
}
} // namespace

mpc_data dolfinx_mpc::create_contact_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>> nh)
{
  dolfinx::common::Timer timer("~MPC: Create slip constraint");

  MPI_Comm comm = meshtags.mesh()->comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  auto mesh = V->mesh();
  assert(mesh == meshtags.mesh());
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V->dofmap()->index_map_bs();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  mesh->topology_mutable().create_connectivity(fdim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, tdim);
  mesh->topology_mutable().create_entity_permutations();

  // Find all slave dofs and split them into locally owned and ghosted blocks
  std::vector<std::int32_t> local_slave_blocks;
  std::vector<std::int32_t> ghost_slave_blocks;
  {
    std::vector<std::int32_t> slave_dofs
        = locate_slave_dofs(V, meshtags, slave_marker);

    local_slave_blocks.reserve(slave_dofs.size());
    std::for_each(slave_dofs.begin(), slave_dofs.end(),
                  [&local_slave_blocks, &ghost_slave_blocks, bs = block_size,
                   sl = size_local](const std::int32_t dof)
                  {
                    std::div_t div = std::div(dof, bs);
                    if (div.quot < sl)
                      local_slave_blocks.push_back(div.quot);
                    else
                      ghost_slave_blocks.push_back(div.quot);
                  });
  }

  // Data structures to hold information about slave data local to process
  std::vector<std::int32_t> local_slaves(local_slave_blocks.size());
  std::vector<std::int32_t> local_rems(local_slave_blocks.size());
  dolfinx_mpc::mpc_data mpc_local;

  // Find all local contributions to MPC, meaning:
  // 1. Degrees of freedom from the same block as the slave
  // 2. Degrees of freedom from the other interface

  // Helper function
  // Determine component of each block has the largest normal value, and use
  // it as slave dofs to avoid zero division in constraint
  std::vector<std::int32_t> dofs(block_size);
  tcb::span<const PetscScalar> normal_array = nh->x()->array();
  const auto largest_normal_component
      = [&dofs, block_size, &normal_array,
         gdim](const std::int32_t block,
               xt::xtensor_fixed<PetscScalar, xt::xshape<3>>& normal)
  {
    std::iota(dofs.begin(), dofs.end(), block * block_size);
    for (std::int32_t j = 0; j < gdim; ++j)
      normal(j) = normal_array[dofs[j]];
    normal /= xt::sqrt(xt::sum(xt::norm(normal)));
    return xt::argmax(xt::abs(normal))[0];
  };

  // Determine which dof in local slave block is the actual slave

  xt::xtensor<PetscScalar, 2> normals({local_slave_blocks.size(), 3});
  xt::xtensor_fixed<PetscScalar, xt::xshape<3>> normal;
  std::fill(normal.begin() + gdim, normal.end(), 0);
  assert(block_size == gdim);
  for (std::size_t i = 0; i < local_slave_blocks.size(); ++i)
  {
    const std::int32_t slave = local_slave_blocks[i];
    const auto block = largest_normal_component(slave, normal);
    local_slaves[i] = block_size * slave + block;
    local_rems[i] = block;
    xt::row(normals, i) = normal;
  }

  // Compute local contributions to constraint using helper function
  // i.e. compute dot(u, n) on slave side
  mpc_data mpc_in_cell = compute_block_contributions(
      local_slaves, local_slave_blocks, normals, imap, block_size, rank);

  dolfinx::geometry::BoundingBoxTree bb_tree
      = create_boundingbox_tree(meshtags, fdim, master_marker);

  // Compute contributions on other side local to process
  mpc_data mpc_master_local;

  // Create map from slave dof blocks to a cell containing them
  std::vector<std::int32_t> slave_cells
      = dolfinx_mpc::create_block_to_cell_map(*V, local_slave_blocks);
  xt::xtensor<double, 2> slave_coordinates
      = xt::transpose(dolfinx_mpc::tabulate_dof_coordinates(
          *V, local_slave_blocks, slave_cells));
  {
    std::vector<std::int32_t> local_cell_collisions
        = dolfinx_mpc::find_local_collisions(*mesh, bb_tree, slave_coordinates);
    xt::xtensor<double, 3> tabulated_basis_values
        = dolfinx_mpc::evaluate_basis_functions(V, slave_coordinates,
                                                local_cell_collisions);

    mpc_master_local = compute_master_contributions(
        local_rems, local_cell_collisions, normals, V, tabulated_basis_values);
  }
  // Find slave indices were contributions are not found on the process
  std::vector<std::int32_t>& l_offsets = mpc_master_local.offsets;
  std::vector<std::int32_t> slave_indices_remote;
  slave_indices_remote.reserve(local_rems.size());
  for (std::size_t i = 0; i < local_rems.size(); i++)
  {
    if (l_offsets[i + 1] - l_offsets[i] == 0)
      slave_indices_remote.push_back((int)i);
  }

  // Structure storing mpc arrays mpc_local
  mpc_local = concatenate(mpc_in_cell, mpc_master_local);

  // If serial, we gather the resulting mpc data as one constraint
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);

  if (mpi_size == 1)
  {
    // Serial assumptions
    mpc_local.slaves = local_slaves;
    return mpc_local;
  }

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = !local_slave_blocks.empty();
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, has_slave, master_marker);

  // Get the  slave->master recv from and send to ranks
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);

  // Convert slaves missing master contributions to global index
  // and prepare data (coordinates and normals) to send to other procs
  xt::xtensor<double, 2> coordinates_send({slave_indices_remote.size(), 3});
  xt::xtensor<PetscScalar, 2> normals_send({slave_indices_remote.size(), 3});
  std::vector<std::int32_t> send_rems(slave_indices_remote.size());
  for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
  {
    const std::int32_t slave_idx = slave_indices_remote[i];
    send_rems[i] = local_rems[slave_idx];
    xt::row(coordinates_send, i) = xt::row(slave_coordinates, slave_idx);
    xt::row(normals_send, i) = xt::row(normals, slave_idx);
  }

  // Figure out how much data to receive from each neighbor
  const std::size_t out_collision_slaves = slave_indices_remote.size();
  std::vector<std::int32_t> num_slaves_recv(indegree);
  MPI_Neighbor_allgather(
      &out_collision_slaves, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_slaves_recv.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[0]);

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
  std::transform(num_slaves_recv.begin(), num_slaves_recv.end(),
                 std::back_inserter(num_slaves_recv3),
                 [](std::int32_t num_slaves) { return 3 * num_slaves; });
  std::vector<int> disp3(indegree + 1, 0);
  std::partial_sum(num_slaves_recv3.begin(), num_slaves_recv3.end(),
                   disp3.begin() + 1);

  // Send slave normal and coordinate to neighbors
  std::vector<double> coord_recv(disp3.back());
  MPI_Neighbor_allgatherv(coordinates_send.data(), (int)coordinates_send.size(),
                          dolfinx::MPI::mpi_type<double>(), coord_recv.data(),
                          num_slaves_recv3.data(), disp3.data(),
                          dolfinx::MPI::mpi_type<double>(),
                          neighborhood_comms[0]);
  std::vector<PetscScalar> normal_recv(disp3.back());
  MPI_Neighbor_allgatherv(normals_send.data(), (int)normals_send.size(),
                          dolfinx::MPI::mpi_type<PetscScalar>(),
                          normal_recv.data(), num_slaves_recv3.data(),
                          disp3.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
                          neighborhood_comms[0]);

  // Compute off-process contributions
  mpc_data remote_data;
  {
    std::vector<std::size_t> shape_3 = {std::size_t(disp.back()), 3};
    auto slave_normals = xt::adapt(normal_recv.data(), normal_recv.size(),
                                   xt::no_ownership(), shape_3);
    auto slave_coords = xt::adapt(coord_recv.data(), coord_recv.size(),
                                  xt::no_ownership(), shape_3);

    std::vector<std::int32_t> recv_slave_cells_collisions
        = dolfinx_mpc::find_local_collisions(*mesh, bb_tree, slave_coords);
    xt::xtensor<double, 3> recv_tabulated_basis_values
        = dolfinx_mpc::evaluate_basis_functions(V, slave_coords,
                                                recv_slave_cells_collisions);

    remote_data = compute_master_contributions(
        recv_rems, recv_slave_cells_collisions, slave_normals, V,
        recv_tabulated_basis_values);
  }

  // Get info about reverse communicator
  int indegree_rev(-1);
  int outdegree_rev(-2);
  int weighted_rev(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree_rev,
                                 &outdegree_rev, &weighted_rev);
  const auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  // Count number of masters found on the process and convert the offsets to
  // be per process
  std::vector<std::int32_t> num_collision_masters(indegree, 0);
  std::vector<int> num_out_offsets;
  num_out_offsets.reserve(indegree);
  std::transform(num_slaves_recv.begin(), num_slaves_recv.end(),
                 std::back_inserter(num_out_offsets),
                 [](std::int32_t num_slaves) { return num_slaves + 1; });
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
  std::vector<int> inc_num_collision_masters(indegree_rev);
  MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
                        inc_num_collision_masters.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

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
  std::vector<PetscScalar> remote_colliding_coeffs(disp_inc_masters.back());
  MPI_Ineighbor_alltoallv(
      remote_data.coeffs.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      remote_colliding_coeffs.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
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

  std::vector<bool> slave_found(slave_indices_remote.size(), false);
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
  assert(std::find(slave_found.begin(), slave_found.end(), false)
         == slave_found.end());

  /// Wait for all communication to finish
  MPI_Waitall(4, requests.data(), status.data());

  // Move the masters, coeffs and owners from the input adjacency list
  // to one where each node corresponds to an entry in slave_indices_remote
  std::vector<std::int32_t> offproc_offsets(slave_indices_remote.size() + 1, 0);
  std::partial_sum(num_inc_masters.begin(), num_inc_masters.end(),
                   offproc_offsets.begin() + 1);
  std::vector<std::int64_t> offproc_masters(offproc_offsets.back());
  std::vector<PetscScalar> offproc_coeffs(offproc_offsets.back());
  std::vector<std::int32_t> offproc_owners(offproc_offsets.back());

  std::fill(slave_found.begin(), slave_found.end(), false);
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
        std::copy(remote_colliding_masters.begin() + proc_start + slave_min,
                  remote_colliding_masters.begin() + proc_start + slave_max,
                  offproc_masters.begin() + offproc_offsets[c]);

        std::copy(remote_colliding_coeffs.begin() + proc_start + slave_min,
                  remote_colliding_coeffs.begin() + proc_start + slave_max,
                  offproc_coeffs.begin() + offproc_offsets[c]);

        std::copy(remote_colliding_owners.begin() + proc_start + slave_min,
                  remote_colliding_owners.begin() + proc_start + slave_max,
                  offproc_owners.begin() + offproc_offsets[c]);
      }
    }
  }
  // Merge local data with incoming data
  // First count number of local masters
  std::vector<std::int32_t>& masters_offsets = mpc_local.offsets;
  std::vector<std::int64_t>& masters_out = mpc_local.masters;
  std::vector<PetscScalar>& coefficients_out = mpc_local.coeffs;
  std::vector<std::int32_t>& owners_out = mpc_local.owners;

  std::vector<std::int32_t> num_masters_total(local_slaves.size(), 0);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
    num_masters_total[i] += masters_offsets[i + 1] - masters_offsets[i];
  // Then add the remote masters
  for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
    num_masters_total[slave_indices_remote[i]]
        += offproc_offsets[i + 1] - offproc_offsets[i];

  // Create new offset array
  std::vector<std::int32_t> local_offsets(local_slaves.size() + 1, 0);
  std::partial_sum(num_masters_total.begin(), num_masters_total.end(),
                   local_offsets.begin() + 1);

  // Reuse num_masters_total for input indices
  std::fill(num_masters_total.begin(), num_masters_total.end(), 0);
  std::vector<std::int64_t> local_masters(local_offsets.back());
  std::vector<std::int32_t> local_owners(local_offsets.back());
  std::vector<PetscScalar> local_coeffs(local_offsets.back());

  // Insert local contributions
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    const std::int32_t master_min = masters_offsets[i];
    const std::int32_t master_max = masters_offsets[i + 1];
    std::copy(masters_out.begin() + master_min,
              masters_out.begin() + master_max,
              local_masters.begin() + local_offsets[i] + num_masters_total[i]);
    std::copy(coefficients_out.begin() + master_min,
              coefficients_out.begin() + master_max,
              local_coeffs.begin() + local_offsets[i] + num_masters_total[i]);
    std::copy(owners_out.begin() + master_min, owners_out.begin() + master_max,
              local_owners.begin() + local_offsets[i] + num_masters_total[i]);
    num_masters_total[i] += master_max - master_min;
  }

  // Insert remote contributions
  for (std::size_t i = 0; i < slave_indices_remote.size(); ++i)
  {
    const std::int32_t master_min = offproc_offsets[i];
    const std::int32_t master_max = offproc_offsets[i + 1];
    const std::int32_t slave_index = slave_indices_remote[i];
    std::copy(offproc_masters.begin() + master_min,
              offproc_masters.begin() + master_max,
              local_masters.begin() + local_offsets[slave_index]
                  + num_masters_total[slave_index]);
    std::copy(offproc_coeffs.begin() + master_min,
              offproc_coeffs.begin() + master_max,
              local_coeffs.begin() + local_offsets[slave_index]
                  + num_masters_total[slave_index]);
    std::copy(offproc_owners.begin() + master_min,
              offproc_owners.begin() + master_max,
              local_owners.begin() + local_offsets[slave_index]
                  + num_masters_total[slave_index]);
    num_masters_total[slave_index] += master_max - master_min;
  }

  // Create new index-map where there are only ghosts for slaves
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    // Compute quotients and remainders for all ghost blocks
    std::vector<std::int64_t> ghosts_as_global(ghost_slave_blocks.size());
    imap->local_to_global(ghost_slave_blocks, ghosts_as_global);

    std::vector<int> slave_ranks(ghost_slave_blocks.size());
    for (std::size_t i = 0; i < ghost_slave_blocks.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_slave_blocks[i] - size_local];
    slave_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, imap->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(slave_ranks.begin(), slave_ranks.end())),
        ghosts_as_global, slave_ranks);
  }
  MPI_Comm slave_to_ghost = create_owner_to_ghost_comm(
      local_slave_blocks, ghost_slave_blocks, imap);

  // Distribute data for ghosted slaves (the coeffs, owners and  offsets)
  auto [src_ranks_ghost, dest_ranks_ghost]
      = dolfinx::MPI::neighbors(slave_to_ghost);
  const std::size_t num_inc_proc = src_ranks_ghost.size();
  const std::size_t num_out_proc = dest_ranks_ghost.size();

  // Count number of outgoing slaves and masters
  std::map<std::int32_t, std::set<int>> shared_indices
      = slave_index_map->compute_shared_indices();
  std::vector<std::int32_t> out_num_slaves(num_out_proc, 0);
  std::vector<std::int32_t> out_num_masters(num_out_proc, 0);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    // Find ghost processes for the ith local slave
    std::set<int> ghost_procs = shared_indices[local_slaves[i] / block_size];
    const std::int32_t num_masters = local_offsets[i + 1] - local_offsets[i];
    for (auto proc : ghost_procs)
    {
      // Find index of process in local MPI communicator
      auto it
          = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(), proc);
      const auto index = std::distance(dest_ranks_ghost.begin(), it);
      out_num_masters[index] += num_masters;
      out_num_slaves[index]++;
    }
  }

  // Create outgoing displacements for slaves
  std::vector<std::int32_t> disp_dest_ghost_slaves(num_out_proc + 1, 0);
  std::partial_sum(out_num_slaves.begin(), out_num_slaves.end(),
                   disp_dest_ghost_slaves.begin() + 1);

  // Count number of incoming slaves from each process
  std::vector<std::int32_t> recv_num_slaves(num_inc_proc, 0);
  for (const auto& block : ghost_slave_blocks)
  {
    const std::int32_t owner = ghost_owners[block - size_local];
    auto it = std::find(src_ranks_ghost.begin(), src_ranks_ghost.end(), owner);
    const auto index = std::distance(src_ranks_ghost.begin(), it);
    recv_num_slaves[index]++;
  }
  std::vector<std::int64_t> send_slaves(disp_dest_ghost_slaves.back());

  // Incoming displacement for slaves
  std::vector<std::int32_t> disp_recv_ghost_slaves(num_inc_proc + 1, 0);
  std::partial_sum(recv_num_slaves.begin(), recv_num_slaves.end(),
                   disp_recv_ghost_slaves.begin() + 1);

  // Prepare local slaves for communciation to other processes
  std::vector<std::int32_t> slave_counter(num_out_proc, 0);
  std::vector<std::int64_t> global_slave_blocks(local_slaves.size());
  imap->local_to_global(local_slave_blocks, global_slave_blocks);
  const std::int64_t bs = block_size;
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    // Find ghost processes for the ith local slave
    std::set<int> ghost_procs = shared_indices[local_slaves[i] / block_size];
    for (auto proc : ghost_procs)
    {
      // Find index of process in local MPI communicator
      auto it
          = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(), proc);
      const auto index = std::distance(dest_ranks_ghost.begin(), it);
      // Insert global slave dof to send
      send_slaves[disp_dest_ghost_slaves[index] + slave_counter[index]++]
          = global_slave_blocks[i] * bs + local_rems[i];
    }
  }

  MPI_Request slave_request;
  MPI_Status slave_status;

  // Receive slaves from owning process
  std::vector<std::int64_t> recv_ghost_slaves(disp_recv_ghost_slaves.back());
  MPI_Ineighbor_alltoallv(
      send_slaves.data(), out_num_slaves.data(), disp_dest_ghost_slaves.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), recv_ghost_slaves.data(),
      recv_num_slaves.data(), disp_recv_ghost_slaves.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), slave_to_ghost, &slave_request);

  // Communicate number of incoming masters to each process owning the ghost
  std::vector<int> inc_num_ghost_masters(num_inc_proc);
  MPI_Neighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
                        inc_num_ghost_masters.data(), 1, MPI_INT,
                        slave_to_ghost);

  // Create incoming displacements for masters and coefficients
  std::vector<int> disp_inc_ghost_masters(num_inc_proc + 1, 0);
  std::partial_sum(inc_num_ghost_masters.begin(), inc_num_ghost_masters.end(),
                   disp_inc_ghost_masters.begin() + 1);

  // Create outgoing displacements for offsets
  std::vector<std::int32_t> out_num_offsets(num_out_proc);
  std::transform(out_num_slaves.begin(), out_num_slaves.end(),
                 out_num_offsets.begin(), [](std::int32_t n) { return n + 1; });
  std::vector<std::int32_t> disp_dest_ghost_offsets(num_out_proc + 1, 0);
  std::partial_sum(out_num_offsets.begin(), out_num_offsets.end(),
                   disp_dest_ghost_offsets.begin() + 1);

  // Create outgoing displacements for masters
  std::vector<std::int32_t> disp_dest_ghost_masters(num_out_proc + 1, 0);
  std::partial_sum(out_num_masters.begin(), out_num_masters.end(),
                   disp_dest_ghost_masters.begin() + 1);

  // Incoming displacement for offsets
  std::vector<std::int32_t> recv_num_offsets(num_inc_proc);
  std::transform(recv_num_slaves.begin(), recv_num_slaves.end(),
                 recv_num_offsets.begin(),
                 [](std::int32_t n) { return n + 1; });
  std::vector<std::int32_t> disp_recv_ghost_offsets(num_inc_proc + 1, 0);
  std::partial_sum(recv_num_offsets.begin(), recv_num_offsets.end(),
                   disp_recv_ghost_offsets.begin() + 1);

  // Prepare data for sending
  std::vector<std::int64_t> send_masters(disp_dest_ghost_masters.back());
  std::vector<std::int32_t> send_owners(disp_dest_ghost_masters.back());
  std::vector<PetscScalar> send_coeffs(disp_dest_ghost_masters.back());
  std::vector<std::int32_t> send_offsets(disp_dest_ghost_offsets.back(), 0);

  // Start offset counter at 1 as first entry is always going to be 0
  std::vector<std::int32_t> master_counter(num_out_proc, 0);
  std::vector<std::int32_t> offset_counter(num_out_proc, 1);
  for (std::size_t i = 0; i < local_slaves.size(); ++i)
  {
    // Find ghost processes for the ith local slave
    std::div_t div = std::div(local_slaves[i], block_size);
    std::set<int> ghost_procs = shared_indices[div.quot];
    const std::int32_t num_masters = local_offsets[i + 1] - local_offsets[i];

    for (auto proc : ghost_procs)
    {
      // Find index of process in local MPI communicator
      auto it
          = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(), proc);
      const auto index = std::distance(dest_ranks_ghost.begin(), it);
      // Insert global master dofs to send
      std::copy(local_masters.begin() + local_offsets[i],
                local_masters.begin() + local_offsets[i + 1],
                send_masters.begin() + disp_dest_ghost_masters[index]
                    + master_counter[index]);
      // Insert coefficients to send
      std::copy(local_coeffs.begin() + local_offsets[i],
                local_coeffs.begin() + local_offsets[i + 1],
                send_coeffs.begin() + disp_dest_ghost_masters[index]
                    + master_counter[index]);
      // Insert owners to send
      std::copy(local_owners.begin() + local_offsets[i],
                local_owners.begin() + local_offsets[i + 1],
                send_owners.begin() + disp_dest_ghost_masters[index]
                    + master_counter[index]);
      // Increase master counter
      master_counter[index] += num_masters;

      // Insert offsets in accumulative fashion
      send_offsets[disp_dest_ghost_offsets[index] + offset_counter[index]]
          = send_offsets[disp_dest_ghost_offsets[index] + offset_counter[index]
                         - 1]
            + num_masters;
      offset_counter[index]++;
    }
  }

  std::vector<MPI_Request> ghost_requests(4);
  std::vector<MPI_Status> ghost_status(4);

  // Receive master offsets from owning process
  std::vector<std::int32_t> recv_ghost_offsets(disp_recv_ghost_offsets.back());
  MPI_Ineighbor_alltoallv(
      send_offsets.data(), out_num_offsets.data(),
      disp_dest_ghost_offsets.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_ghost_offsets.data(), recv_num_offsets.data(),
      disp_recv_ghost_offsets.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_ghost, &ghost_requests[0]);

  // Receive masters, owners and coeffs from other process
  std::vector<std::int64_t> recv_ghost_masters(disp_inc_ghost_masters.back());
  MPI_Ineighbor_alltoallv(
      send_masters.data(), out_num_masters.data(),
      disp_dest_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      recv_ghost_masters.data(), inc_num_ghost_masters.data(),
      disp_inc_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      slave_to_ghost, &ghost_requests[1]);

  std::vector<PetscScalar> recv_ghost_coeffs(disp_inc_ghost_masters.back());
  MPI_Ineighbor_alltoallv(
      send_coeffs.data(), out_num_masters.data(),
      disp_dest_ghost_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      recv_ghost_coeffs.data(), inc_num_ghost_masters.data(),
      disp_inc_ghost_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      slave_to_ghost, &ghost_requests[2]);

  std::vector<std::int32_t> recv_ghost_owners(disp_inc_ghost_masters.back());
  MPI_Ineighbor_alltoallv(
      send_owners.data(), out_num_masters.data(),
      disp_dest_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_ghost_owners.data(), inc_num_ghost_masters.data(),
      disp_inc_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_ghost, &ghost_requests[3]);

  /// Wait for all communication to finish
  MPI_Waitall(4, ghost_requests.data(), ghost_status.data());
  MPI_Wait(&slave_request, &slave_status);
  assert(recv_ghost_slaves.size() == ghost_slave_blocks.size());

  // Map input ghosts to local index
  std::vector<std::int32_t> inc_ghosts(recv_ghost_slaves.size());
  std::vector<std::int64_t> recv_block(recv_ghost_slaves.size());
  std::vector<std::int32_t> recv_rem(recv_ghost_slaves.size());
  for (std::size_t i = 0; i < inc_ghosts.size(); ++i)
  {
    std::ldiv_t div = std::ldiv(recv_ghost_slaves[i], block_size);
    recv_block[i] = div.quot;
    recv_rem[i] = std::int32_t(div.rem);
  }
  imap->global_to_local(recv_block, inc_ghosts);
  for (std::size_t i = 0; i < inc_ghosts.size(); ++i)
    inc_ghosts[i] = inc_ghosts[i] * block_size + recv_rem[i];

  // Resize arrays to append ghosts
  const std::size_t num_local_slaves = local_slaves.size();
  local_slaves.resize(num_local_slaves + inc_ghosts.size());
  const std::size_t num_local_masters = local_masters.size();
  local_masters.resize(num_local_masters + recv_ghost_masters.size());
  local_coeffs.resize(num_local_masters + recv_ghost_coeffs.size());
  local_owners.resize(num_local_masters + recv_ghost_coeffs.size());
  local_offsets.resize(num_local_slaves + 1 + inc_ghosts.size());
  std::int32_t s_counter = 0;
  std::int32_t m_counter = 0;
  // Loop through incomming procesess
  for (std::size_t i = 0; i < src_ranks_ghost.size(); ++i)
  {
    const std::int32_t min = disp_recv_ghost_offsets[i];
    const std::int32_t max = disp_recv_ghost_offsets[i + 1];
    const std::int32_t m_pos = disp_inc_ghost_masters[i];
    const std::int32_t s_pos = disp_recv_ghost_slaves[i];
    std::int32_t c = 0;

    // Add data for each slave
    for (std::int32_t j = min; j < max - 1; ++j)
    {
      const std::int32_t num_incoming_masters
          = recv_ghost_offsets[j + 1] - recv_ghost_offsets[j];
      assert(num_incoming_masters > 0);
      std::copy(recv_ghost_masters.begin() + m_pos + recv_ghost_offsets[j],
                recv_ghost_masters.begin() + m_pos + recv_ghost_offsets[j + 1],
                local_masters.begin() + num_local_masters + m_counter);
      std::copy(recv_ghost_coeffs.begin() + m_pos + recv_ghost_offsets[j],
                recv_ghost_coeffs.begin() + m_pos + recv_ghost_offsets[j + 1],
                local_coeffs.begin() + num_local_masters + m_counter);
      std::copy(recv_ghost_owners.begin() + m_pos + recv_ghost_offsets[j],
                recv_ghost_owners.begin() + m_pos + recv_ghost_offsets[j + 1],
                local_owners.begin() + num_local_masters + m_counter);
      m_counter += num_incoming_masters;

      // Insert slave and increase offset
      local_slaves[num_local_slaves + s_counter++] = inc_ghosts[s_pos + c++];
      local_offsets[num_local_slaves + s_counter]
          = local_offsets[num_local_slaves + s_counter - 1]
            + num_incoming_masters;
    }
  }

  mpc_data mpc;
  mpc.slaves = local_slaves;
  mpc.masters = local_masters;
  mpc.offsets = local_offsets;
  mpc.owners = local_owners;
  mpc.coeffs = local_coeffs;
  return mpc;
}
//-----------------------------------------------------------------------------
mpc_data dolfinx_mpc::create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker)
{
  dolfinx::common::Timer timer("~MPC: Inelastic condition");

  MPI_Comm comm = meshtags.mesh()->comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int tdim = V->mesh()->topology().dim();
  const int fdim = tdim - 1;
  const int gdim = V->mesh()->geometry().dim();
  const int block_size = V->dofmap()->index_map_bs();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  // Create entity permutations needed in evaluate_basis_functions
  V->mesh()->topology_mutable().create_entity_permutations();
  // Create connectivities needed for evaluate_basis_functions and
  // select_colliding cells
  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
  V->mesh()->topology_mutable().create_connectivity(tdim, tdim);

  std::vector<std::int32_t> slave_blocks
      = locate_slave_dofs(V, meshtags, slave_marker);
  std::for_each(slave_blocks.begin(), slave_blocks.end(),
                [block_size](std::int32_t& d) { d /= block_size; });

  // Vector holding what blocks local to process are slaves
  std::vector<std::int32_t> local_blocks;

  // Array holding ghost slaves blocks (masters,coeffs and offsets will be
  // received)
  std::vector<std::int32_t> ghost_blocks;

  // Map slave blocks to arrays holding local bocks and ghost blocks
  std::for_each(
      slave_blocks.begin(), slave_blocks.end(),
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
      = create_neighborhood_comms(meshtags, has_slave, master_marker);

  // Create communicator local_blocks -> ghost_block
  MPI_Comm slave_to_ghost
      = create_owner_to_ghost_comm(local_blocks, ghost_blocks, imap);

  // Create new index-map where there are only ghosts for slaves
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    std::vector<int> slave_ranks(ghost_blocks.size());
    for (std::size_t i = 0; i < ghost_blocks.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_blocks[i] - size_local];
    std::vector<std::int64_t> ghosts_as_global(ghost_blocks.size());
    imap->local_to_global(ghost_blocks, ghosts_as_global);
    slave_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, imap->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(slave_ranks.begin(), slave_ranks.end())),
        ghosts_as_global, slave_ranks);
  }

  auto facet_to_cell = V->mesh()->topology().connectivity(fdim, tdim);
  assert(facet_to_cell);
  // Find all cells connected to master facets for collision detection
  std::int32_t num_local_cells
      = V->mesh()->topology().index_map(tdim)->size_local();

  std::set<std::int32_t> master_cells;
  {
    auto facet_values = meshtags.values();
    auto facet_indices = meshtags.indices();
    for (std::size_t i = 0; i < facet_indices.size(); ++i)
    {
      if (facet_values[i] == master_marker)
      {
        auto cell = facet_to_cell->links(facet_indices[i]);
        assert(cell.size() == 1);
        if (cell[0] < num_local_cells)
          master_cells.insert(cell[0]);
      }
    }
  }
  // Loop through all masters on current processor and check if they
  // collide with a local master facet

  std::map<std::int32_t, std::vector<std::int32_t>> local_owners;
  std::map<std::int32_t, std::vector<std::int64_t>> local_masters;
  std::map<std::int32_t, std::vector<PetscScalar>> local_coeffs;

  xt::xtensor<double, 2> coordinates = V->tabulate_dof_coordinates(false);
  std::vector<std::int64_t> blocks_wo_local_collision;

  // Map to get coordinates and normals of slaves that should be
  // distributed
  std::vector<int> master_cells_vec(master_cells.begin(), master_cells.end());
  dolfinx::geometry::BoundingBoxTree bb_tree(*V->mesh(), tdim, master_cells_vec,
                                             1e-12);

  std::vector<size_t> collision_to_local;
  xt::xtensor<double, 2> slave_coordinates({local_blocks.size(), 3});
  // Get coordinate for slave and save in send array
  for (std::size_t i = 0; i < local_blocks.size(); ++i)
    xt::row(slave_coordinates, i) = xt::row(coordinates, local_blocks[i]);
  dolfinx::graph::AdjacencyList<std::int32_t> bbox_collision
      = dolfinx::geometry::compute_collisions(bb_tree, slave_coordinates);
  dolfinx::graph::AdjacencyList<std::int32_t> verified_collisions
      = dolfinx::geometry::compute_colliding_cells(*V->mesh(), bbox_collision,
                                                   slave_coordinates);

  for (std::size_t i = 0; i < local_blocks.size(); ++i)
  {
    auto verified_cells = verified_collisions.links((int)i);
    if (!verified_cells.empty())
    {
      // Compute interpolation on other cell
      xt::xtensor<PetscScalar, 2> basis_values = get_basis_functions(
          V,
          xt::reshape_view(xt::view(slave_coordinates, i, xt::xrange(0, gdim)),
                           {1, gdim}),
          verified_cells[0]);
      auto cell_blocks = V->dofmap()->cell_dofs(verified_cells[0]);

      std::vector<std::vector<std::int32_t>> l_master(3);
      std::vector<std::vector<PetscScalar>> l_coeff(3);
      std::vector<std::vector<std::int32_t>> l_owner(3);
      // Create arrays per topological dimension
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        for (std::size_t k = 0; k < cell_blocks.size(); ++k)
        {
          for (std::int32_t block = 0; block < block_size; ++block)
          {
            if (const PetscScalar coeff
                = basis_values(k * block_size + block, j);
                std::abs(coeff) > 1e-6)
            {
              l_master[j].push_back(cell_blocks[k] * block_size + block);
              l_coeff[j].push_back(coeff);
              l_owner[j].push_back(
                  cell_blocks[k] < size_local
                      ? rank
                      : ghost_owners[cell_blocks[k] - size_local]);
            }
          }
        }
      }
      bool found_masters = false;
      // Add masters for each master in the block
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        const std::vector<std::int32_t>& masters_j = l_master[j];
        if (!masters_j.empty())
        {
          const size_t num_masters = masters_j.size();
          const std::int32_t local_slave = local_blocks[i] * block_size + j;
          // Convert local masteer dofs to their corresponding global dof
          std::vector<std::int64_t> masters_as_global(num_masters);
          std::vector<std::int32_t> master_block_loc(num_masters);
          std::vector<std::int32_t> masters_rem(num_masters);
          for (std::size_t k = 0; k < num_masters; ++k)
          {
            std::div_t div = std::div(masters_j[k], block_size);
            master_block_loc[k] = div.quot;
            masters_rem[k] = div.rem;
          }
          imap->local_to_global(master_block_loc, masters_as_global);
          for (size_t k = 0; k < num_masters; ++k)
          {
            masters_as_global[k]
                = masters_as_global[k] * block_size + masters_rem[k];
          }
          local_masters[local_slave] = masters_as_global;
          local_coeffs[local_slave] = l_coeff[j];
          local_owners[local_slave] = l_owner[j];
          found_masters = true;
        }
      }
      if (!found_masters)
      {
        blocks_wo_local_collision.push_back(local_blocks_as_glob[i]);
        collision_to_local.push_back(i);
      }
    }
    else
    {
      blocks_wo_local_collision.push_back(local_blocks_as_glob[i]);
      collision_to_local.push_back(i);
    }
  }
  // Extract coordinates and normals to distribute
  xt::xtensor<double, 2> distribute_coordinates(
      {blocks_wo_local_collision.size(), 3});
  for (std::size_t i = 0; i < collision_to_local.size(); ++i)
  {
    xt::row(distribute_coordinates, i)
        = xt::row(slave_coordinates, collision_to_local[i]);
  }

  dolfinx_mpc::mpc_data mpc;

  // If serial, we only have to gather slaves, masters, coeffs in 1D
  // arrays
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1)
  {
    assert(blocks_wo_local_collision.empty());
    std::vector<std::int64_t> masters_out;
    std::vector<PetscScalar> coeffs_out;
    std::vector<std::int32_t> offsets_out = {0};
    std::vector<std::int32_t> slaves_out;
    // Flatten the maps to 1D arrays (assuming all slaves are local
    // slaves)
    std::vector<std::int32_t> slaves;
    slaves.reserve(tdim * local_blocks.size());
    std::for_each(
        local_blocks.begin(), local_blocks.end(),
        [block_size, tdim, &masters_out, &local_masters, &coeffs_out,
         &local_coeffs, &offsets_out, &slaves](const std::int32_t block)
        {
          for (std::int32_t j = 0; j < tdim; ++j)
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
    std::vector<std::int32_t> owners(masters_out.size());
    std::fill(owners.begin(), owners.end(), 0);
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
  std::vector<std::int32_t> num_slave_blocks(indegree);
  MPI_Neighbor_allgather(
      &num_colliding_blocks, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_slave_blocks.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[0]);

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
  std::vector<double> coord_recv(coordinate_disp.back());
  MPI_Neighbor_allgatherv(
      distribute_coordinates.data(), 3 * (int)distribute_coordinates.size(),
      dolfinx::MPI::mpi_type<double>(), coord_recv.data(),
      num_block_coordinates.data(), coordinate_disp.data(),
      dolfinx::MPI::mpi_type<double>(), neighborhood_comms[0]);

  // Vector for processes with slaves, mapping slaves with
  // collision on this process
  std::vector<std::vector<std::int64_t>> collision_slaves(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int64_t>>>
      collision_masters(indegree);
  std::vector<std::map<std::int64_t, std::vector<PetscScalar>>>
      collision_coeffs(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
      collision_owners(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
      collision_block_offsets(indegree);
  xt::xtensor_fixed<double, xt::xshape<1, 3>> slave_coord;

  // Loop over slaves per incoming processor
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
    {
      // Extract slave coordinates and normals from incoming data
      for (std::int32_t k = 0; k < 3; k++)
        slave_coord[k] = coord_recv[3 * j + k];

      // Use collision detection algorithm (GJK) to check distance from
      // cell to point and select one within 1e-10
      dolfinx::graph::AdjacencyList<std::int32_t> bbox_collisions
          = dolfinx::geometry::compute_collisions(bb_tree, slave_coord);
      dolfinx::graph::AdjacencyList<std::int32_t> verified_candidates
          = dolfinx::geometry::compute_colliding_cells(
              *V->mesh(), bbox_collisions, slave_coord);
      auto verified_cells = verified_candidates.links(0);
      if (!verified_cells.empty())
      {
        // Compute local contribution of slave on master facet
        xt::xtensor<PetscScalar, 2> basis_values = get_basis_functions(
            V,
            xt::reshape_view(xt::view(slave_coord, i, xt::xrange(0, gdim)),
                             {1, gdim}),
            verified_cells[0]);

        auto cell_blocks = V->dofmap()->cell_dofs(verified_cells[0]);
        std::vector<std::vector<std::int32_t>> l_master(3);
        std::vector<std::vector<PetscScalar>> l_coeff(3);
        std::vector<std::vector<std::int32_t>> l_owner(3);
        // Create arrays per topological dimension
        for (std::int32_t t = 0; t < tdim; ++t)
        {
          for (std::size_t k = 0; k < cell_blocks.size(); ++k)
          {
            for (std::int32_t block = 0; block < block_size; ++block)
            {
              if (const PetscScalar coeff
                  = basis_values(k * block_size + block, t);
                  std::abs(coeff) > 1e-6)
              {
                l_master[t].push_back(cell_blocks[k] * block_size + block);
                l_coeff[t].push_back(coeff);
                l_owner[t].push_back(
                    cell_blocks[k] < size_local
                        ? rank
                        : ghost_owners[cell_blocks[k] - size_local]);
              }
            }
          }
        }
        // Collect masters per block
        std::vector<std::int32_t> remote_owners;
        std::vector<std::int64_t> remote_masters;
        std::vector<PetscScalar> remote_coeffs;
        std::vector<std::int32_t> num_masters_per_slave(tdim, 0);
        for (std::int32_t t = 0; t < tdim; ++t)
        {
          if (!l_master[t].empty())
          {
            const std::vector<std::int32_t>& masters_t = l_master[t];

            // Convert local masteer dofs to their corresponding global
            // dof
            const size_t num_masters = masters_t.size();
            std::vector<std::int32_t> masters_block_loc(num_masters);
            std::vector<std::int64_t> masters_as_global(num_masters);
            std::vector<std::int32_t> masters_rem(num_masters);

            for (size_t k = 0; k < num_masters; ++k)
            {
              std::div_t div = std::div(masters_t[k], block_size);
              masters_block_loc[k] = div.quot;
              masters_rem[k] = div.rem;
            }
            imap->local_to_global(masters_block_loc, masters_as_global);
            for (size_t k = 0; k < num_masters; ++k)
            {
              masters_as_global[k]
                  = masters_as_global[k] * block_size + masters_rem[k];
            }
            remote_masters.insert(remote_masters.end(),
                                  masters_as_global.begin(),
                                  masters_as_global.end());
            remote_coeffs.insert(remote_coeffs.end(), l_coeff[t].begin(),
                                 l_coeff[t].end());
            remote_owners.insert(remote_owners.end(), l_owner[t].begin(),
                                 l_owner[t].end());
            num_masters_per_slave[t] = (std::int32_t)masters_as_global.size();
          }
        }
        collision_slaves[i].push_back(remote_slave_blocks[j]);
        collision_masters[i][remote_slave_blocks[j]] = remote_masters;
        collision_coeffs[i][remote_slave_blocks[j]] = remote_coeffs;
        collision_owners[i][remote_slave_blocks[j]] = remote_owners;
        collision_block_offsets[i][remote_slave_blocks[j]]
            = num_masters_per_slave;
      }
    }
  }
  // Flatten data structures before send/recv
  std::vector<std::int32_t> num_found_slave_blocks(indegree);
  for (std::int32_t i = 0; i < indegree; ++i)
    num_found_slave_blocks[i] = (std::int32_t)collision_slaves[i].size();

  // Get info about reverse communicator (masters->slaves)
  int indegree_rev(-1);
  int outdegree_rev(-2);
  int weighted_rev(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree_rev,
                                 &outdegree_rev, &weighted_rev);
  const auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  // Communicate number of incoming slaves and masters after coll
  // detection
  std::vector<int> inc_num_found_slave_blocks(indegree_rev);
  MPI_Neighbor_alltoall(num_found_slave_blocks.data(), 1, MPI_INT,
                        inc_num_found_slave_blocks.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

  std::vector<std::int32_t> num_collision_masters(indegree);
  std::vector<std::int64_t> found_slave_blocks;
  std::vector<std::int64_t> found_masters;
  std::vector<std::int32_t> offset_for_blocks;
  std::vector<std::int32_t> offsets_in_blocks;
  std::vector<std::int32_t> found_owners;
  std::vector<PetscScalar> found_coefficients;
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

      std::vector<PetscScalar>& coeffs_ij = collision_coeffs[i][slave];
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

  std::vector<int> num_inc_masters(indegree_rev);
  MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
                        num_inc_masters.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

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
  std::vector<PetscScalar> remote_colliding_coeffs(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      found_coefficients.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      remote_colliding_coeffs.data(), num_inc_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
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

          std::vector<PetscScalar> coeffs_(
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

  // Distribute data for ghosted slaves (the coeffs, owners and offsets)
  std::vector<std::int32_t> ghost_slaves(tdim * ghost_blocks.size());
  for (std::size_t i = 0; i < ghost_blocks.size(); ++i)
    for (std::int32_t j = 0; j < tdim; ++j)
      ghost_slaves[i * tdim + j] = ghost_blocks[i] * block_size + j;

  std::array<std::vector<std::int32_t>, 2> neighbors
      = dolfinx::MPI::neighbors(slave_to_ghost);
  auto& src_ranks_ghost = neighbors[0];
  auto& dest_ranks_ghost = neighbors[1];

  // Count number of incoming slaves
  std::vector<std::int32_t> inc_num_slaves(src_ranks_ghost.size(), 0);
  std::for_each(ghost_slaves.begin(), ghost_slaves.end(),
                [block_size, size_local, &ghost_owners, &inc_num_slaves,
                 &src_ranks_ghost](std::int32_t slave)
                {
                  const std::int32_t owner
                      = ghost_owners[slave / block_size - size_local];
                  const auto it = std::find(src_ranks_ghost.begin(),
                                            src_ranks_ghost.end(), owner);
                  const auto index = std::distance(src_ranks_ghost.begin(), it);
                  inc_num_slaves[index]++;
                });
  // Count number of outgoing slaves and masters
  std::map<std::int32_t, std::set<int>> shared_indices
      = slave_index_map->compute_shared_indices();

  std::vector<std::int32_t> out_num_slaves(dest_ranks_ghost.size(), 0);
  std::vector<std::int32_t> out_num_masters(dest_ranks_ghost.size(), 0);
  // Create mappings from ghosted process to the data to recv
  // (can include repeats of data)
  std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost;
  std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost_masters;
  std::map<std::int32_t, std::vector<PetscScalar>> proc_to_ghost_coeffs;
  std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_owners;
  std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_offsets;
  std::vector<std::int32_t> loc_block(1);
  std::vector<std::int64_t> glob_block(1);

  std::for_each(
      local_blocks.begin(), local_blocks.end(),
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
          const std::vector<PetscScalar>& coeffs_i = local_coeffs[slave];
          const std::vector<std::int32_t>& owners_i = local_owners[slave];
          const auto num_masters = (std::int32_t)masters_i.size();
          std::set<int> ghost_procs = shared_indices[slave / block_size];
          for (auto proc : ghost_procs)
          {
            const auto it = std::find(dest_ranks_ghost.begin(),
                                      dest_ranks_ghost.end(), proc);
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
  std::vector<PetscScalar> out_ghost_coeffs;
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
  std::vector<int> inc_num_masters(src_ranks_ghost.size());
  MPI_Neighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
                        inc_num_masters.data(), 1, MPI_INT, slave_to_ghost);
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
  std::vector<PetscScalar> in_ghost_coeffs(disp_recv_ghost_masters.back());
  MPI_Neighbor_alltoallv(out_ghost_coeffs.data(), num_send_masters.data(),
                         disp_send_ghost_masters.data(),
                         dolfinx::MPI::mpi_type<PetscScalar>(),
                         in_ghost_coeffs.data(), inc_num_masters.data(),
                         disp_recv_ghost_masters.data(),
                         dolfinx::MPI::mpi_type<PetscScalar>(), slave_to_ghost);
  std::vector<std::int32_t> in_ghost_owners(disp_recv_ghost_masters.back());
  MPI_Neighbor_alltoallv(
      out_ghost_owners.data(), num_send_masters.data(),
      disp_send_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      in_ghost_owners.data(), inc_num_masters.data(),
      disp_recv_ghost_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_ghost);

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
  std::vector<PetscScalar> coeffs_out;
  coeffs_out.reserve(slaves.size());
  std::vector<std::int32_t> owners_out;
  owners_out.reserve(slaves.size());
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(slaves.size() + 1);

  std::for_each(
      local_blocks.begin(), local_blocks.end(),
      [block_size, tdim, &masters, &local_masters, &coeffs_out, &local_coeffs,
       &owners_out, &local_owners, &offsets, &slaves](const std::int32_t block)
      {
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          const std::int32_t slave = block * block_size + j;
          slaves.push_back(slave);
          masters.insert(masters.end(), local_masters[slave].begin(),
                         local_masters[slave].end());
          coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                            local_coeffs[slave].end());
          offsets.push_back(masters.size());
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
  for (std::int32_t i = 0; i < num_ghost_slaves; i++)
  {
    for (std::int32_t j = ghost_offsets[i]; j < ghost_offsets[i + 1]; j++)
    {
      masters.push_back(in_ghost_masters[j]);
      owners_out.push_back(in_ghost_owners[j]);
      coeffs_out.push_back(in_ghost_coeffs[j]);
    }
    offsets.push_back(masters.size());
  }

  // Map Ghost data

  std::vector<std::int64_t> in_ghost_block(num_ghost_slaves);
  std::vector<std::int64_t> in_ghost_rem(num_ghost_slaves);
  for (std::int32_t i = 0; i < num_ghost_slaves; ++i)
  {
    const std::int64_t bs = block_size;
    std::ldiv_t div = std::div(in_ghost_slaves[i], bs);
    in_ghost_block[i] = div.quot;
    in_ghost_rem[i] = div.rem;
  }

  std::vector<std::int32_t> in_ghost_block_loc(in_ghost_block.size());
  imap->global_to_local(in_ghost_block, in_ghost_block_loc);
  slaves.resize(num_loc_slaves + num_ghost_slaves);
  for (std::int32_t i = 0; i < num_ghost_slaves; ++i)
  {
    slaves[num_loc_slaves + i]
        = in_ghost_block_loc[i] * block_size + in_ghost_rem[i];
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

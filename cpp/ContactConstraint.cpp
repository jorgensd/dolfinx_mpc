// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ContactConstraint.h"
#include "MultiPointConstraint.h"
#include "dolfinx/fem/DirichletBC.h"
#include "utils.h"
#include <chrono>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <xtensor/xsort.hpp>
using namespace dolfinx_mpc;

mpc_data dolfinx_mpc::create_contact_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>> nh)
{
  dolfinx::common::Timer timer("~MPC: Create slip constraint");

  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const size_t tdim = V->mesh()->topology().dim();
  const int gdim = V->mesh()->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V->dofmap()->index_map_bs();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
  V->mesh()->topology_mutable().create_connectivity(tdim, tdim);
  V->mesh()->topology_mutable().create_entity_permutations();

  // Extract slave_facets
  std::vector<std::int32_t> slave_facets;
  for (std::int32_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == slave_marker)
      slave_facets.push_back(meshtags.indices()[i]);

  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  std::pair<std::shared_ptr<dolfinx::fem::FunctionSpace>,
            std::vector<std::int32_t>>
      V0_pair = V->sub({0})->collapse();
  auto [V0, map] = V0_pair;

  std::array<std::vector<std::int32_t>, 2> slave_blocks
      = dolfinx::fem::locate_dofs_topological(
          {*V->sub({0}).get(), *V0.get()}, fdim, tcb::make_span(slave_facets));
  for (std::size_t i = 0; i < slave_blocks[0].size(); ++i)
    slave_blocks[0][i] /= block_size;

  // Make maps for local data of slave and master coeffs
  std::vector<std::int32_t> local_slaves;
  std::vector<std::int32_t> local_slaves_as_glob;
  std::map<std::int64_t, std::vector<std::int64_t>> masters;
  std::map<std::int64_t, std::vector<PetscScalar>> coeffs;
  std::map<std::int64_t, std::vector<std::int32_t>> owners;

  // Array holding ghost slaves (masters,coeffs and offsets will be recieved)
  std::vector<std::int32_t> ghost_slaves;
  // Reusable arrays for slave and master dofs
  std::vector<std::int32_t> dofs(block_size);
  std::vector<std::int32_t> m_loc(1);
  std::vector<std::int64_t> m_block_glob(1);

  // Vector holder normal coefficients
  tcb::span<const PetscScalar> normal_array = nh->x()->array();

  // Determine which dof should be a slave dof (based on max component of
  // normal vector normal vector)
  {
    std::vector<std::int64_t> slave_blocks_glob(slave_blocks[0].size());
    imap->local_to_global(slave_blocks[0], slave_blocks_glob);
    std::vector<std::int32_t> masters_i;
    std::vector<PetscScalar> coeffs_i;
    std::vector<std::int32_t> owners_i;
    masters_i.reserve(tdim);
    coeffs_i.reserve(tdim);
    owners_i.reserve(tdim);
    xt::xarray<PetscScalar> l_normal = xt::empty<PetscScalar>({tdim});
    for (std::int32_t i = 0; i < slave_blocks[0].size(); ++i)
    {
      std::iota(dofs.begin(), dofs.end(), slave_blocks[0][i] * block_size);
      // Create normalized local normal vector and find its max index
      for (std::int32_t j = 0; j < tdim; ++j)
        l_normal[j] = normal_array[dofs[j]];

      l_normal /= xt::sqrt(xt::sum(xt::norm(l_normal)));

      auto max_index = xt::argmax(xt::abs(l_normal));

      // Compute coeffs if slave is owned by the processor
      if (dofs[max_index[0]] < size_local * block_size)
      {
        owners_i.clear();
        coeffs_i.clear();
        masters_i.clear();
        std::div_t div = std::div(dofs[max_index[0]], block_size);
        const std::int64_t slave_glob
            = slave_blocks_glob[i] * block_size + div.rem;
        local_slaves.push_back(dofs[max_index[0]]);
        local_slaves_as_glob.push_back(slave_glob);

        for (std::int32_t j = 0; j < tdim; ++j)
          if (j != max_index[0])
          {
            PetscScalar coeff_j = -l_normal[j] / l_normal[max_index];
            if (std::abs(coeff_j) > 1e-6)
            {
              owners_i.push_back(rank);
              masters_i.push_back(dofs[j]);
              coeffs_i.push_back(coeff_j);
            }
          }
        const std::int32_t num_masters = masters_i.size();
        if (num_masters > 0)
        {
          std::vector<std::int64_t> m_glob(num_masters);
          for (std::int32_t j = 0; j < num_masters; ++j)
          {
            std::div_t div = std::div(masters_i[j], block_size);
            m_loc[0] = div.quot;
            imap->local_to_global(m_loc, m_block_glob);
            m_glob[j] = m_block_glob[0] * block_size + div.rem;
          }
          masters[slave_glob] = m_glob;
          coeffs[slave_glob] = coeffs_i;
          owners[slave_glob] = owners_i;
        }
      }
      else
        ghost_slaves.push_back(dofs[max_index[0]]);
    }
  }

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = local_slaves.size() > 0 ? 1 : 0;
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, has_slave, master_marker);
  // Create communicator local_slaves -> ghost_slaves
  MPI_Comm slave_to_ghost = create_owner_to_ghost_comm(
      local_slaves, ghost_slaves, imap, block_size);
  // Create new index-map where there are only ghosts for slaves
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    // Compute quotients and remainders for all ghost blocks
    std::vector<std::int32_t> ghost_blocks(ghost_slaves.size());
    for (std::size_t i = 0; i < ghost_slaves.size(); ++i)
      ghost_blocks[i] = ghost_slaves[i] / block_size;
    std::vector<std::int64_t> ghosts_as_global(ghost_blocks.size());
    imap->local_to_global(ghost_blocks, ghosts_as_global);

    std::vector<int> slave_ranks(ghost_slaves.size());
    for (std::int32_t i = 0; i < ghost_slaves.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_blocks[i] - size_local];
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
  auto facet_values = meshtags.values();
  auto facet_indices = meshtags.indices();
  for (std::int32_t i = 0; i < facet_indices.size(); ++i)
  {
    if (facet_values[i] == master_marker)
    {
      auto cell = facet_to_cell->links(facet_indices[i]);
      assert(cell.size() == 1);
      if (cell[0] < num_local_cells)
        master_cells.insert(cell[0]);
    }
  }
  // Loop through all masters on current processor and check if they collide
  // with a local master facet

  std::map<std::int32_t, std::vector<std::int32_t>> local_owners;
  std::map<std::int32_t, std::vector<std::int64_t>> local_masters;
  std::map<std::int32_t, std::vector<PetscScalar>> local_coeffs;

  xt::xtensor<double, 2> coordinates = V->tabulate_dof_coordinates(false);
  std::vector<std::array<double, 3>> local_slave_coordinates(
      local_slaves.size());
  xt::xtensor<PetscScalar, 2> local_slave_normal({local_slaves.size(), 3});
  std::vector<std::int64_t> slaves_wo_local_collision;

  // Map to get coordinates and normals of slaves that should be distributed
  std::vector<std::int32_t> collision_to_local;
  std::vector<int> master_cells_vec(master_cells.begin(), master_cells.end());
  dolfinx::geometry::BoundingBoxTree bb_tree(*V->mesh(), tdim, master_cells_vec,
                                             1e-12);
  xt::xtensor_fixed<PetscScalar, xt::xshape<3>> slave_coeffs;
  for (std::int32_t i = 0; i < local_slaves.size(); ++i)
  {
    // Get coordinate and coeff for slave and save in send array
    const std::int32_t slave_block = slave_blocks[0][i];
    const std::array<double, 3> coord
        = {coordinates(slave_block, 0), coordinates(slave_block, 1),
           coordinates(slave_block, 2)};
    local_slave_coordinates[i] = coord;
    std::iota(dofs.begin(), dofs.end(), slave_blocks[0][i] * block_size);
    for (std::int32_t j = 0; j < gdim; ++j)
      slave_coeffs[j] = normal_array[dofs[j]];
    // Pad 2D spaces with zero in normal
    for (std::int32_t j = gdim; j < 3; ++j)
      slave_coeffs[j] = 0;
    slave_coeffs /= xt::sqrt(xt::sum(xt::norm(slave_coeffs)));
    auto max_index = xt::argmax(xt::abs(slave_coeffs));

    xt::row(local_slave_normal, i) = slave_coeffs;

    // Use collision detection algorithm (GJK) to check distance from  cell
    // to point and select one within 1e-10

    std::vector<int> bbox_collisions
        = dolfinx::geometry::compute_collisions(bb_tree, coord);
    std::vector<std::int32_t> verified_candidates
        = dolfinx::geometry::select_colliding_cells(*V->mesh(), bbox_collisions,
                                                    coord, 1);
    if (verified_candidates.size() == 1)
    {
      // Compute local contribution of slave on master facet
      xt::xtensor<PetscScalar, 2> basis_values
          = get_basis_functions(V, coord, verified_candidates[0]);
      auto cell_blocks = V->dofmap()->cell_dofs(verified_candidates[0]);
      std::vector<std::int32_t> l_master;
      std::vector<PetscScalar> l_coeff;
      std::vector<std::int32_t> l_owner;
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        for (std::int32_t k = 0; k < cell_blocks.size(); ++k)
        {
          for (std::int32_t block = 0; block < block_size; ++block)
          {
            PetscScalar coeff = basis_values(k * block_size + block, j)
                                * slave_coeffs[j] / slave_coeffs[max_index];
            if (std::abs(coeff) > 1e-6)
            {
              l_master.push_back(cell_blocks[k] * block_size + block);
              l_coeff.push_back(coeff);
              if (cell_blocks[k] < size_local)
                l_owner.push_back(rank);
              else
                l_owner.push_back(ghost_owners[cell_blocks[k] - size_local]);
            }
          }
        }
      }
      const std::int32_t num_masters = l_master.size();
      if (num_masters > 0)
      {
        std::vector<std::int64_t> global_masters(num_masters);
        for (std::int32_t k = 0; k < num_masters; ++k)
        {
          std::div_t div = std::div(l_master[k], block_size);
          m_loc[0] = div.quot;
          imap->local_to_global(m_loc, m_block_glob);
          global_masters[k] = m_block_glob[0] * block_size + div.rem;
        }
        local_masters[local_slaves_as_glob[i]] = global_masters;
        local_coeffs[local_slaves_as_glob[i]] = l_coeff;
        local_owners[local_slaves_as_glob[i]] = l_owner;
      }
      else
      {
        slaves_wo_local_collision.push_back(local_slaves_as_glob[i]);
        collision_to_local.push_back(i);
      }
    }
    else
    {
      slaves_wo_local_collision.push_back(local_slaves_as_glob[i]);
      collision_to_local.push_back(i);
    }
  }
  // Extract coordinates and normals to distribute
  std::vector<std::array<double, 3>> distribute_coordinates(
      slaves_wo_local_collision.size());
  xt::xtensor<PetscScalar, 2> distribute_normals(
      {slaves_wo_local_collision.size(), 3});
  for (std::int32_t i = 0; i < collision_to_local.size(); ++i)
  {
    distribute_coordinates[i] = local_slave_coordinates[collision_to_local[i]];
    xt::row(distribute_normals, i)
        = xt::row(local_slave_normal, collision_to_local[i]);
  }
  // Structure storing mpc arrays
  dolfinx_mpc::mpc_data mpc;
  // If serial, we only have to gather slaves, masters, coeffs in 1D arrays
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1)
  {
    assert(slaves_wo_local_collision.size() == 0);
    std::vector<std::int64_t> masters_out;
    std::vector<PetscScalar> coeffs_out;
    std::vector<std::int32_t> offsets_out = {0};
    // Flatten the maps to 1D arrays (assuming all slaves are local slaves)
    for (std::int32_t i = 0; i < local_slaves_as_glob.size(); i++)
    {
      auto slave = local_slaves_as_glob[i];
      masters_out.insert(masters_out.end(), masters[slave].begin(),
                         masters[slave].end());
      masters_out.insert(masters_out.end(), local_masters[slave].begin(),
                         local_masters[slave].end());
      coeffs_out.insert(coeffs_out.end(), coeffs[slave].begin(),
                        coeffs[slave].end());
      coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                        local_coeffs[slave].end());
      offsets_out.push_back(masters_out.size());
    }

    // Serial assumptions
    std::vector<std::int32_t> owners(masters_out.size());
    std::fill(owners.begin(), owners.end(), 1);
    std::vector<std::int32_t> gs(0);
    std::vector<std::int64_t> gm(0);
    std::vector<PetscScalar> gc(0);
    std::vector<std::int32_t> go(0);
    std::vector<std::int32_t> goff(1);
    std::fill(goff.begin(), goff.end(), 0);
    mpc.local_slaves = local_slaves;
    mpc.ghost_slaves = gs;
    mpc.local_masters = masters_out;
    mpc.ghost_masters = gm;
    mpc.local_offsets = offsets_out;
    mpc.ghost_offsets = goff;
    mpc.local_owners = owners;
    mpc.ghost_owners = go;
    mpc.local_coeffs = coeffs_out;
    mpc.ghost_coeffs = gc;

    return mpc;
  }

  // Get the  slave->master recv from and send to ranks
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);
  const auto [src_ranks, dest_ranks]
      = dolfinx::MPI::neighbors(neighborhood_comms[0]);

  // Figure out how much data to receive from each neighbor
  const std::int32_t out_collision_slaves = slaves_wo_local_collision.size();
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
  std::vector<std::int64_t> slaves_recv(disp.back());
  MPI_Neighbor_allgatherv(
      slaves_wo_local_collision.data(), slaves_wo_local_collision.size(),
      dolfinx::MPI::mpi_type<std::int64_t>(), slaves_recv.data(),
      num_slaves_recv.data(), disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[0]);

  // Multiply recv size by three to accommodate vector coordinates and
  // function data
  std::vector<std::int32_t> num_slaves_recv3(indegree);
  for (std::int32_t i = 0; i < num_slaves_recv.size(); ++i)
    num_slaves_recv3[i] = num_slaves_recv[i] * 3;
  std::vector<int> disp3(indegree + 1, 0);
  std::partial_sum(num_slaves_recv3.begin(), num_slaves_recv3.end(),
                   disp3.begin() + 1);

  // Send slave normal and coordinate to neighbors
  std::vector<double> coord_recv(disp3.back());
  MPI_Neighbor_allgatherv(
      distribute_coordinates.data(), 3 * distribute_coordinates.size(),
      dolfinx::MPI::mpi_type<double>(), coord_recv.data(),
      num_slaves_recv3.data(), disp3.data(), dolfinx::MPI::mpi_type<double>(),
      neighborhood_comms[0]);
  std::vector<PetscScalar> normal_recv(disp3.back());
  MPI_Neighbor_allgatherv(distribute_normals.data(), distribute_normals.size(),
                          dolfinx::MPI::mpi_type<PetscScalar>(),
                          normal_recv.data(), num_slaves_recv3.data(),
                          disp3.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
                          neighborhood_comms[0]);

  // Vector for processes with slaves, mapping slaves with
  // collision on this process
  std::vector<std::vector<std::int64_t>> collision_slaves(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int64_t>>>
      collision_masters(indegree);
  std::vector<std::map<std::int64_t, std::vector<PetscScalar>>>
      collision_coeffs(indegree);
  std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
      collision_owners(indegree);

  std::array<double, 3> slave_coord;
  // Loop over slaves per incoming processor
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
    {
      // Extract slave coordinates and normals from incoming data
      for (std::int32_t k = 0; k < 3; ++k)
        slave_coord[k] = coord_recv[3 * j + k];
      xtl::span<const PetscScalar> slave_normal
          = xtl::span<PetscScalar>(normal_recv.data() + 3 * j, 3);

      // Find index of slave coord by using block size remainder
      std::int32_t local_index = slaves_recv[j] % block_size;

      // Use collision detection algorithm (GJK) to check distance from
      // cell to point and select one within 1e-10
      std::vector<int> bbox_collisions
          = dolfinx::geometry::compute_collisions(bb_tree, slave_coord);
      std::vector<std::int32_t> verified_candidates
          = dolfinx::geometry::select_colliding_cells(
              *V->mesh(), bbox_collisions, slave_coord, 1);
      if (verified_candidates.size() == 1)
      {
        // Compute local contribution of slave on master facet
        xt::xtensor<PetscScalar, 2> basis_values
            = get_basis_functions(V, slave_coord, verified_candidates[0]);
        auto cell_blocks = V->dofmap()->cell_dofs(verified_candidates[0]);
        std::vector<std::int32_t> l_block;
        std::vector<std::int32_t> l_rem;
        std::vector<PetscScalar> l_coeff;
        std::vector<std::int32_t> l_owner;
        for (std::int32_t d = 0; d < tdim; ++d)
        {
          for (std::int32_t k = 0; k < cell_blocks.size(); ++k)
          {
            for (std::int32_t block = 0; block < block_size; ++block)
            {
              const PetscScalar coeff = basis_values(k * block_size + block, d)
                                        * slave_normal[d]
                                        / slave_normal[local_index];
              if (std::abs(coeff) > 1e-6)
              {
                l_block.push_back(cell_blocks[k]);
                l_rem.push_back(block);
                l_coeff.push_back(coeff);
                if (cell_blocks[k] < size_local)
                  l_owner.push_back(rank);
                else
                  l_owner.push_back(ghost_owners[cell_blocks[k] - size_local]);
              }
            }
          }
        }
        std::vector<std::int64_t> global_block(l_block.size());
        imap->local_to_global(l_block, global_block);
        std::vector<std::int64_t> global_masters(l_block.size());
        for (std::int32_t k = 0; k < l_block.size(); ++k)
          global_masters[k] = global_block[k] * block_size + l_rem[k];
        collision_slaves[i].push_back(slaves_recv[j]);
        collision_masters[i][slaves_recv[j]] = global_masters;
        collision_coeffs[i][slaves_recv[j]] = l_coeff;
        collision_owners[i][slaves_recv[j]] = l_owner;
      }
    }
  }

  // Flatten data structures before send/recv
  std::vector<std::int32_t> num_collision_slaves(indegree);
  for (std::int32_t i = 0; i < indegree; ++i)
    num_collision_slaves[i] = collision_slaves[i].size();

  // Get info about reverse communicator
  int indegree_rev(-1), outdegree_rev(-2), weighted_rev(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree_rev,
                                 &outdegree_rev, &weighted_rev);
  const auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  // Communicate number of incoming slaves and masters after coll detection
  std::vector<int> inc_num_collision_slaves(indegree_rev);
  MPI_Neighbor_alltoall(num_collision_slaves.data(), 1, MPI_INT,
                        inc_num_collision_slaves.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

  std::vector<std::int32_t> num_collision_masters(indegree);
  std::vector<std::int64_t> collision_slaves_out;
  std::vector<std::int64_t> collision_masters_out;
  std::vector<std::int32_t> collision_offsets_out;
  std::vector<std::int32_t> collision_owners_out;
  std::vector<PetscScalar> collision_coeffs_out;
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    std::int32_t master_offset = 0;
    std::vector<std::int64_t>& slaves_i = collision_slaves[i];
    collision_slaves_out.insert(collision_slaves_out.end(), slaves_i.begin(),
                                slaves_i.end());
    for (auto slave : slaves_i)
    {
      std::vector<std::int64_t>& masters_ij = collision_masters[i][slave];
      num_collision_masters[i] += masters_ij.size();
      collision_masters_out.insert(collision_masters_out.end(),
                                   masters_ij.begin(), masters_ij.end());

      std::vector<PetscScalar>& coeffs_ij = collision_coeffs[i][slave];
      collision_coeffs_out.insert(collision_coeffs_out.end(), coeffs_ij.begin(),
                                  coeffs_ij.end());

      std::vector<std::int32_t>& owners_ij = collision_owners[i][slave];
      collision_owners_out.insert(collision_owners_out.end(), owners_ij.begin(),
                                  owners_ij.end());
      master_offset += masters_ij.size();
      collision_offsets_out.push_back(master_offset);
    }
  }

  std::vector<int> inc_num_collision_masters(indegree_rev);
  MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
                        inc_num_collision_masters.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

  // Create displacement vector for slaves and masters
  std::vector<int> disp_inc_slaves(indegree_rev + 1, 0);
  std::partial_sum(inc_num_collision_slaves.begin(),
                   inc_num_collision_slaves.end(), disp_inc_slaves.begin() + 1);
  std::vector<int> disp_inc_masters(indegree_rev + 1, 0);
  std::partial_sum(inc_num_collision_masters.begin(),
                   inc_num_collision_masters.end(),
                   disp_inc_masters.begin() + 1);

  // Compute send offsets
  std::vector<int> send_disp_slaves(indegree + 1, 0);
  std::partial_sum(num_collision_slaves.begin(), num_collision_slaves.end(),
                   send_disp_slaves.begin() + 1);
  std::vector<int> send_disp_masters(indegree + 1, 0);
  std::partial_sum(num_collision_masters.begin(), num_collision_masters.end(),
                   send_disp_masters.begin() + 1);

  // Receive colliding slaves from other processor
  std::vector<std::int64_t> remote_colliding_slaves(disp_inc_slaves.back());
  MPI_Neighbor_alltoallv(
      collision_slaves_out.data(), num_collision_slaves.data(),
      send_disp_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      remote_colliding_slaves.data(), inc_num_collision_slaves.data(),
      disp_inc_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      neighborhood_comms[1]);
  std::vector<std::int32_t> remote_colliding_offsets(disp_inc_slaves.back());
  MPI_Neighbor_alltoallv(
      collision_offsets_out.data(), num_collision_slaves.data(),
      send_disp_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      remote_colliding_offsets.data(), inc_num_collision_slaves.data(),
      disp_inc_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);
  // Receive colliding masters and relevant data from other processor
  std::vector<std::int64_t> remote_colliding_masters(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      collision_masters_out.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      remote_colliding_masters.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
      neighborhood_comms[1]);
  std::vector<PetscScalar> remote_colliding_coeffs(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      collision_coeffs_out.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      remote_colliding_coeffs.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      neighborhood_comms[1]);
  std::vector<std::int32_t> remote_colliding_owners(disp_inc_masters.back());
  MPI_Neighbor_alltoallv(
      collision_owners_out.data(), num_collision_masters.data(),
      send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      remote_colliding_owners.data(), inc_num_collision_masters.data(),
      disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);

  // Convert received slaves from other processors into local dofs
  const std::int32_t num_recv_slaves = remote_colliding_slaves.size();
  std::vector<std::int64_t> glob_slave_blocks(num_recv_slaves);
  std::vector<std::int64_t> glob_slave_rems(num_recv_slaves);
  for (std::int32_t i = 0; i < num_recv_slaves; ++i)
  {
    const std::int64_t bs = block_size;
    std::ldiv_t div = std::div(remote_colliding_slaves[i], bs);
    glob_slave_blocks[i] = div.quot;
    glob_slave_rems[i] = div.rem;
  }
  std::vector<std::int32_t> recv_blocks_as_local(glob_slave_blocks.size());
  imap->global_to_local(glob_slave_blocks, recv_blocks_as_local);
  std::vector<std::int32_t> recv_slaves_as_local(num_recv_slaves);
  for (std::int32_t i = 0; i < num_recv_slaves; ++i)
  {
    recv_slaves_as_local[i]
        = recv_blocks_as_local[i] * block_size + glob_slave_rems[i];
  }

  // Iterate through the processors
  for (std::int32_t i = 0; i < src_ranks_rev.size(); ++i)
  {
    const std::int32_t min = disp_inc_slaves[i];
    const std::int32_t max = disp_inc_slaves[i + 1];
    // Find offsets for masters on given proc
    std::vector<std::int32_t> master_offsets = {0};
    master_offsets.insert(master_offsets.end(),
                          remote_colliding_offsets.begin() + min,
                          remote_colliding_offsets.begin() + max);

    // Loop through slaves and add them if they havent already been added
    for (std::int32_t j = 0; j < master_offsets.size() - 1; ++j)
    {
      const std::int32_t slave = recv_slaves_as_local[min + j];
      const std::int64_t key = remote_colliding_slaves[min + j];
      assert(slave != -1);

      // Skip if already found on other incoming processor
      if (local_masters[key].size() == 0)
      {
        std::vector<std::int64_t> masters_(
            remote_colliding_masters.begin() + disp_inc_masters[i]
                + master_offsets[j],
            remote_colliding_masters.begin() + disp_inc_masters[i]
                + master_offsets[j + 1]);
        local_masters[key].insert(local_masters[key].end(), masters_.begin(),
                                  masters_.end());

        std::vector<PetscScalar> coeffs_(
            remote_colliding_coeffs.begin() + disp_inc_masters[i]
                + master_offsets[j],
            remote_colliding_coeffs.begin() + disp_inc_masters[i]
                + master_offsets[j + 1]);
        local_coeffs[key].insert(local_coeffs[key].end(), coeffs_.begin(),
                                 coeffs_.end());

        std::vector<std::int32_t> owners_(
            remote_colliding_owners.begin() + disp_inc_masters[i]
                + master_offsets[j],
            remote_colliding_owners.begin() + disp_inc_masters[i]
                + master_offsets[j + 1]);
        local_owners[key].insert(local_owners[key].end(), owners_.begin(),
                                 owners_.end());
      }
    }
  }
  // Merge arrays from other processors with those of the same block
  for (const auto& block_masters : masters)
  {
    std::int64_t key = block_masters.first;
    std::vector<std::int64_t> masters_ = block_masters.second;
    std::vector<PetscScalar> coeffs_ = coeffs[key];
    std::vector<std::int32_t> owners_ = owners[key];

    // Check that we have actually found a collision
    if (local_masters[key].size() == 0)
      throw std::runtime_error("Could not find contact for slave dof.");
    local_masters[key].insert(local_masters[key].end(), masters_.begin(),
                              masters_.end());
    local_coeffs[key].insert(local_coeffs[key].end(), coeffs_.begin(),
                             coeffs_.end());
    local_owners[key].insert(local_owners[key].end(), owners_.begin(),
                             owners_.end());
  }

  // Distribute data for ghosted slaves (the coeffs, owners and offsets)
  auto [src_ranks_ghost, dest_ranks_ghost]
      = dolfinx::MPI::neighbors(slave_to_ghost);
  // Count number of incoming slaves
  std::vector<std::int32_t> inc_num_slaves(src_ranks_ghost.size(), 0);
  for (std::int32_t i = 0; i < ghost_slaves.size(); ++i)
  {
    const std::int32_t owner
        = ghost_owners[ghost_slaves[i] / block_size - size_local];
    std::vector<int>::iterator it
        = std::find(src_ranks_ghost.begin(), src_ranks_ghost.end(), owner);
    std::int32_t index = std::distance(src_ranks_ghost.begin(), it);
    inc_num_slaves[index]++;
  }
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

  for (std::int32_t i = 0; i < local_slaves_as_glob.size(); ++i)
  {
    const std::int64_t glob_slave = local_slaves_as_glob[i];
    std::vector<std::int64_t> masters_i = local_masters[glob_slave];
    std::vector<PetscScalar> coeffs_i = local_coeffs[glob_slave];
    std::vector<std::int32_t> owners_i = local_owners[glob_slave];
    std::int32_t num_masters = masters_i.size();
    std::set<int> ghost_procs = shared_indices[local_slaves[i] / block_size];
    for (auto proc : ghost_procs)
    {
      auto it
          = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(), proc);
      std::int32_t index = std::distance(dest_ranks_ghost.begin(), it);
      out_num_masters[index] += num_masters;
      out_num_slaves[index]++;
      // Map slaves to global dof to be recognized by recv proc
      proc_to_ghost[index].push_back(glob_slave);
      // Add master data in process-wise fashion
      proc_to_ghost_masters[index].insert(proc_to_ghost_masters[index].end(),
                                          masters_i.begin(), masters_i.end());
      proc_to_ghost_coeffs[index].insert(proc_to_ghost_coeffs[index].end(),
                                         coeffs_i.begin(), coeffs_i.end());
      proc_to_ghost_owners[index].insert(proc_to_ghost_owners[index].end(),
                                         owners_i.begin(), owners_i.end());
      proc_to_ghost_offsets[index].push_back(
          proc_to_ghost_masters[index].size());
    }
  }

  // Flatten map of global slave ghost dofs to use alltoallv
  std::vector<std::int64_t> out_ghost_slaves;
  std::vector<std::int64_t> out_ghost_masters;
  std::vector<PetscScalar> out_ghost_coeffs;
  std::vector<std::int32_t> out_ghost_owners;
  std::vector<std::int32_t> out_ghost_offsets;
  std::vector<std::int32_t> num_send_slaves(dest_ranks_ghost.size());
  std::vector<std::int32_t> num_send_masters(dest_ranks_ghost.size());
  for (std::int32_t i = 0; i < dest_ranks_ghost.size(); ++i)
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
  for (std::int32_t i = 0; i < src_ranks_ghost.size(); ++i)
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

  std::vector<std::int64_t> masters_out;
  std::vector<PetscScalar> coeffs_out;
  std::vector<std::int32_t> owners_out;
  std::vector<std::int32_t> offsets_out = {0};
  // Flatten local slaves
  for (auto i : local_slaves_as_glob)
  {
    masters_out.insert(masters_out.end(), local_masters[i].begin(),
                       local_masters[i].end());
    coeffs_out.insert(coeffs_out.end(), local_coeffs[i].begin(),
                      local_coeffs[i].end());
    offsets_out.push_back(masters_out.size());
    owners_out.insert(owners_out.end(), local_owners[i].begin(),
                      local_owners[i].end());
  }

  // Map Ghost data
  const std::int32_t num_ghost_slaves = in_ghost_slaves.size();
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
  std::vector<std::int32_t> in_ghost_slaves_loc(num_ghost_slaves);
  for (std::int32_t i = 0; i < num_ghost_slaves; ++i)
  {
    in_ghost_slaves_loc[i]
        = in_ghost_block_loc[i] * block_size + in_ghost_rem[i];
  }

  mpc.local_slaves = local_slaves;
  mpc.ghost_slaves = in_ghost_slaves_loc;
  mpc.local_masters = masters_out;
  mpc.ghost_masters = in_ghost_masters;
  mpc.local_offsets = offsets_out;
  mpc.ghost_offsets = ghost_offsets;
  mpc.local_owners = owners_out;
  mpc.ghost_owners = in_ghost_owners;
  mpc.local_coeffs = coeffs_out;
  mpc.ghost_coeffs = in_ghost_coeffs;
  timer.stop();
  return mpc;
}
//-----------------------------------------------------------------------------
mpc_data dolfinx_mpc::create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker)
{
  dolfinx::common::Timer timer("~MPC: Inelastic condition");

  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int tdim = V->mesh()->topology().dim();
  const int gdim = V->mesh()->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V->dofmap()->index_map_bs();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  // Create entity permutations needed in evaluate_basis_functions
  V->mesh()->topology_mutable().create_entity_permutations();
  // Create connectivities needed for evaluate_basis_functions and
  // select_colliding cells
  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
  V->mesh()->topology_mutable().create_connectivity(tdim, tdim);

  // Extract slave_facets
  std::vector<std::int32_t> slave_facets;
  for (std::int32_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == slave_marker)
      slave_facets.push_back(meshtags.indices()[i]);

  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  std::pair<std::shared_ptr<dolfinx::fem::FunctionSpace>,
            std::vector<std::int32_t>>
      V0_pair = V->sub({0})->collapse();
  auto [V0, map] = V0_pair;
  std::array<std::vector<std::int32_t>, 2> slave_blocks
      = dolfinx::fem::locate_dofs_topological(
          {*V->sub({0}).get(), *V0.get()}, fdim, tcb::make_span(slave_facets));
  for (std::size_t i = 0; i < slave_blocks[0].size(); ++i)
    slave_blocks[0][i] /= block_size;

  // Make maps for local data of slave and master coeffs
  std::vector<std::int32_t> local_block;
  std::vector<std::vector<std::int32_t>> local_slaves(tdim);

  // Array holding ghost slaves blocks (masters,coeffs and offsets will be
  // received)
  std::vector<std::int32_t> ghost_blocks;

  for (std::int32_t i = 0; i < slave_blocks[0].size(); ++i)
  {
    if (slave_blocks[0][i] < size_local)
    {
      local_block.push_back(slave_blocks[0][i]);
      for (std::int32_t j = 0; j < tdim; ++j)
        local_slaves[j].push_back(block_size * slave_blocks[0][i] + j);
    }
    else
      ghost_blocks.push_back(slave_blocks[0][i]);
  }
  std::vector<std::int64_t> local_block_as_glob(local_block.size());
  imap->local_to_global(local_block, local_block_as_glob);

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = local_block.size() > 0 ? 1 : 0;
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, has_slave, master_marker);

  // Create communicator local_block -> ghost_block
  MPI_Comm slave_to_ghost
      = create_owner_to_ghost_comm(local_block, ghost_blocks, imap, 1);

  // Create new index-map where there are only ghosts for slaves
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    std::vector<int> slave_ranks(ghost_blocks.size());
    for (std::int32_t i = 0; i < ghost_blocks.size(); ++i)
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
    for (std::int32_t i = 0; i < facet_indices.size(); ++i)
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
  // Loop through all masters on current processor and check if they collide
  // with a local master facet

  std::map<std::int32_t, std::vector<std::int32_t>> local_owners;
  std::map<std::int32_t, std::vector<std::int64_t>> local_masters;
  std::map<std::int32_t, std::vector<PetscScalar>> local_coeffs;

  xt::xtensor<double, 2> coordinates = V->tabulate_dof_coordinates(false);
  std::vector<std::array<double, 3>> local_slave_coordinates(
      local_block.size());
  std::vector<std::int64_t> blocks_wo_local_collision;

  // Map to get coordinates and normals of slaves that should be distributed
  std::vector<int> master_cells_vec(master_cells.begin(), master_cells.end());
  dolfinx::geometry::BoundingBoxTree bb_tree(*V->mesh(), tdim, master_cells_vec,
                                             1e-12);

  std::vector<std::int32_t> collision_to_local;
  for (std::int32_t i = 0; i < local_block.size(); ++i)
  {
    // Get coordinate and coeff for slave and save in send array
    const std::int32_t block = local_block[i];
    const std::array<double, 3> coord
        = {coordinates(block, 0), coordinates(block, 1), coordinates(block, 2)};
    local_slave_coordinates[i] = coord;

    // Use collision detection algorithm (GJK) to check distance from  cell
    // to point and select one within 1e-10
    std::vector<int> bbox_collisions
        = dolfinx::geometry::compute_collisions(bb_tree, coord);
    std::vector<std::int32_t> verified_candidates
        = dolfinx::geometry::select_colliding_cells(*V->mesh(), bbox_collisions,
                                                    coord, 1);
    if (verified_candidates.size() == 1)
    {

      // Compute interpolation on other cell
      xt::xtensor<PetscScalar, 2> basis_values
          = dolfinx_mpc::get_basis_functions(V, coord, verified_candidates[0]);
      auto cell_blocks = V->dofmap()->cell_dofs(verified_candidates[0]);

      std::vector<std::vector<std::int32_t>> l_master(3);
      std::vector<std::vector<PetscScalar>> l_coeff(3);
      std::vector<std::vector<std::int32_t>> l_owner(3);
      // Create arrays per topological dimension
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        for (std::int32_t k = 0; k < cell_blocks.size(); ++k)
        {
          for (std::int32_t block = 0; block < block_size; ++block)
          {
            const PetscScalar coeff = basis_values(k * block_size + block, j);
            if (std::abs(coeff) > 1e-6)
            {
              l_master[j].push_back(cell_blocks[k] * block_size + block);
              l_coeff[j].push_back(coeff);
              if (cell_blocks[k] < size_local)
                l_owner[j].push_back(rank);
              else
                l_owner[j].push_back(ghost_owners[cell_blocks[k] - size_local]);
            }
          }
        }
      }
      bool found_masters = false;
      // Add masters for each master in the block
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        const std::vector<std::int32_t>& masters_j = l_master[j];
        const int32_t num_masters = masters_j.size();
        if (num_masters > 0)
        {
          const std::int32_t local_slave = local_slaves[j][i];
          // Convert local masteer dofs to their corresponding global dof
          std::vector<std::int64_t> masters_as_global(num_masters);
          std::vector<std::int32_t> master_block_loc(num_masters);
          std::vector<std::int64_t> master_block_glob(num_masters);
          std::vector<std::int32_t> masters_rem(num_masters);
          for (std::int32_t k = 0; k < num_masters; ++k)
          {
            std::div_t div = std::div(masters_j[k], block_size);
            master_block_loc[k] = div.quot;
            masters_rem[k] = div.rem;
          }
          imap->local_to_global(master_block_loc, master_block_glob);
          for (std::int32_t k = 0; k < num_masters; ++k)
          {
            masters_as_global[k]
                = master_block_glob[k] * block_size + masters_rem[k];
          }
          local_masters[local_slave] = masters_as_global;
          local_coeffs[local_slave] = l_coeff[j];
          local_owners[local_slave] = l_owner[j];
          found_masters = true;
        }
      }
      if (!found_masters)
      {
        blocks_wo_local_collision.push_back(local_block_as_glob[i]);
        collision_to_local.push_back(i);
      }
    }
    else
    {
      blocks_wo_local_collision.push_back(local_block_as_glob[i]);
      collision_to_local.push_back(i);
    }
  }
  // Extract coordinates and normals to distribute
  std::vector<std::array<double, 3>> distribute_coordinates(
      blocks_wo_local_collision.size());
  for (std::int32_t i = 0; i < collision_to_local.size(); ++i)
  {
    distribute_coordinates[i] = local_slave_coordinates[collision_to_local[i]];
  }

  dolfinx_mpc::mpc_data mpc;

  // If serial, we only have to gather slaves, masters, coeffs in 1D arrays
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1)
  {
    assert(blocks_wo_local_collision.size() == 0);
    std::vector<std::int64_t> masters_out;
    std::vector<PetscScalar> coeffs_out;
    std::vector<std::int32_t> offsets_out = {0};
    std::vector<std::int32_t> slaves_out;
    // Flatten the maps to 1D arrays (assuming all slaves are local slaves)
    std::vector<std::int32_t> slaves;
    for (std::int32_t j = 0; j < tdim; ++j)
    {
      for (std::int32_t i = 0; i < local_slaves[j].size(); ++i)
      {
        auto slave = local_slaves[j][i];
        slaves.push_back(slave);
        masters_out.insert(masters_out.end(), local_masters[slave].begin(),
                           local_masters[slave].end());
        coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                          local_coeffs[slave].end());
        offsets_out.push_back(masters_out.size());
      }
    }

    // Serial assumptions
    std::vector<std::int32_t> owners(masters_out.size());
    std::fill(owners.begin(), owners.end(), 1);
    std::vector<std::int32_t> gs(0);
    std::vector<std::int64_t> gm(0);
    std::vector<PetscScalar> gc(0);
    std::vector<std::int32_t> go(0);
    std::vector<std::int32_t> goff(1);
    std::fill(goff.begin(), goff.end(), 0);
    mpc.local_slaves = slaves;
    mpc.ghost_slaves = gs;
    mpc.local_masters = masters_out;
    mpc.ghost_masters = gm;
    mpc.local_offsets = offsets_out;
    mpc.ghost_offsets = goff;
    mpc.local_owners = owners;
    mpc.ghost_owners = go;
    mpc.local_coeffs = coeffs_out;
    mpc.ghost_coeffs = gc;

    return mpc;
  }

  // Get the slave->master recv from and send to ranks
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);
  const auto [src_ranks, dest_ranks]
      = dolfinx::MPI::neighbors(neighborhood_comms[0]);

  // Figure out how much data to receive from each neighbor
  const std::int32_t num_colliding_blocks = blocks_wo_local_collision.size();
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
      blocks_wo_local_collision.data(), blocks_wo_local_collision.size(),
      dolfinx::MPI::mpi_type<std::int64_t>(), remote_slave_blocks.data(),
      num_slave_blocks.data(), disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[0]);

  // Multiply recv size by three to accommodate block coordinates
  std::vector<std::int32_t> num_block_coordinates(indegree);
  for (std::int32_t i = 0; i < num_slave_blocks.size(); ++i)
    num_block_coordinates[i] = num_slave_blocks[i] * 3;
  std::vector<int> coordinate_disp(indegree + 1, 0);
  std::partial_sum(num_block_coordinates.begin(), num_block_coordinates.end(),
                   coordinate_disp.begin() + 1);

  // Send slave coordinates to neighbors
  std::vector<double> coord_recv(coordinate_disp.back());
  MPI_Neighbor_allgatherv(
      distribute_coordinates.data(), 3 * distribute_coordinates.size(),
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
  // Loop over slaves per incoming processor
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
    {
      // Extract slave coordinates and normals from incoming data
      const std::array<double, 3> slave_coord
          = {coord_recv[3 * j], coord_recv[3 * j + 1], coord_recv[3 * j + 2]};
      // Use collision detection algorithm (GJK) to check distance from
      // cell to point and select one within 1e-10
      std::vector<int> bbox_collisions
          = dolfinx::geometry::compute_collisions(bb_tree, slave_coord);
      std::vector<std::int32_t> verified_candidates
          = dolfinx::geometry::select_colliding_cells(
              *V->mesh(), bbox_collisions, slave_coord, 1);
      if (verified_candidates.size() == 1)
      {
        // Compute local contribution of slave on master facet
        xt::xtensor<PetscScalar, 2> basis_values
            = get_basis_functions(V, slave_coord, verified_candidates[0]);
        auto cell_blocks = V->dofmap()->cell_dofs(verified_candidates[0]);
        std::vector<std::vector<std::int32_t>> l_master(3);
        std::vector<std::vector<PetscScalar>> l_coeff(3);
        std::vector<std::vector<std::int32_t>> l_owner(3);
        // Create arrays per topological dimension
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          for (std::int32_t k = 0; k < cell_blocks.size(); ++k)
          {
            for (std::int32_t block = 0; block < block_size; ++block)
            {
              const PetscScalar coeff = basis_values(k * block_size + block, j);
              if (std::abs(coeff) > 1e-6)
              {
                l_master[j].push_back(cell_blocks[k] * block_size + block);
                l_coeff[j].push_back(coeff);
                if (cell_blocks[k] < size_local)
                  l_owner[j].push_back(rank);
                else
                {
                  l_owner[j].push_back(
                      ghost_owners[cell_blocks[k] - size_local]);
                }
              }
            }
          }
        }
        // Collect masters per block
        std::vector<std::int32_t> remote_owners;
        std::vector<std::int64_t> remote_masters;
        std::vector<PetscScalar> remote_coeffs;
        std::vector<std::int32_t> num_masters_per_slave(tdim);
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          const std::vector<std::int32_t>& masters_j = l_master[j];
          const int32_t num_masters = masters_j.size();
          if (masters_j.size() > 0)
          {
            // Convert local masteer dofs to their corresponding global dof
            std::vector<std::int64_t> global_masters(num_masters);
            std::vector<std::int32_t> masters_block_loc(num_masters);
            std::vector<std::int64_t> masters_block_glob(num_masters);
            std::vector<std::int32_t> masters_rem(num_masters);
            for (std::int32_t k = 0; k < num_masters; ++k)
            {
              std::div_t div = std::div(masters_j[k], block_size);
              masters_block_loc[k] = div.quot;
              masters_rem[k] = div.rem;
            }
            imap->local_to_global(masters_block_loc, masters_block_glob);
            for (std::int32_t k = 0; k < num_masters; ++k)
            {
              global_masters[k]
                  = masters_block_glob[k] * block_size + masters_rem[k];
            }
            remote_masters.insert(remote_masters.end(), global_masters.begin(),
                                  global_masters.end());
            remote_coeffs.insert(remote_coeffs.end(), l_coeff[j].begin(),
                                 l_coeff[j].end());
            remote_owners.insert(remote_owners.end(), l_owner[j].begin(),
                                 l_owner[j].end());
            num_masters_per_slave[j] = global_masters.size();
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
    num_found_slave_blocks[i] = collision_slaves[i].size();

  // Get info about reverse communicator (masters->slaves)
  int indegree_rev(-1), outdegree_rev(-2), weighted_rev(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree_rev,
                                 &outdegree_rev, &weighted_rev);
  const auto [src_ranks_rev, dest_ranks_rev]
      = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  // Communicate number of incoming slaves and masters after coll detection
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
  for (std::int32_t i = 0; i < recv_num_found_blocks.size(); ++i)
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
  for (std::int32_t i = 0; i < src_ranks_rev.size(); ++i)
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
    for (std::int32_t j = 0; j < master_offsets.size() - 1; ++j)
    {
      // Get the slave block
      const std::int32_t local_block = recv_blocks_as_local[min + j];
      const std::int32_t global_block = remote_colliding_blocks[min + j];
      std::vector<std::int32_t> block_offsets(tdim + 1, 0);
      std::partial_sum(num_dofs_per_block.begin() + j * tdim,
                       num_dofs_per_block.begin() + (j + 1) * tdim,
                       block_offsets.begin() + 1);

      for (std::int32_t k = 0; k < tdim; ++k)
      {
        std::int32_t local_slave = local_block * block_size + k;

        // Skip if already found on other incoming processor
        if (local_masters[local_slave].size() == 0)
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
  for (std::int32_t i = 0; i < ghost_blocks.size(); ++i)
    for (std::int32_t j = 0; j < tdim; ++j)
      ghost_slaves[i * tdim + j] = ghost_blocks[i] * block_size + j;

  auto [src_ranks_ghost, dest_ranks_ghost]
      = dolfinx::MPI::neighbors(slave_to_ghost);
  // Count number of incoming slaves
  std::vector<std::int32_t> inc_num_slaves(src_ranks_ghost.size(), 0);
  for (std::int32_t i = 0; i < ghost_slaves.size(); ++i)
  {
    const std::int32_t owner
        = ghost_owners[ghost_slaves[i] / block_size - size_local];
    std::vector<int>::iterator it
        = std::find(src_ranks_ghost.begin(), src_ranks_ghost.end(), owner);
    std::int32_t index = std::distance(src_ranks_ghost.begin(), it);
    inc_num_slaves[index]++;
  }

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
  for (std::int32_t j = 0; j < tdim; ++j)
  {
    for (std::int32_t i = 0; i < local_slaves[j].size(); ++i)
    {
      auto slave = local_slaves[j][i];

      std::vector<std::int64_t> masters_i = local_masters[slave];
      std::vector<PetscScalar> coeffs_i = local_coeffs[slave];
      std::vector<std::int32_t> owners_i = local_owners[slave];
      std::int32_t num_masters = masters_i.size();
      std::set<int> ghost_procs = shared_indices[slave / block_size];
      for (auto proc : ghost_procs)
      {
        auto it
            = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(), proc);
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
        proc_to_ghost_masters[index].insert(proc_to_ghost_masters[index].end(),
                                            masters_i.begin(), masters_i.end());
        proc_to_ghost_coeffs[index].insert(proc_to_ghost_coeffs[index].end(),
                                           coeffs_i.begin(), coeffs_i.end());
        proc_to_ghost_owners[index].insert(proc_to_ghost_owners[index].end(),
                                           owners_i.begin(), owners_i.end());
        proc_to_ghost_offsets[index].push_back(
            proc_to_ghost_masters[index].size());
      }
    }
  }
  // Flatten map of global slave ghost dofs to use alltoallv
  std::vector<std::int64_t> out_ghost_slaves;
  std::vector<std::int64_t> out_ghost_masters;
  std::vector<PetscScalar> out_ghost_coeffs;
  std::vector<std::int32_t> out_ghost_owners;
  std::vector<std::int32_t> out_ghost_offsets;
  std::vector<std::int32_t> num_send_slaves(dest_ranks_ghost.size());
  std::vector<std::int32_t> num_send_masters(dest_ranks_ghost.size());
  for (std::int32_t i = 0; i < dest_ranks_ghost.size(); ++i)
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
  for (std::int32_t i = 0; i < src_ranks_ghost.size(); ++i)
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
  std::vector<std::int64_t> masters;
  std::vector<PetscScalar> coeffs_out;
  std::vector<std::int32_t> owners_out;
  std::vector<std::int32_t> offsets = {0};
  for (std::int32_t j = 0; j < tdim; ++j)
  {
    for (std::int32_t i = 0; i < local_slaves[j].size(); ++i)
    {
      auto slave = local_slaves[j][i];
      slaves.push_back(slave);
      masters.insert(masters.end(), local_masters[slave].begin(),
                     local_masters[slave].end());
      coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                        local_coeffs[slave].end());
      offsets.push_back(masters.size());
      owners_out.insert(owners_out.end(), local_owners[slave].begin(),
                        local_owners[slave].end());
    }
  }

  // Map Ghost data
  const std::int32_t num_ghost_slaves = in_ghost_slaves.size();
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
  std::vector<std::int32_t> in_ghost_slaves_loc(num_ghost_slaves);
  for (std::int32_t i = 0; i < num_ghost_slaves; ++i)
  {
    in_ghost_slaves_loc[i]
        = in_ghost_block_loc[i] * block_size + in_ghost_rem[i];
  }
  mpc.local_slaves = slaves;
  mpc.ghost_slaves = in_ghost_slaves_loc;
  mpc.local_masters = masters;
  mpc.ghost_masters = in_ghost_masters;
  mpc.local_offsets = offsets;
  mpc.ghost_offsets = ghost_offsets;
  mpc.local_owners = owners_out;
  mpc.ghost_owners = in_ghost_owners;
  mpc.local_coeffs = coeffs_out;
  mpc.ghost_coeffs = in_ghost_coeffs;
  timer.stop();
  return mpc;
}
//-----------------------------------------------------------------------------

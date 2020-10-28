// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ContactConstraint.h"
#include "MultiPointConstraint.h"
#include "utils.h"
#include <chrono>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/geometry/utils.h>

using namespace dolfinx_mpc;

mpc_data dolfinx_mpc::create_contact_slip_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> nh)
{
  dolfinx::common::Timer timer0("~DEBUG: 0. Total time");

  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);
  dolfinx::common::Timer timer("~DEBUG: 1. Condition on own proc");
  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int tdim = V->mesh()->topology().dim();
  const int gdim = V->mesh()->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V->dofmap()->index_map->block_size();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  // Extract slave_facets
  std::vector<std::int32_t> _slave_facets;
  for (std::int32_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == slave_marker)
      _slave_facets.push_back(meshtags.indices()[i]);
  Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>> slave_facets(
      _slave_facets.data(), _slave_facets.size());

  // Extract dofs on slave facets per dimension
  std::vector<Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>>
      dofs_on_slave_facet(tdim);

  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  std::pair<std::shared_ptr<dolfinx::function::FunctionSpace>,
            std::vector<std::int32_t>>
      V0_pair = V->sub({0})->collapse();
  auto [V0, map] = V0_pair;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> slave_blocks
      = dolfinx::fem::locate_dofs_topological({*V->sub({0}).get(), *V0.get()},
                                              fdim, slave_facets);
  slave_blocks /= block_size;

  // Make maps for local data of slave and master coeffs
  std::vector<std::int32_t> local_slaves;
  std::vector<std::int32_t> local_slaves_as_glob;
  std::map<std::int64_t, std::vector<std::int64_t>> masters;
  std::map<std::int64_t, std::vector<PetscScalar>> coeffs;
  std::map<std::int64_t, std::vector<std::int32_t>> owners;

  // Array holding ghost slaves (masters,coeffs and offsets will be recieved)
  std::vector<std::int32_t> ghost_slaves;

  // Determine which dof should be a slave dof (based on max component of
  // normal vector normal vector)
  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> normal_array
      = nh->x()->array();
  for (std::int32_t i = 0; i < slave_blocks.rows(); ++i)
  {
    std::vector<std::int32_t> dofs(block_size);
    std::iota(dofs.begin(), dofs.end(), slave_blocks(i, 0) * block_size);
    // Create normalized local normal vector and fin its max index
    Eigen::Array<PetscScalar, 1, Eigen::Dynamic> l_normal(1, tdim);
    for (std::int32_t j = 0; j < tdim; ++j)
      l_normal[j] = normal_array[dofs[j]];
    l_normal /= std::sqrt(l_normal.abs2().sum());

    Eigen::Index max_index;
    const double coeff = l_normal.abs().maxCoeff(&max_index);

    // Compute coeffs if slave is owned by the processor
    if (dofs[max_index] < size_local * block_size)
    {
      std::vector<std::int32_t> masters_i;
      std::vector<PetscScalar> coeffs_i;
      std::vector<std::int32_t> owners_i;
      auto slave_glob = imap->local_to_global({dofs[max_index]}, false);
      local_slaves.push_back(dofs[max_index]);
      local_slaves_as_glob.push_back(slave_glob[0]);

      for (Eigen::Index j = 0; j < tdim; ++j)
        if (j != max_index)
        {
          PetscScalar coeff_j = -l_normal[j] / l_normal[max_index];
          if (std::abs(coeff_j) > 1e-6)
          {
            owners_i.push_back(rank);
            masters_i.push_back(dofs[j]);
            coeffs_i.push_back(coeff_j);
          }
        }
      if (masters_i.size() > 0)
      {
        std::vector<std::int64_t> m_glob
            = imap->local_to_global(masters_i, false);
        masters.insert({slave_glob[0], m_glob});
        coeffs.insert({slave_glob[0], coeffs_i});
        owners.insert({slave_glob[0], owners_i});
      }
    }
    else
      ghost_slaves.push_back(dofs[max_index]);
  }

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = local_slaves.size() > 0 ? 1 : 0;
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, has_slave, master_marker);
  // Create communicator local_slaves -> ghost_slaves
  MPI_Comm slave_to_ghost
      = create_owner_to_ghost_comm(local_slaves, ghost_slaves, imap);

  // Create new index-map where there are only ghosts for slaves
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners
      = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    std::vector<int> slave_ranks(ghost_slaves.size());
    for (std::int32_t i = 0; i < ghost_slaves.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_slaves[i] / block_size - size_local];
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> ghosts_eigen(
        ghost_slaves.data(), ghost_slaves.size());
    auto ghosts_as_global = imap->local_to_global(ghosts_eigen, false);
    ghosts_as_global /= block_size;
    slave_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, imap->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(slave_ranks.begin(), slave_ranks.end())),
        ghosts_as_global, slave_ranks, block_size);
  }

  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
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

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> coordinates
      = V->tabulate_dof_coordinates();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      local_slave_coordinates(local_slaves.size(), 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 3, Eigen::RowMajor>
      local_slave_normal(local_slaves.size(), 3);
  std::vector<std::int64_t> slaves_wo_local_collision;

  // Map to get coordinates and normals of slaves that should be distributed
  std::vector<std::int32_t> collision_to_local;

  for (std::int32_t i = 0; i < local_slaves.size(); ++i)
  {
    // Get coordinate and coeff for slave and save in send array
    auto coord = coordinates.row(local_slaves[i]);
    local_slave_coordinates.row(i) = coord;
    Eigen::Array<PetscScalar, 1, 3> coeffs;
    std::vector<std::int32_t> dofs(block_size);
    std::iota(dofs.begin(), dofs.end(), slave_blocks(i, 0) * block_size);
    for (std::int32_t j = 0; j < gdim; ++j)
      coeffs[j] = normal_array[dofs[j]];
    // Pad 2D spaces with zero in normal
    for (std::int32_t j = gdim; j < 3; ++j)
      coeffs[j] = 0;
    coeffs /= std::sqrt(coeffs.abs2().sum());
    Eigen::Index max_index;
    const double _c = coeffs.abs().maxCoeff(&max_index);

    local_slave_normal.row(i) = coeffs;

    // Use collision detection algorithm (GJK) to check distance from  cell
    // to point and select one within 1e-10
    std::vector<std::int32_t> candidates;
    for (auto cell : master_cells)
      candidates.push_back(cell);
    std::vector<std::int32_t> verified_candidates
        = dolfinx::geometry::select_colliding_cells(*V->mesh(), candidates,
                                                    coord, 1);
    if (verified_candidates.size() == 1)
    {
      // Compute local contribution of slave on master facet
      Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> basis_values
          = get_basis_functions(V, coord, verified_candidates[0]);
      auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);
      std::vector<std::int32_t> l_master;
      std::vector<PetscScalar> l_coeff;
      std::vector<std::int32_t> l_owner;
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        for (std::int32_t k = 0; k < cell_dofs.size(); ++k)
        {
          PetscScalar coeff
              = basis_values(k, j) * coeffs[j] / coeffs[max_index];
          if (std::abs(coeff) > 1e-6)
          {
            l_master.push_back(cell_dofs(k));
            l_coeff.push_back(coeff);
            if (cell_dofs[k] < size_local * block_size)
              l_owner.push_back(rank);
            else
              l_owner.push_back(
                  ghost_owners[cell_dofs[k] / block_size - size_local]);
          }
        }
      }
      if (l_master.size() > 0)
      {
        auto global_masters = imap->local_to_global(l_master, false);
        local_masters.insert(
            std::pair(local_slaves_as_glob[i], global_masters));
        local_coeffs.insert(std::pair(local_slaves_as_glob[i], l_coeff));
        local_owners.insert(std::pair(local_slaves_as_glob[i], l_owner));
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
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      distribute_coordinates(slaves_wo_local_collision.size(), 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 3, Eigen::RowMajor>
      distribute_normals(slaves_wo_local_collision.size(), 3);
  for (std::int32_t i = 0; i < collision_to_local.size(); ++i)
  {
    distribute_coordinates.row(i)
        = local_slave_coordinates.row(collision_to_local[i]);
    distribute_normals.row(i) = local_slave_normal.row(collision_to_local[i]);
  }
  timer.stop();
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
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gowners(0);
    // Map vectors to Eigen arrays
    Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> masters(
        masters_out.data(), masters_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> slaves(
        local_slaves.data(), local_slaves.size());
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coeffs(
        coeffs_out.data(), coeffs_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> offsets(
        offsets_out.data(), offsets_out.size());

    // Serial assumptions
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(masters_out.size());
    owners.fill(1);
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> gm(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gs(0, 1);
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> gc(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> go(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> goff(1, 1);
    goff.fill(0);
    mpc.local_slaves = slaves;
    mpc.ghost_slaves = gs;
    mpc.local_masters = masters;
    mpc.ghost_masters = gm;
    mpc.local_offsets = offsets;
    mpc.ghost_offsets = goff;
    mpc.local_owners = owners;
    mpc.ghost_owners = go;
    mpc.local_coeffs = coeffs;
    mpc.ghost_coeffs = gc;

    return mpc;
  }

  dolfinx::common::Timer tt("~DEBUG: 2. First communication ");

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
      distribute_coordinates.data(), distribute_coordinates.size(),
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

  tt.stop();
  auto timer_start2 = std::chrono::system_clock::now();
  std::chrono::duration<double> dt_select = (timer_start2 - timer_start2);

  dolfinx::common::Timer ts("~DEBUG: 3. Remote collision");
  // Loop over slaves per incoming processor
  for (std::int32_t i = 0; i < indegree; ++i)
  {
    for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
    {
      // Extract slave coordinates and normals from incoming data
      Eigen::Map<const Eigen::Array<double, 1, 3>> slave_coord(
          coord_recv.data() + 3 * j, 3);
      Eigen::Map<const Eigen::Array<PetscScalar, 3, 1>> slave_normal(
          normal_recv.data() + 3 * j, 3);

      // Find index of slave coord by using block size remainder
      std::int32_t local_index = slaves_recv[j] % block_size;

      std::vector<std::int32_t> candidates(master_cells.begin(),
                                           master_cells.end());

      // Use collision detection algorithm (GJK) to check distance from
      // cell to point and select one within 1e-10
      auto timer_start_select = std::chrono::system_clock::now();

      std::vector<std::int32_t> verified_candidates
          = dolfinx::geometry::select_colliding_cells(*V->mesh(), candidates,
                                                      slave_coord, 1);
      auto timer_end_select = std::chrono::system_clock::now();
      dt_select += (timer_end_select - timer_start_select);
      if (verified_candidates.size() == 1)
      {
        // Compute local contribution of slave on master facet
        Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> basis_values
            = get_basis_functions(V, slave_coord, verified_candidates[0]);
        auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);
        std::vector<std::int32_t> l_master;
        std::vector<PetscScalar> l_coeff;
        std::vector<std::int32_t> l_owner;
        for (std::int32_t d = 0; d < tdim; ++d)
        {
          for (std::int32_t k = 0; k < cell_dofs.size(); ++k)
          {
            PetscScalar coeff = basis_values(k, d) * slave_normal[d]
                                / slave_normal[local_index];
            if (std::abs(coeff) > 1e-6)
            {
              l_master.push_back(cell_dofs(k));
              l_coeff.push_back(coeff);
              if (cell_dofs[k] < size_local * block_size)
                l_owner.push_back(rank);
              else

                l_owner.push_back(
                    ghost_owners[cell_dofs[k] / block_size - size_local]);
            }
          }
        }
        auto global_masters = imap->local_to_global(l_master, false);
        collision_slaves[i].push_back(slaves_recv[j]);
        collision_masters[i].insert(std::pair(slaves_recv[j], global_masters));
        collision_coeffs[i].insert(std::pair(slaves_recv[j], l_coeff));
        collision_owners[i].insert(std::pair(slaves_recv[j], l_owner));
      }
    }
  }
  ts.stop();
  auto timer_end2 = std::chrono::system_clock::now();
  std::chrono::duration<double> dt2 = (timer_end2 - timer_start2);

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

  std::stringstream ss;
  dolfinx::common::Timer tm("~DEBUG: 4. Masters->slaves (Send incoming sizes)");
  // Communicate number of incoming slaves and masters after coll detection
  auto timer_start = std::chrono::system_clock::now();
  std::vector<int> inc_num_collision_slaves(indegree_rev);
  MPI_Neighbor_alltoall(num_collision_slaves.data(), 1, MPI_INT,
                        inc_num_collision_slaves.data(), 1, MPI_INT,
                        neighborhood_comms[1]);

  auto timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt = (timer_end - timer_start);

  ss << " " << rank << " Loc slaves" << local_slaves.size() << " Collisions "
     << std::accumulate(num_collision_slaves.begin(),
                        num_collision_slaves.end(), 0)
     << " "
     << "Remote collision. " << dt2.count() << "GJK " << dt_select.count()
     << " MAster cells " << master_cells.size() << " Found local "
     << local_masters.size() << " Comm. " << dt.count() << "\n";
  std::cout << ss.str() << "\n";
  tm.stop();
  dolfinx::common::Timer tz("~DEBUG: 5. Flatten data");
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
  tz.stop();
  dolfinx::common::Timer timer2("~DEBUG: 6. Masters-> slaves");

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

  timer2.stop();
  dolfinx::common::Timer tl("~DEBUG: 7. Add masters from remote procs");

  auto recv_slaves_as_local
      = imap->global_to_local(remote_colliding_slaves, false);
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
  tl.stop();

  dolfinx::common::Timer timer3("~DEBUG: 8. Distribute ghosts (part 1)");
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

  dolfinx::common::Timer timer44(
      "~DEBUG: 8.2 Distribute ghosts (new shared indices)");

  // Count number of outgoing slaves and masters
  std::map<std::int32_t, std::set<int>> shared_indices
      = slave_index_map->compute_shared_indices();
  timer44.stop();

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
      auto global_slaves = imap->local_to_global({local_slaves[i]}, false);
      proc_to_ghost[index].push_back(global_slaves[0]);
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
  timer3.stop();
  dolfinx::common::Timer timerg("~DEBUG: 8.5 Ghost alltoallv");

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
  timerg.stop();
  dolfinx::common::Timer tzz("~DEBUG: 9. Final flattening");
  // Accumulate offsets of masters from different processors
  std::vector<std::int32_t> ghost_offsets_ = {0};
  for (std::int32_t i = 0; i < src_ranks_ghost.size(); ++i)
  {
    const std::int32_t min = disp_recv_ghost_slaves[i];
    const std::int32_t max = disp_recv_ghost_slaves[i + 1];
    std::vector<std::int32_t> inc_offset;
    inc_offset.insert(inc_offset.end(), in_ghost_offsets.begin() + min,
                      in_ghost_offsets.begin() + max);
    for (std::int32_t& offset : inc_offset)
      offset += *(ghost_offsets_.end() - 1);
    ghost_offsets_.insert(ghost_offsets_.end(), inc_offset.begin(),
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
  // Map info of locally owned slaves to Eigen arrays
  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> loc_masters(
      masters_out.data(), masters_out.size());
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> slaves(
      local_slaves.data(), local_slaves.size());
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> loc_coeffs(
      coeffs_out.data(), coeffs_out.size());
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> offsets(
      offsets_out.data(), offsets_out.size());
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> loc_owners(
      owners_out.data(), owners_out.size());

  // Map Ghost data
  std::vector<std::int32_t> in_ghost_slaves_loc
      = imap->global_to_local(in_ghost_slaves, false);
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> ghost_slaves_eigen(
      in_ghost_slaves_loc.data(), in_ghost_slaves_loc.size());
  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> ghost_masters(
      in_ghost_masters.data(), in_ghost_masters.size());
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> ghost_coeffs(
      in_ghost_coeffs.data(), in_ghost_coeffs.size());
  Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>>
      ghost_offsets(ghost_offsets_.data(), ghost_offsets_.size());
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> m_ghost_owners(
      in_ghost_owners.data(), in_ghost_owners.size());
  mpc.local_slaves = slaves;
  mpc.ghost_slaves = ghost_slaves_eigen;
  mpc.local_masters = loc_masters;
  mpc.ghost_masters = ghost_masters;
  mpc.local_offsets = offsets;
  mpc.ghost_offsets = ghost_offsets;
  mpc.local_owners = loc_owners;
  mpc.ghost_owners = m_ghost_owners;
  mpc.local_coeffs = loc_coeffs;
  mpc.ghost_coeffs = ghost_coeffs;
  tzz.stop();
  return mpc;
}
//-----------------------------------------------------------------------------
mpc_data dolfinx_mpc::create_contact_inelastic_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker)
{
  dolfinx::common::Timer timer0("~DEBUG: 0. Total time");

  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);
  dolfinx::common::Timer timer("~DEBUG: 1. Condition on own proc");
  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int tdim = V->mesh()->topology().dim();
  const int gdim = V->mesh()->geometry().dim();
  const int fdim = tdim - 1;
  const int block_size = V->dofmap()->index_map->block_size();
  std::int32_t size_local = V->dofmap()->index_map->size_local();

  // Extract slave_facets
  std::vector<std::int32_t> _slave_facets;
  for (std::int32_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == slave_marker)
      _slave_facets.push_back(meshtags.indices()[i]);
  Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>> slave_facets(
      _slave_facets.data(), _slave_facets.size());

  // Extract dofs on slave facets per dimension
  std::vector<Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>>
      dofs_on_slave_facet(tdim);

  // NOTE: Assumption that we are only working with vector spaces, which is
  // ordered as xyz,xyz
  std::pair<std::shared_ptr<dolfinx::function::FunctionSpace>,
            std::vector<std::int32_t>>
      V0_pair = V->sub({0})->collapse();
  auto [V0, map] = V0_pair;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> slave_blocks
      = dolfinx::fem::locate_dofs_topological({*V->sub({0}).get(), *V0.get()},
                                              fdim, slave_facets);
  slave_blocks /= block_size;

  // Make maps for local data of slave and master coeffs
  std::vector<std::int32_t> local_block;
  std::vector<std::int32_t> local_block_as_glob;
  std::vector<std::vector<std::int32_t>> local_slaves(tdim);
  std::vector<std::vector<std::int64_t>> local_slave_as_glob(tdim);

  // Array holding ghost slaves blocks (masters,coeffs and offsets will be
  // recieved)
  std::vector<std::int32_t> ghost_blocks;

  for (std::int32_t i = 0; i < slave_blocks.rows(); ++i)
  {
    if (slave_blocks(i, 0) < size_local)
    {
      local_block.push_back(slave_blocks(i, 0));
      auto block_glob = imap->local_to_global({slave_blocks(i, 0)}, true);
      local_block_as_glob.push_back(block_glob[0]);
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        local_slaves[j].push_back(block_size * slave_blocks(i, 0) + j);
        auto slave_glob = imap->local_to_global(
            {block_size * slave_blocks(i, 0) + j}, false);
        local_slave_as_glob[j].push_back(slave_glob[0]);
      }
    }
    else
      ghost_blocks.push_back(slave_blocks(i, 0));
  }

  // Create slave_dofs->master facets and master->slave dofs neighborhood comms
  const bool has_slave = local_block.size() > 0 ? 1 : 0;
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, has_slave, master_marker);
  // Create communicator local_block -> ghost_block
  MPI_Comm slave_to_ghost
      = create_owner_to_ghost_comm(local_block, ghost_blocks, imap, true);

  // Create new index-map where there are only ghosts for slaves
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners
      = imap->ghost_owner_rank();
  std::shared_ptr<const dolfinx::common::IndexMap> slave_index_map;
  {
    std::vector<int> slave_ranks(ghost_blocks.size());
    for (std::int32_t i = 0; i < ghost_blocks.size(); ++i)
      slave_ranks[i] = ghost_owners[ghost_blocks[i] - size_local];
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> ghosts_eigen(
        ghost_blocks.data(), ghost_blocks.size());
    auto ghosts_as_global = imap->local_to_global(ghosts_eigen, true);
    slave_index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, imap->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(slave_ranks.begin(), slave_ranks.end())),
        ghosts_as_global, slave_ranks, block_size);
  }

  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
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

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> coordinates
      = V->tabulate_dof_coordinates();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      local_slave_coordinates(local_block.size(), 3);
  std::vector<std::int64_t> blocks_wo_local_collision;

  // Map to get coordinates and normals of slaves that should be distributed
  std::vector<std::int32_t> collision_to_local;
  std::stringstream cc;
  cc << rank << "\n\n";
  std::vector<std::int32_t> candidates;
  for (auto cell : master_cells)
    candidates.push_back(cell);

  for (std::int32_t i = 0; i < local_block.size(); ++i)
  {
    // Get coordinate and coeff for slave and save in send array
    auto coord = coordinates.row(local_block[i] * block_size);
    local_slave_coordinates.row(i) = coord;

    // Use collision detection algorithm (GJK) to check distance from  cell
    // to point and select one within 1e-10
    std::cout << rank << " " << i << " Search" << candidates.size() << "\n\n";
    std::vector<std::int32_t> verified_candidates
        = dolfinx::geometry::select_colliding_cells(*V->mesh(), candidates,
                                                    coord, 1);
    std::cout << rank << " " << i << " Done" << verified_candidates.size()
              << "\n\n";

    if (verified_candidates.size() == 1)
    {
      // Compute interpolation on other cell
      Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> basis_values
          = get_basis_functions(V, coord, verified_candidates[0]);
      auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);

      std::vector<std::vector<std::int32_t>> l_master(3);
      std::vector<std::vector<PetscScalar>> l_coeff(3);
      std::vector<std::vector<std::int32_t>> l_owner(3);
      // Create arrays per topological dimension
      // std::cout << "BASIS VALUES" << basis_values << "\n\n";
      for (std::int32_t j = 0; j < tdim; ++j)
      {
        for (std::int32_t k = 0; k < cell_dofs.size(); ++k)
        {
          if (std::abs(basis_values(k, j)) > 1e-6)
          {
            l_master[j].push_back(cell_dofs[k]);
            l_coeff[j].push_back(basis_values(k, j));
            if (cell_dofs[k] < size_local * block_size)
              l_owner[j].push_back(rank);
            else
              l_owner[j].push_back(
                  ghost_owners[cell_dofs[k] / block_size - size_local]);
          }
        }
      }
      //   bool found_masters = false;
      //   // Add masters for each master in the block
      //   for (std::int32_t j = 0; j < tdim; ++j)
      //   {
      //     if (l_master[j].size() > 0)
      //     {
      //       auto global_masters = imap->local_to_global(l_master[j], false);
      //       local_masters.insert(
      //           std::pair(local_slave_as_glob[j][i], global_masters));
      //       local_coeffs.insert(std::pair(local_slave_as_glob[j][i],
      //       l_coeff[j]));
      //       local_owners.insert(std::pair(local_slave_as_glob[j][i],
      //       l_owner[j])); found_masters = true;
      //     }
      //   }
      //   if (!found_masters)
      //   {
      //     blocks_wo_local_collision.push_back(local_block_as_glob[i]);
      //     collision_to_local.push_back(i);
      //   }
    }
    // else
    // {
    //   blocks_wo_local_collision.push_back(local_block_as_glob[i]);
    //   collision_to_local.push_back(i);
    // }
  }
  // Structure storing mpc arrays
  dolfinx_mpc::mpc_data mpc2;
  return mpc2;

  // Extract coordinates and normals to distribute
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      distribute_coordinates(blocks_wo_local_collision.size(), 3);
  for (std::int32_t i = 0; i < collision_to_local.size(); ++i)
  {
    distribute_coordinates.row(i)
        = local_slave_coordinates.row(collision_to_local[i]);
  }
  timer.stop();

  // Structure storing mpc arrays
  dolfinx_mpc::mpc_data mpc;
  return mpc;

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
    std::vector<std::int32_t> slaves_flattened;
    for (std::int32_t j = 0; j < tdim; ++j)
    {
      for (std::int32_t i = 0; i < local_slaves[j].size(); ++i)
      {
        auto slave = local_slave_as_glob[j][i];
        slaves_flattened.push_back(local_slaves[j][i]);
        masters_out.insert(masters_out.end(), local_masters[slave].begin(),
                           local_masters[slave].end());
        coeffs_out.insert(coeffs_out.end(), local_coeffs[slave].begin(),
                          local_coeffs[slave].end());
        offsets_out.push_back(masters_out.size());
      }
    }
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gowners(0);
    // Map vectors to Eigen arrays
    Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> masters(
        masters_out.data(), masters_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> slaves(
        slaves_flattened.data(), slaves_flattened.size());
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coeffs(
        coeffs_out.data(), coeffs_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> offsets(
        offsets_out.data(), offsets_out.size());

    // Serial assumptions
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(masters_out.size());
    owners.fill(1);
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> gm(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gs(0, 1);
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> gc(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> go(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> goff(1, 1);
    goff.fill(0);
    mpc.local_slaves = slaves;
    mpc.ghost_slaves = gs;
    mpc.local_masters = masters;
    mpc.ghost_masters = gm;
    mpc.local_offsets = offsets;
    mpc.ghost_offsets = goff;
    mpc.local_owners = owners;
    mpc.ghost_owners = go;
    mpc.local_coeffs = coeffs;
    mpc.ghost_coeffs = gc;

    return mpc;
  }
  dolfinx::common::Timer tt("~DEBUG: 2. First communication ");
  return mpc;
  // Get the  slave->master recv from and send to ranks
  // int indegree(-1), outdegree(-2), weighted(-1);
  // MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree,
  // &outdegree,
  //                                &weighted);
  // const auto [src_ranks, dest_ranks]
  //     = dolfinx::MPI::neighbors(neighborhood_comms[0]);

  // // Figure out how much data to receive from each neighbor
  // const std::int32_t out_collision_slaves =
  // blocks_wo_local_collision.size(); std::vector<std::int32_t>
  // num_slaves_recv(indegree); MPI_Neighbor_allgather(
  //     &out_collision_slaves, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
  //     num_slaves_recv.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
  //     neighborhood_comms[0]);

  // // Compute displacements for data to receive
  // std::vector<int> disp(indegree + 1, 0);
  // std::partial_sum(num_slaves_recv.begin(), num_slaves_recv.end(),
  //                  disp.begin() + 1);

  // // Send data to neighbors and receive data
  // std::vector<std::int64_t> slaves_recv(disp.back());
  // MPI_Neighbor_allgatherv(
  //     slaves_wo_local_collision.data(), slaves_wo_local_collision.size(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), slaves_recv.data(),
  //     num_slaves_recv.data(), disp.data(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[0]);

  // // Multiply recv size by three to accommodate vector coordinates and
  // // function data
  // std::vector<std::int32_t> num_slaves_recv3(indegree);
  // for (std::int32_t i = 0; i < num_slaves_recv.size(); ++i)
  //   num_slaves_recv3[i] = num_slaves_recv[i] * 3;
  // std::vector<int> disp3(indegree + 1, 0);
  // std::partial_sum(num_slaves_recv3.begin(), num_slaves_recv3.end(),
  //                  disp3.begin() + 1);

  // // Send slave normal and coordinate to neighbors
  // std::vector<double> coord_recv(disp3.back());
  // MPI_Neighbor_allgatherv(
  //     distribute_coordinates.data(), distribute_coordinates.size(),
  //     dolfinx::MPI::mpi_type<double>(), coord_recv.data(),
  //     num_slaves_recv3.data(), disp3.data(),
  //     dolfinx::MPI::mpi_type<double>(), neighborhood_comms[0]);
  // std::vector<PetscScalar> normal_recv(disp3.back());
  // MPI_Neighbor_allgatherv(distribute_normals.data(),
  // distribute_normals.size(),
  //                         dolfinx::MPI::mpi_type<PetscScalar>(),
  //                         normal_recv.data(), num_slaves_recv3.data(),
  //                         disp3.data(),
  //                         dolfinx::MPI::mpi_type<PetscScalar>(),
  //                         neighborhood_comms[0]);

  // // Vector for processes with slaves, mapping slaves with
  // // collision on this process
  // std::vector<std::vector<std::int64_t>> collision_slaves(indegree);
  // std::vector<std::map<std::int64_t, std::vector<std::int64_t>>>
  //     collision_masters(indegree);
  // std::vector<std::map<std::int64_t, std::vector<PetscScalar>>>
  //     collision_coeffs(indegree);
  // std::vector<std::map<std::int64_t, std::vector<std::int32_t>>>
  //     collision_owners(indegree);

  // tt.stop();
  // auto timer_start2 = std::chrono::system_clock::now();
  // std::chrono::duration<double> dt_select = (timer_start2 - timer_start2);

  // dolfinx::common::Timer ts("~DEBUG: 3. Remote collision");
  // // Loop over slaves per incoming processor
  // for (std::int32_t i = 0; i < indegree; ++i)
  // {
  //   for (std::int32_t j = disp[i]; j < disp[i + 1]; ++j)
  //   {
  //     // Extract slave coordinates and normals from incoming data
  //     Eigen::Map<const Eigen::Array<double, 1, 3>> slave_coord(
  //         coord_recv.data() + 3 * j, 3);
  //     Eigen::Map<const Eigen::Array<PetscScalar, 3, 1>> slave_normal(
  //         normal_recv.data() + 3 * j, 3);

  //     // Find index of slave coord by using block size remainder
  //     std::int32_t local_index = slaves_recv[j] % block_size;

  //     std::vector<std::int32_t> candidates(master_cells.begin(),
  //                                          master_cells.end());

  //     // Use collision detection algorithm (GJK) to check distance from
  //     // cell to point and select one within 1e-10
  //     auto timer_start_select = std::chrono::system_clock::now();

  //     std::vector<std::int32_t> verified_candidates
  //         = dolfinx::geometry::select_colliding_cells(*V->mesh(),
  //         candidates,
  //                                                     slave_coord, 1);
  //     auto timer_end_select = std::chrono::system_clock::now();
  //     dt_select += (timer_end_select - timer_start_select);
  //     if (verified_candidates.size() == 1)
  //     {
  //       // Compute local contribution of slave on master facet
  //       Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic>
  //       basis_values
  //           = get_basis_functions(V, slave_coord, verified_candidates[0]);
  //       auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);
  //       std::vector<std::int32_t> l_master;
  //       std::vector<PetscScalar> l_coeff;
  //       std::vector<std::int32_t> l_owner;
  //       for (std::int32_t d = 0; d < tdim; ++d)
  //       {
  //         for (std::int32_t k = 0; k < cell_dofs.size(); ++k)
  //         {
  //           PetscScalar coeff = basis_values(k, d) * slave_normal[d]
  //                               / slave_normal[local_index];
  //           if (std::abs(coeff) > 1e-6)
  //           {
  //             l_master.push_back(cell_dofs(k));
  //             l_coeff.push_back(coeff);
  //             if (cell_dofs[k] < size_local * block_size)
  //               l_owner.push_back(rank);
  //             else

  //               l_owner.push_back(
  //                   ghost_owners[cell_dofs[k] / block_size - size_local]);
  //           }
  //         }
  //       }
  //       auto global_masters = imap->local_to_global(l_master, false);
  //       collision_slaves[i].push_back(slaves_recv[j]);
  //       collision_masters[i].insert(std::pair(slaves_recv[j],
  //       global_masters));
  //       collision_coeffs[i].insert(std::pair(slaves_recv[j], l_coeff));
  //       collision_owners[i].insert(std::pair(slaves_recv[j], l_owner));
  //     }
  //   }
  // }
  // ts.stop();
  // auto timer_end2 = std::chrono::system_clock::now();
  // std::chrono::duration<double> dt2 = (timer_end2 - timer_start2);

  // // Flatten data structures before send/recv
  // std::vector<std::int32_t> num_collision_slaves(indegree);
  // for (std::int32_t i = 0; i < indegree; ++i)
  //   num_collision_slaves[i] = collision_slaves[i].size();

  // // Get info about reverse communicator
  // int indegree_rev(-1), outdegree_rev(-2), weighted_rev(-1);
  // MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree_rev,
  //                                &outdegree_rev, &weighted_rev);
  // const auto [src_ranks_rev, dest_ranks_rev]
  //     = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  // std::stringstream ss;
  // dolfinx::common::Timer tm("~DEBUG: 4. Masters->slaves (Send incoming
  // sizes)");
  // // Communicate number of incoming slaves and masters after coll detection
  // auto timer_start = std::chrono::system_clock::now();
  // std::vector<int> inc_num_collision_slaves(indegree_rev);
  // MPI_Neighbor_alltoall(num_collision_slaves.data(), 1, MPI_INT,
  //                       inc_num_collision_slaves.data(), 1, MPI_INT,
  //                       neighborhood_comms[1]);

  // auto timer_end = std::chrono::system_clock::now();
  // std::chrono::duration<double> dt = (timer_end - timer_start);

  // ss << " " << rank << " Loc slaves" << local_slaves.size() << " Collisions
  // "
  //    << std::accumulate(num_collision_slaves.begin(),
  //                       num_collision_slaves.end(), 0)
  //    << " "
  //    << "Remote collision. " << dt2.count() << "GJK " << dt_select.count()
  //    << " MAster cells " << master_cells.size() << " Found local "
  //    << local_masters.size() << " Comm. " << dt.count() << "\n";
  // std::cout << ss.str() << "\n";
  // tm.stop();
  // dolfinx::common::Timer tz("~DEBUG: 5. Flatten data");
  // std::vector<std::int32_t> num_collision_masters(indegree);
  // std::vector<std::int64_t> collision_slaves_out;
  // std::vector<std::int64_t> collision_masters_out;
  // std::vector<std::int32_t> collision_offsets_out;
  // std::vector<std::int32_t> collision_owners_out;
  // std::vector<PetscScalar> collision_coeffs_out;
  // for (std::int32_t i = 0; i < indegree; ++i)
  // {
  //   std::int32_t master_offset = 0;
  //   std::vector<std::int64_t>& slaves_i = collision_slaves[i];
  //   collision_slaves_out.insert(collision_slaves_out.end(),
  //   slaves_i.begin(),
  //                               slaves_i.end());
  //   for (auto slave : slaves_i)
  //   {
  //     std::vector<std::int64_t>& masters_ij = collision_masters[i][slave];
  //     num_collision_masters[i] += masters_ij.size();
  //     collision_masters_out.insert(collision_masters_out.end(),
  //                                  masters_ij.begin(), masters_ij.end());

  //     std::vector<PetscScalar>& coeffs_ij = collision_coeffs[i][slave];
  //     collision_coeffs_out.insert(collision_coeffs_out.end(),
  //     coeffs_ij.begin(),
  //                                 coeffs_ij.end());

  //     std::vector<std::int32_t>& owners_ij = collision_owners[i][slave];
  //     collision_owners_out.insert(collision_owners_out.end(),
  //     owners_ij.begin(),
  //                                 owners_ij.end());
  //     master_offset += masters_ij.size();
  //     collision_offsets_out.push_back(master_offset);
  //   }
  // }
  // tz.stop();
  // dolfinx::common::Timer timer2("~DEBUG: 6. Masters-> slaves");

  // std::vector<int> inc_num_collision_masters(indegree_rev);
  // MPI_Neighbor_alltoall(num_collision_masters.data(), 1, MPI_INT,
  //                       inc_num_collision_masters.data(), 1, MPI_INT,
  //                       neighborhood_comms[1]);

  // // Create displacement vector for slaves and masters
  // std::vector<int> disp_inc_slaves(indegree_rev + 1, 0);
  // std::partial_sum(inc_num_collision_slaves.begin(),
  //                  inc_num_collision_slaves.end(), disp_inc_slaves.begin()
  //                  + 1);
  // std::vector<int> disp_inc_masters(indegree_rev + 1, 0);
  // std::partial_sum(inc_num_collision_masters.begin(),
  //                  inc_num_collision_masters.end(),
  //                  disp_inc_masters.begin() + 1);

  // // Compute send offsets
  // std::vector<int> send_disp_slaves(indegree + 1, 0);
  // std::partial_sum(num_collision_slaves.begin(),
  // num_collision_slaves.end(),
  //                  send_disp_slaves.begin() + 1);
  // std::vector<int> send_disp_masters(indegree + 1, 0);
  // std::partial_sum(num_collision_masters.begin(),
  // num_collision_masters.end(),
  //                  send_disp_masters.begin() + 1);

  // // Receive colliding slaves from other processor
  // std::vector<std::int64_t>
  // remote_colliding_slaves(disp_inc_slaves.back()); MPI_Neighbor_alltoallv(
  //     collision_slaves_out.data(), num_collision_slaves.data(),
  //     send_disp_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
  //     remote_colliding_slaves.data(), inc_num_collision_slaves.data(),
  //     disp_inc_slaves.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
  //     neighborhood_comms[1]);
  // std::vector<std::int32_t>
  // remote_colliding_offsets(disp_inc_slaves.back()); MPI_Neighbor_alltoallv(
  //     collision_offsets_out.data(), num_collision_slaves.data(),
  //     send_disp_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
  //     remote_colliding_offsets.data(), inc_num_collision_slaves.data(),
  //     disp_inc_slaves.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
  //     neighborhood_comms[1]);
  // // Receive colliding masters and relevant data from other processor
  // std::vector<std::int64_t>
  // remote_colliding_masters(disp_inc_masters.back());
  // MPI_Neighbor_alltoallv(
  //     collision_masters_out.data(), num_collision_masters.data(),
  //     send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
  //     remote_colliding_masters.data(), inc_num_collision_masters.data(),
  //     disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
  //     neighborhood_comms[1]);
  // std::vector<PetscScalar>
  // remote_colliding_coeffs(disp_inc_masters.back()); MPI_Neighbor_alltoallv(
  //     collision_coeffs_out.data(), num_collision_masters.data(),
  //     send_disp_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
  //     remote_colliding_coeffs.data(), inc_num_collision_masters.data(),
  //     disp_inc_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
  //     neighborhood_comms[1]);
  // std::vector<std::int32_t>
  // remote_colliding_owners(disp_inc_masters.back()); MPI_Neighbor_alltoallv(
  //     collision_owners_out.data(), num_collision_masters.data(),
  //     send_disp_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
  //     remote_colliding_owners.data(), inc_num_collision_masters.data(),
  //     disp_inc_masters.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
  //     neighborhood_comms[1]);

  // timer2.stop();
  // dolfinx::common::Timer tl("~DEBUG: 7. Add masters from remote procs");

  // auto recv_slaves_as_local
  //     = imap->global_to_local(remote_colliding_slaves, false);
  // // Iterate through the processors
  // for (std::int32_t i = 0; i < src_ranks_rev.size(); ++i)
  // {
  //   const std::int32_t min = disp_inc_slaves[i];
  //   const std::int32_t max = disp_inc_slaves[i + 1];
  //   // Find offsets for masters on given proc
  //   std::vector<std::int32_t> master_offsets = {0};
  //   master_offsets.insert(master_offsets.end(),
  //                         remote_colliding_offsets.begin() + min,
  //                         remote_colliding_offsets.begin() + max);

  //   // Loop through slaves and add them if they havent already been added
  //   for (std::int32_t j = 0; j < master_offsets.size() - 1; ++j)
  //   {
  //     const std::int32_t slave = recv_slaves_as_local[min + j];
  //     const std::int64_t key = remote_colliding_slaves[min + j];
  //     assert(slave != -1);

  //     // Skip if already found on other incoming processor
  //     if (local_masters[key].size() == 0)
  //     {
  //       std::vector<std::int64_t> masters_(
  //           remote_colliding_masters.begin() + disp_inc_masters[i]
  //               + master_offsets[j],
  //           remote_colliding_masters.begin() + disp_inc_masters[i]
  //               + master_offsets[j + 1]);
  //       local_masters[key].insert(local_masters[key].end(),
  //       masters_.begin(),
  //                                 masters_.end());

  //       std::vector<PetscScalar> coeffs_(
  //           remote_colliding_coeffs.begin() + disp_inc_masters[i]
  //               + master_offsets[j],
  //           remote_colliding_coeffs.begin() + disp_inc_masters[i]
  //               + master_offsets[j + 1]);
  //       local_coeffs[key].insert(local_coeffs[key].end(), coeffs_.begin(),
  //                                coeffs_.end());

  //       std::vector<std::int32_t> owners_(
  //           remote_colliding_owners.begin() + disp_inc_masters[i]
  //               + master_offsets[j],
  //           remote_colliding_owners.begin() + disp_inc_masters[i]
  //               + master_offsets[j + 1]);
  //       local_owners[key].insert(local_owners[key].end(), owners_.begin(),
  //                                owners_.end());
  //     }
  //   }
  // }
  // // Merge arrays from other processors with those of the same block
  // for (const auto& block_masters : masters)
  // {
  //   std::int64_t key = block_masters.first;
  //   std::vector<std::int64_t> masters_ = block_masters.second;
  //   std::vector<PetscScalar> coeffs_ = coeffs[key];
  //   std::vector<std::int32_t> owners_ = owners[key];

  //   // Check that we have actually found a collision
  //   if (local_masters[key].size() == 0)
  //     throw std::runtime_error("Could not find contact for slave dof.");
  //   local_masters[key].insert(local_masters[key].end(), masters_.begin(),
  //                             masters_.end());
  //   local_coeffs[key].insert(local_coeffs[key].end(), coeffs_.begin(),
  //                            coeffs_.end());
  //   local_owners[key].insert(local_owners[key].end(), owners_.begin(),
  //                            owners_.end());
  // }
  // tl.stop();

  // dolfinx::common::Timer timer3("~DEBUG: 8. Distribute ghosts (part 1)");
  // // Distribute data for ghosted slaves (the coeffs, owners and offsets)

  // auto [src_ranks_ghost, dest_ranks_ghost]
  //     = dolfinx::MPI::neighbors(slave_to_ghost);
  // // Count number of incoming slaves
  // std::vector<std::int32_t> inc_num_slaves(src_ranks_ghost.size(), 0);
  // for (std::int32_t i = 0; i < ghost_slaves.size(); ++i)
  // {
  //   const std::int32_t owner
  //       = ghost_owners[ghost_slaves[i] / block_size - size_local];
  //   std::vector<int>::iterator it
  //       = std::find(src_ranks_ghost.begin(), src_ranks_ghost.end(), owner);
  //   std::int32_t index = std::distance(src_ranks_ghost.begin(), it);
  //   inc_num_slaves[index]++;
  // }

  // dolfinx::common::Timer timer44(
  //     "~DEBUG: 8.2 Distribute ghosts (new shared indices)");

  // // Count number of outgoing slaves and masters
  // std::map<std::int32_t, std::set<int>> shared_indices
  //     = slave_index_map->compute_shared_indices();
  // timer44.stop();

  // std::vector<std::int32_t> out_num_slaves(dest_ranks_ghost.size(), 0);
  // std::vector<std::int32_t> out_num_masters(dest_ranks_ghost.size(), 0);
  // // Create mappings from ghosted process to the data to recv
  // // (can include repeats of data)
  // std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost;
  // std::map<std::int32_t, std::vector<std::int64_t>> proc_to_ghost_masters;
  // std::map<std::int32_t, std::vector<PetscScalar>> proc_to_ghost_coeffs;
  // std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_owners;
  // std::map<std::int32_t, std::vector<std::int32_t>> proc_to_ghost_offsets;

  // for (std::int32_t i = 0; i < local_slaves_as_glob.size(); ++i)
  // {
  //   const std::int64_t glob_slave = local_slaves_as_glob[i];
  //   std::vector<std::int64_t> masters_i = local_masters[glob_slave];
  //   std::vector<PetscScalar> coeffs_i = local_coeffs[glob_slave];
  //   std::vector<std::int32_t> owners_i = local_owners[glob_slave];
  //   std::int32_t num_masters = masters_i.size();
  //   std::set<int> ghost_procs = shared_indices[local_slaves[i] /
  //   block_size]; for (auto proc : ghost_procs)
  //   {
  //     auto it
  //         = std::find(dest_ranks_ghost.begin(), dest_ranks_ghost.end(),
  //         proc);
  //     std::int32_t index = std::distance(dest_ranks_ghost.begin(), it);
  //     out_num_masters[index] += num_masters;
  //     out_num_slaves[index]++;
  //     // Map slaves to global dof to be recognized by recv proc
  //     auto global_slaves = imap->local_to_global({local_slaves[i]}, false);
  //     proc_to_ghost[index].push_back(global_slaves[0]);
  //     // Add master data in process-wise fashion
  //     proc_to_ghost_masters[index].insert(proc_to_ghost_masters[index].end(),
  //                                         masters_i.begin(),
  //                                         masters_i.end());
  //     proc_to_ghost_coeffs[index].insert(proc_to_ghost_coeffs[index].end(),
  //                                        coeffs_i.begin(), coeffs_i.end());
  //     proc_to_ghost_owners[index].insert(proc_to_ghost_owners[index].end(),
  //                                        owners_i.begin(), owners_i.end());
  //     proc_to_ghost_offsets[index].push_back(
  //         proc_to_ghost_masters[index].size());
  //   }
  // }

  // // Flatten map of global slave ghost dofs to use alltoallv
  // std::vector<std::int64_t> out_ghost_slaves;
  // std::vector<std::int64_t> out_ghost_masters;
  // std::vector<PetscScalar> out_ghost_coeffs;
  // std::vector<std::int32_t> out_ghost_owners;
  // std::vector<std::int32_t> out_ghost_offsets;
  // std::vector<std::int32_t> num_send_slaves(dest_ranks_ghost.size());
  // std::vector<std::int32_t> num_send_masters(dest_ranks_ghost.size());
  // for (std::int32_t i = 0; i < dest_ranks_ghost.size(); ++i)
  // {
  //   num_send_slaves[i] = proc_to_ghost[i].size();
  //   num_send_masters[i] = proc_to_ghost_masters[i].size();
  //   out_ghost_slaves.insert(out_ghost_slaves.end(),
  //   proc_to_ghost[i].begin(),
  //                           proc_to_ghost[i].end());
  //   out_ghost_masters.insert(out_ghost_masters.end(),
  //                            proc_to_ghost_masters[i].begin(),
  //                            proc_to_ghost_masters[i].end());
  //   out_ghost_coeffs.insert(out_ghost_coeffs.end(),
  //                           proc_to_ghost_coeffs[i].begin(),
  //                           proc_to_ghost_coeffs[i].end());
  //   out_ghost_owners.insert(out_ghost_owners.end(),
  //                           proc_to_ghost_owners[i].begin(),
  //                           proc_to_ghost_owners[i].end());
  //   out_ghost_offsets.insert(out_ghost_offsets.end(),
  //                            proc_to_ghost_offsets[i].begin(),
  //                            proc_to_ghost_offsets[i].end());
  // }
  // timer3.stop();
  // dolfinx::common::Timer timerg("~DEBUG: 8.5 Ghost alltoallv");

  // // Receive global slave dofs for ghosts structured as on src proc
  // // Compute displacements for data to send and receive
  // std::vector<int> disp_recv_ghost_slaves(src_ranks_ghost.size() + 1, 0);
  // std::partial_sum(inc_num_slaves.begin(), inc_num_slaves.end(),
  //                  disp_recv_ghost_slaves.begin() + 1);

  // std::vector<int> disp_send_ghost_slaves(dest_ranks_ghost.size() + 1, 0);
  // std::partial_sum(num_send_slaves.begin(), num_send_slaves.end(),
  //                  disp_send_ghost_slaves.begin() + 1);

  // std::vector<std::int64_t> in_ghost_slaves(disp_recv_ghost_slaves.back());
  // MPI_Neighbor_alltoallv(
  //     out_ghost_slaves.data(), num_send_slaves.data(),
  //     disp_send_ghost_slaves.data(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), in_ghost_slaves.data(),
  //     inc_num_slaves.data(), disp_recv_ghost_slaves.data(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), slave_to_ghost);
  // std::vector<std::int32_t>
  // in_ghost_offsets(disp_recv_ghost_slaves.back()); MPI_Neighbor_alltoallv(
  //     out_ghost_offsets.data(), num_send_slaves.data(),
  //     disp_send_ghost_slaves.data(),
  //     dolfinx::MPI::mpi_type<std::int32_t>(), in_ghost_offsets.data(),
  //     inc_num_slaves.data(), disp_recv_ghost_slaves.data(),
  //     dolfinx::MPI::mpi_type<std::int32_t>(), slave_to_ghost);

  // // Communicate size of communication of masters
  // std::vector<int> inc_num_masters(src_ranks_ghost.size());
  // MPI_Neighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
  //                       inc_num_masters.data(), 1, MPI_INT,
  //                       slave_to_ghost);
  // // Send and receive the masters (the proc owning the master) and the
  // // corresponding coeffs from the processor owning the slave
  // std::vector<int> disp_recv_ghost_masters(src_ranks_ghost.size() + 1, 0);
  // std::partial_sum(inc_num_masters.begin(), inc_num_masters.end(),
  //                  disp_recv_ghost_masters.begin() + 1);
  // std::vector<int> disp_send_ghost_masters(dest_ranks_ghost.size() + 1, 0);
  // std::partial_sum(num_send_masters.begin(), num_send_masters.end(),
  //                  disp_send_ghost_masters.begin() + 1);
  // std::vector<std::int64_t>
  // in_ghost_masters(disp_recv_ghost_masters.back()); MPI_Neighbor_alltoallv(
  //     out_ghost_masters.data(), num_send_masters.data(),
  //     disp_send_ghost_masters.data(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), in_ghost_masters.data(),
  //     inc_num_masters.data(), disp_recv_ghost_masters.data(),
  //     dolfinx::MPI::mpi_type<std::int64_t>(), slave_to_ghost);
  // std::vector<PetscScalar> in_ghost_coeffs(disp_recv_ghost_masters.back());
  // MPI_Neighbor_alltoallv(out_ghost_coeffs.data(), num_send_masters.data(),
  //                        disp_send_ghost_masters.data(),
  //                        dolfinx::MPI::mpi_type<PetscScalar>(),
  //                        in_ghost_coeffs.data(), inc_num_masters.data(),
  //                        disp_recv_ghost_masters.data(),
  //                        dolfinx::MPI::mpi_type<PetscScalar>(),
  //                        slave_to_ghost);
  // std::vector<std::int32_t>
  // in_ghost_owners(disp_recv_ghost_masters.back()); MPI_Neighbor_alltoallv(
  //     out_ghost_owners.data(), num_send_masters.data(),
  //     disp_send_ghost_masters.data(),
  //     dolfinx::MPI::mpi_type<std::int32_t>(), in_ghost_owners.data(),
  //     inc_num_masters.data(), disp_recv_ghost_masters.data(),
  //     dolfinx::MPI::mpi_type<std::int32_t>(), slave_to_ghost);
  // timerg.stop();
  // dolfinx::common::Timer tzz("~DEBUG: 9. Final flattening");
  // // Accumulate offsets of masters from different processors
  // std::vector<std::int32_t> ghost_offsets_ = {0};
  // for (std::int32_t i = 0; i < src_ranks_ghost.size(); ++i)
  // {
  //   const std::int32_t min = disp_recv_ghost_slaves[i];
  //   const std::int32_t max = disp_recv_ghost_slaves[i + 1];
  //   std::vector<std::int32_t> inc_offset;
  //   inc_offset.insert(inc_offset.end(), in_ghost_offsets.begin() + min,
  //                     in_ghost_offsets.begin() + max);
  //   for (std::int32_t& offset : inc_offset)
  //     offset += *(ghost_offsets_.end() - 1);
  //   ghost_offsets_.insert(ghost_offsets_.end(), inc_offset.begin(),
  //                         inc_offset.end());
  // }

  // std::vector<std::int64_t> masters_out;
  // std::vector<PetscScalar> coeffs_out;
  // std::vector<std::int32_t> owners_out;
  // std::vector<std::int32_t> offsets_out = {0};
  // // Flatten local slaves
  // for (auto i : local_slaves_as_glob)
  // {
  //   masters_out.insert(masters_out.end(), local_masters[i].begin(),
  //                      local_masters[i].end());
  //   coeffs_out.insert(coeffs_out.end(), local_coeffs[i].begin(),
  //                     local_coeffs[i].end());
  //   offsets_out.push_back(masters_out.size());
  //   owners_out.insert(owners_out.end(), local_owners[i].begin(),
  //                     local_owners[i].end());
  // }
  // // Map info of locally owned slaves to Eigen arrays
  // Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> loc_masters(
  //     masters_out.data(), masters_out.size());
  // Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> slaves(
  //     local_slaves.data(), local_slaves.size());
  // Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> loc_coeffs(
  //     coeffs_out.data(), coeffs_out.size());
  // Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> offsets(
  //     offsets_out.data(), offsets_out.size());
  // Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> loc_owners(
  //     owners_out.data(), owners_out.size());

  // // Map Ghost data
  // std::vector<std::int32_t> in_ghost_slaves_loc
  //     = imap->global_to_local(in_ghost_slaves, false);
  // Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  // ghost_slaves_eigen(
  //     in_ghost_slaves_loc.data(), in_ghost_slaves_loc.size());
  // Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> ghost_masters(
  //     in_ghost_masters.data(), in_ghost_masters.size());
  // Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> ghost_coeffs(
  //     in_ghost_coeffs.data(), in_ghost_coeffs.size());
  // Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>>
  //     ghost_offsets(ghost_offsets_.data(), ghost_offsets_.size());
  // Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> m_ghost_owners(
  //     in_ghost_owners.data(), in_ghost_owners.size());
  // mpc.local_slaves = slaves;
  // mpc.ghost_slaves = ghost_slaves_eigen;
  // mpc.local_masters = loc_masters;
  // mpc.ghost_masters = ghost_masters;
  // mpc.local_offsets = offsets;
  // mpc.ghost_offsets = ghost_offsets;
  // mpc.local_owners = loc_owners;
  // mpc.ghost_owners = m_ghost_owners;
  // mpc.local_coeffs = loc_coeffs;
  // mpc.ghost_coeffs = ghost_coeffs;
  // tzz.stop();
  // return mpc;
}
//-----------------------------------------------------------------------------
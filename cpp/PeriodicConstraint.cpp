// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "PeriodicConstraint.h"
#include "utils.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
namespace
{

/// Create a periodic MPC condition given a set of slave degrees of freedom
/// (blocked wrt. the input function space)
/// @param[in] V The input function space (possibly a collapsed sub space)
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] scale Scaling of the periodic condition
/// @param[in] parent_map Map from collapsed space to parent space (Identity if
/// not collapsed)
/// @param[in] parent_space The parent space (The same space as V if not
/// collapsed)
/// @returns The multi point constraint
template <typename T>
dolfinx_mpc::mpc_data _create_periodic_condition(
    const dolfinx::fem::FunctionSpace& V, std::span<std::int32_t> slave_blocks,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    double scale,
    const std::function<const std::int32_t(const std::int32_t&)>& parent_map,
    const dolfinx::fem::FunctionSpace& parent_space)
{
  // Map a list of indices in collapsed space back to the parent space
  auto sub_to_parent = [&parent_map](const std::vector<std::int32_t>& sub_dofs)
  {
    std::vector<std::int32_t> parent_dofs(sub_dofs.size());
    std::transform(sub_dofs.cbegin(), sub_dofs.cend(), parent_dofs.begin(),
                   [&parent_map](auto& dof) { return parent_map(dof); });
    return parent_dofs;
  };

  // Map a list of parent dofs (local to process) to its global dof
  auto parent_dofmap = parent_space.dofmap();
  auto parent_to_global
      = [&parent_dofmap](const std::vector<std::int32_t>& dofs)
  {
    std::vector<std::int32_t> parent_blocks;
    std::vector<std::int32_t> parent_rems;
    parent_blocks.reserve(dofs.size());
    parent_rems.reserve(dofs.size());
    const auto bs = parent_dofmap->index_map_bs();
    // Split parent dofs into blocks and rems
    std::for_each(dofs.cbegin(), dofs.cend(),
                  [&parent_blocks, &parent_rems, &bs](auto dof)
                  {
                    std::div_t div = std::div(dof, bs);
                    parent_blocks.push_back(div.quot);
                    parent_rems.push_back(div.rem);
                  });
    // Map blocks to global index
    std::vector<std::int64_t> parents_glob(dofs.size());
    parent_dofmap->index_map->local_to_global(parent_blocks, parents_glob);
    // Unroll block size
    for (std::size_t i = 0; i < dofs.size(); i++)
      parents_glob[i] = parents_glob[i] * bs + parent_rems[i];
    return parents_glob;
  };
  if (const std::size_t value_size
      = V.element()->value_size() / V.element()->block_size();
      value_size > 1)
    throw std::runtime_error(
        "Periodic conditions for vector valued spaces are not "
        "implemented");

  // Tolerance for adding scaled basis values to MPC. Any scaled basis
  // value with lower absolute value than the tolerance is ignored
  const double tol = 1e-13;

  auto mesh = V.mesh();
  auto dofmap = V.dofmap();
  auto imap = dofmap->index_map;
  const int bs = dofmap->index_map_bs();
  const int size_local = imap->size_local();

  /// Compute which rank (relative to neighbourhood) to send each ghost to
  const std::vector<int>& ghost_owners = imap->owners();

  // Only work with local blocks
  std::vector<std::int32_t> local_blocks;
  local_blocks.reserve(slave_blocks.size());
  std::for_each(slave_blocks.begin(), slave_blocks.end(),
                [&local_blocks, size_local](auto block)
                {
                  if (block < size_local)
                    local_blocks.push_back(block);
                });

  // Create map from slave dof blocks to a cell containing them
  std::vector<std::int32_t> slave_cells
      = dolfinx_mpc::create_block_to_cell_map(V, local_blocks);

  // Compute relation(slave_blocks)
  std::vector<double> mapped_T_b(local_blocks.size() * 3);
  const std::array<std::size_t, 2> T_shape = {local_blocks.size(), 3};

  {
    std::experimental::mdspan<
        double, std::experimental::extents<
                    std::size_t, std::experimental::dynamic_extent, 3>>
        mapped_T(mapped_T_b.data(), T_shape);

    // Tabulate dof coordinates for each dof
    auto [x, x_shape]
        = dolfinx_mpc::tabulate_dof_coordinates(V, local_blocks, slave_cells);
    // Map all slave coordinates using the relation
    std::vector<double> mapped_x = relation(x);
    std::experimental::mdspan<
        double, std::experimental::extents<std::size_t, 3,
                                           std::experimental::dynamic_extent>>
        x_(mapped_x.data(), x_shape);
    for (std::size_t i = 0; i < mapped_T.extent(0); ++i)
      for (std::size_t j = 0; j < mapped_T.extent(1); ++j)
        mapped_T(i, j) = x_(j, i);
  }
  // Get mesh info
  const int tdim = mesh->topology().dim();
  auto cell_imap = mesh->topology().index_map(tdim);
  const int num_cells_local = cell_imap->size_local();

  // Create bounding-box tree over owned cells
  std::vector<std::int32_t> r(num_cells_local);
  std::iota(r.begin(), r.end(), 0);
  dolfinx::geometry::BoundingBoxTree tree(*mesh.get(), tdim, r, 1e-15);
  auto process_tree = tree.create_global_tree(mesh->comm());
  auto colliding_bbox_processes
      = dolfinx::geometry::compute_collisions(process_tree, mapped_T_b);

  std::vector<std::int32_t> local_cell_collisions
      = dolfinx_mpc::find_local_collisions(*mesh, tree, mapped_T_b, 1e-20);
  dolfinx::common::Timer t0("~~Periodic: Local cell and eval basis");
  auto [basis_values, basis_shape] = dolfinx_mpc::evaluate_basis_functions(
      V, mapped_T_b, local_cell_collisions);
  std::experimental::mdspan<const double,
                            std::experimental::dextents<std::size_t, 3>>
      tabulated_basis_values(basis_values.data(), basis_shape);
  t0.stop();
  // Create output arrays
  std::vector<std::int32_t> slaves;
  slaves.reserve(slave_blocks.size() * bs);
  std::vector<std::int64_t> masters;
  masters.reserve(slave_blocks.size() * bs);
  std::vector<std::int32_t> owners;
  owners.reserve(slave_blocks.size() * bs);
  // FIXME: This should really be templated
  // Requires changes all over the place for mpc_data
  std::vector<PetscScalar> coeffs;
  coeffs.reserve(slave_blocks.size() * bs);
  std::vector<std::int32_t> num_masters_per_slave;
  num_masters_per_slave.reserve(slave_blocks.size() * bs);

  // Temporary array holding global indices
  std::vector<std::int32_t> sub_dofs;

  const int rank = dolfinx::MPI::rank(mesh->comm());
  // Create counter for blocks that will be sent to other processes
  const int num_procs = dolfinx::MPI::size(mesh->comm());
  std::vector<std::int32_t> off_process_counter(num_procs, 0);
  for (std::size_t i = 0; i < local_cell_collisions.size(); i++)
  {
    if (const std::int32_t cell = local_cell_collisions[i]; cell != -1)
    {

      // Map local dofs on master cell to global indices
      auto cell_blocks = dofmap->cell_dofs(cell);
      // Unroll local master dofs
      sub_dofs.resize(cell_blocks.size() * bs);
      for (std::size_t j = 0; j < cell_blocks.size(); j++)
        for (int b = 0; b < bs; b++)
          sub_dofs[j * bs + b] = cell_blocks[j] * bs + b;
      auto parent_dofs = sub_to_parent(sub_dofs);
      auto global_parent_dofs = parent_to_global(parent_dofs);

      // Check if basis values are not zero, and add master, coeff and
      // owner info for each dof in the block
      for (int b = 0; b < bs; b++)
      {
        slaves.push_back(parent_map(local_blocks[i] * bs + b));
        int num_masters = 0;
        for (std::size_t j = 0; j < cell_blocks.size(); j++)
        {
          const std::int32_t cell_block = cell_blocks[j];
          // NOTE: Assuming 0 value size
          if (const double val = scale * tabulated_basis_values(i, j, 0);
              std::abs(val) > tol)
          {
            num_masters++;
            masters.push_back(global_parent_dofs[j * bs + b]);
            coeffs.push_back(val);
            // NOTE: Assuming same ownership of dofs in collapsed sub space
            if (cell_block < size_local)
              owners.push_back(rank);
            else
              owners.push_back(ghost_owners[cell_block - size_local]);
          }
        }
        num_masters_per_slave.push_back(num_masters);
      }
    }
    else
    {
      // Count number of incoming blocks from other processes
      auto procs = colliding_bbox_processes.links((int)i);
      for (auto proc : procs)
      {
        if (rank == proc)
          continue;
        off_process_counter[proc]++;
      }
    }
  }
  // Communicate s_to_m
  std::vector<std::int32_t> s_to_m_ranks;
  std::vector<std::int8_t> s_to_m_indicator(num_procs, 0);
  std::vector<std::int32_t> num_out_slaves;
  for (int i = 0; i < num_procs; i++)
  {
    if (off_process_counter[i] > 0 && i != rank)
    {
      s_to_m_indicator[i] = 1;
      s_to_m_ranks.push_back(i);
      num_out_slaves.push_back(off_process_counter[i]);
    }
  }
  // Push back to avoid null_ptr
  num_out_slaves.push_back(0);

  std::vector<std::int8_t> indicator(num_procs);
  MPI_Alltoall(s_to_m_indicator.data(), 1, MPI_INT8_T, indicator.data(), 1,
               MPI_INT8_T, mesh->comm());

  std::vector<std::int32_t> m_to_s_ranks;
  for (int i = 0; i < num_procs; i++)
    if (indicator[i])
      m_to_s_ranks.push_back(i);

  // Create neighborhood communicator
  // Slave block owners -> Process with possible masters
  std::vector<int> s_to_m_weights(s_to_m_ranks.size(), 1);
  std::vector<int> m_to_s_weights(m_to_s_ranks.size(), 1);
  auto slave_to_master = MPI_COMM_NULL;
  MPI_Dist_graph_create_adjacent(
      mesh->comm(), (int)m_to_s_ranks.size(), m_to_s_ranks.data(),
      m_to_s_weights.data(), (int)s_to_m_ranks.size(), s_to_m_ranks.data(),
      s_to_m_weights.data(), MPI_INFO_NULL, false, &slave_to_master);

  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(slave_to_master, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == (int)m_to_s_ranks.size());
  assert(outdegree == (int)s_to_m_ranks.size());

  // Compute number of receiving slaves
  std::vector<std::int32_t> num_recv_slaves(indegree + 1);
  MPI_Neighbor_alltoall(
      num_out_slaves.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_recv_slaves.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_master);
  num_out_slaves.pop_back();
  num_recv_slaves.pop_back();
  // Prepare data structures for sending information
  std::vector<int> disp_out(outdegree + 1, 0);
  std::partial_sum(num_out_slaves.begin(), num_out_slaves.end(),
                   disp_out.begin() + 1);

  // Prepare data structures for receiving information
  std::vector<int> disp_in(indegree + 1, 0);
  std::partial_sum(num_recv_slaves.begin(), num_recv_slaves.end(),
                   disp_in.begin() + 1);

  // Array holding all dofs (index local to process) for coordinates sent
  // out
  std::vector<std::int32_t> searching_dofs(bs * disp_out.back());

  // Communicate coordinates of slave blocks
  std::vector<std::int32_t> out_placement(outdegree, 0);
  std::vector<double> coords_out(disp_out.back() * 3);

  for (std::size_t i = 0; i < local_cell_collisions.size(); i++)
  {
    if (local_cell_collisions[i] == -1)
    {
      auto procs = colliding_bbox_processes.links((int)i);
      for (auto proc : procs)
      {
        if (rank == proc)
          continue;
        // Find position in neighborhood communicator
        auto it = std::find(s_to_m_ranks.begin(), s_to_m_ranks.end(), proc);
        assert(it != s_to_m_ranks.end());
        auto dist = std::distance(s_to_m_ranks.begin(), it);
        const std::int32_t insert_location
            = disp_out[dist] + out_placement[dist]++;
        // Copy coordinates and dofs to output arrays
        std::copy_n(std::next(mapped_T_b.begin(), 3 * i), 3,
                    std::next(coords_out.begin(), 3 * insert_location));
        for (int b = 0; b < bs; b++)
        {
          searching_dofs[insert_location * bs + b]
              = parent_map(local_blocks[i] * bs + b);
        }
      }
    }
  }

  // Communciate coordinates with other process
  const std::array<std::size_t, 2> cr_shape = {(std::size_t)disp_in.back(), 3};
  std::vector<double> coords_recvb(cr_shape.front() * cr_shape.back());
  std::experimental::mdspan<
      const double, std::experimental::extents<
                        std::size_t, std::experimental::dynamic_extent, 3>>
      coords_recv(coords_recvb.data(), cr_shape);

  // Take into account that we send three values per slave
  auto m_3 = [](auto& num) { num *= 3; };
  std::for_each(disp_out.begin(), disp_out.end(), m_3);
  std::for_each(num_out_slaves.begin(), num_out_slaves.end(), m_3);
  std::for_each(disp_in.begin(), disp_in.end(), m_3);
  std::for_each(num_recv_slaves.begin(), num_recv_slaves.end(), m_3);

  // Communicate coordinates
  MPI_Neighbor_alltoallv(coords_out.data(), num_out_slaves.data(),
                         disp_out.data(), dolfinx::MPI::mpi_type<double>(),
                         coords_recvb.data(), num_recv_slaves.data(),
                         disp_in.data(), dolfinx::MPI::mpi_type<double>(),
                         slave_to_master);

  // Reset in_displacements to be per block for later usage
  auto d_3 = [](auto& num) { num /= 3; };
  std::for_each(disp_in.begin(), disp_in.end(), d_3);

  // Reset out_displacments to be for every slave
  auto m_bs_d_3 = [bs](auto& num) { num = num * bs / 3; };
  std::for_each(disp_out.begin(), disp_out.end(), m_bs_d_3);
  std::for_each(num_out_slaves.begin(), num_out_slaves.end(), m_bs_d_3);

  // Create remote arrays
  std::vector<std::int64_t> masters_remote;
  masters_remote.reserve(coords_recvb.size());
  std::vector<std::int32_t> owners_remote;
  owners_remote.reserve(coords_recvb.size());
  std::vector<PetscScalar> coeffs_remote;
  coeffs_remote.reserve(coords_recvb.size());
  std::vector<std::int32_t> num_masters_per_slave_remote;
  num_masters_per_slave_remote.reserve(bs * coords_recvb.size() / 3);

  std::vector<std::int32_t> remote_cell_collisions
      = dolfinx_mpc::find_local_collisions(*mesh, tree, coords_recvb, 1e-20);
  auto [remote_basis_valuesb, r_basis_shape]
      = dolfinx_mpc::evaluate_basis_functions(V, coords_recvb,
                                              remote_cell_collisions);
  std::experimental::mdspan<const double,
                            std::experimental::dextents<std::size_t, 3>>
      remote_basis_values(remote_basis_valuesb.data(), r_basis_shape);

  // Find remote masters and count how many to send to each process
  std::vector<std::int32_t> num_remote_masters(indegree, 0);
  std::vector<std::int32_t> num_remote_slaves(indegree);
  for (int i = 0; i < indegree; i++)
  {
    // Count number of masters added and number of slaves for output
    std::int32_t r_masters = 0;
    num_remote_slaves[i] = bs * (disp_in[i + 1] - disp_in[i]);

    for (std::int32_t j = disp_in[i]; j < disp_in[i + 1]; j++)
    {
      if (const std::int32_t cell = remote_cell_collisions[j]; j == -1)
      {
        for (int b = 0; b < bs; b++)
          num_masters_per_slave_remote.push_back(0);
      }
      else
      {
        // Compute basis functions at mapped point
        // Check if basis values are not zero, and add master, coeff and
        // owner info for each dof in the block
        auto cell_blocks = dofmap->cell_dofs(cell);

        // Unroll local master dofs
        sub_dofs.resize(cell_blocks.size() * bs);
        for (std::size_t k = 0; k < cell_blocks.size(); k++)
          for (int b = 0; b < bs; b++)
            sub_dofs[k * bs + b] = cell_blocks[k] * bs + b;
        auto parent_dofs = sub_to_parent(sub_dofs);
        auto global_parent_dofs = parent_to_global(parent_dofs);

        for (int b = 0; b < bs; b++)
        {
          int num_masters = 0;
          for (std::size_t k = 0; k < cell_blocks.size(); k++)
          {
            // NOTE: Assuming value_size 0
            if (const double val = scale * remote_basis_values(j, k, 0);
                std::abs(val) > tol)
            {
              num_masters++;
              masters_remote.push_back(global_parent_dofs[k * bs + b]);
              coeffs_remote.push_back(val);
              // NOTE assuming same ownership in collapsed sub as parent space
              if (cell_blocks[k] < size_local)
                owners_remote.push_back(rank);
              else
                owners_remote.push_back(
                    ghost_owners[cell_blocks[k] - size_local]);
            }
          }
          r_masters += num_masters;
          num_masters_per_slave_remote.push_back(num_masters);
        }
      }
    }
    num_remote_masters[i] += r_masters;
  }

  // Create inverse communicator
  // Procs with possible masters -> slave block owners
  auto master_to_slave = MPI_COMM_NULL;
  MPI_Dist_graph_create_adjacent(
      mesh->comm(), (int)s_to_m_ranks.size(), s_to_m_ranks.data(),
      s_to_m_weights.data(), (int)m_to_s_ranks.size(), m_to_s_ranks.data(),
      m_to_s_weights.data(), MPI_INFO_NULL, false, &master_to_slave);

  // Send data back to owning process
  dolfinx_mpc::recv_data recv_data
      = dolfinx_mpc::send_master_data_to_owner<PetscScalar>(
          master_to_slave, num_remote_masters, num_remote_slaves,
          num_out_slaves, num_masters_per_slave_remote, masters_remote,
          coeffs_remote, owners_remote);

  // Append found slaves/master pairs
  dolfinx_mpc::append_master_data<PetscScalar>(
      recv_data, searching_dofs, slaves, masters, coeffs, owners,
      num_masters_per_slave, parent_space.dofmap()->index_map->size_local(),
      parent_space.dofmap()->index_map_bs());

  // Distribute ghost data
  dolfinx_mpc::mpc_data ghost_data
      = dolfinx_mpc::distribute_ghost_data<PetscScalar>(
          slaves, masters, coeffs, owners, num_masters_per_slave,
          parent_space.dofmap()->index_map,
          parent_space.dofmap()->index_map_bs());

  // Add ghost data to existing arrays
  std::vector<std::int32_t>& ghost_slaves = ghost_data.slaves;
  slaves.insert(std::end(slaves), std::begin(ghost_slaves),
                std::end(ghost_slaves));
  std::vector<std::int64_t>& ghost_masters = ghost_data.masters;
  masters.insert(std::end(masters), std::begin(ghost_masters),
                 std::end(ghost_masters));
  std::vector<std::int32_t>& ghost_num = ghost_data.offsets;
  num_masters_per_slave.insert(std::end(num_masters_per_slave),
                               std::begin(ghost_num), std::end(ghost_num));
  std::vector<PetscScalar>& ghost_coeffs = ghost_data.coeffs;
  coeffs.insert(std::end(coeffs), std::begin(ghost_coeffs),
                std::end(ghost_coeffs));
  std::vector<std::int32_t>& ghost_owner_ranks = ghost_data.owners;
  owners.insert(std::end(owners), std::begin(ghost_owner_ranks),
                std::end(ghost_owner_ranks));

  // Compute offsets
  std::vector<std::int32_t> offsets(num_masters_per_slave.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   offsets.begin() + 1);

  dolfinx_mpc::mpc_data output;
  output.offsets = offsets;
  output.masters = masters;
  output.coeffs = coeffs;
  output.owners = owners;
  output.slaves = slaves;
  return output;
}

/// Create a periodic MPC condition given a geometrical relation between the
/// slave and master surface
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] indicator Function marking tabulated degrees of freedom
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @returns The multi point constraint
template <typename T>
dolfinx_mpc::mpc_data geometrical_condition(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<std::size_t, 3,
                                       std::experimental::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    double scale, bool collapse)
{
  std::vector<std::int32_t> reduced_blocks;
  if (collapse)
  {
    // Locate dofs in sub and parent space
    std::pair<dolfinx::fem::FunctionSpace, std::vector<int32_t>> sub_space
        = V->collapse();
    const dolfinx::fem::FunctionSpace& V_sub = sub_space.first;
    const std::vector<std::int32_t>& parent_map = sub_space.second;
    std::array<std::vector<std::int32_t>, 2> slave_blocks
        = dolfinx::fem::locate_dofs_geometrical({*V.get(), V_sub}, indicator);
    reduced_blocks.reserve(slave_blocks[0].size());
    // Remove blocks in Dirichlet bcs
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_blocks[0], bcs);
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[1][i]);
    // Create sub space to parent map
    auto sub_map
        = [&parent_map](const std::int32_t& i) { return parent_map[i]; };
    return _create_periodic_condition<T>(V_sub, std::span(reduced_blocks),
                                         relation, scale, sub_map, *V);
  }
  else
  {
    std::vector<std::int32_t> slave_blocks
        = dolfinx::fem::locate_dofs_geometrical(*V, indicator);
    reduced_blocks.reserve(slave_blocks.size());
    // Remove blocks in Dirichlet bcs
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_blocks, bcs);
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[i]);
    auto sub_map = [](const std::int32_t& dof) { return dof; };
    return _create_periodic_condition<T>(*V, std::span(reduced_blocks),
                                         relation, scale, sub_map, *V);
  }
}

/// Create a periodic MPC on a given set of mesh entities, mapped to the
/// master surface by a relation function.
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] meshtag Meshtag with set of entities
/// @param[in] tag The value of the mesh tag entities that should bec
/// considered as entities
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @returns The multi point constraint
template <typename T>
dolfinx_mpc::mpc_data topological_condition(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    double scale, bool collapse)
{
  std::vector<std::int32_t> entities = meshtag->find(tag);

  if (collapse)
  {
    // Locate dofs in sub and parent space
    std::pair<dolfinx::fem::FunctionSpace, std::vector<int32_t>> sub_space
        = V->collapse();
    const dolfinx::fem::FunctionSpace& V_sub = sub_space.first;
    const std::vector<std::int32_t>& parent_map = sub_space.second;
    std::array<std::vector<std::int32_t>, 2> slave_blocks
        = dolfinx::fem::locate_dofs_topological({*V.get(), V_sub},
                                                meshtag->dim(), entities);
    // Remove DirichletBC dofs from sub space
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_blocks[0], bcs);
    std::vector<std::int32_t> reduced_blocks;
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[1][i]);
    // Create sub space to parent map
    const auto sub_map
        = [&parent_map](const std::int32_t& i) { return parent_map[i]; };
    // Create mpc on sub space
    dolfinx_mpc::mpc_data sub_data = _create_periodic_condition<T>(
        V_sub, std::span(reduced_blocks), relation, scale, sub_map, *V);
    return sub_data;
  }
  else
  {
    std::vector<std::int32_t> slave_blocks
        = dolfinx::fem::locate_dofs_topological(*V.get(), meshtag->dim(),
                                                entities);
    const std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<TCB_SPAN_HAVE_CPP17>(*V, slave_blocks, bcs);
    std::vector<std::int32_t> reduced_blocks;
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[i]);
    // Identity map
    const auto sub_map = [](const std::int32_t& dof) { return dof; };

    return _create_periodic_condition<T>(*V, std::span(reduced_blocks),
                                         relation, scale, sub_map, *V);
  }
};

} // namespace

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<std::size_t, 3,
                                       std::experimental::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale, bool collapse)
{
  return geometrical_condition<double>(V, indicator, relation, bcs, scale,
                                       collapse);
}

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<std::size_t, 3,
                                       std::experimental::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale, bool collapse)
{
  return geometrical_condition<std::complex<double>>(V, indicator, relation,
                                                     bcs, scale, collapse);
}

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale, bool collapse)
{
  return topological_condition<double>(V, meshtag, tag, relation, bcs, scale,
                                       collapse);
};

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<double>(std::span<const double>)>& relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale, bool collapse)
{
  return topological_condition<std::complex<double>>(V, meshtag, tag, relation,
                                                     bcs, scale, collapse);
}

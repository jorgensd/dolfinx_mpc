// Copyright (C) 2020-2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "mpi_utils.h"
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <xtensor/xtensor.hpp>
namespace dolfinx_mpc
{

/// Structure to hold data after mpi communication
template <typename T>
struct recv_data
{
  std::vector<std::int32_t> num_masters_per_slave;
  std::vector<std::int64_t> masters;
  std::vector<std::int32_t> owners;
  std::vector<T> coeffs;
};

struct mpc_data
{
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<PetscScalar> coeffs;
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> owners;
};

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
dolfinx::graph::AdjacencyList<int>
compute_shared_indices(std::shared_ptr<dolfinx::fem::FunctionSpace> V);

dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc1,
    const std::string& type = std::string());

dolfinx::la::petsc::Matrix create_matrix(
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

/// Create a map from dof blocks to one of the cells that contains the degree of
/// freedom
/// @param[in] V The function space
/// @param[in] blocks The blocks (local to process) we want to map
std::vector<std::int32_t>
create_block_to_cell_map(const dolfinx::fem::FunctionSpace& V,
                         xtl::span<const std::int32_t> blocks);

/// Create sparsity pattern with multi point constraint additions to the rows
/// and the columns
/// @param[in] a bi-linear form for the current variational problem
/// (The one used to generate the standard sparsity-pattern)
/// @param[in] mpc0 The multi point constraint to apply to the rows of the
/// matrix.
/// @param[in] mpc1 The multi point constraint to apply to the columns of the
/// matrix.
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc1);

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
}

/// Create an function space with an extended index map, where all input dofs
/// (global index) is added to the local index map as ghosts.
/// @param[in] V The original function space
/// @param[in] global_dofs The list of master dofs (global index)
/// @param[in] owners The owners of the master degrees of freedom
dolfinx::fem::FunctionSpace create_extended_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::vector<std::int64_t>& global_dofs, std::vector<std::int32_t>& owners);

/// Send masters, coefficients, owners and offsets to the process owning the
/// slave degree of freedom.
/// @param[in] master_to_slave MPI neighborhood communicator from processes with
/// master dofs to those owning the slave dofs
/// @param[in] num_remote_masters Number of masters that will be sent to each
/// destination of the neighboorhod communicator
/// @param[in] num_remote_slaves Number of slaves that will be sent to each
/// destination of the neighborhood communicator
/// @param[in] num_incoming_slaves Number of slaves that wil be received from
/// each source of the neighboorhod communicator
/// @param[in] num_masters_per_slave The number of masters for each slave that
/// will be sent to the owning rank
/// @param[in] masters The masters to send to the owning process
/// @param[in] coeffs The corresponding coefficients to send to the owning
/// process
/// @param[in] owners The owning rank of each master
/// @returns Data structure with the number of masters per slave, the master
/// dofs (global indices), the coefficients and owners
template <typename T>
recv_data<T> send_master_data_to_owner(
    MPI_Comm& master_to_slave, std::vector<std::int32_t>& num_remote_masters,
    const std::vector<std::int32_t>& num_remote_slaves,
    const std::vector<std::int32_t>& num_incoming_slaves,
    const std::vector<std::int32_t>& num_masters_per_slave,
    const std::vector<std::int64_t>& masters, const std::vector<T>& coeffs,
    const std::vector<std::int32_t>& owners)
{
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(master_to_slave, &indegree, &outdegree,
                                 &weighted);

  // Communicate how many masters has been found on the other process
  std::vector<std::int32_t> num_recv_masters(indegree + 1);
  num_remote_masters.push_back(0);
  MPI_Request request_m;
  MPI_Ineighbor_alltoall(
      num_remote_masters.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_recv_masters.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      master_to_slave, &request_m);
  num_recv_masters.pop_back();
  num_remote_masters.pop_back();
  std::vector<std::int32_t> remote_slave_disp_out(outdegree + 1, 0);
  std::partial_sum(num_remote_slaves.begin(), num_remote_slaves.end(),
                   remote_slave_disp_out.begin() + 1);

  // Send num masters per slave
  std::vector<int> slave_disp_in(indegree + 1, 0);
  std::partial_sum(num_incoming_slaves.begin(), num_incoming_slaves.end(),
                   slave_disp_in.begin() + 1);
  std::vector<std::int32_t> recv_num_masters_per_slave(slave_disp_in.back());
  MPI_Neighbor_alltoallv(
      num_masters_per_slave.data(), num_remote_slaves.data(),
      remote_slave_disp_out.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_num_masters_per_slave.data(), num_incoming_slaves.data(),
      slave_disp_in.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      master_to_slave);

  // Wait for number of remote masters to be received
  MPI_Status status_m;
  MPI_Wait(&request_m, &status_m);

  // Compute in/out displacements for masters/coeffs/owners
  std::vector<std::int32_t> master_recv_disp(indegree + 1, 0);
  std::partial_sum(num_recv_masters.begin(), num_recv_masters.end(),
                   master_recv_disp.begin() + 1);
  std::vector<std::int32_t> master_send_disp(outdegree + 1, 0);
  std::partial_sum(num_remote_masters.begin(), num_remote_masters.end(),
                   master_send_disp.begin() + 1);

  // Send masters/coeffs/owners to slave process
  std::vector<std::int64_t> recv_masters(master_recv_disp.back());
  std::vector<std::int32_t> recv_owners(master_recv_disp.back());
  std::vector<T> recv_coeffs(master_recv_disp.back());
  std::array<MPI_Status, 3> data_status;
  std::array<MPI_Request, 3> data_request;

  MPI_Ineighbor_alltoallv(
      masters.data(), num_remote_masters.data(), master_send_disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), recv_masters.data(),
      num_recv_masters.data(), master_recv_disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), master_to_slave,
      &data_request[0]);
  MPI_Ineighbor_alltoallv(coeffs.data(), num_remote_masters.data(),
                          master_send_disp.data(), dolfinx::MPI::mpi_type<T>(),
                          recv_coeffs.data(), num_recv_masters.data(),
                          master_recv_disp.data(), dolfinx::MPI::mpi_type<T>(),
                          master_to_slave, &data_request[1]);
  MPI_Ineighbor_alltoallv(
      owners.data(), num_remote_masters.data(), master_send_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), recv_owners.data(),
      num_recv_masters.data(), master_recv_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), master_to_slave,
      &data_request[2]);

  /// Wait for all communication to finish
  MPI_Waitall(3, data_request.data(), data_status.data());
  dolfinx_mpc::recv_data<T> output;
  output.masters = recv_masters;
  output.coeffs = recv_coeffs;
  output.num_masters_per_slave = recv_num_masters_per_slave;
  output.owners = recv_owners;
  return output;
}

/// Append received slaves to arrays holding slave, master, coeffs and
/// num_masters_per_slave received from other processes
/// @param[in] in_data Structure holding incoming masters, coeffs, owners and
/// number of masters per slave
/// @param[in] local_slaves The slave dofs (local to process), where the ith
/// entry corresponds to the ith entry of the vectors in in-data.
/// @note local_slaves can have duplicates
/// @param[in] masters Array to append the masters to
/// @param[in] coeffs Array to append owner ranks to
/// @param[in] owners Array to append owner ranks to
/// @param[in] num_masters_per_slave Array to append num masters per slave to
/// @param[in] size_local The local size of the index map
/// @param[in] bs The block size of the index map
template <typename T>
void append_master_data(recv_data<T> in_data,
                        const std::vector<std::int32_t>& local_slaves,
                        std::vector<std::int32_t>& slaves,
                        std::vector<std::int64_t>& masters,
                        std::vector<T>& coeffs,
                        std::vector<std::int32_t>& owners,
                        std::vector<std::int32_t>& num_masters_per_slave,
                        std::int32_t size_local, std::int32_t bs)
{
  std::vector<std::int8_t> slave_found(size_local * bs, false);
  std::vector<std::int32_t>& m_per_slave = in_data.num_masters_per_slave;
  std::vector<std::int64_t>& inc_masters = in_data.masters;
  std::vector<T>& inc_coeffs = in_data.coeffs;
  std::vector<std::int32_t>& inc_owners = in_data.owners;
  assert(m_per_slave.size() == local_slaves.size());

  // Compute accumulated position
  std::vector<std::int32_t> disp_m(m_per_slave.size() + 1, 0);
  std::partial_sum(m_per_slave.begin(), m_per_slave.end(), disp_m.begin() + 1);

  // NOTE: outdegree is really in degree as we are using the reverse comm
  for (std::size_t i = 0; i < m_per_slave.size(); i++)
  {
    // Only add slave to list if it hasn't been found on another proc and the
    // number of incoming masters is nonzero
    if (auto dof = local_slaves[i]; !slave_found[dof] && m_per_slave[i] > 0)
    {
      slaves.push_back(dof);
      for (std::int32_t j = disp_m[i]; j < disp_m[i + 1]; j++)
      {
        masters.push_back(inc_masters[j]);
        owners.push_back(inc_owners[j]);
        coeffs.push_back(inc_coeffs[j]);
      }
      num_masters_per_slave.push_back(m_per_slave[i]);
      slave_found[dof] = true;
    }
  }

  // Check that all local blocks has found its master
  [[maybe_unused]] const auto num_found
      = std::accumulate(std::begin(slave_found), std::end(slave_found), 0.0);
  [[maybe_unused]] const std::size_t num_unique
      = std::set<std::int32_t>(local_slaves.begin(), local_slaves.end()).size();
  assert(num_found == num_unique);
}

/// Distribute local slave->master data from owning process to ghost processes
/// @param[in] slaves List of local slaves indices (local to process, unrolled)
/// @param[in] masters The corresponding master dofs (global indices, unrolled)
/// @param[in] coeffs The master coefficients
/// @param[in] owners The owners of the corresponding master dof
/// @param[in] num_masters_per_slave The number of masters owned by each slave
/// @param[in] imap The index map
/// @param[in] bs The index map block size
/// @returns Data structure holding the received slave->master data
template <typename T>
dolfinx_mpc::mpc_data distribute_ghost_data(
    const std::vector<std::int32_t>& slaves,
    const std::vector<std::int64_t>& masters, const std::vector<T>& coeffs,
    const std::vector<std::int32_t>& owners,
    const std::vector<std::int32_t>& num_masters_per_slave,
    std::shared_ptr<const dolfinx::common::IndexMap> imap, const int bs)
{
  std::shared_ptr<const dolfinx::common::IndexMap> slave_to_ghost;
  std::vector<int> parent_to_sub;
  parent_to_sub.reserve(slaves.size());
  std::vector<std::int32_t> slave_blocks;
  slave_blocks.reserve(slaves.size());
  std::vector<std::int32_t> slave_rems;
  slave_rems.reserve(slaves.size());

  // Create new index map for each slave block
  {
    std::vector<std::int32_t> blocks;
    blocks.reserve(slaves.size());
    std::transform(slaves.cbegin(), slaves.cend(), std::back_inserter(blocks),
                   [bs](auto& dof) { return dof / bs; });
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());

    std::pair<dolfinx::common::IndexMap, std::vector<int32_t>> compressed_map
        = imap->create_submap(blocks);
    slave_to_ghost = std::make_shared<const dolfinx::common::IndexMap>(
        std::move(compressed_map.first));

    // Build map from new index map to slave indices (unrolled)
    for (std::size_t i = 0; i < slaves.size(); i++)
    {
      std::div_t div = std::div(slaves[i], bs);
      slave_blocks[i] = div.quot;
      slave_rems[i] = div.rem;
      auto it = std::find(blocks.begin(), blocks.end(), div.quot);
      assert(it != blocks.end());
      auto index = std::distance(blocks.begin(), it);
      parent_to_sub.push_back((int)index);
    }
  }

  // Get communicator for owner->ghost
  MPI_Comm local_to_ghost = create_owner_to_ghost_comm(*slave_to_ghost);
  const std::vector<std::int32_t>& src_ranks_ghosts = slave_to_ghost->src();
  const std::vector<std::int32_t>& dest_ranks_ghosts = slave_to_ghost->dest();

  // Compute number of outgoing slaves and masters for each process
  dolfinx::graph::AdjacencyList<int> shared_indices
      = slave_to_ghost->index_to_dest_ranks();
  const std::size_t num_inc_proc = src_ranks_ghosts.size();
  const std::size_t num_out_proc = dest_ranks_ghosts.size();
  std::vector<std::int32_t> out_num_slaves(num_out_proc + 1, 0);
  std::vector<std::int32_t> out_num_masters(num_out_proc + 1, 0);
  for (std::size_t i = 0; i < slaves.size(); ++i)
  {
    for (auto proc : shared_indices.links(parent_to_sub[i]))
    {
      // Find index of process in local MPI communicator
      auto it
          = std::find(dest_ranks_ghosts.begin(), dest_ranks_ghosts.end(), proc);
      const auto index = std::distance(dest_ranks_ghosts.begin(), it);
      out_num_masters[index] += num_masters_per_slave[i];
      out_num_slaves[index]++;
    }
  }

  // Communicate number of incoming slaves and masters
  std::vector<int> in_num_slaves(num_inc_proc + 1);
  std::vector<int> in_num_masters(num_inc_proc + 1);
  std::array<MPI_Request, 2> requests;
  std::array<MPI_Status, 2> states;
  MPI_Ineighbor_alltoall(out_num_slaves.data(), 1, MPI_INT,
                         in_num_slaves.data(), 1, MPI_INT, local_to_ghost,
                         &requests[0]);
  out_num_slaves.pop_back();
  in_num_slaves.pop_back();
  MPI_Ineighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
                         in_num_masters.data(), 1, MPI_INT, local_to_ghost,
                         &requests[1]);
  out_num_masters.pop_back();
  in_num_masters.pop_back();
  // Compute out displacements for slaves and masters
  std::vector<std::int32_t> disp_out_masters(num_out_proc + 1, 0);
  std::partial_sum(out_num_masters.begin(), out_num_masters.end(),
                   disp_out_masters.begin() + 1);
  std::vector<std::int32_t> disp_out_slaves(num_out_proc + 1, 0);
  std::partial_sum(out_num_slaves.begin(), out_num_slaves.end(),
                   disp_out_slaves.begin() + 1);

  // Compute displacement of masters to able to insert them correctly
  std::vector<std::int32_t> local_offsets(slaves.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   local_offsets.begin() + 1);

  // Insertion counter
  std::vector<std::int32_t> insert_slaves(num_out_proc, 0);
  std::vector<std::int32_t> insert_masters(num_out_proc, 0);

  // Prepare arrays for sending ghost information
  std::vector<std::int64_t> masters_out(disp_out_masters.back());
  std::vector<PetscScalar> coeffs_out(disp_out_masters.back());
  std::vector<std::int32_t> owners_out(disp_out_masters.back());
  std::vector<std::int32_t> slaves_out_loc(disp_out_slaves.back());
  std::vector<std::int64_t> slaves_out(disp_out_slaves.back());
  std::vector<std::int32_t> masters_per_slave(disp_out_slaves.back());
  for (std::size_t i = 0; i < slaves.size(); ++i)
  {
    // Find ghost processes for the ith local slave
    const std::int32_t master_start = local_offsets[i];
    const std::int32_t master_end = local_offsets[i + 1];
    for (auto proc : shared_indices.links(parent_to_sub[i]))
    {
      // Find index of process in local MPI communicator
      auto it
          = std::find(dest_ranks_ghosts.begin(), dest_ranks_ghosts.end(), proc);
      const auto index = std::distance(dest_ranks_ghosts.begin(), it);

      // Insert slave and num masters per slave
      slaves_out_loc[disp_out_slaves[index] + insert_slaves[index]] = slaves[i];
      masters_per_slave[disp_out_slaves[index] + insert_slaves[index]]
          = num_masters_per_slave[i];
      insert_slaves[index]++;

      // Insert global master dofs to send
      std::copy(masters.begin() + master_start, masters.begin() + master_end,
                masters_out.begin() + disp_out_masters[index]
                    + insert_masters[index]);
      // Insert owners to send
      std::copy(owners.begin() + master_start, owners.begin() + master_end,
                owners_out.begin() + disp_out_masters[index]
                    + insert_masters[index]);
      // Insert coeffs to send
      std::copy(coeffs.begin() + master_start, coeffs.begin() + master_end,
                coeffs_out.begin() + disp_out_masters[index]
                    + insert_masters[index]);
      insert_masters[index] += num_masters_per_slave[i];
    }
  }
  // Map slaves to global index
  {
    std::vector<std::int32_t> blocks(slaves_out_loc.size());
    std::vector<std::int32_t> rems(slaves_out_loc.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
      std::div_t pos = std::div(slaves_out_loc[i], bs);
      blocks[i] = pos.quot;
      rems[i] = pos.rem;
    }
    imap->local_to_global(blocks, slaves_out);
    std::transform(slaves_out.cbegin(), slaves_out.cend(), rems.cbegin(),
                   slaves_out.begin(),
                   [bs](auto dof, auto rem) { return dof * bs + rem; });
  }

  // Create in displacements for slaves
  MPI_Wait(&requests[0], &states[0]);
  std::vector<std::int32_t> disp_in_slaves(num_inc_proc + 1, 0);
  std::partial_sum(in_num_slaves.begin(), in_num_slaves.end(),
                   disp_in_slaves.begin() + 1);

  // Create in displacements for masters
  MPI_Wait(&requests[1], &states[1]);
  std::vector<std::int32_t> disp_in_masters(num_inc_proc + 1, 0);
  std::partial_sum(in_num_masters.begin(), in_num_masters.end(),
                   disp_in_masters.begin() + 1);

  // Send data to ghost processes
  std::vector<MPI_Request> ghost_requests(5);
  std::vector<MPI_Status> ghost_status(5);

  // Receive slaves from owner
  std::vector<std::int64_t> recv_slaves(disp_in_slaves.back());
  MPI_Ineighbor_alltoallv(
      slaves_out.data(), out_num_slaves.data(), disp_out_slaves.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), recv_slaves.data(),
      in_num_slaves.data(), disp_in_slaves.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), local_to_ghost,
      &ghost_requests[0]);

  // Receive number of masters from owner
  std::vector<std::int32_t> recv_num(disp_in_slaves.back());
  MPI_Ineighbor_alltoallv(
      masters_per_slave.data(), out_num_slaves.data(), disp_out_slaves.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), recv_num.data(),
      in_num_slaves.data(), disp_in_slaves.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), local_to_ghost,
      &ghost_requests[1]);

  // Convert slaves to local index
  MPI_Wait(&ghost_requests[0], &ghost_status[0]);
  std::vector<std::int64_t> recv_block;
  recv_block.reserve(recv_slaves.size());
  std::vector<std::int32_t> recv_rem;
  recv_rem.reserve(recv_slaves.size());
  std::for_each(recv_slaves.cbegin(), recv_slaves.cend(),
                [bs, &recv_rem, &recv_block](const auto dof)
                {
                  recv_rem.push_back(dof % bs);
                  recv_block.push_back(dof / bs);
                });
  std::vector<std::int32_t> recv_local(recv_slaves.size());
  imap->global_to_local(recv_block, recv_local);
  for (std::size_t i = 0; i < recv_local.size(); i++)
    recv_local[i] = recv_local[i] * bs + recv_rem[i];

  MPI_Wait(&ghost_requests[1], &ghost_status[1]);

  // Receive masters, coeffs and owners from owning processes
  std::vector<std::int64_t> recv_masters(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(
      masters_out.data(), out_num_masters.data(), disp_out_masters.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), recv_masters.data(),
      in_num_masters.data(), disp_in_masters.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), local_to_ghost,
      &ghost_requests[2]);
  std::vector<std::int32_t> recv_owners(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(
      owners_out.data(), out_num_masters.data(), disp_out_masters.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), recv_owners.data(),
      in_num_masters.data(), disp_in_masters.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), local_to_ghost,
      &ghost_requests[3]);
  std::vector<PetscScalar> recv_coeffs(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(
      coeffs_out.data(), out_num_masters.data(), disp_out_masters.data(),
      dolfinx::MPI::mpi_type<PetscScalar>(), recv_coeffs.data(),
      in_num_masters.data(), disp_in_masters.data(),
      dolfinx::MPI::mpi_type<PetscScalar>(), local_to_ghost,
      &ghost_requests[4]);

  mpc_data ghost_data;
  ghost_data.slaves = recv_local;
  ghost_data.offsets = recv_num;

  MPI_Wait(&ghost_requests[2], &ghost_status[2]);
  ghost_data.masters = recv_masters;
  MPI_Wait(&ghost_requests[3], &ghost_status[3]);
  ghost_data.owners = recv_owners;
  MPI_Wait(&ghost_requests[4], &ghost_status[4]);
  ghost_data.coeffs = recv_coeffs;
  return ghost_data;
}

//-----------------------------------------------------------------------------
/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3).
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] u The values at the points. Values are not computed
/// for points with a negative cell index. This argument must be
/// passed with the correct size.
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size)
xt::xtensor<double, 3>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace& V,
                         const xt::xtensor<double, 2>& x,
                         const xtl::span<const std::int32_t>& cells);

//-----------------------------------------------------------------------------
/// Tabuilate dof coordinates (not unrolled for block size) for a set of points
/// and corresponding cells.
/// @param[in] V The function space
/// @param[in] dofs Array of dofs (not unrolled with block size)
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of a cell that contains dofs[i]
/// @returns The dof coordinates, where the ith row corresponds to the ith dof
xt::xtensor<double, 2>
tabulate_dof_coordinates(const dolfinx::fem::FunctionSpace& V,
                         xtl::span<const std::int32_t> dofs,
                         xtl::span<const std::int32_t> cells);

/// From a Mesh, find which cells collide with a set of points.
/// @note Uses the GJK algorithm, see dolfinx::geometry::compute_distance_gjk
/// for details
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points The points to check for collision
/// (shape=(num_points, 3))
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
/// @return Adjacency list where the ith node is the closest entity whose
/// squared distance is within eps2
/// @note There may be nodes with no entries in the adjacency list
dolfinx::graph::AdjacencyList<int> compute_colliding_cells(
    const dolfinx::mesh::Mesh& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    const xt::xtensor<double, 2>& points, const double eps2);

/// Given a mesh and corresponding bounding box tree and a set of points,check
/// which cells (local to process) collide with each point.
/// Return an array of the same size as the number of points, where the ith
/// entry corresponds to the first cell colliding with the ith point.
/// @note If no colliding point is found, the index -1 is returned.
/// @param[in] mesh The mesh
/// @param[in] tree The boundingbox tree of all cells (local to process) in the
/// mesh
/// @param[in] points The points to check collision with, shape (num_points, 3)
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
std::vector<std::int32_t>
find_local_collisions(const dolfinx::mesh::Mesh& mesh,
                      const dolfinx::geometry::BoundingBoxTree& tree,
                      const xt::xtensor<double, 2>& points, const double eps2);

/// Given an input array of dofs from a function space, return an array with
/// true/false if the degree of freedom is in a DirichletBC
/// @param[in] V The function space
/// @param[in] blocks The degrees of freedom (not unrolled for dofmap block
/// size)
/// @param[in] bcs List of Dirichlet BCs on V
template <typename T>
std::vector<std::int8_t> is_bc(
    const dolfinx::fem::FunctionSpace& V, xtl::span<const std::int32_t> blocks,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{
  auto dofmap = V.dofmap();
  assert(dofmap);
  auto imap = dofmap->index_map;
  assert(imap);
  const int bs = dofmap->index_map_bs();
  std::int32_t dim = bs * (imap->size_local() + imap->num_ghosts());
  std::vector<std::int8_t> dof_marker(dim, false);
  std::for_each(bcs.begin(), bcs.end(),
                [&dof_marker, &V](auto bc)
                {
                  assert(bc);
                  assert(bc->function_space());
                  if (bc->function_space()->contains(V))
                    bc->mark_dofs(dof_marker);
                });
  // Remove slave blocks contained in DirichletBC
  std::vector<std::int8_t> bc_marker(blocks.size(), 0);
  const int dofmap_bs = dofmap->bs();
  for (std::size_t i = 0; i < blocks.size(); i++)
  {
    auto& block = blocks[i];
    for (int j = 0; j < dofmap_bs; j++)
    {
      if (dof_marker[block * dofmap_bs + j])
      {
        bc_marker[i] = 1;
        break;
      }
    }
  }
  return bc_marker;
}

} // namespace dolfinx_mpc

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
#include <dolfinx/mesh/MeshTags.h>
#include <span>

namespace impl
{
/// Create a map from each dof (block) found on the set of facets
/// topologically,
/// to the connecting facets
/// @param[in] V The function space
/// @param[in] dim The dimension of the entities
/// @param[in] entities The list of entities
/// @returns The map from each block (local + ghost) to the set of facets
dolfinx::graph::AdjacencyList<std::int32_t>
create_block_to_facet_map(dolfinx::mesh::Topology& topology,
                          const dolfinx::fem::DofMap& dofmap, std::int32_t dim,
                          std::span<const std::int32_t> entities)
{
  std::shared_ptr<const dolfinx::common::IndexMap> imap = dofmap.index_map;
  const std::int32_t tdim = topology.dim();
  // Locate all dofs for each facet
  topology.create_connectivity(dim, tdim);
  topology.create_connectivity(tdim, dim);
  auto e_to_c = topology.connectivity(dim, tdim);
  auto c_to_e = topology.connectivity(tdim, dim);

  const std::int32_t num_dofs = imap->size_local() + imap->num_ghosts();
  std::vector<std::int32_t> num_facets_per_dof(num_dofs);

  // Count how many facets each dof on process relates to
  std::vector<std::int32_t> local_indices(entities.size());
  std::vector<std::int32_t> cells(entities.size());
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    auto cell = e_to_c->links(entities[i]);
    assert(cell.size() == 1);
    cells[i] = cell[0];

    // Get local index of facet with respect to the cell
    auto cell_entities = c_to_e->links(cell[0]);
    const auto* it
        = std::find(cell_entities.data(),
                    cell_entities.data() + cell_entities.size(), entities[i]);
    assert(it != (cell_entities.data() + cell_entities.size()));
    const int local_entity = std::distance(cell_entities.data(), it);
    local_indices[i] = local_entity;
    auto cell_blocks = dofmap.cell_dofs(cell[0]);
    auto closure_blocks
        = dofmap.element_dof_layout().entity_closure_dofs(dim, local_entity);
    for (std::size_t j = 0; j < closure_blocks.size(); ++j)
    {
      const int dof = cell_blocks[closure_blocks[j]];
      num_facets_per_dof[dof]++;
    }
  }

  // Compute offsets
  std::vector<std::int32_t> offsets(num_dofs + 1);
  offsets[0] = 0;
  std::partial_sum(num_facets_per_dof.begin(), num_facets_per_dof.end(),
                   offsets.begin() + 1);
  // Reuse data structure for insertion
  std::fill(num_facets_per_dof.begin(), num_facets_per_dof.end(), 0);

  // Create dof->entities map
  std::vector<std::int32_t> data(offsets.back());
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    auto cell_blocks = dofmap.cell_dofs(cells[i]);
    auto closure_blocks = dofmap.element_dof_layout().entity_closure_dofs(
        dim, local_indices[i]);
    for (std::size_t j = 0; j < closure_blocks.size(); ++j)
    {
      const int dof = cell_blocks[closure_blocks[j]];
      data[offsets[dof] + num_facets_per_dof[dof]++] = entities[i];
    }
  }
  return dolfinx::graph::AdjacencyList<std::int32_t>(data, offsets);
}

} // namespace impl

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

template <typename T>
struct mpc_data
{
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<T> coeffs;
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> owners;
};

template <typename T, std::floating_point U>
class MultiPointConstraint;

/// Given a function space, compute its shared entities
template <std::floating_point U>
dolfinx::graph::AdjacencyList<int>
compute_shared_indices(std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V)
{
  return V->dofmap()->index_map->index_to_dest_ranks();
}

template <std::floating_point U>
dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc1,
    const std::string& type = std::string())
{
  dolfinx::common::Timer timer("~MPC: Create Matrix");

  // Build sparsitypattern
  dolfinx::la::SparsityPattern pattern = create_sparsity_pattern(a, mpc0, mpc1);

  // Finalise communication
  dolfinx::common::Timer timer_s("~MPC: Assemble sparsity pattern");
  pattern.finalize();
  timer_s.stop();

  // Initialize matrix
  dolfinx::la::petsc::Matrix A(a.mesh()->comm(), pattern, type);

  return A;
}

template <std::floating_point U>
dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc,
    const std::string& type = std::string())
{
  return dolfinx_mpc::create_matrix(a, mpc, mpc, type);
}

/// Create neighborhood communicators from every processor with a slave dof on
/// it, to the processors with a set of master facets.
/// @param[in] comm The MPI communicator to base communications on
/// @param[in] meshtags The meshtag
/// @param[in] has_slaves Boolean saying if the processor owns slave dofs
/// @param[in] master_marker Tag for the other interface
std::array<MPI_Comm, 2>
create_neighborhood_comms(MPI_Comm comm,
                          const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
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
/// FIXME: Remove petsc dependency here
template <std::floating_point U>
dolfinx::fem::Function<PetscScalar>
create_normal_approximation(std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V,
                            std::int32_t dim,
                            std::span<const std::int32_t> entities)
{
  dolfinx::graph::AdjacencyList<std::int32_t> block_to_entities
      = impl::create_block_to_facet_map(*V->mesh()->topology_mutable(),
                                        *V->dofmap(), dim, entities);

  // Create normal vector function and get local span
  dolfinx::fem::Function<PetscScalar> nh(V);
  Vec n_local;
  dolfinx::la::petsc::Vector n_vec(
      dolfinx::la::petsc::create_vector_wrap(*nh.x()), false);
  VecGhostGetLocalForm(n_vec.vec(), &n_local);
  PetscInt n = 0;
  VecGetSize(n_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(n_local, &array);
  std::span<PetscScalar> _n(array, n);

  const std::int32_t bs = V->dofmap()->index_map_bs();
  std::array<U, 3> normal;
  std::array<U, 3> n_0;
  for (std::int32_t i = 0; i < block_to_entities.num_nodes(); i++)
  {
    auto ents = block_to_entities.links(i);
    if (ents.empty())
      continue;
    // Sum all normal for entities
    std::vector<U> normals = dolfinx::mesh::cell_normals(*V->mesh(), dim, ents);
    std::copy_n(normals.begin(), 3, n_0.begin());
    std::copy_n(n_0.begin(), 3, normal.begin());
    for (std::size_t j = 1; j < normals.size() / 3; ++j)
    {
      // Align direction of normal vectors n_0 and n_j
      U n_nj = std::transform_reduce(
          n_0.begin(), n_0.end(), std::next(normals.begin(), 3 * j), 0.,
          std::plus{}, [](auto x, auto y) { return x * y; });
      auto sign = n_nj / std::abs(n_nj);
      // NOTE: Could probably use std::transform for this operation
      for (std::size_t k = 0; k < 3; ++k)
        normal[k] += sign * normals[3 * j + k];
    }
    std::copy_n(normal.begin(), bs, std::next(_n.begin(), i * bs));
  }
  // Receive normals from other processes with dofs on the facets
  VecGhostUpdateBegin(n_vec.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(n_vec.vec(), ADD_VALUES, SCATTER_REVERSE);
  // Normalize nh
  auto imap = V->dofmap()->index_map;
  std::int32_t num_blocks = imap->size_local();
  for (std::int32_t i = 0; i < num_blocks; i++)
  {
    PetscScalar acc = 0;
    for (std::int32_t j = 0; j < bs; j++)
      acc += _n[i * bs + j] * _n[i * bs + j];
    if (U abs = std::abs(acc); abs > 1e-10)
    {
      for (std::int32_t j = 0; j < bs; j++)
        _n[i * bs + j] /= abs;
    }
  }

  VecGhostUpdateBegin(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  return nh;
}

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
///
/// @note This function is almost a copy of
/// dolfinx::fem::utils.h::create_sparsity_pattern
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
template <typename T>
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<T>& a)
{

  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (Classic)");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  // Get dof maps and mesh
  std::array<std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmap(),
      *a.function_spaces().at(1)->dofmap()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<dolfinx::fem::IntegralType> types = a.integral_types();
  if (types.find(dolfinx::fem::IntegralType::interior_facet) != types.end()
      or types.find(dolfinx::fem::IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  for (auto type : types)
  {
    std::vector<int> ids = a.integral_ids(type);
    switch (type)
    {
    case dolfinx::fem::IntegralType::cell:
      for (int id : ids)
      {
        std::span<const std::int32_t> cells = a.domain(type, id);
        dolfinx::fem::sparsitybuild::cells(pattern, cells,
                                           {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case dolfinx::fem::IntegralType::interior_facet:
      for (int id : ids)
      {
        std::span<const std::int32_t> facets = a.domain(type, id);
        std::vector<std::int32_t> f;
        f.reserve(facets.size() / 2);
        for (std::size_t i = 0; i < facets.size(); i += 4)
          f.insert(f.end(), {facets[i], facets[i + 2]});
        dolfinx::fem::sparsitybuild::interior_facets(
            pattern, f, {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case dolfinx::fem::IntegralType::exterior_facet:
      for (int id : ids)
      {
        std::span<const std::int32_t> facets = a.domain(type, id);
        std::vector<std::int32_t> cells;
        cells.reserve(facets.size() / 2);
        for (std::size_t i = 0; i < facets.size(); i += 2)
          cells.push_back(facets[i]);
        dolfinx::fem::sparsitybuild::cells(pattern, cells,
                                           {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  timer.stop();
}

/// Create a map from dof blocks to one of the cells that contains the degree of
/// freedom
/// @param[in] topology The mesh topology
/// @param[in] dofmap The dofmap
/// @param[in] blocks The blocks (local to process) we want to map
std::vector<std::int32_t>
create_block_to_cell_map(const dolfinx::mesh::Topology& topology,
                         const dolfinx::fem::DofMap& dofmap,
                         std::span<const std::int32_t> blocks);

/// Create sparsity pattern with multi point constraint additions to the rows
/// and the columns
/// @param[in] a bi-linear form for the current variational problem
/// (The one used to generate the standard sparsity-pattern)
/// @param[in] mpc0 The multi point constraint to apply to the rows of the
/// matrix.
/// @param[in] mpc1 The multi point constraint to apply to the columns of the
/// matrix.
template <typename T, std::floating_point U>
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> mpc1)
{
  {
    LOG(INFO) << "Generating MPC sparsity pattern";
    dolfinx::common::Timer timer("~MPC: Create sparsity pattern");
    if (a.rank() != 2)
    {
      throw std::runtime_error(
          "Cannot create sparsity pattern. Form is not a bilinear form");
    }

    // Extract function space and index map from mpc
    auto V0 = mpc0->function_space();
    auto V1 = mpc1->function_space();

    auto bs0 = V0->dofmap()->index_map_bs();
    auto bs1 = V1->dofmap()->index_map_bs();

    const auto& mesh = *(a.mesh());

    std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
    new_maps[0] = V0->dofmap()->index_map;
    new_maps[1] = V1->dofmap()->index_map;
    std::array<int, 2> bs = {bs0, bs1};
    dolfinx::la::SparsityPattern pattern(mesh.comm(), new_maps, bs);

    LOG(INFO) << "Build standard pattern\n";
    ///  Create and build sparsity pattern for original form. Should be
    ///  equivalent to calling create_sparsity_pattern(Form a)
    build_standard_pattern<T>(pattern, a);
    LOG(INFO) << "Build new pattern\n";

    // Arrays replacing slave dof with master dof in sparsity pattern
    auto pattern_populator
        = [](dolfinx::la::SparsityPattern& pattern,
             const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> mpc,
             const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>
                 mpc_off_axis,
             const auto& pattern_inserter, const auto& master_inserter)
    {
      const auto& V = mpc->function_space();
      const auto& V_off_axis = mpc_off_axis->function_space();

      // Data structures used for insert
      std::array<std::int32_t, 1> master_block;
      std::array<std::int32_t, 1> other_master_block;

      // Map from cell index (local to mpc) to slave indices in the cell
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
          cell_to_slaves = mpc->cell_to_slaves();

      // For each cell (local to process) having a slave, get all slaves in main
      // constraint, and all dofs in off-axis constraint in the cell
      for (std::int32_t i = 0; i < cell_to_slaves->num_nodes(); ++i)
      {
        std::span<const std::int32_t> slaves = cell_to_slaves->links(i);
        if (slaves.empty())
          continue;

        std::span<const std::int32_t> cell_dofs
            = V_off_axis->dofmap()->cell_dofs(i);

        // Arrays for flattened master slave data
        std::vector<std::int32_t> flattened_masters;
        flattened_masters.reserve(slaves.size());

        // For each slave find all master degrees of freedom and flatten them
        for (auto slave : slaves)
        {
          for (auto master : mpc->masters()->links(slave))
          {
            const std::div_t div
                = std::div(master, V->dofmap()->index_map_bs());
            flattened_masters.push_back(div.quot);
          }
        }

        // Loop over all masters and insert all cell dofs for each master
        for (std::size_t j = 0; j < flattened_masters.size(); ++j)
        {
          master_block[0] = flattened_masters[j];
          pattern_inserter(pattern, std::span(master_block), cell_dofs);
          // Add sparsity pattern for all master dofs of any slave on this cell
          for (std::size_t k = j + 1; k < flattened_masters.size(); ++k)
          {
            other_master_block[0] = flattened_masters[k];
            master_inserter(pattern, std::span(other_master_block),
                            std::span(master_block));
          }
        }
      }
    };

    if (mpc0 == mpc1) // TODO: should this be
                      // mpc0.function_space().contains(mpc1.function_space()) ?
    {
      // Only need to loop through once
      const auto square_inserter
          = [](auto& pattern, const auto& dofs_m, const auto& dofs_s)
      {
        pattern.insert(dofs_m, dofs_s);
        pattern.insert(dofs_s, dofs_m);
      };
      pattern_populator(pattern, mpc0, mpc1, square_inserter, square_inserter);
    }
    else
    {
      const auto do_nothing_inserter
          = []([[maybe_unused]] auto& pattern,
               [[maybe_unused]] const auto& dofs_m,
               [[maybe_unused]] const auto& dofs_s) {};
      // Potentially rectangular pattern needs each axis inserted separately
      pattern_populator(
          pattern, mpc0, mpc1,
          [](auto& pattern, const auto& dofs_m, const auto& dofs_s)
          { pattern.insert(dofs_m, dofs_s); },
          do_nothing_inserter);
      pattern_populator(
          pattern, mpc1, mpc0,
          [](auto& pattern, const auto& dofs_m, const auto& dofs_s)
          { pattern.insert(dofs_s, dofs_m); },
          do_nothing_inserter);
    }

    return pattern;
  }
}

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
dolfinx_mpc::mpc_data<T> distribute_ghost_data(
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
        = dolfinx::common::create_sub_index_map(*imap, blocks);
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
  std::span src_ranks_ghosts = slave_to_ghost->src();
  std::span dest_ranks_ghosts = slave_to_ghost->dest();

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
  std::vector<T> coeffs_out(disp_out_masters.back());
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
  std::vector<T> recv_coeffs(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(coeffs_out.data(), out_num_masters.data(),
                          disp_out_masters.data(), dolfinx::MPI::mpi_type<T>(),
                          recv_coeffs.data(), in_num_masters.data(),
                          disp_in_masters.data(), dolfinx::MPI::mpi_type<T>(),
                          local_to_ghost, &ghost_requests[4]);

  mpc_data<T> ghost_data;
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
/// (num_points, 3), flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] u The values at the points. Values are not computed
/// for points with a negative cell index. This argument must be
/// passed with the correct size.
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size). Flattened row major
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 3>>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace<U>& V,
                         std::span<const U> x,
                         std::span<const std::int32_t> cells)
{
  assert(x.size() % 3 == 0);
  const std::size_t num_points = x.size() / 3;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get mesh
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get geometry data
  namespace stdex = std::experimental;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();

  auto cmaps = mesh->geometry().cmaps();
  if (cmaps.size() > 1)
  {
    throw std::runtime_error(
        "Multiple coordinate maps in evaluate basis functions");
  }
  const std::size_t num_dofs_g = cmaps[0].dim();
  std::span<const U> x_g = mesh->geometry().x();

  // Get element
  auto element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error(
        "Evaluation of basis functions is not supported for mixed "
        "elements. Extract subspaces.");
  }

  // Return early if we have no points
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);

  assert(basis_shape[2]
         == std::size_t(element->space_dimension() / bs_element));
  assert(basis_shape[3] == std::size_t(element->value_size() / bs_element));
  std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<U> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), 1, std::multiplies{}));

  if (num_points == 0)
    return {output_basis, reference_shape};

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;

  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmaps[0].tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = stdex::submdspan(phi0, std::pair(1, tdim + 1), 0,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), 0,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Reference coordinates for each point
  std::vector<U> Xb(num_points * tdim);
  mdspan2_t X(Xb.data(), num_points, tdim);

  // Geometry data at each point
  std::vector<U> J_b(num_points * gdim * tdim);
  mdspan3_t J(J_b.data(), num_points, gdim, tdim);
  std::vector<U> K_b(num_points * tdim * gdim);
  mdspan3_t K(K_b.data(), num_points, tdim, gdim);
  std::vector<U> detJ(num_points);
  std::vector<U> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = stdex::submdspan(x_dofmap, cell_index,
                                   MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[3 * p + j];

    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    std::array<U, 3> Xpb = {0, 0, 0};
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmaps[0].is_affine())
    {
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs,
                                                           _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      std::array<U, 3> x0 = {0, 0, 0};
      for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
        x0[i] += coord_dofs(0, i);
      dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K, x0, xp);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }
    else
    {
      // Pull-back physical point xp to reference coordinate Xp
      cmaps[0].pull_back_nonaffine(Xp, xp, coord_dofs,
                                   500 * std::numeric_limits<U>::epsilon(), 15);

      cmaps[0].tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi, coord_dofs,
                                                           _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Compute basis on reference element
  std::vector<U> reference_basisb(std::reduce(
      basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  element->tabulate(reference_basisb, Xb, {X.extent(0), X.extent(1)}, 0);

  // Data structure to hold basis for transformation
  const std::size_t num_basis_values = basis_shape[2] * basis_shape[3];
  std::vector<U> basis_valuesb(num_basis_values);
  mdspan2_t basis_values(basis_valuesb.data(), basis_shape[2], basis_shape[3]);

  using xu_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xU_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xJ_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xK_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation
      = element->template get_pre_dof_transformation_function<U>();

  mdspan3_t full_basis(output_basis.data(), reference_shape);
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];
    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    std::copy_n(std::next(reference_basisb.begin(), num_basis_values * p),
                num_basis_values, basis_valuesb.begin());
    apply_dof_transformation(basis_valuesb, cell_info, cell_index,
                             (int)reference_value_size);

    auto _U = stdex::submdspan(full_basis, p,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    push_forward_fn(_U, basis_values, _J, detJ[p], _K);
  }
  return {output_basis, reference_shape};
}

//-----------------------------------------------------------------------------
/// Tabuilate dof coordinates (not unrolled for block size) for a set of points
/// and corresponding cells.
/// @param[in] V The function space
/// @param[in] dofs Array of dofs (not unrolled with block size)
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of a cell that contains dofs[i]
/// @param[in] transposed If true return coordiantes in xxyyzz format. Else
/// xyzxzyxzy
/// @returns The dof coordinates flattened in the appropriate format
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 2>> tabulate_dof_coordinates(
    const dolfinx::fem::FunctionSpace<U>& V, std::span<const std::int32_t> dofs,
    std::span<const std::int32_t> cells, bool transposed = false)
{
  if (!V.component().empty())
  {
    throw std::runtime_error("Cannot tabulate coordinates for a "
                             "FunctionSpace that is a subspace.");
  }
  auto element = V.element();
  assert(element);
  if (V.element()->is_mixed())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a mixed FunctionSpace.");
  }

  auto mesh = V.mesh();
  assert(mesh);

  const std::size_t gdim = mesh->geometry().dim();

  // Get dofmap local size
  auto dofmap = V.dofmap();
  assert(dofmap);
  std::shared_ptr<const dolfinx::common::IndexMap> index_map
      = V.dofmap()->index_map;
  assert(index_map);

  const int element_block_size = element->block_size();
  const std::size_t space_dimension
      = element->space_dimension() / element_block_size;

  // Get the dof coordinates on the reference element
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  auto [X_b, X_shape] = element->interpolation_points();

  // Get coordinate map
  auto cmaps = mesh->geometry().cmaps();
  if (cmaps.size() > 1)
  {
    throw std::runtime_error(
        "Multiple coordinate maps in evaluate tabulate dof coordinates");
  }

  // Prepare cell geometry
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();
  std::span<const U> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = x_dofmap.extent(1);

  // Array to hold coordinates to return
  namespace stdex = std::experimental;
  std::array<std::size_t, 2> coord_shape = {dofs.size(), 3};
  if (transposed)
    coord_shape = {3, dofs.size()};
  std::vector<U> coordsb(std::reduce(coord_shape.cbegin(), coord_shape.cend(),
                                     1, std::multiplies{}));

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

  // Loop over cells and tabulate dofs
  assert(space_dimension == X_shape[0]);
  std::vector<U> xb(space_dimension * gdim);
  mdspan2_t x(xb.data(), space_dimension, gdim);

  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coordinate_dofs_b(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  const auto apply_dof_transformation
      = element->template get_pre_dof_transformation_function<U>();

  const std::array<std::size_t, 4> bsize
      = cmaps[0].tabulate_shape(0, X_shape[0]);
  std::vector<U> phi_b(
      std::reduce(bsize.begin(), bsize.end(), 1, std::multiplies{}));
  cmaps[0].tabulate(0, X_b, X_shape, phi_b);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>
      phi_full(phi_b.data(), bsize);
  auto phi = stdex::submdspan(phi_full, 0,
                              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Create insertion function
  std::function<void(std::size_t, std::size_t, std::ptrdiff_t)> inserter;
  if (transposed)
  {
    inserter = [&coordsb, &xb, gdim, coord_shape](std::size_t c, std::size_t j,
                                                  std::ptrdiff_t loc)
    { coordsb[j * coord_shape[1] + c] = xb[loc * gdim + j]; };
  }
  else
  {
    inserter = [&coordsb, &xb, gdim, coord_shape](std::size_t c, std::size_t j,
                                                  std::ptrdiff_t loc)
    { coordsb[c * coord_shape[1] + j] = xb[loc * gdim + j]; };
  }

  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Fetch the coordinates of the cell
    auto x_dofs = stdex::submdspan(x_dofmap, cells[c],
                                   MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    // Tabulate dof coordinates on cell
    dolfinx::fem::CoordinateElement<U>::push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(std::span(xb.data(), x.size()),
                             std::span(cell_info.data(), cell_info.size()),
                             (std::int32_t)c, (int)gdim);

    // Get cell dofmap
    auto cell_dofs = dofmap->cell_dofs(cells[c]);
    auto it = std::find(cell_dofs.begin(), cell_dofs.end(), dofs[c]);
    auto loc = std::distance(cell_dofs.begin(), it);

    // Copy dof coordinates into vector
    for (std::size_t j = 0; j < gdim; ++j)
      inserter(c, j, loc);
  }

  return {coordsb, coord_shape};
}

/// From a Mesh, find which cells collide with a set of points.
/// @note Uses the GJK algorithm, see dolfinx::geometry::compute_distance_gjk
/// for details
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points The points to check for collision, shape=(num_points, 3).
/// Flattened row major.
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
/// @return Adjacency list where the ith node is the closest entity whose
/// squared distance is within eps2
/// @note There may be nodes with no entries in the adjacency list
template <std::floating_point U>
dolfinx::graph::AdjacencyList<int> compute_colliding_cells(
    const dolfinx::mesh::Mesh<U>& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    std::span<const U> points, const U eps2)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  const int tdim = mesh.topology()->dim();
  std::vector<std::int32_t> result;
  for (std::int32_t i = 0; i < candidate_cells.num_nodes(); i++)
  {
    auto cells = candidate_cells.links(i);
    if (cells.empty())
    {
      offsets.push_back((std::int32_t)colliding_cells.size());
      continue;
    }
    // Create span of
    std::vector<U> distances_sq(cells.size());
    for (std::size_t j = 0; j < cells.size(); j++)
    {
      distances_sq[j]
          = dolfinx::geometry::squared_distance(mesh, tdim, cells.subspan(j, 1),
                                                points.subspan(3 * i, 3))
                .front();
    }
    // Only push back closest cell
    if (auto cell_idx
        = std::min_element(distances_sq.cbegin(), distances_sq.cend());
        *cell_idx < eps2)
    {
      auto pos = std::distance(distances_sq.cbegin(), cell_idx);
      colliding_cells.push_back(cells[pos]);
    }
    offsets.push_back((std::int32_t)colliding_cells.size());
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(colliding_cells),
                                                     std::move(offsets));
}

/// Given a mesh and corresponding bounding box tree and a set of points,check
/// which cells (local to process) collide with each point.
/// Return an array of the same size as the number of points, where the ith
/// entry corresponds to the first cell colliding with the ith point.
/// @note If no colliding point is found, the index -1 is returned.
/// @param[in] mesh The mesh
/// @param[in] tree The boundingbox tree of all cells (local to process) in
/// the mesh
/// @param[in] points The points to check collision with, shape (num_points,
/// 3). Flattened row major.
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
template <std::floating_point U>
std::vector<std::int32_t>
find_local_collisions(const dolfinx::mesh::Mesh<U>& mesh,
                      const dolfinx::geometry::BoundingBoxTree<U>& tree,
                      std::span<const U> points, const U eps2)
{
  assert(points.size() % 3 == 0);

  // Compute collisions for each point with BoundingBoxTree
  dolfinx::graph::AdjacencyList<std::int32_t> bbox_collisions
      = dolfinx::geometry::compute_collisions(tree, points);

  // Compute exact collision
  auto cell_collisions = dolfinx_mpc::compute_colliding_cells(
      mesh, bbox_collisions, points, eps2);

  // Extract first collision
  std::vector<std::int32_t> collisions(points.size() / 3, -1);
  for (int i = 0; i < cell_collisions.num_nodes(); i++)
  {
    auto local_cells = cell_collisions.links(i);
    if (!local_cells.empty())
      collisions[i] = local_cells[0];
  }
  return collisions;
}

/// Given an input array of dofs from a function space, return an array with
/// true/false if the degree of freedom is in a DirichletBC
/// @param[in] V The function space
/// @param[in] blocks The degrees of freedom (not unrolled for dofmap block
/// size)
/// @param[in] bcs List of Dirichlet BCs on V
template <typename T, std::floating_point U>
std::vector<std::int8_t> is_bc(
    const dolfinx::fem::FunctionSpace<U>& V,
    std::span<const std::int32_t> blocks,
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

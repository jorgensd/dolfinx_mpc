// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx_mpc;

//-----------------------------------------------------------------------------
void dolfinx_mpc::build_standard_pattern(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx::fem::Form<PetscScalar>& a)
{
  dolfinx::common::Timer timer("~MPC: Build classic sparsity pattern");
  // Get dof maps
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    dolfinx::fem::SparsityPatternBuilder::cells(pattern, mesh.topology(),
                                                {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::interior_facet)
      > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::interior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(dolfinx::fem::IntegralType::exterior_facet)
      > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::exterior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
dolfinx_mpc::get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index)
{

  // TODO: This could be easily made more efficient by exploiting points
  // being ordered by the cell to which they belong.
  // Get mesh
  assert(V);
  assert(V->mesh());
  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate mapping
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  assert(V->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const int block_size = element->block_size();
  const int reference_value_size = element->reference_value_size() / block_size;
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;

  // Prepare geometry data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
  Eigen::Array<double, Eigen::Dynamic, 1> detJ(1);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(1,
                                                                          tdim);

  // Prepare basis function data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      1, space_dimension, reference_value_size);
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(1, space_dimension,
                                                         value_size);

  // Get dofmap
  assert(V->dofmap());
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();

  mesh->topology_mutable().create_entity_permutations();

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& permutation_info
      = mesh->topology().get_cell_permutation_info();
  // Skip negative cell indices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_array(space_dimension * block_size, value_size * block_size);
  basis_array.setZero();
  if (index < 0)
    return basis_array;

  // Get cell geometry (coordinate dofs)
  auto x_dofs = x_dofmap.links(index);
  for (int i = 0; i < num_dofs_g; ++i)
    coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

  // Compute reference coordinates X, and J, detJ and K
  cmap.compute_reference_geometry(X, J, detJ, K, x.head(gdim), coordinate_dofs);

  // Compute basis on reference element
  element->evaluate_reference_basis(basis_reference_values, X);

  // Push basis forward to physical element
  element->transform_reference_basis(basis_values, basis_reference_values, X, J,
                                     detJ, K, permutation_info[index]);

  // Get the degrees of freedom for the current cell
  auto dofs = dofmap->cell_dofs(index);

  // Expand basis values for each dof
  for (int block = 0; block < block_size; ++block)
  {
    for (int i = 0; i < space_dimension; ++i)
    {
      for (int j = 0; j < value_size; ++j)
      {
        basis_array(i * block_size + block, j * block_size + block)
            = basis_values(0, i, j);
      }
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<dolfinx::function::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
};
//-----------------------------------------------------------------------------
void dolfinx_mpc::add_pattern_diagonal(
    dolfinx::la::SparsityPattern& pattern,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> blocks,
    std::int32_t block_size)
{
  for (std::int32_t i = 0; i < blocks.rows(); ++i)
  {
    Eigen::Array<PetscInt, Eigen::Dynamic, 1> diag_block(block_size);
    for (std::size_t comp = 0; comp < block_size; comp++)
      diag_block(comp) = block_size * blocks[i] + comp;
    pattern.insert(diag_block, diag_block);
  }
}

//-----------------------------------------------------------------------------
std::array<MPI_Comm, 2> dolfinx_mpc::create_neighborhood_comms(
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker)
{
  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // Check if entities for either master or slave are on this processor
  std::vector<std::uint8_t> has_slaves(mpi_size, 0);
  if (std::find(meshtags.values().begin(), meshtags.values().end(),
                slave_marker)
      != meshtags.values().end())
    std::fill(has_slaves.begin(), has_slaves.end(), 1);

  std::vector<std::uint8_t> has_masters(mpi_size, 0);
  if (std::find(meshtags.values().begin(), meshtags.values().end(),
                master_marker)
      != meshtags.values().end())
    std::fill(has_masters.begin(), has_masters.end(), 1);

  // Get received data sizes from each rank
  std::vector<std::uint8_t> procs_with_masters(mpi_size, -1);
  MPI_Alltoall(has_masters.data(), 1, MPI_UINT8_T, procs_with_masters.data(), 1,
               MPI_UINT8_T, comm);
  std::vector<std::uint8_t> procs_with_slaves(mpi_size, -1);
  MPI_Alltoall(has_slaves.data(), 1, MPI_UINT8_T, procs_with_slaves.data(), 1,
               MPI_UINT8_T, comm);

  // Create communicator with edges slaves (sources) -> masters (destinations)
  MPI_Comm slaves_to_master = MPI_COMM_NULL;
  std::vector<std::int32_t> source_edges;
  std::vector<std::int32_t> dest_edges;
  // If current rank owns masters add all slaves as source edges
  if (procs_with_masters[rank] == 1)
    for (std::size_t i = 0; i < mpi_size; ++i)
      if ((i != rank) && (procs_with_slaves[i] == 1))
        source_edges.push_back(i);

  // If current rank owns a slave add all masters as destinations
  if (procs_with_slaves[rank] == 1)
    for (std::size_t i = 0; i < mpi_size; ++i)
      if ((i != rank) && (procs_with_masters[i] == 1))
        dest_edges.push_back(i);
  std::array comms{MPI_COMM_NULL, MPI_COMM_NULL};
  // Create communicator with edges slaves (sources) -> masters (destinations)
  {
    std::vector<int> source_weights(source_edges.size(), 1);
    std::vector<int> dest_weights(dest_edges.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, source_edges.size(), source_edges.data(), source_weights.data(),
        dest_edges.size(), dest_edges.data(), dest_weights.data(),
        MPI_INFO_NULL, false, &comms[0]);
  }
  // Create communicator with edges masters (sources) -> slaves (destinations)
  {
    std::vector<int> source_weights(dest_edges.size(), 1);
    std::vector<int> dest_weights(source_edges.size(), 1);

    MPI_Dist_graph_create_adjacent(comm, dest_edges.size(), dest_edges.data(),
                                   source_weights.data(), source_edges.size(),
                                   source_edges.data(), dest_weights.data(),
                                   MPI_INFO_NULL, false, &comms[1]);
  }
  return comms;
}
//-----------------------------------------------------------------------------
mpc_data dolfinx_mpc::create_contact_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> nh)
{
  // Create slave->master facets and master->slave facets neihborhood comms
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, slave_marker, master_marker);

  // Extract some const information from function-space
  const std::shared_ptr<const dolfinx::common::IndexMap> imap
      = V->dofmap()->index_map;
  const int tdim = V->mesh()->topology().dim();
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

  // Arrays for holding local and ghost data of slave and master coeffs
  std::vector<std::int32_t> local_slaves;
  std::vector<std::int32_t> ghost_slaves;
  std::vector<std::int32_t> local_masters;
  std::vector<std::int32_t> ghost_masters;
  std::vector<PetscScalar> local_coeffs;
  std::vector<PetscScalar> ghost_coeffs;
  std::vector<std::int32_t> local_master_offset = {0};
  std::vector<std::int32_t> ghost_master_offset = {0};

  // Determine which dof should be a slave dof (based on max component of
  // normal vector normal vector)
  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> normal_array
      = nh->x()->array();
  std::vector<std::int32_t> local_slaves_to_index;
  for (std::int32_t i = 0; i < slave_blocks.rows(); ++i)
  {
    std::vector<std::int32_t> dofs(block_size);
    std::iota(dofs.begin(), dofs.end(), slave_blocks(i, 0) * block_size);
    Eigen::Array<PetscScalar, 1, Eigen::Dynamic> coeffs(1, tdim);
    for (std::int32_t j = 0; j < tdim; ++j)
      coeffs[j] = std::abs(normal_array[dofs[j]]);
    Eigen::Index max_index;
    const PetscScalar coeff = coeffs.maxCoeff(&max_index);
    bool is_local = true;
    if (size_local * block_size <= dofs[max_index])
      is_local = false;
    // Compute local coefficients for MPC
    for (Eigen::Index j = 0; j < tdim; ++j)
      if (j == max_index)
      {
        if (is_local)
        {
          local_slaves_to_index.push_back(i);
          local_slaves.push_back(dofs[j]);
        }
        else
          ghost_slaves.push_back(dofs[j]);
      }
      else
      {
        PetscScalar coeff_j
            = -normal_array[dofs[j]] / normal_array[dofs[max_index]];
        if (is_local)
        {
          local_masters.push_back(dofs[j]);
          local_coeffs.push_back(coeff_j);
        }
        else
        {
          ghost_masters.push_back(dofs[j]);
          ghost_coeffs.push_back(coeff_j);
        }
      }

    if (is_local)
      local_master_offset.push_back(local_masters.size());
    else
      ghost_master_offset.push_back(ghost_masters.size());
  }

  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  V->mesh()->topology_mutable().create_connectivity(fdim, tdim);
  auto facet_to_cell = V->mesh()->topology().connectivity(fdim, tdim);
  assert(facet_to_cell);
  // Find all cells connected to master facets for collision detection
  std::set<std::int32_t> master_cells;
  auto facet_values = meshtags.values();
  auto facet_indices = meshtags.indices();
  for (std::int32_t i = 0; i < facet_indices.size(); ++i)
  {
    if (facet_values[i] == master_marker)
    {
      auto cell = facet_to_cell->links(facet_indices[i]);
      assert(cell.size() == 1);
      master_cells.insert(cell[0]);
    }
  }
  // Loop through all masters on current processor and check if they collide
  // with a local master facet
  dolfinx::geometry::BoundingBoxTree tree(*V->mesh(), tdim);
  std::int32_t num_local_cells
      = V->mesh()->topology().index_map(tdim)->size_local();
  std::map<std::int32_t, std::vector<std::int32_t>> slave_index_to_master_owner;
  std::map<std::int32_t, std::vector<std::int64_t>> slave_index_to_master_dof;
  std::map<std::int32_t, std::vector<PetscScalar>> slave_index_to_master_coeff;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners
      = V->dofmap()->index_map->ghost_owner_rank();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> coordinates
      = V->tabulate_dof_coordinates();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      local_slave_coordinates(local_slaves.size(), 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 3, Eigen::RowMajor>
      local_slave_normal(local_slaves.size(), 3);

  for (std::int32_t i = 0; i < local_slaves.size(); ++i)
  {
    // Get coordinate and coeff for slave and save in send array
    auto coord = coordinates.row(local_slaves[i]);
    local_slave_coordinates.row(i) = coord;
    Eigen::Array<PetscScalar, 1, Eigen::Dynamic> coeffs(1, tdim);
    std::vector<std::int32_t> dofs(block_size);
    std::iota(dofs.begin(), dofs.end(), slave_blocks(i, 0) * block_size);
    for (std::int32_t j = 0; j < tdim; ++j)
      coeffs[j] = normal_array[dofs[j]];
    local_slave_normal.row(i) = coeffs;

    // Narrow number of candidates by using BB collision detection
    std::vector<std::int32_t> possible_cells
        = dolfinx::geometry::compute_collisions(tree, coord);
    if (possible_cells.size() > 0)
    {
      // Filter out cell that are not connect to master facet or owned by
      // processor
      std::vector<std::int32_t> candidates;
      for (auto cell : possible_cells)
      {
        if ((std::find(master_cells.begin(), master_cells.end(), cell)
             != master_cells.end())
            && (cell < num_local_cells))
          candidates.push_back(cell);
      }
      // Use collision detection algorithm (GJK) to check distance from  cell
      // to point and select one within 1e-10
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
            PetscScalar coeff = basis_values(k, j) * coeffs[j]
                                / normal_array[local_slaves[i]];
            if (std::abs(coeff) > 1e-14)
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
        slave_index_to_master_dof.insert(std::pair(i, global_masters));
        slave_index_to_master_coeff.insert(std::pair(i, l_coeff));
        slave_index_to_master_owner.insert(std::pair(i, l_owner));
      }
    }
  }
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  // Structure storing mpc arrays
  dolfinx_mpc::mpc_data mpc;

  // If serial, we only have to gather slaves, masters, coeffs in 1D arrays
  if (mpi_size == 1)
  {
    std::vector<std::int64_t> masters_out;
    std::vector<PetscScalar> coeffs_out;
    std::vector<std::int32_t> offsets_out = {0};
    for (std::int32_t i = 0; i < local_slaves.size(); i++)
    {
      std::vector<std::int32_t> master_i(
          local_masters.begin() + local_master_offset[i],
          local_masters.begin() + local_master_offset[i + 1]);
      masters_out.insert(masters_out.end(), master_i.begin(), master_i.end());
      std::vector<PetscScalar> coeffs_i(
          local_coeffs.begin() + local_master_offset[i],
          local_coeffs.begin() + local_master_offset[i + 1]);
      coeffs_out.insert(coeffs_out.end(), coeffs_i.begin(), coeffs_i.end());
      std::vector<std::int64_t> other_masters = slave_index_to_master_dof[i];
      std::vector<PetscScalar> other_coeffs = slave_index_to_master_coeff[i];
      masters_out.insert(masters_out.end(), other_masters.begin(),
                         other_masters.end());
      coeffs_out.insert(coeffs_out.end(), other_coeffs.begin(),
                        other_coeffs.end());
      offsets_out.push_back(masters_out.size());
    }
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(masters_out.size());
    owners.fill(1);
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
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> gm(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gs(0, 1);
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> gc(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> go(0, 1);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> goff(1, 1);
    go.fill(0);
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

  // Get the  slave->master recv from and send to ranks
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[0], &indegree, &outdegree,
                                 &weighted);
  const auto [src_ranks, dest_ranks]
      = dolfinx::MPI::neighbors(neighborhood_comms[0]);

  // Figure out how much data to receive from each neighbor
  const std::int32_t num_slaves = local_slaves.size();
  std::vector<std::int32_t> num_slaves_recv(indegree);
  MPI_Neighbor_allgather(&num_slaves, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
                         num_slaves_recv.data(), 1,
                         dolfinx::MPI::mpi_type<std::int32_t>(),
                         neighborhood_comms[0]);

  // Compute displacements for data to receive
  std::vector<int> disp(indegree + 1, 0);
  std::partial_sum(num_slaves_recv.begin(), num_slaves_recv.end(),
                   disp.begin() + 1);

  // Send data to neihbors and receive data
  auto global_slaves = imap->local_to_global(local_slaves, false);
  std::vector<std::int64_t> slaves_recv(disp.back());
  MPI_Neighbor_allgatherv(global_slaves.data(), global_slaves.size(),
                          dolfinx::MPI::mpi_type<std::int64_t>(),
                          slaves_recv.data(), num_slaves_recv.data(),
                          disp.data(), dolfinx::MPI::mpi_type<std::int64_t>(),
                          neighborhood_comms[0]);
  // Multiply recv size by three to accommodate vector coordinates and function
  // data
  std::vector<int> num_slaves_recv3(indegree + 1);
  for (std::int32_t i = 0; i < num_slaves_recv.size(); ++i)
    num_slaves_recv3[i] = num_slaves_recv[i] * 3;
  std::vector<int> disp3(indegree + 1, 0);
  std::partial_sum(num_slaves_recv3.begin(), num_slaves_recv3.end(),
                   disp3.begin() + 1);

  // Send slave normal and coordinate to neighbors
  std::vector<double> coord_recv(disp3.back());
  MPI_Neighbor_allgatherv(
      local_slave_coordinates.data(), local_slave_coordinates.size(),
      dolfinx::MPI::mpi_type<double>(), coord_recv.data(),
      num_slaves_recv3.data(), disp3.data(), dolfinx::MPI::mpi_type<double>(),
      neighborhood_comms[0]);
  std::vector<PetscScalar> normal_recv(disp3.back());
  MPI_Neighbor_allgatherv(local_slave_normal.data(), local_slave_normal.size(),
                          dolfinx::MPI::mpi_type<PetscScalar>(),
                          normal_recv.data(), num_slaves_recv3.data(),
                          disp3.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
                          neighborhood_comms[0]);

  // Loop through received slaves and compute collisions and coefficients on
  // this proc
  std::map<std::int32_t, std::vector<std::int64_t>>
      recv_slave_index_to_master_dof;
  std::map<std::int32_t, std::vector<PetscScalar>>
      recv_slave_index_to_master_coeff;
  std::map<std::int32_t, std::vector<std::int32_t>>
      recv_slave_index_to_master_owner;
  for (std::int32_t i = 0; i < slaves_recv.size(); ++i)
  {

    Eigen::Map<const Eigen::Array<double, 1, 3>> slave_coord(
        coord_recv.data() + 3 * i, 3);
    Eigen::Map<const Eigen::Array<PetscScalar, 3, 1>> slave_normal(
        normal_recv.data() + 3 * i, 3);

    // Find index of slave coord by using block size remainder
    const int slave_as_int = slaves_recv[i];
    const std::div_t l = std::div(slave_as_int, block_size);
    std::int32_t local_index = l.rem;

    std::vector<std::int32_t> possible_cells
        = dolfinx::geometry::compute_collisions(tree, slave_coord);
    if (possible_cells.size() > 0)
    {
      // Filter out cell that are not connect to master facet or owned by
      // processor
      std::vector<std::int32_t> candidates;
      for (auto cell : possible_cells)
      {
        if ((std::find(master_cells.begin(), master_cells.end(), cell)
             != master_cells.end())
            && (cell < num_local_cells))
          candidates.push_back(cell);
      }
      // Use collision detection algorithm (GJK) to check distance from
      // cell to point and select one within 1e-10
      std::vector<std::int32_t> verified_candidates
          = dolfinx::geometry::select_colliding_cells(*V->mesh(), candidates,
                                                      slave_coord, 1);
      if (verified_candidates.size() == 1)
      {
        // Compute local contribution of slave on master facet
        Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> basis_values
            = get_basis_functions(V, slave_coord, verified_candidates[0]);
        auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);
        std::vector<std::int32_t> l_master;
        std::vector<PetscScalar> l_coeff;
        std::vector<std::int32_t> l_owner;
        for (std::int32_t j = 0; j < tdim; ++j)
        {
          for (std::int32_t k = 0; k < cell_dofs.size(); ++k)
          {
            PetscScalar coeff = basis_values(k, j) * slave_normal[j]
                                / slave_normal[local_index];
            if (std::abs(coeff) > 1e-14)
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
        recv_slave_index_to_master_dof.insert(std::pair(i, global_masters));
        recv_slave_index_to_master_coeff.insert(std::pair(i, l_coeff));
        recv_slave_index_to_master_owner.insert(std::pair(i, l_owner));
      }
    }
  }

  // Flatten structure by removing those global slaves not found on this
  // processor
  std::vector<std::int64_t> slaves_in;
  std::vector<std::int64_t> masters_in;
  std::vector<std::int32_t> owners_in;
  std::vector<PetscScalar> coeffs_in;
  std::vector<std::int32_t> offsets_in;

  for (std::int32_t i = 0; i < slaves_recv.size(); ++i)
  {
    auto global_masters = recv_slave_index_to_master_dof[i];
    if (global_masters.size() > 0)
      slaves_in.push_back(slaves_recv[i]);

    std::vector<std::int32_t> owners_i = recv_slave_index_to_master_owner[i];
    std::vector<PetscScalar> coeffs_i = recv_slave_index_to_master_coeff[i];
    masters_in.insert(masters_in.end(), global_masters.begin(),
                      global_masters.end());
    coeffs_in.insert(coeffs_in.end(), coeffs_i.begin(), coeffs_i.end());
    owners_in.insert(owners_in.end(), owners_i.begin(), owners_i.end());
    if ((masters_in.size() > 0) && (global_masters.size() > 0))
      offsets_in.push_back(masters_in.size());
  }

  // Figure out how many slaves and masters to receive from other procs (those
  // that have masters on that proc) (Procs with masters -> Procs with slaves)
  int indegree2(-1), outdegree2(-2), weighted2(-1);
  MPI_Dist_graph_neighbors_count(neighborhood_comms[1], &indegree2, &outdegree2,
                                 &weighted2);
  const auto [src_ranks2, dest_ranks2]
      = dolfinx::MPI::neighbors(neighborhood_comms[1]);

  const std::int32_t returning_num_slaves = slaves_in.size();
  const std::int32_t returning_num_masters = masters_in.size();
  std::vector<std::int32_t> recv_num_global_slaves(indegree2);
  MPI_Neighbor_allgather(
      &returning_num_slaves, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_num_global_slaves.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);
  std::vector<std::int32_t> recv_num_global_masters(indegree2);
  MPI_Neighbor_allgather(
      &returning_num_masters, 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_num_global_masters.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      neighborhood_comms[1]);
  // Slave processor receives slaves from other processor (where they have
  // masters)
  // Compute displacements for data to receive
  std::vector<int> disp_slaves(indegree2 + 1, 0);
  std::partial_sum(recv_num_global_slaves.begin(), recv_num_global_slaves.end(),
                   disp_slaves.begin() + 1);

  // Send and recv the global dof number of slaves whose masters has been
  // found
  std::vector<std::int64_t> slaves_in_recv(disp_slaves.back());
  MPI_Neighbor_allgatherv(
      slaves_in.data(), slaves_in.size(),
      dolfinx::MPI::mpi_type<std::int64_t>(), slaves_in_recv.data(),
      recv_num_global_slaves.data(), disp_slaves.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[1]);

  // Receive offsets of masters for each processor (without the first zero)
  std::vector<std::int32_t> master_offsets_in_recv(disp_slaves.back());
  MPI_Neighbor_allgatherv(
      offsets_in.data(), offsets_in.size(),
      dolfinx::MPI::mpi_type<std::int32_t>(), master_offsets_in_recv.data(),
      recv_num_global_slaves.data(), disp_slaves.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), neighborhood_comms[1]);

  // Recieve all data of masters (global dof number, coeff, owner)
  std::vector<int> disp_masters(indegree2 + 1, 0);
  std::partial_sum(recv_num_global_masters.begin(),
                   recv_num_global_masters.end(), disp_masters.begin() + 1);

  std::vector<std::int64_t> masters_in_recv(disp_masters.back());
  MPI_Neighbor_allgatherv(
      masters_in.data(), masters_in.size(),
      dolfinx::MPI::mpi_type<std::int64_t>(), masters_in_recv.data(),
      recv_num_global_masters.data(), disp_masters.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), neighborhood_comms[1]);
  std::vector<PetscScalar> coeffs_in_recv(disp_masters.back());
  MPI_Neighbor_allgatherv(
      coeffs_in.data(), coeffs_in.size(), dolfinx::MPI::mpi_type<PetscScalar>(),
      coeffs_in_recv.data(), recv_num_global_masters.data(),
      disp_masters.data(), dolfinx::MPI::mpi_type<PetscScalar>(),
      neighborhood_comms[1]);
  std::vector<std::int32_t> owners_in_recv(disp_masters.back());
  MPI_Neighbor_allgatherv(
      owners_in.data(), owners_in.size(),
      dolfinx::MPI::mpi_type<std::int32_t>(), owners_in_recv.data(),
      recv_num_global_masters.data(), disp_masters.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), neighborhood_comms[1]);

  auto recv_slaves_as_local = imap->global_to_local(slaves_in_recv, false);
  // Create look-up for which slaves have been located either:
  // - Locally on this processor (in slave_index_to_master_dof)
  // - Other other processor earlier in the following loop
  std::vector<bool> found(local_slaves.size(), false);
  for (std::map<std::int32_t, std::vector<std::int64_t>>::iterator iter
       = slave_index_to_master_dof.begin();
       iter != slave_index_to_master_dof.end(); ++iter)
    found[iter->first] = true;

  // Iterate through the processors
  for (std::int32_t i = 0; i < src_ranks2.size(); ++i)
  {
    const std::int32_t min = disp_slaves[i];
    const std::int32_t max = disp_slaves[i + 1];
    // Find offsets for masters on given proc
    std::vector<std::int32_t> master_offsets = {0};
    master_offsets.insert(master_offsets.end(),
                          master_offsets_in_recv.begin() + min,
                          master_offsets_in_recv.begin() + max);
    // Loop through slaves and add them if they havent already been added
    for (std::int32_t j = 0; j < master_offsets.size() - 1; ++j)
    {
      const std::int32_t slave = recv_slaves_as_local[min + j];
      if (slave != -1)
      {
        std::vector<std::int32_t>::iterator it
            = std::find(local_slaves.begin(), local_slaves.end(), slave);
        std::int32_t index = std::distance(local_slaves.begin(), it);
        // Skip if already found
        if (!found[index])
        {
          std::vector<std::int64_t> masters_(
              masters_in_recv.begin() + master_offsets[j],
              masters_in_recv.begin() + master_offsets[j + 1]);
          slave_index_to_master_dof.insert(std::pair(index, masters_));
          std::vector<PetscScalar> coeffs_(
              coeffs_in_recv.begin() + master_offsets[j],
              coeffs_in_recv.begin() + master_offsets[j + 1]);
          slave_index_to_master_coeff.insert(std::pair(index, coeffs_));
          std::vector<std::int32_t> owners_(
              owners_in_recv.begin() + master_offsets[j],
              owners_in_recv.begin() + master_offsets[j + 1]);
          slave_index_to_master_owner.insert(std::pair(index, owners_));
          found[index] = true;
        }
      }
    }
  }

  // Receive
  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gowners(0);

  // Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters(0, 1);
  // Eigen::Array<std::int64_t, Eigen::Dynamic, 1> gm(0, 1);

  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves(0, 1);
  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> gs(0, 1);

  // Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeffs(0, 1);
  // Eigen::Array<PetscScalar, Eigen::Dynamic, 1> gc(0, 1);

  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets(1, 1);
  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> goff(1, 1);

  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(0);
  // Eigen::Array<std::int32_t, Eigen::Dynamic, 1> go(0);

  // mpc.local_slaves = slaves;
  // mpc.ghost_slaves = gs;
  // mpc.local_masters = masters;
  // mpc.ghost_masters = gm;
  // mpc.local_offsets = offsets;
  // mpc.ghost_offsets = goff;
  // mpc.local_owners = owners;
  // mpc.ghost_owners = go;
  // mpc.local_coeffs = coeffs;
  // mpc.ghost_coeffs = gc;

  return mpc;
  // const auto [src_ranks, dest_ranks] =
  // dolfinx::MPI::neighbors(comms[0]);
}

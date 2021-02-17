// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "MultiPointConstraint.h"
#include <Eigen/Core>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx_mpc;

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
dolfinx_mpc::get_basis_functions(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::array<double, 3>& x, const int index)
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
  const dolfinx::common::array2d<double>& x_g = mesh->geometry().x();
  dolfinx::common::array2d<double> coordinate_dofs(num_dofs_g, gdim);

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
  std::vector<double> J(gdim * tdim);
  std::array<double, 1> detJ;
  std::vector<double> K(tdim * gdim);
  dolfinx::common::array2d<double> X(1, tdim);
  // Prepare basis function data structures
  std::vector<double> basis_reference_values(space_dimension
                                             * reference_value_size);
  std::vector<double> basis_values(space_dimension * value_size);

  // Get dofmap
  assert(V->dofmap());
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();

  mesh->topology_mutable().create_entity_permutations();

  const std::vector<std::uint32_t> permutation_info
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
  {
    tcb::span<const double> coord = x_g.row(x_dofs[i]);
    for (int j = 0; j < gdim; ++j)
      coordinate_dofs(i, j) = coord[j];
  }
  dolfinx::common::array2d<double> xp(1, gdim);
  for (int j = 0; j < gdim; ++j)
    xp(0, j) = x[j];
  // Compute reference coordinates X, and J, detJ and K
  cmap.compute_reference_geometry(X, J, detJ, K, xp, coordinate_dofs);

  // Compute basis on reference element
  element->evaluate_reference_basis(basis_reference_values, X);

  element->apply_dof_transformation(basis_reference_values.data(),
                                    permutation_info[index],
                                    reference_value_size);

  // Push basis forward to physical element
  element->transform_reference_basis(basis_values, basis_reference_values, X, J,
                                     detJ, K);

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
            = basis_values[i * value_size + j];
      }
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
};
//-----------------------------------------------------------------------------
dolfinx::la::PETScMatrix dolfinx_mpc::create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc,
    const std::string& type)
{
  dolfinx::common::Timer timer("~MPC: Create Matrix");

  // Build sparsitypattern
  dolfinx::la::SparsityPattern pattern = mpc->create_sparsity_pattern(a);

  // Finalise communication
  dolfinx::common::Timer timer_s("~MPC: Assemble sparsity pattern");
  pattern.assemble();
  timer_s.stop();

  // Initialize matrix
  dolfinx::la::PETScMatrix A(a.mesh()->mpi_comm(), pattern, type);

  return A;
}
//-----------------------------------------------------------------------------
std::array<MPI_Comm, 2> dolfinx_mpc::create_neighborhood_comms(
    dolfinx::mesh::MeshTags<std::int32_t>& meshtags, const bool has_slave,
    std::int32_t& master_marker)
{
  MPI_Comm comm = meshtags.mesh()->mpi_comm();
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  std::uint8_t slave_val = has_slave ? 1 : 0;
  std::vector<std::uint8_t> has_slaves(mpi_size, slave_val);
  // Check if entities if master entities are on this processor
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
MPI_Comm dolfinx_mpc::create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_dofs,
    std::vector<std::int32_t>& ghost_dofs,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map, int block_size)
{
  // Get data from IndexMap
  const std::vector<std::int32_t>& ghost_owners = index_map->ghost_owner_rank();
  const std::int32_t size_local = index_map->size_local();
  std::map<std::int32_t, std::set<int>> shared_indices
      = index_map->compute_shared_indices();
  MPI_Comm comm
      = index_map->comm(dolfinx::common::IndexMap::Direction::symmetric);

  // Array of processors sending to the ghost_dofs
  std::set<std::int32_t> src_edges;
  // Array of processors the local_dofs are sent to
  std::set<std::int32_t> dst_edges;
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  for (auto dof : local_dofs)
  {
    std::int32_t block = dof / block_size;

    const std::set<int> procs = shared_indices[block];
    for (auto proc : procs)
      dst_edges.insert(proc);
  }

  for (auto dof : ghost_dofs)
  {
    std::int32_t block = dof / block_size;

    const std::int32_t proc = ghost_owners[block - size_local];
    src_edges.insert(proc);
  }
  MPI_Comm comm_loc = MPI_COMM_NULL;
  // Create communicator with edges owners (sources) -> ghosts (destinations)
  std::vector<std::int32_t> source_edges;
  source_edges.assign(src_edges.begin(), src_edges.end());
  std::vector<std::int32_t> dest_edges;
  dest_edges.assign(dst_edges.begin(), dst_edges.end());
  std::vector<int> source_weights(source_edges.size(), 1);
  std::vector<int> dest_weights(dest_edges.size(), 1);
  MPI_Dist_graph_create_adjacent(comm, source_edges.size(), source_edges.data(),
                                 source_weights.data(), dest_edges.size(),
                                 dest_edges.data(), dest_weights.data(),
                                 MPI_INFO_NULL, false, &comm_loc);

  return comm_loc;
};
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::vector<std::int32_t>>
dolfinx_mpc::create_dof_to_facet_map(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    const tcb::span<const std::int32_t>& facets)
{

  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const int element_bs = dofmap->element_dof_layout->block_size();
  const std::int32_t tdim = mesh->topology().dim();
  // Locate all dofs for each facet
  mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
  auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  // Initialize empty map for the dofs located topologically
  std::map<std::int32_t, std::vector<std::int32_t>> dofs_to_facets;

  // For each facet, find which dofs is on the given facet
  for (std::int32_t i = 0; i < facets.size(); ++i)
  {
    auto cell = f_to_c->links(facets[i]);
    assert(cell.size() == 1);
    // Get local index of facet with respect to the cell
    auto cell_facets = c_to_f->links(cell[0]);
    const auto* it = std::find(
        cell_facets.data(), cell_facets.data() + cell_facets.size(), facets[i]);
    assert(it != (cell_facets.data() + cell_facets.size()));
    const int local_facet = std::distance(cell_facets.data(), it);
    auto cell_blocks = dofmap->cell_dofs(cell[0]);
    auto closure_blocks = dofmap->element_dof_layout->entity_closure_dofs(
        tdim - 1, local_facet);
    for (std::int32_t j = 0; j < closure_blocks.size(); ++j)
    {
      for (int block = 0; block < element_bs; ++block)
      {
        const int dof = element_bs * cell_blocks[closure_blocks[j]] + block;
        dofs_to_facets[dof].push_back(facets[i]);
      }
    }
  }
  return dofs_to_facets;
};
//-----------------------------------------------------------------------------
Eigen::Vector3d dolfinx_mpc::create_average_normal(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, std::int32_t dof,
    std::int32_t dim, const tcb::span<const std::int32_t>& entities)
{
  assert(entities.size() > 0);
  dolfinx::common::array2d<double> normals
      = dolfinx::mesh::cell_normals(*V->mesh(), dim, entities);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      normals_eigen(normals.data(), normals.shape[0], normals.shape[1]);
  Eigen::Vector3d normal(normals_eigen.row(0));
  for (std::int32_t i = 1; i < normals.shape[0]; ++i)
  {
    Eigen::Vector3d normal_i = normals_eigen.row(i);
    double sign = normal.dot(normal_i) / std::abs(normal.dot(normal_i));
    normal += sign * normal_i;
  }
  normal /= std::sqrt(normal.dot(normal));
  return normal;
};
//-----------------------------------------------------------------------------
void dolfinx_mpc::create_normal_approximation(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    const tcb::span<const std::int32_t>& entities,
    tcb::span<PetscScalar> vector)
{
  const std::int32_t tdim = V->mesh()->topology().dim();
  const std::int32_t block_size = V->dofmap()->index_map_bs();

  std::map<std::int32_t, std::vector<std::int32_t>> facets_to_dofs
      = create_dof_to_facet_map(V, entities);
  for (auto const& pair : facets_to_dofs)
  {
    Eigen::Vector3d normal = create_average_normal(
        V, pair.first, tdim - 1,
        tcb::span(pair.second.data(), pair.second.size()));
    const std::div_t div = std::div(pair.first, block_size);
    const int& block = div.rem;
    vector[pair.first] = normal[block];
  }
};

// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// Modified by Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "MultiPointConstraint.h"
#include <array>
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
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx_mpc;

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> dolfinx_mpc::get_basis_functions(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::array<double, 3>& x, const int index)
{

  // Get mesh
  assert(V);
  assert(V->mesh());
  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  const size_t gdim = mesh->geometry().dim();
  const size_t tdim = mesh->topology().dim();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});

  // Get coordinate mapping
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  assert(V->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const size_t block_size = element->block_size();
  const size_t reference_value_size
      = element->reference_value_size() / block_size;
  const size_t value_size = element->value_size() / block_size;
  const size_t space_dimension = element->space_dimension() / block_size;

  // Create data structures for Jacobian info
  xt::xtensor<double, 2> X = xt::empty<double>({(size_t)1, tdim});
  xt::xtensor<double, 3> J = xt::empty<double>({X.shape(0), gdim, tdim});
  xt::xtensor<double, 3> K = xt::empty<double>({X.shape(0), tdim, gdim});
  xt::xtensor<double, 1> detJ = xt::empty<double>({X.shape(0)});

  // Prepare basis function data structures
  xt::xtensor<double, 4> tabulated_data(
      {1, 1, space_dimension, reference_value_size});
  auto reference_basis_values
      = xt::view(tabulated_data, 0, xt::all(), xt::all(), xt::all());
  xt::xtensor<double, 3> basis_values(
      {static_cast<std::size_t>(1), space_dimension, value_size});

  // Get dofmap
  assert(V->dofmap());
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();

  mesh->topology_mutable().create_entity_permutations();

  const std::vector<std::uint32_t> permutation_info
      = mesh->topology().get_cell_permutation_info();
  // Skip negative cell indices
  xt::xtensor<double, 2> basis_array = xt::zeros<double>(
      {space_dimension * block_size, value_size * block_size});
  if (index < 0)
    return basis_array;

  // Get cell geometry (coordinate dofs)
  auto x_dofs = x_dofmap.links(index);
  for (int i = 0; i < num_dofs_g; ++i)
  {
    auto coord = xt::row(x_g, x_dofs[i]);
    for (int j = 0; j < gdim; ++j)
      coordinate_dofs(i, j) = coord[j];
  }
  xt::xtensor<double, 2> xp({1, gdim});
  for (int j = 0; j < gdim; ++j)
    xp(0, j) = x[j];

  // Compute reference coordinates X, and J, detJ and K
  cmap.pull_back(X, J, detJ, K, xp, coordinate_dofs);

  // Compute basis on reference element

  element->tabulate(tabulated_data, X, 0);

  element->apply_dof_transformation(
      xtl::span<double>(tabulated_data.data(), tabulated_data.size()),
      permutation_info[index], reference_value_size);

  // Push basis forward to physical element
  element->transform_reference_basis(basis_values, reference_basis_values, J,
                                     detJ, K);

  // Get the degrees of freedom for the current cell
  auto dofs = dofmap->cell_dofs(index);

  // Expand basis values for each dof
  for (int block = 0; block < block_size; ++block)
  {
    for (int i = 0; i < space_dimension; ++i)
    {
      const std::int32_t row = i * block_size + block;
      for (int j = 0; j < value_size; ++j)
        basis_array(row, j * block_size + block) = basis_values(0, i, j);
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
};
//-----------------------------------------------------------------------------
dolfinx::la::SparsityPattern dolfinx_mpc::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc1)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  /// Check that we are using the correct function-space in the bilinear
  /// form otherwise the index map will be wrong
  auto _V0 = mpc0->function_space();
  auto _V1 = mpc1->function_space();

  assert(a.function_spaces().at(0) == _V0);
  assert(a.function_spaces().at(1) == _V1);
  auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = mpc0->index_map();
  new_maps[1] = mpc1->index_map();
  std::array<int, 2> bs = {bs0, bs1};
  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps, bs);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);

  // Arrays replacing slave dof with master dof in sparsity pattern
  auto pattern_populator
      = [](dolfinx::la::SparsityPattern& pattern,
           const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>
               mpc,
           const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>
               mpc_off_axis,
           const auto& pattern_inserter, const auto& master_inserter) -> void
  {
    // Data structures used for insert
    std::array<std::int32_t, 1> master_block;
    std::array<std::int32_t, 1> other_master_block;

    // Map from cell index (local to mpc) to slave indices in the cell
    const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
        cell_to_slaves = mpc->cell_to_slaves();
    const std::vector<std::int32_t>& slave_cells = mpc->slave_cells();

    // For each cell (local to process) having a slave, get all slaves in main
    // constraint, and all dofs in off-axis constraint in the cell
    for (std::int32_t i = 0; i < cell_to_slaves->num_nodes(); ++i)
    {

      xtl::span<int32_t> slaves = cell_to_slaves->links(i);
      xtl::span<const int32_t> cell_dofs
          = mpc_off_axis->dofmap()->cell_dofs(slave_cells[i]);

      // Arrays for flattened master slave data
      // There are at least one master per slave
      std::vector<std::int32_t> flattened_masters;
      flattened_masters.reserve(slaves.size());

      // For each slave find all master degrees of freedom and flatten them
      const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
          master_block_map = mpc->master_block_map();
      for (auto slave : slaves)
      {
        const xtl::span<std::int32_t> local_masters
            = master_block_map->links(slave);
        for (auto master : local_masters)
          flattened_masters.push_back(master);
      }

      // Remove duplicate master blocks
      std::sort(flattened_masters.begin(), flattened_masters.end());
      flattened_masters.erase(
          std::unique(flattened_masters.begin(), flattened_masters.end()),
          flattened_masters.end());

      // Loop over all masters and insert all cell dofs for each master
      for (std::int32_t j = 0; j < flattened_masters.size(); ++j)
      {
        master_block[0] = flattened_masters[j];
        pattern_inserter(pattern, tcb::make_span(master_block), cell_dofs);
        // Add sparsity pattern for all master dofs of any slave on this cell
        for (std::int32_t k = j + 1; k < flattened_masters.size(); ++k)
        {
          other_master_block[0] = flattened_masters[k];
          master_inserter(pattern, tcb::make_span(other_master_block),
                          tcb::make_span(master_block));
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
        = [](auto& pattern, const auto& dofs_m, const auto& dofs_s) {};
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
//-----------------------------------------------------------------------------
dolfinx::la::SparsityPattern dolfinx_mpc::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc)
{
  return dolfinx_mpc::create_sparsity_pattern(a, mpc, mpc);
}
//-----------------------------------------------------------------------------
dolfinx::la::PETScMatrix dolfinx_mpc::create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc1,
    const std::string& type)
{
  dolfinx::common::Timer timer("~MPC: Create Matrix");

  // Build sparsitypattern
  dolfinx::la::SparsityPattern pattern = create_sparsity_pattern(a, mpc0, mpc1);

  // Finalise communication
  dolfinx::common::Timer timer_s("~MPC: Assemble sparsity pattern");
  pattern.assemble();
  timer_s.stop();

  // Initialize matrix
  dolfinx::la::PETScMatrix A(a.mesh()->mpi_comm(), pattern, type);

  return A;
}
//-----------------------------------------------------------------------------
dolfinx::la::PETScMatrix dolfinx_mpc::create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc,
    const std::string& type)
{
  return dolfinx_mpc::create_matrix(a, mpc, mpc, type);
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
      = index_map->comm(dolfinx::common::IndexMap::Direction::forward);

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
    const xtl::span<const std::int32_t>& facets)
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
xt::xtensor_fixed<double, xt::xshape<3>> dolfinx_mpc::create_average_normal(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, std::int32_t dof,
    std::int32_t dim, const xtl::span<const std::int32_t>& entities)
{
  assert(entities.size() > 0);
  xt::xtensor<double, 2> normals
      = dolfinx::mesh::cell_normals(*V->mesh(), dim, entities);
  xt::xtensor_fixed<double, xt::xshape<3>> normal = xt::row(normals, 0);
  xt::xtensor_fixed<double, xt::xshape<3>> normal_i;
  for (std::int32_t i = 1; i < normals.shape(0); ++i)
  {
    normal_i = xt::row(normals, i);
    auto sign = xt::linalg::dot(normal, normal_i)
                / xt::abs(xt::linalg::dot(normal, normal_i));
    normal += sign[0] * normal_i;
  }
  normal /= std::sqrt(xt::linalg::dot(normal, normal)[0]);
  return normal;
};
//-----------------------------------------------------------------------------
void dolfinx_mpc::create_normal_approximation(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    const xtl::span<const std::int32_t>& entities,
    xtl::span<PetscScalar> vector)
{
  const std::int32_t tdim = V->mesh()->topology().dim();
  const std::int32_t block_size = V->dofmap()->index_map_bs();

  std::map<std::int32_t, std::vector<std::int32_t>> facets_to_dofs
      = create_dof_to_facet_map(V, entities);
  for (auto const& pair : facets_to_dofs)
  {
    xt::xtensor_fixed<double, xt::xshape<3>> normal = create_average_normal(
        V, pair.first, tdim - 1,
        xtl::span<const std::int32_t>(pair.second.data(), pair.second.size()));
    const std::div_t div = std::div(pair.first, block_size);
    const int& block = div.rem;
    vector[pair.first] = normal[block];
  }
};

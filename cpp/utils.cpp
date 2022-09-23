// Copyright (C) 2020 Jorgen S. Dokken and Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx_mpc;

namespace
{

/// Create a map from each dof (block) found on the set of facets topologically,
/// to the connecting facets
/// @param[in] V The function space
/// @param[in] dim The dimension of the entities
/// @param[in] entities The list of entities
/// @returns The map from each block (local + ghost) to the set of facets
dolfinx::graph::AdjacencyList<std::int32_t>
create_block_to_facet_map(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                          std::int32_t dim,
                          const std::span<const std::int32_t>& entities)
{
  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  std::shared_ptr<const dolfinx::common::IndexMap> imap = dofmap->index_map;
  const std::int32_t tdim = mesh->topology().dim();
  // Locate all dofs for each facet
  mesh->topology_mutable().create_connectivity(dim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, dim);
  auto e_to_c = mesh->topology().connectivity(dim, tdim);
  auto c_to_e = mesh->topology().connectivity(tdim, dim);

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
    auto cell_blocks = dofmap->cell_dofs(cell[0]);
    auto closure_blocks
        = dofmap->element_dof_layout().entity_closure_dofs(dim, local_entity);
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
    auto cell_blocks = dofmap->cell_dofs(cells[i]);
    auto closure_blocks = dofmap->element_dof_layout().entity_closure_dofs(
        dim, local_indices[i]);
    for (std::size_t j = 0; j < closure_blocks.size(); ++j)
    {
      const int dof = cell_blocks[closure_blocks[j]];
      data[offsets[dof] + num_facets_per_dof[dof]++] = entities[i];
    }
  }
  return dolfinx::graph::AdjacencyList<std::int32_t>(data, offsets);
}

} // namespace

//-----------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<int> dolfinx_mpc::compute_shared_indices(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V)
{
  return V->dofmap()->index_map->index_to_dest_ranks();
}
//-----------------------------------------------------------------------------
dolfinx::la::petsc::Matrix dolfinx_mpc::create_matrix(
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
  dolfinx::la::petsc::Matrix A(a.mesh()->comm(), pattern, type);

  return A;
}
//-----------------------------------------------------------------------------
dolfinx::la::petsc::Matrix dolfinx_mpc::create_matrix(
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
  MPI_Comm comm = meshtags.mesh()->comm();
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
  std::vector<std::int32_t> source_edges;
  std::vector<std::int32_t> dest_edges;
  // If current rank owns masters add all slaves as source edges
  if (procs_with_masters[rank] == 1)
    for (int i = 0; i < mpi_size; ++i)
      if ((i != rank) && (procs_with_slaves[i] == 1))
        source_edges.push_back(i);

  // If current rank owns a slave add all masters as destinations
  if (procs_with_slaves[rank] == 1)
    for (int i = 0; i < mpi_size; ++i)
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
    std::vector<std::int32_t>& local_blocks,
    std::vector<std::int32_t>& ghost_blocks,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map)
{
  // Get data from IndexMap
  const std::vector<int>& ghost_owners = index_map->owners();
  const std::int32_t size_local = index_map->size_local();
  dolfinx::graph::AdjacencyList<int> shared_indices
      = index_map->index_to_dest_ranks();

  MPI_Comm comm = create_owner_to_ghost_comm(*index_map);

  // Array of processors sending to the ghost_dofs
  std::set<std::int32_t> src_edges;
  // Array of processors the local_dofs are sent to
  std::set<std::int32_t> dst_edges;
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  for (auto block : local_blocks)
    for (auto proc : shared_indices.links(block))
      dst_edges.insert(proc);

  for (auto block : ghost_blocks)
    src_edges.insert(ghost_owners[block - size_local]);

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
}
//-----------------------------------------------------------------------------
dolfinx::fem::Function<PetscScalar> dolfinx_mpc::create_normal_approximation(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, std::int32_t dim,
    const std::span<const std::int32_t>& entities)
{
  dolfinx::graph::AdjacencyList<std::int32_t> block_to_entities
      = create_block_to_facet_map(V, dim, entities);

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
  std::array<double, 3> normal;
  std::array<double, 3> n_0;
  for (std::int32_t i = 0; i < block_to_entities.num_nodes(); i++)
  {
    auto ents = block_to_entities.links(i);
    if (ents.empty())
      continue;
    // Sum all normal for entities
    std::vector<double> normals
        = dolfinx::mesh::cell_normals(*V->mesh(), dim, ents);
    std::copy_n(normals.begin(), 3, n_0.begin());
    std::copy_n(n_0.begin(), 3, normal.begin());
    for (std::size_t j = 1; j < normals.size() / 3; ++j)
    {
      // Align direction of normal vectors n_0 and n_j
      double n_nj = std::transform_reduce(
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
    if (double abs = std::abs(acc); abs > 1e-10)
    {
      for (std::int32_t j = 0; j < bs; j++)
        _n[i * bs + j] /= abs;
    }
  }

  VecGhostUpdateBegin(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  return nh;
}
//-----------------------------------------------------------------------------

std::vector<std::int32_t>
dolfinx_mpc::create_block_to_cell_map(const dolfinx::fem::FunctionSpace& V,
                                      std::span<const std::int32_t> blocks)
{
  std::vector<std::int32_t> cells;
  cells.reserve(blocks.size());
  // Create block -> cells map

  // Compute number of cells each dof is in
  auto mesh = V.mesh();
  auto dofmap = V.dofmap();
  auto imap = dofmap->index_map;
  const int size_local = imap->size_local();
  const std::vector<int>& ghost_owners = imap->owners();
  std::vector<std::int32_t> num_cells_per_dof(size_local + ghost_owners.size());

  const int tdim = mesh->topology().dim();
  auto cell_imap = mesh->topology().index_map(tdim);
  const int num_cells_local = cell_imap->size_local();
  const int num_ghost_cells = cell_imap->num_ghosts();
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap->cell_dofs(i);
    for (auto dof : dofs)
      num_cells_per_dof[dof]++;
  }
  std::vector<std::int32_t> cell_dofs_disp(num_cells_per_dof.size() + 1, 0);
  std::partial_sum(num_cells_per_dof.begin(), num_cells_per_dof.end(),
                   cell_dofs_disp.begin() + 1);
  std::vector<std::int32_t> cell_map(cell_dofs_disp.back());
  // Reuse num_cells_per_dof for insertion
  std::fill(num_cells_per_dof.begin(), num_cells_per_dof.end(), 0);

  // Create the block -> cells map
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap->cell_dofs(i);
    for (auto dof : dofs)
      cell_map[cell_dofs_disp[dof] + num_cells_per_dof[dof]++] = i;
  }

  // Populate map from slaves to corresponding cell (choose first cell in map)
  std::for_each(blocks.begin(), blocks.end(),
                [&cell_dofs_disp, &cell_map, &cells](const auto dof)
                { cells.push_back(cell_map[cell_dofs_disp[dof]]); });
  assert(cells.size() == blocks.size());
  return cells;
}

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

  // Extract function space and index map from mpc
  auto V0 = mpc0->function_space();
  auto V1 = mpc1->function_space();

  auto bs0 = V0->dofmap()->index_map_bs();
  auto bs1 = V1->dofmap()->index_map_bs();

  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = V0->dofmap()->index_map;
  new_maps[1] = V1->dofmap()->index_map;
  std::array<int, 2> bs = {bs0, bs1};
  dolfinx::la::SparsityPattern pattern(mesh.comm(), new_maps, bs);

  LOG(INFO) << "Build standard pattern\n";
  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  build_standard_pattern<PetscScalar>(pattern, a);
  LOG(INFO) << "Build new pattern\n";

  // Arrays replacing slave dof with master dof in sparsity pattern
  auto pattern_populator
      = [](dolfinx::la::SparsityPattern& pattern,
           const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>
               mpc,
           const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>
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
          const std::div_t div = std::div(master, V->dofmap()->index_map_bs());
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
    const auto do_nothing_inserter = []([[maybe_unused]] auto& pattern,
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

std::pair<std::vector<double>, std::array<std::size_t, 3>>
dolfinx_mpc::evaluate_basis_functions(
    const dolfinx::fem::FunctionSpace& V, std::span<const double> x,
    const std::span<const std::int32_t>& cells)
{
  assert(x.size() % 3 == 0);
  const std::size_t num_points = x.size() / 3;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();
  auto map = mesh->topology().index_map(tdim);

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = mesh->geometry().cmap().dim();
  std::span<const double> x_g = mesh->geometry().x();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error("Function::eval is not supported for mixed "
                             "elements. Extract subspaces.");
  }

  // Return early if we have no points
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);

  assert(basis_shape[2] == element->space_dimension() / bs_element);
  assert(basis_shape[3] == element->value_size() / bs_element);
  std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<double> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), 1, std::multiplies{}));

  if (num_points == 0)
    return {output_basis, reference_shape};

  // If the space has sub elements, concatenate the evaluations on the sub
  // elements
  if (const int num_sub_elements = element->num_sub_elements();
      num_sub_elements > 1 && num_sub_elements != bs_element)
  {
    throw std::runtime_error("Function::eval is not supported for mixed "
                             "elements. Extract subspaces.");
  }

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  namespace stdex = std::experimental;
  using cmdspan4_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;

  // Create buffer for coordinate dofs and point in physical space
  std::vector<double> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<double> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<double> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmap.tabulate(1, std::vector<double>(tdim), {1, tdim}, phi0_b);
  auto dphi0 = stdex::submdspan(phi0, std::pair(1, tdim + 1), 0,
                                stdex::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
  std::vector<double> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  auto dphi
      = stdex::submdspan(phi, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);

  // Reference coordinates for each point
  std::vector<double> Xb(num_points * tdim);
  mdspan2_t X(Xb.data(), num_points, tdim);

  // Geometry data at each point
  std::vector<double> J_b(num_points * gdim * tdim);
  mdspan3_t J(J_b.data(), num_points, gdim, tdim);
  std::vector<double> K_b(num_points * tdim * gdim);
  mdspan3_t K(K_b.data(), num_points, tdim, gdim);
  std::vector<double> detJ(num_points);
  std::vector<double> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell_index);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[3 * p + j];

    auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
    auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);

    std::array<double, 3> Xpb = {0, 0, 0};
    stdex::mdspan<double, stdex::extents<std::size_t, 1, stdex::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine())
    {
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi0, coord_dofs, _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      std::array<double, 3> x0 = {0, 0, 0};
      for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
        x0[i] += coord_dofs(0, i);
      dolfinx::fem::CoordinateElement::pull_back_affine(Xp, _K, x0, xp);
      detJ[p] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          _J, det_scratch);
    }
    else
    {
      // Pull-back physical point xp to reference coordinate Xp
      cmap.pull_back_nonaffine(Xp, xp, coord_dofs);

      cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coord_dofs, _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      detJ[p] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          _J, det_scratch);
    }

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Compute basis on reference element
  std::vector<double> reference_basisb(std::reduce(
      basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  element->tabulate(reference_basisb, Xb, {X.extent(0), X.extent(1)}, 0);

  // Data structure to hold basis for transformation
  const std::size_t num_basis_values = basis_shape[2] * basis_shape[3];
  std::vector<double> basis_valuesb(num_basis_values);
  mdspan2_t basis_values(basis_valuesb.data(), basis_shape[2], basis_shape[3]);

  using xu_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using xU_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using xJ_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using xK_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation
      = element->get_dof_transformation_function<double>();

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

    auto _U = stdex::submdspan(full_basis, p, stdex::full_extent,
                               stdex::full_extent);
    auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
    auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
    push_forward_fn(_U, basis_values, _J, detJ[p], _K);
  }
  return {output_basis, reference_shape};
}

//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
dolfinx_mpc::tabulate_dof_coordinates(const dolfinx::fem::FunctionSpace& V,
                                      std::span<const std::int32_t> dofs,
                                      std::span<const std::int32_t> cells,
                                      bool transposed)
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
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  std::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = x_dofmap.num_links(0);

  // Array to hold coordinates to return
  namespace stdex = std::experimental;
  std::array<std::size_t, 2> coord_shape = {dofs.size(), 3};
  if (transposed)
    coord_shape = {3, dofs.size()};
  std::vector<double> coordsb(std::reduce(
      coord_shape.cbegin(), coord_shape.cend(), 1, std::multiplies{}));

  using cmdspan4_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;

  // Loop over cells and tabulate dofs
  assert(space_dimension == X_shape[0]);
  std::vector<double> xb(space_dimension * gdim);
  mdspan2_t x(xb.data(), space_dimension, gdim);

  // Create buffer for coordinate dofs and point in physical space
  std::vector<double> coordinate_dofs_b(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  const std::function<void(const std::span<double>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  const std::array<std::size_t, 4> bsize = cmap.tabulate_shape(0, X_shape[0]);
  std::vector<double> phi_b(
      std::reduce(bsize.begin(), bsize.end(), 1, std::multiplies{}));
  cmap.tabulate(0, X_b, X_shape, phi_b);
  cmdspan4_t phi_full(phi_b.data(), bsize);
  auto phi = stdex::submdspan(phi_full, 0, stdex::full_extent,
                              stdex::full_extent, 0);

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
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(cells[c]);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    // Tabulate dof coordinates on cell
    dolfinx::fem::CoordinateElement::push_forward(x, coordinate_dofs, phi);
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

//-----------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_mpc::compute_colliding_cells(
    const dolfinx::mesh::Mesh& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    std::span<const double> points, const double eps2)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  const int tdim = mesh.topology().dim();
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
    std::vector<double> distances_sq(cells.size());
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
//-----------------------------------------------------------------------------
std::vector<std::int32_t> dolfinx_mpc::find_local_collisions(
    const dolfinx::mesh::Mesh& mesh,
    const dolfinx::geometry::BoundingBoxTree& tree,
    std::span<const double> points, const double eps2)
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

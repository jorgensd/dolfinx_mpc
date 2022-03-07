// Copyright (C) 2020 Jorgen S. Dokken and Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "utils.h"
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xsort.hpp>
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
                          const xtl::span<const std::int32_t>& entities)
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
xt::xtensor<double, 2> dolfinx_mpc::get_basis_functions(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const xt::xtensor<double, 2>& x, const int index)
{
  // Get mesh
  assert(V);
  assert(V->mesh());
  const std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  const size_t gdim = mesh->geometry().dim();
  const size_t tdim = mesh->topology().dim();
  assert(x.shape(0) == 1);
  assert(x.shape(1) == gdim);

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const size_t num_dofs_g = x_dofmap.num_links(0);
  xtl::span<const double> x_g = mesh->geometry().x();
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
  for (std::size_t i = 0; i < num_dofs_g; ++i)
  {
    const int pos = 3 * x_dofs[i];
    for (std::size_t j = 0; j < gdim; ++j)
      coordinate_dofs(i, j) = x_g[pos + j];
  }
  xt::xtensor<double, 2> xp({1, gdim});
  for (std::size_t j = 0; j < gdim; ++j)
    xp(0, j) = x[j];

  // -- Lambda function for affine pull-backs
  auto pull_back_affine
      = [&cmap, tdim,
         X0 = xt::xtensor<double, 2>(xt::zeros<double>({std::size_t(1), tdim})),
         data = xt::xtensor<double, 4>(cmap.tabulate_shape(1, 1)),
         dphi = xt::xtensor<double, 2>({tdim, cmap.tabulate_shape(1, 1)[2]})](
            auto&& X, const auto& cell_geometry, auto&& J, auto&& K,
            const auto& x) mutable
  {
    cmap.tabulate(1, X0, data);
    dphi = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
    cmap.compute_jacobian(dphi, cell_geometry, J);
    cmap.compute_jacobian_inverse(J, K);
    cmap.pull_back_affine(X, K, cmap.x0(cell_geometry), x);
  };
  // FIXME: Move initialization out of J, detJ, K out of function
  xt::xtensor<double, 2> dphi;
  xt::xtensor<double, 2> X({1, tdim});
  xt::xtensor<double, 3> J = xt::zeros<double>({std::size_t(1), gdim, tdim});
  xt::xtensor<double, 3> K = xt::zeros<double>({std::size_t(1), tdim, gdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({1});
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
  if (cmap.is_affine())
  {
    J.fill(0);
    pull_back_affine(X, coordinate_dofs, xt::view(J, 0, xt::all(), xt::all()),
                     xt::view(K, 0, xt::all(), xt::all()), xp);
    detJ[0] = cmap.compute_jacobian_determinant(
        xt::view(J, 0, xt::all(), xt::all()));
  }
  else
  {
    cmap.pull_back_nonaffine(X, xp, coordinate_dofs);
    cmap.tabulate(1, X, phi);
    dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
    J.fill(0);
    auto _J = xt::view(J, 0, xt::all(), xt::all());
    cmap.compute_jacobian(dphi, coordinate_dofs, _J);
    cmap.compute_jacobian_inverse(_J, xt::view(K, 0, xt::all(), xt::all()));
    detJ[0] = cmap.compute_jacobian_determinant(_J);
  }

  // Compute basis on reference element
  element->tabulate(tabulated_data, X, 0);

  element->apply_dof_transformation(
      xtl::span<double>(tabulated_data.data(), tabulated_data.size()),
      permutation_info[index], reference_value_size);

  // Push basis forward to physical element
  using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(reference_basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
  for (std::size_t i = 0; i < basis_values.shape(0); ++i)
  {
    auto _K = xt::view(K, i, xt::all(), xt::all());
    auto _J = xt::view(J, i, xt::all(), xt::all());
    auto _u = xt::view(basis_values, i, xt::all(), xt::all());
    auto _U = xt::view(reference_basis_values, i, xt::all(), xt::all());
    push_forward_fn(_u, _U, _J, detJ[i], _K);
  }

  // Expand basis values for each dof
  for (std::size_t block = 0; block < block_size; ++block)
  {
    for (std::size_t i = 0; i < space_dimension; ++i)
    {
      for (std::size_t j = 0; j < value_size; ++j)
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
    std::shared_ptr<dolfinx::fem::FunctionSpace> V)
{
  return V->dofmap()->index_map->compute_shared_indices();
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

  for (auto block : local_blocks)
  {
    const std::set<int> procs = shared_indices[block];
    for (auto proc : procs)
      dst_edges.insert(proc);
  }

  for (auto block : ghost_blocks)
  {
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
}
//-----------------------------------------------------------------------------
dolfinx::fem::Function<PetscScalar> dolfinx_mpc::create_normal_approximation(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, std::int32_t dim,
    const xtl::span<const std::int32_t>& entities)
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
  xtl::span<PetscScalar> _n(array, n);

  const std::int32_t bs = V->dofmap()->index_map_bs();
  xt::xtensor_fixed<double, xt::xshape<3>> normal;
  for (std::int32_t i = 0; i < block_to_entities.num_nodes(); i++)
  {
    auto ents = block_to_entities.links(i);
    if (ents.empty())
      continue;
    // Sum all normal for entities
    xt::xtensor<double, 2> normals
        = dolfinx::mesh::cell_normals(*V->mesh(), dim, ents);

    auto n_0 = xt::row(normals, 0);
    normal = n_0;
    for (std::size_t i = 1; i < normals.shape(0); ++i)
    {
      // Align direction of normal vectors
      double n_ni = dot(n_0, xt::row(normals, i));
      auto sign = n_ni / std::abs(n_ni);
      normal += sign * xt::row(normals, i);
    }
    for (std::int32_t j = 0; j < bs; j++)
    {
      _n[i * bs + j] = normal[j];
    }
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
    if (std::sqrt(xt::norm(acc)) > 1e-10)
    {
      acc = std::sqrt(xt::norm(acc));
      for (std::int32_t j = 0; j < bs; j++)
        _n[i * bs + j] /= acc;
    }
  }

  VecGhostUpdateBegin(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  return nh;
}
//-----------------------------------------------------------------------------

std::vector<std::int32_t>
dolfinx_mpc::create_block_to_cell_map(const dolfinx::fem::FunctionSpace& V,
                                      tcb::span<const std::int32_t> blocks)
{
  std::vector<std::int32_t> cells;
  cells.reserve(blocks.size());
  // Create block -> cells map

  // Compute number of cells each dof is in
  auto mesh = V.mesh();
  auto dofmap = V.dofmap();
  auto imap = dofmap->index_map;
  const int size_local = imap->size_local();
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();
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
    const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
        cell_to_slaves = mpc->cell_to_slaves();

    // For each cell (local to process) having a slave, get all slaves in main
    // constraint, and all dofs in off-axis constraint in the cell
    for (std::int32_t i = 0; i < cell_to_slaves->num_nodes(); ++i)
    {
      if (cell_to_slaves->num_links(i) == 0)
        continue;

      xtl::span<int32_t> slaves = cell_to_slaves->links(i);
      xtl::span<const int32_t> cell_dofs = V_off_axis->dofmap()->cell_dofs(i);

      // Arrays for flattened master slave data
      std::vector<std::int32_t> flattened_masters;
      flattened_masters.reserve(slaves.size());

      // For each slave find all master degrees of freedom and flatten them
      for (auto slave : slaves)
      {
        const auto& local_masters = mpc->masters()->links(slave);
        for (auto master : local_masters)
        {
          const std::div_t div = std::div(master, V->dofmap()->index_map_bs());
          flattened_masters.push_back(div.quot);
        }
      }

      // Loop over all masters and insert all cell dofs for each master
      for (std::size_t j = 0; j < flattened_masters.size(); ++j)
      {
        master_block[0] = flattened_masters[j];
        pattern_inserter(pattern, tcb::make_span(master_block), cell_dofs);
        // Add sparsity pattern for all master dofs of any slave on this cell
        for (std::size_t k = j + 1; k < flattened_masters.size(); ++k)
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

xt::xtensor<double, 3> dolfinx_mpc::evaluate_basis_functions(
    const dolfinx::fem::FunctionSpace& V, const xt::xtensor<double, 2>& x,
    const xtl::span<const std::int32_t>& cells)
{
  if (x.shape(0) != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }
  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();
  auto map = mesh->topology().index_map((int)tdim);

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  xtl::span<const double> x_g = mesh->geometry().x();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  assert(V.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;
  const std::size_t value_size = element->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;

  // Return early if we have no points
  xt::xtensor<double, 3> basis_values(
      {x.shape(0), space_dimension, value_size});
  if (x.shape(0) == 0)
    return basis_values;

  // If the space has sub elements, concatenate the evaluations on the sub
  // elements
  if (const int num_sub_elements = element->num_sub_elements();
      num_sub_elements > 1 && num_sub_elements != bs_element)
  {
    throw std::runtime_error("Function::eval is not supported for mixed "
                             "elements. Extract subspaces.");
  }

  // Get dofmap
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  assert(dofmap);

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, gdim});
  xt::xtensor<double, 2> xp = xt::zeros<double>({std::size_t(1), gdim});

  // -- Lambda function for affine pull-backs
  xt::xtensor<double, 4> data(cmap.tabulate_shape(1, 1));
  const xt::xtensor<double, 2> X0(xt::zeros<double>({std::size_t(1), tdim}));
  cmap.tabulate(1, X0, data);
  const xt::xtensor<double, 2> dphi_i
      = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
  auto pull_back_affine = [&dphi_i](auto&& X, const auto& cell_geometry,
                                    auto&& J, auto&& K, const auto& x) mutable
  {
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_i, cell_geometry, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    dolfinx::fem::CoordinateElement::pull_back_affine(
        X, K, dolfinx::fem::CoordinateElement::x0(cell_geometry), x);
  };

  xt::xtensor<double, 2> dphi;
  xt::xtensor<double, 2> X({x.shape(0), tdim});
  xt::xtensor<double, 3> J = xt::zeros<double>({x.shape(0), gdim, tdim});
  xt::xtensor<double, 3> K = xt::zeros<double>({x.shape(0), tdim, gdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({x.shape(0)});
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));

  xt::xtensor<double, 2> _Xp({1, tdim});
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell_index);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x(p, j);

    auto _J = xt::view(J, p, xt::all(), xt::all());
    auto _K = xt::view(K, p, xt::all(), xt::all());

    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine())
    {
      pull_back_affine(_Xp, coordinate_dofs, _J, _K, xp);
      detJ[p]
          = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
    }
    else
    {
      cmap.pull_back_nonaffine(_Xp, xp, coordinate_dofs);
      cmap.tabulate(1, _Xp, phi);
      dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      detJ[p]
          = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
    }
    xt::row(X, p) = xt::row(_Xp, 0);
  }

  // Prepare basis function data structures
  xt::xtensor<double, 4> basis_reference_values(
      {1, x.shape(0), space_dimension, reference_value_size});

  // Compute basis on reference element
  element->tabulate(basis_reference_values, X, 0);

  using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t
      = xt::xview<decltype(basis_reference_values)&, std::size_t, std::size_t,
                  xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
  const std::function<void(const xtl::span<double>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();
  const std::size_t num_basis_values = space_dimension * reference_value_size;
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    apply_dof_transformation(
        xtl::span(basis_reference_values.data() + p * num_basis_values,
                  num_basis_values),
        cell_info, cell_index, (int)reference_value_size);

    // Push basis forward to physical element
    auto _K = xt::view(K, p, xt::all(), xt::all());
    auto _J = xt::view(J, p, xt::all(), xt::all());
    auto _u = xt::view(basis_values, p, xt::all(), xt::all());
    auto _U = xt::view(basis_reference_values, (std::size_t)0, p, xt::all(),
                       xt::all());
    push_forward_fn(_u, _U, _J, detJ[p], _K);
  }
  return basis_values;
}

//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
dolfinx_mpc::tabulate_dof_coordinates(const dolfinx::fem::FunctionSpace& V,
                                      tcb::span<const std::int32_t> dofs,
                                      tcb::span<const std::int32_t> cells)
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
  const std::size_t scalar_dofs
      = element->space_dimension() / element_block_size;

  // Get the dof coordinates on the reference element
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  const xt::xtensor<double, 2>& X = element->interpolation_points();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  xtl::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = x_dofmap.num_links(0);

  // Array to hold coordinates to return
  const std::size_t shape_c0 = 3;
  const std::size_t shape_c1 = dofs.size();
  xt::xtensor<double, 2> coords = xt::zeros<double>({shape_c0, shape_c1});

  // Loop over cells and tabulate dofs
  xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  const std::function<void(const xtl::span<double>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(cells[c]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate dof coordinates on cell
    dolfinx::fem::CoordinateElement::push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(xtl::span(x.data(), x.size()),
                             xtl::span(cell_info.data(), cell_info.size()),
                             (std::int32_t)c, (int)x.shape(1));

    // Get cell dofmap
    auto cell_dofs = dofmap->cell_dofs(cells[c]);
    auto it = std::find(cell_dofs.begin(), cell_dofs.end(), dofs[c]);
    auto loc = std::distance(cell_dofs.begin(), it);

    // Copy dof coordinates into vector
    for (std::size_t j = 0; j < gdim; ++j)
      coords(j, c) = x(loc, j);
  }

  return coords;
}

//-----------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_mpc::compute_colliding_cells(
    const dolfinx::mesh::Mesh& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    const xt::xtensor<double, 2>& points, const double eps2)
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
    xt::xtensor<double, 2> _point({cells.size(), 3});
    for (std::size_t j = 0; j < cells.size(); j++)
      xt::row(_point, j) = xt::row(points, i);

    xt::xtensor<double, 1> distances_sq
        = dolfinx::geometry::squared_distance(mesh, tdim, cells, _point);
    // Only push back closest cell
    if (std::size_t cell_idx = xt::argmin(distances_sq)();
        distances_sq[cell_idx] < eps2)
    {
      colliding_cells.push_back(cells[cell_idx]);
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
    const xt::xtensor<double, 2>& points, const double eps2)
{
  assert(points.shape(1) == 3);

  // Compute collisions for each point with BoundingBoxTree
  auto bbox_collisions = dolfinx::geometry::compute_collisions(tree, points);

  // Compute exact collision
  auto cell_collisions = dolfinx_mpc::compute_colliding_cells(
      mesh, bbox_collisions, points, eps2);

  // Extract first collision
  std::vector<std::int32_t> collisions(points.shape(0), -1);
  for (int i = 0; i < cell_collisions.num_nodes(); i++)
  {
    auto local_cells = cell_collisions.links(i);
    if (!local_cells.empty())
      collisions[i] = local_cells[0];
  }
  return collisions;
}

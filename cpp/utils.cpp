// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
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
    std::vector<int> source_weights(source_edges.size(), 1);
    std::vector<int> dest_weights(dest_edges.size(), 1);

    MPI_Dist_graph_create_adjacent(
        comm, source_edges.size(), source_edges.data(), source_weights.data(),
        dest_edges.size(), dest_edges.data(), dest_weights.data(),
        MPI_INFO_NULL, false, &comms[1]);
  }
  return comms;
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::create_contact_condition(
    std::shared_ptr<dolfinx::function::FunctionSpace> V,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t slave_marker,
    std::int32_t master_marker,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> nh)
{
  // Create slave->master facets and master->slave facets neihborhood comms
  std::array<MPI_Comm, 2> neighborhood_comms
      = create_neighborhood_comms(meshtags, slave_marker, master_marker);

  // Extract some const information from function-space
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

  // std::vector<std::shared_ptr<dolfinx::function::FunctionSpace>> V_sub(tdim);
  for (std::int32_t i = 0; i < tdim; ++i)
  {
    std::pair<std::shared_ptr<dolfinx::function::FunctionSpace>,
              std::vector<std::int32_t>>
        Vi_pair = V->sub({i})->collapse();
    auto [Vi, map] = Vi_pair;
    // V_sub[i] = Vi;

    dofs_on_slave_facet[i] = dolfinx::fem::locate_dofs_topological(
        {*V->sub({i}).get(), *Vi.get()}, fdim, slave_facets);
  }

  // Sort dofs on slave facets from each topological dimension by physical
  // coordinates
  Eigen::Array<double, Eigen::Dynamic, 3> coordinates
      = V->tabulate_dof_coordinates();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 3> index_pairs(
      dofs_on_slave_facet[0].rows(), 3);
  for (std::int32_t i = 0; i < dofs_on_slave_facet[0].rows(); ++i)
  {
    index_pairs(i, 0) = dofs_on_slave_facet[0].row(i)[0];
    const Eigen::Array<double, Eigen::Dynamic, 3> coordinate
        = coordinates.row(dofs_on_slave_facet[0].row(i)[0]);
    for (std::int32_t d = 1; d < tdim; ++d)
    {
      for (std::int32_t j = 0; j < dofs_on_slave_facet[d].rows(); ++j)
      {
        const Eigen::Array<double, Eigen::Dynamic, 3> other_coordinate
            = coordinates.row(dofs_on_slave_facet[d].row(j)[0]);
        if (coordinate.isApprox(other_coordinate))
        {
          index_pairs(i, d) = dofs_on_slave_facet[d].row(j)[0];
          break;
        }
      }
    }
  }

  // Arrays for holding local and ghost data of slave and master coeffs
  std::vector<std::int32_t> local_slaves;
  std::vector<std::int32_t> ghost_slaves;
  std::vector<std::int32_t> local_masters;
  std::vector<std::int32_t> ghost_masters;
  std::vector<PetscScalar> local_coeffs;
  std::vector<PetscScalar> ghost_coeffs;
  std::vector<std::int32_t> local_master_offset = {0};
  std::vector<std::int32_t> ghost_master_offset = {0};

  // Determine which dof should be a slave dof (based on max component of normal
  // vector normal vector)
  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> normal_array
      = nh->x()->array();
  std::vector<std::int32_t> local_slaves_to_index;
  for (std::int32_t i = 0; i < index_pairs.rows(); ++i)
  {
    auto dofs = index_pairs.row(i);
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

  std::cout << "\n Rank " << rank << " " << local_slaves.size() << " "
            << ghost_slaves.size() << "\n";
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
  std::map<std::int32_t, std::vector<std::int32_t>> slave_index_to_master_dof;
  std::map<std::int32_t, std::vector<PetscScalar>> slave_index_to_master_coeff;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owners
      = V->dofmap()->index_map->ghost_owner_rank();
  for (std::int32_t i = 0; i < local_slaves.size(); ++i)
  {
    auto coord = coordinates.row(local_slaves[i]);
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
        // Get coefficients from slave facet
        Eigen::Array<PetscScalar, 1, Eigen::Dynamic> coeffs(1, tdim);
        auto slave_facet_indices = index_pairs.row(local_slaves_to_index[i]);
        for (std::int32_t j = 0; j < tdim; ++j)
          coeffs[j] = normal_array[slave_facet_indices[j]];
        // Compute local contribution of slave on master facet
        Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> basis_values
            = get_basis_functions(V, coord, verified_candidates[0]);
        auto cell_dofs = V->dofmap()->cell_dofs(verified_candidates[0]);
        std::vector<std::int32_t> l_master;
        std::vector<PetscScalar> l_coeff;
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
            }
          }
        }
        slave_index_to_master_coeff.insert(std::pair(i, l_coeff));
        slave_index_to_master_dof.insert(std::pair(i, l_master));
      }
    }
  }
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);

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
      std::vector<std::int32_t> coeffs_i(
          local_coeffs.begin() + local_master_offset[i],
          local_coeffs.begin() + local_master_offset[i + 1]);
      coeffs_out.insert(coeffs_out.end(), coeffs_i.begin(), coeffs_i.end());
      std::vector<std::int32_t> other_masters = slave_index_to_master_dof[i];
      std::vector<std::int32_t> other_coeffs = slave_index_to_master_dof[i];
      masters_out.insert(masters_out.end(), other_masters.begin(),
                         other_masters.end());
      coeffs_out.insert(coeffs_out.end(), other_coeffs.begin(),
                        other_coeffs.end());
      offsets_out.push_back(masters_out.size());
    }
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners(masters_out.size());
    owners.fill(1);

    // Map vectors to Eigen arrays

    Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> masters(
        masters_out.data(), masters_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> slaves(
        local_slaves.data(), local_slaves.size());
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coeffs(
        coeffs_out.data(), coeffs_out.size());
    Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> offsets(
        offsets_out.data(), offsets_out.size());
    Eigen::Array<std::int64_t, 0, 1> gm;
    Eigen::Array<std::int32_t, 0, 1> gs;
    Eigen::Array<PetscScalar, 0, 1> gc;
    Eigen::Array<std::int64_t, 1, 1> go(1, 1);
    go.fill(0);
  }

  // const auto [src_ranks, dest_ranks] = dolfinx::MPI::neighbors(comms[0]);
}
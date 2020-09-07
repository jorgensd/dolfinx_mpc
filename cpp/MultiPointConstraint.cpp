// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>
using namespace dolfinx_mpc;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves,
    std::int32_t num_local_slaves)
    : _V(V), _slaves(), _slave_cells(), _cell_to_slaves_map(),
      _slave_to_cells_map(), _num_local_slaves(num_local_slaves), _master_map(),
      _coeff_map(), _owner_map(), _master_local_map(), _master_block_map(),
      _dofmap()
{
  _slaves
      = std::make_shared<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(slaves);
  auto [slave_to_cells, cell_to_slaves_map] = create_cell_maps(slaves);
  auto [unique_cells, cell_to_slaves] = cell_to_slaves_map;
  _slave_cells
      = std::make_shared<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
          unique_cells);
  _cell_to_slaves_map = cell_to_slaves;
  _slave_to_cells_map = slave_to_cells;
}

std::pair<
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
    std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
MultiPointConstraint::create_cell_maps(
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs)
{
  dolfinx::common::Timer timer("~MPC: Create slave->cell  and cell->slave map");
  const dolfinx::mesh::Mesh& mesh = *(_V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  // Create an array for the whole function space, where
  // dof index contains the index of the dof if it is in the list,
  // otherwise -1
  std::vector<std::int64_t> dof_index(_V->dim(), -1);
  std::map<std::int32_t, std::vector<std::int32_t>> idx_to_cell;
  for (std::int32_t i = 0; i < dofs.rows(); ++i)
  {
    std::vector<std::int32_t> cell_vec;
    idx_to_cell.insert({i, cell_vec});
    dof_index[dofs[i]] = i;
  }
  // Loop through all cells and make cell to dof index and index to cell map
  std::vector<std::int32_t> unique_cells;
  std::map<std::int32_t, std::vector<std::int32_t>> cell_to_idx;
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    auto cell_dofs = dofmap.cell_dofs(i);
    bool any_dof_in_cell = false;
    std::vector<std::int32_t> dofs_in_cell;
    for (std::int32_t j = 0; j < cell_dofs.size(); ++j)
    {
      // Check if cell dof in dofs list
      if (dof_index[cell_dofs[j]] != -1)
      {
        any_dof_in_cell = true;
        dofs_in_cell.push_back(dof_index[cell_dofs[j]]);
        idx_to_cell[dof_index[cell_dofs[j]]].push_back(i);
      }
    }
    if (any_dof_in_cell)
    {
      unique_cells.push_back(i);
      cell_to_idx.insert({i, dofs_in_cell});
    }
  }
  // Flatten unstructured maps
  std::vector<std::int32_t> indices_to_cells;
  std::vector<std::int32_t> cells_to_indices;
  std::vector<std::int32_t> offsets_cells{0};
  std::vector<std::int32_t> offsets_idx{0};
  for (auto it = cell_to_idx.begin(); it != cell_to_idx.end(); ++it)
  {
    auto idx = it->second;
    cells_to_indices.insert(cells_to_indices.end(), idx.begin(), idx.end());
    offsets_cells.push_back(cells_to_indices.size());
  }
  for (auto it = idx_to_cell.begin(); it != idx_to_cell.end(); ++it)
  {
    auto idx = it->second;
    indices_to_cells.insert(indices_to_cells.end(), idx.begin(), idx.end());
    offsets_idx.push_back(cells_to_indices.size());
  }

  // Create dof to cells adjacency list
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj_ptr
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          indices_to_cells, offsets_idx);

  // Create cell to dofs adjacency list
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> dof_cells(
      unique_cells.data(), unique_cells.size());
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj2_ptr;
  adj2_ptr = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      cells_to_indices, offsets_cells);

  return std::make_pair(adj_ptr, std::make_pair(dof_cells, adj2_ptr));
}
//-----------------------------------------------------------------------------
void MultiPointConstraint::add_masters(
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters,
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeffs,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> owners,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets)
{
  _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
      masters, offsets);
  _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<PetscScalar>>(
      coeffs, offsets);
  _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      owners, offsets);

  // Create new index map with all masters
  create_new_index_map();
}
//-----------------------------------------------------------------------------
void MultiPointConstraint::create_new_index_map()
{
  dolfinx::common::Timer timer("~MPC: Create new index map");
  MPI_Comm comm = _V->mesh()->mpi_comm();
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());

  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;

  // Compute local master index before creating new index-map
  const std::int32_t& block_size = _V->dofmap()->index_map->block_size();
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> master_array
      = _master_map->array();
  master_array /= block_size;
  const std::vector<std::int32_t> master_as_local
      = _V->dofmap()->index_map->global_to_local(master_array);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> masters_block(
      master_as_local.data(), master_as_local.size(), 1);
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      old_master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_block, _master_map->offsets());
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1)
  {
    _index_map = index_map;
    _master_block_map = old_master_block_map;
  }
  else
  {
    std::vector<std::int64_t> new_ghosts;
    std::vector<std::int32_t> new_ranks;

    for (Eigen::Index i = 0; i < old_master_block_map->num_nodes(); ++i)
    {
      for (Eigen::Index j = 0; j < old_master_block_map->links(i).size(); ++j)
      {
        // Check if master is already ghosted
        if (old_master_block_map->links(i)[j] == -1)
        {
          const int& master_as_int = _master_map->links(i)[j];
          const std::div_t div = std::div(master_as_int, block_size);
          const int& block = div.quot;
          auto already_ghosted
              = std::find(new_ghosts.begin(), new_ghosts.end(), block);

          if (already_ghosted == new_ghosts.end())
          {
            new_ghosts.push_back(block);
            new_ranks.push_back(_owner_map->links(i)[j]);
          }
        }
      }
    }
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& old_ranks
        = index_map->ghost_owner_rank();
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> old_ghosts
        = index_map->ghosts();
    std::int32_t num_ghosts = old_ghosts.size();
    old_ghosts.conservativeResize(num_ghosts + new_ghosts.size());
    for (std::int64_t i = 0; i < new_ghosts.size(); i++)
      old_ghosts(num_ghosts + i) = new_ghosts[i];

    // Append rank of new ghost owners
    std::vector<int> ghost_ranks(old_ghosts.size());
    for (std::int64_t i = 0; i < old_ranks.size(); i++)
      ghost_ranks[i] = old_ranks[i];

    for (int i = 0; i < new_ghosts.size(); ++i)
    {
      ghost_ranks[num_ghosts + i] = new_ranks[i];
    }
    dolfinx::common::Timer timer_indexmap("MPC: Init indexmap");

    _index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, index_map->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(ghost_ranks.begin(), ghost_ranks.end())),
        old_ghosts, ghost_ranks, block_size);

    timer_indexmap.stop();
  }
  // Create new dofmap
  const dolfinx::fem::DofMap* old_dofmap = _V->dofmap().get();
  const int& bs = old_dofmap->element_dof_layout->block_size();
  dolfinx::mesh::Topology topology = _V->mesh()->topology();
  dolfinx::fem::ElementDofLayout layout = *old_dofmap->element_dof_layout;
  auto [unused_indexmap, o_dofmap] = dolfinx::fem::DofMapBuilder::build(
      _V->mesh()->mpi_comm(), topology, layout);
  _dofmap = std::make_shared<dolfinx::fem::DofMap>(
      old_dofmap->element_dof_layout, _index_map, o_dofmap);

  // Compute local master index before creating new index-map
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> master_blocks
      = _master_map->array();
  master_blocks /= block_size;
  const std::vector<std::int32_t>& master_block_local
      = _index_map->global_to_local(master_blocks);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
      masters_block_eigen(master_block_local.data(), master_block_local.size(),
                          1);
  _master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_block_eigen, _master_map->offsets());

  // Go from blocks to actual local index
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> masters_local(
      master_block_local.size());
  std::int32_t c = 0;
  for (Eigen::Index i = 0; i < _master_map->num_nodes(); ++i)
  {
    auto masters = _master_map->links(i);
    for (Eigen::Index j = 0; j < masters.size(); ++j)
    {
      const int master_as_int = masters[j];
      const std::div_t div = std::div(master_as_int, block_size);
      const int rem = div.rem;
      masters_local(c) = masters_block_eigen(c) * block_size + rem;
      c++;
    }
  }
  _master_local_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_local, _master_map->offsets());
}
//-----------------------------------------------------------------------------
/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern MultiPointConstraint::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (MPC)");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  /// Check that we are using the correct function-space in the bilinear
  /// form otherwise the index map will be wrong
  assert(a.function_space(0) == _V);
  assert(a.function_space(1) == _V);

  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  const int& block_size = _index_map->block_size();

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = _index_map;
  new_maps[1] = _index_map;
  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);

  /// Arrays replacing slave dof with master dof in sparsity pattern
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_slave(block_size);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_other_slave(block_size);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_on_cell(block_size);

  // Add non-zeros for each slave cell to sparsity pattern.
  // For the i-th cell with a slave, all local entries has to be from the
  // j-th slave to the k-th master degree of freedom
  for (Eigen::Index i = 0; i < _cell_to_slaves_map->num_nodes(); ++i)
  {
    // Find index for slave in test and trial space
    auto cell_dofs = _dofmap->cell_dofs(_slave_cells->array()[i]);
    auto slaves = _cell_to_slaves_map->links(i);
    // Loop over slaves in cell
    for (Eigen::Index j = 0; j < slaves.size(); ++j)
    {
      auto masters = _master_local_map->links(slaves[j]);
      auto blocks = _master_block_map->links(slaves[j]);
      // Insert pattern for each master
      for (Eigen::Index k = 0; k < masters.size(); ++k)
      {
        // Find dofs for full master block
        const std::int32_t block = blocks[k];
        for (std::size_t comp = 0; comp < block_size; ++comp)
          master_for_slave(comp) = block_size * block + comp;
        // Add all values on cell (including slave), to get complete blocks
        pattern.insert(master_for_slave, cell_dofs);
        pattern.insert(cell_dofs, master_for_slave);

        // Add pattern for master owned by other slave on same cell
        for (Eigen::Index k = j + 1; k < slaves.size(); ++k)
        {
          auto other_masters = _master_local_map->links(slaves[k]);
          auto other_blocks = _master_block_map->links(slaves[k]);
          for (Eigen::Index l = 0; l < other_masters.size(); ++l)
          {
            const int other_block = other_blocks[l];
            for (std::size_t comp = 0; comp < block_size; ++comp)
              master_for_other_slave(comp) = block_size * other_block + comp;

            pattern.insert(master_for_slave, master_for_other_slave);
            pattern.insert(master_for_other_slave, master_for_slave);
          }
        }
      }
    }
  }
  // Add pattern for all local masters for the same slave
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> local_master_dof(block_size);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_dof(block_size);

  for (std::int64_t i = 0; i < _master_local_map->num_nodes(); ++i)
  {
    auto master_block = _master_block_map->links(i);
    for (std::int64_t j = 0; j < master_block.size(); ++j)
    {
      const int& block = master_block[j];
      for (std::int64_t comp = 0; comp < block_size; comp++)
        local_master_dof[comp] = block_size * block + comp;
      for (std::int64_t k = j + 1; k < master_block.size(); ++k)
      {
        // Map other master to local dof and add remainder of block
        const int& other_block = master_block[k];

        for (std::int64_t comp = 0; comp < block_size; comp++)
          other_master_dof[comp] = block_size * other_block + comp;

        pattern.insert(local_master_dof, other_master_dof);
        pattern.insert(other_master_dof, local_master_dof);
      }
    }
  }
  return pattern;
};
//-----------------------------------------------------------------------------
/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern MultiPointConstraint::create_sparsity_pattern_new(
    const dolfinx::fem::Form<PetscScalar>& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (MPC) (NEW!)");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  /// Check that we are using the correct function-space in the bilinear
  /// form otherwise the index map will be wrong
  assert(a.function_space(0) == _V);
  assert(a.function_space(1) == _V);
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = _index_map;
  new_maps[1] = _index_map;
  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);

  // Arrays replacing slave dof with master dof in sparsity pattern
  const int& block_size = _index_map->block_size();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_slave(block_size);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_other_slave(block_size);

  for (Eigen::Index i = 0; i < _cell_to_slaves_map->num_nodes(); ++i)
  {

    auto cell_dofs = _dofmap->cell_dofs(_slave_cells->array()[i]);
    auto slaves = _cell_to_slaves_map->links(i);

    // Arrays for flattened master slave data
    std::vector<std::int32_t> flattened_masters;
    for (std::int32_t j = 0; j < slaves.size(); ++j)
    {
      auto local_masters = _master_block_map->links(slaves[j]);
      for (std::int32_t k = 0; k < local_masters.size(); ++k)
        flattened_masters.push_back(local_masters[k]);
    }
    // Remove duplicate master blocks
    std::sort(flattened_masters.begin(), flattened_masters.end());
    flattened_masters.erase(
        std::unique(flattened_masters.begin(), flattened_masters.end()),
        flattened_masters.end());
    for (std::int32_t j = 0; j < flattened_masters.size(); ++j)
    {
      const std::int32_t& master_block = flattened_masters[j];
      // Add all dofs of the slave cell to the master blocks sparsity pattern
      for (std::size_t comp = 0; comp < block_size; ++comp)
        master_for_slave(comp) = block_size * master_block + comp;

      pattern.insert(master_for_slave, cell_dofs);
      pattern.insert(cell_dofs, master_for_slave);
      // Add sparsity pattern for all master dofs of any slave on this cell
      for (std::int32_t k = j + 1; k < flattened_masters.size(); ++k)
      {
        const std::int32_t& other_master_block = flattened_masters[k];
        for (std::size_t comp = 0; comp < block_size; ++comp)
          master_for_other_slave(comp) = block_size * other_master_block + comp;

        pattern.insert(master_for_slave, master_for_other_slave);
        pattern.insert(master_for_other_slave, master_for_slave);
      }
    }
  }
  return pattern;
};

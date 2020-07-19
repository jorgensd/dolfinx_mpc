// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ContactConstraint.h"
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

ContactConstraint::ContactConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves,
    std::int32_t num_local_slaves)
    : _V(V), _slaves(slaves), _slave_cells(), _cell_to_slaves_map(),
      _slave_to_cells_map(), _num_local_slaves(num_local_slaves), _master_map(),
      _coeff_map(), _owner_map(), _master_local_map(), _master_block_map(),
      _dofmap()
{
  auto [slave_to_cells, cell_to_slaves_map] = create_cell_maps(slaves);
  auto [unique_cells, cell_to_slaves] = cell_to_slaves_map;
  _slave_cells = unique_cells;
  _cell_to_slaves_map = cell_to_slaves;
  _slave_to_cells_map = slave_to_cells;
}

std::pair<
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
    std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
ContactConstraint::create_cell_maps(
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs)
{
  dolfinx::common::Timer timer("MPC: Create slave->cell  and cell->slave map");
  const dolfinx::mesh::Mesh& mesh = *(_V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  std::vector<std::int32_t> dofs_to_cell;
  std::vector<std::int32_t> offsets{0};
  std::map<std::int32_t, std::vector<std::int32_t>> cell_to_dof;

  // Loop over all dofs
  for (std::int32_t i = 0; i < dofs.rows(); ++i)
  {
    bool was_added = false;
    // Loop over all cells
    for (std::int32_t j = 0; j < num_cells; ++j)
    {
      auto cell_dofs = dofmap.cell_dofs(j);
      if ((cell_dofs.array() == dofs[i]).any())
      {
        was_added = true;
        dofs_to_cell.push_back(j);
        // Check if cell allready added to unstructured map
        auto vec = cell_to_dof.find(j);
        if (vec == cell_to_dof.end())
        {
          std::vector<std::int32_t> dof_vec{dofs[i]};
          cell_to_dof.insert({j, dof_vec});
        }
        else
          vec->second.push_back(dofs[i]);
      }
    }
    // Add offsets for AdjacencyList
    if (was_added)
    {
      offsets.push_back(dofs_to_cell.size());
    }
  }
  // Create dof to cells adjacency list
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj_ptr
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          dofs_to_cell, offsets);
  std::vector<std::int32_t> cells_to_dof;
  std::vector<std::int32_t> offsets_2{0};
  std::vector<std::int32_t> unique_cells;

  // Flatten unstructured map to array
  for (auto it = cell_to_dof.begin(); it != cell_to_dof.end(); ++it)
  {
    unique_cells.push_back(it->first);
    auto vec = it->second;
    cells_to_dof.insert(cells_to_dof.end(), vec.begin(), vec.end());
    offsets_2.push_back(cells_to_dof.size());
  }

  // Create cell to dofs adjacency list
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> dof_cells(
      unique_cells.data(), unique_cells.size());
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj2_ptr;
  adj2_ptr = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      cells_to_dof, offsets_2);

  return std::make_pair(adj_ptr, std::make_pair(dof_cells, adj2_ptr));
}

void ContactConstraint::add_masters(
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
  // Compute local master index before creating new index-map
  std::int32_t block_size = _V->dofmap()->index_map->block_size();
  masters /= block_size;
  std::vector<std::int32_t> master_as_local
      = _V->dofmap()->index_map->global_to_local(masters);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> masters_block(
      master_as_local.data(), master_as_local.size(), 1);
  _master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_block, offsets);
  // Create new index map with all masters
  create_new_index_map();
}

void ContactConstraint::create_new_index_map()
{
  dolfinx::common::Timer timer("MPC: Create new index map");
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());

  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> old_ghosts
      = index_map->ghosts();

  std::vector<std::int64_t> new_ghosts;
  std::vector<std::int32_t> new_ranks;

  int block_size = index_map->block_size();
  for (Eigen::Index i = 0; i < _master_block_map->num_nodes(); ++i)
  {
    for (Eigen::Index j = 0; j < _master_block_map->links(i).size(); ++j)
    {
      // Check if master is already ghosted
      if (_master_block_map->links(i)[j] == -1)
      {
        const int master_as_int = _master_map->links(i)[j];
        const std::div_t div = std::div(master_as_int, block_size);
        const int block = div.quot;
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
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> old_ranks
      = index_map->ghost_owner_rank();
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
  MPI_Comm comm = _V->mesh()->mpi_comm();

  _index_map = std::make_shared<dolfinx::common::IndexMap>(
      comm, index_map->size_local(),
      dolfinx::MPI::compute_graph_edges(
          MPI_COMM_WORLD,
          std::set<int>(ghost_ranks.begin(), ghost_ranks.end())),
      old_ghosts, ghost_ranks, block_size);
  timer_indexmap.stop();

  dolfinx::common::Timer timer_glob_to_loc("MPC: Update local masters");
  // Compute local master index before creating new index-map
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> master_blocks
      = _master_map->array();
  master_blocks /= block_size;
  std::vector<std::int32_t> master_block_local
      = _index_map->global_to_local(master_blocks);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
      masters_block_eigen(master_block_local.data(), master_block_local.size(),
                          1);
  _master_block_map = _master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_block_eigen, _master_map->offsets());

  // Go from blocks to actual local index
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> masters_local(
      master_block_local.size());
  std::int32_t c = 0;
  for (Eigen::Index i = 0; i < _master_block_map->num_nodes(); ++i)
  {
    for (Eigen::Index j = 0; j < _master_block_map->links(i).size(); ++j)
    {
      const int master_as_int = _master_map->links(i)[j];
      const std::div_t div = std::div(master_as_int, block_size);
      const int rem = div.rem;
      masters_local(c) = masters_block_eigen(c) * block_size + rem;
      c++;
    }
  }
  _master_local_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_local, _master_map->offsets());
  timer_glob_to_loc.stop();
}

/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern ContactConstraint::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("MPC: Sparsitypattern Total");
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

  int block_size = _index_map->block_size();
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts = _index_map->ghosts();
  const std::vector<std::int64_t> global_indices
      = _index_map->global_indices(false);

  /// Create new dofmap using the MPC index-maps
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = _index_map;
  new_maps[1] = _index_map;
  const dolfinx::fem::DofMap* old_dofmap = a.function_space(0)->dofmap().get();

  /// Get AdjacencyList for old dofmap
  const int bs = old_dofmap->element_dof_layout->block_size();

  dolfinx::mesh::Topology topology = mesh.topology();
  dolfinx::fem::ElementDofLayout layout = *old_dofmap->element_dof_layout;
  if (bs != 1)
  {
    layout = *old_dofmap->element_dof_layout->sub_dofmap({0});
  }
  auto [unused_indexmap, o_dofmap] = dolfinx::fem::DofMapBuilder::build(
      mesh.mpi_comm(), topology, layout, bs);
  _dofmap = std::make_shared<dolfinx::fem::DofMap>(
      old_dofmap->element_dof_layout, _index_map, o_dofmap);

  std::array<std::int64_t, 2> local_range = _dofmap->index_map->local_range();
  std::int64_t local_size = local_range[1] - local_range[0];
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{_dofmap.get(), _dofmap.get()}};

  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);

  /// Arrays replacing slave dof with master dof in sparsity pattern
  std::vector<Eigen::Array<PetscInt, Eigen::Dynamic, 1>> master_for_slave(2);
  master_for_slave[0].resize(block_size);
  master_for_slave[1].resize(block_size);

  std::vector<Eigen::Array<PetscInt, Eigen::Dynamic, 1>> master_for_other_slave(
      2);
  master_for_other_slave[0].resize(block_size);
  master_for_other_slave[1].resize(block_size);

  std::vector<Eigen::Array<PetscInt, Eigen::Dynamic, 1>> other_master_on_cell(
      2);
  other_master_on_cell[0].resize(block_size);
  other_master_on_cell[1].resize(block_size);

  // Add non-zeros for each slave cell to sparsity pattern.
  // For the i-th cell with a slave, all local entries has to be from the
  // j-th slave to the k-th master degree of freedom
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    // Find index for slave in test and trial space
    std::vector<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_dof_lists(
        2);
    for (std::size_t l = 0; l < 2; l++)
      cell_dof_lists[l] = dofmaps[l]->cell_dofs(_slave_cells[i]);

    // Loop over slaves in cell
    for (Eigen::Index j = 0; j < _cell_to_slaves_map->links(i).size(); j++)
    {
      // Insert pattern for each master
      for (Eigen::Index k = 0;
           k
           < _master_local_map->links(_cell_to_slaves_map->links(i)[j]).size();
           k++)
      {
        // Loop over test and trial space
        for (std::size_t l = 0; l < 2; l++)
        {
          // Find dofs for full master block
          std::int32_t block
              = _master_block_map->links(_cell_to_slaves_map->links(i)[j])[k];
          for (std::size_t comp = 0; comp < block_size; comp++)
            master_for_slave[l](comp) = block_size * block + comp;
        }
        // Add all values on cell (including slave), to get complete blocks
        pattern.insert(master_for_slave[0], cell_dof_lists[1]);
        pattern.insert(cell_dof_lists[0], master_for_slave[1]);

        // Add pattern for master owned by other slave on same cell
        for (Eigen::Index k = j + 1; k < _cell_to_slaves_map->links(i).size();
             k++)
        {
          for (Eigen::Index l = 0;
               l < _master_local_map->links(_cell_to_slaves_map->links(i)[k])
                       .size();
               l++)
          {
            const int other_block
                = _master_block_map->links(_cell_to_slaves_map->links(i)[k])[l];
            for (std::size_t m = 0; m < 2; m++)
            {

              for (std::size_t comp = 0; comp < block_size; comp++)
                master_for_other_slave[m](comp)
                    = block_size * other_block + comp;
            }
            pattern.insert(master_for_slave[0], master_for_other_slave[1]);
            pattern.insert(master_for_other_slave[0], master_for_slave[1]);
          }
        }
      }
    }
  }
  // Add pattern for all local masters for the same slave
  for (std::int64_t i = 0; i < _master_local_map->num_nodes(); i++)
  {
    for (std::int64_t j = 0; j < _master_local_map->links(i).size(); j++)
    {
      const int block = _master_block_map->links(i)[j];
      Eigen::Array<PetscInt, Eigen::Dynamic, 1> local_master_dof(block_size);
      for (std::int64_t comp = 0; comp < block_size; comp++)
        local_master_dof[comp] = block_size * block + comp;
      for (std::int64_t k = j + 1; k < _master_local_map->links(i).size(); k++)
      {
        // Map other master to local dof and add remainder of block
        const int other_block = _master_block_map->links(i)[k];
        Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_dof(block_size);

        for (std::int64_t comp = 0; comp < block_size; comp++)
          other_master_dof[comp] = block_size * other_block + comp;

        pattern.insert(local_master_dof, other_master_dof);
        pattern.insert(other_master_dof, local_master_dof);
      }
    }
  }
  return pattern;
};
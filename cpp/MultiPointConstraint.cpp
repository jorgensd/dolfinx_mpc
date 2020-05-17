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
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FormIntegrals.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
using namespace dolfinx_mpc;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters,
    Eigen::Array<double, Eigen::Dynamic, 1> coefficients,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets_master)
    : _function_space(V), _index_map(), _mpc_dofmap(), _slaves(slaves),
      _masters(), _masters_local(), _slaves_local(),
      _coefficients(coefficients), _cells_to_dofs(2, 1), _cell_to_slave_index(),
      _slave_cells(), _master_cells()
{
  LOG(INFO) << "Initializing MPC class";
  dolfinx::common::Timer timer("MPC: Initialize MultiPointConstraint class");
  _masters = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
      masters, offsets_master);
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> dof_lists(2);
  dof_lists[0] = slaves;
  dof_lists[1] = masters;
  /// Locate all cells containing slaves and masters locally and create a
  /// cell_to_slave/master map
  auto cell_info = dolfinx_mpc::locate_cells_with_dofs(V, dof_lists);
  auto [q, cell_to_slave] = cell_info[0];

  _cells_to_dofs(0) = cell_to_slave;
  _slave_cells = q;

  auto [t, cell_to_master] = cell_info[1];
  _cells_to_dofs(1) = cell_to_master;
  _master_cells = t;

  /// Create reuseable map from slave_cells to the corresponding slave_index
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slave_cell_indices(
      _cells_to_dofs[0]->array().size());
  for (Eigen::Index i = 0; i < _cells_to_dofs[0]->array().size(); i++)
  {
    for (std::uint64_t counter = 0; counter < _slaves.size(); counter++)
    {
      if (_slaves[counter] == _cells_to_dofs[0]->array()[i])
      {
        slave_cell_indices(i) = counter;
        break;
      }
    }
  }
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_cell_offsets
      = _cells_to_dofs[0]->offsets();
  _cell_to_slave_index
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
          slave_cell_indices, slave_cell_offsets);

  /// Generate MPC specific index map
  _index_map = generate_index_map();

  /// Find all local indices for master
  std::vector<std::int64_t> master_vec(masters.data(),
                                       masters.data() + masters.size());
  std::vector<std::int32_t> masters_local
      = _index_map->global_to_local(master_vec, false);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> masters_eigen(
      masters_local.data(), masters_local.size(), 1);
  _masters_local
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_eigen, offsets_master);
  /// Find all local indices for slaves
  std::vector<std::int64_t> slaves_vec(slaves.data(),
                                       slaves.data() + slaves.size());
  std::vector<std::int32_t> slaves_local
      = _index_map->global_to_local(slaves_vec, false);
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
      __slaves_local(slaves_local.data(), slaves_local.size(), 1);
  _slaves_local = __slaves_local;
}

// /// Generate MPC specific index map with the correct ghosting
std::shared_ptr<dolfinx::common::IndexMap>
MultiPointConstraint::generate_index_map()
{
  LOG(INFO) << "Generating MPC index map with additional ghosts";
  dolfinx::common::Timer timer("MPC: Generate index map with correct ghosting");
  const dolfinx::mesh::Mesh& mesh = *(_function_space->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_function_space->dofmap());

  // Get common::IndexMaps for each dimension
  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;
  std::array<std::int64_t, 2> local_range = index_map->local_range();

  int block_size = index_map->block_size();

  // Get old dofs that will be appended to
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> new_ghosts
      = index_map->ghosts();

  std::vector<std::int64_t> additional_ghosts;

  int num_ghosts = new_ghosts.size();
  // Offsets for slave and master
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_offsets
      = _cells_to_dofs[0]->offsets();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> master_cell_offsets
      = _cells_to_dofs[1]->offsets();
  dolfinx::common::Timer timer2(
      "MPC: SLAVE_MASTER index map with correct ghosting");
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    // Loop over slaves in cell
    for (Eigen::Index j = 0; j < slave_offsets[i + 1] - slave_offsets[i]; j++)
    {
      // // Loop over all masters for given slave
      for (Eigen::Index k = 0;
           k < _masters->links(_cell_to_slave_index->links(i)[j]).size(); k++)
      {
        // Check if master is already owned by the processor by looping over
        if (!((block_size * local_range[0]
               <= _masters->links(_cell_to_slave_index->links(i)[j])[k])
              && (_masters->links(_cell_to_slave_index->links(i)[j])[k]
                  < block_size * local_range[1])))
        {
          const int master_as_int
              = _masters->links(_cell_to_slave_index->links(i)[j])[k];
          const std::div_t div = std::div(master_as_int, block_size);
          const int index = div.quot;
          std::int32_t is_already_ghost = (new_ghosts == index).count();
          auto it = std::find(additional_ghosts.begin(),
                              additional_ghosts.end(), index);
          if ((is_already_ghost == 0) && (it == additional_ghosts.end()))
          {
            // Ghost insert should be the global number of the block
            // containing the master
            additional_ghosts.push_back(index);
          }
        }
      }
    }
  }
  timer2.stop();
  dolfinx::common::Timer timer3(
      "MPC: MASTER_MASTER index map with correct ghosting");

  // Add ghosts for all other master for same slave
  for (std::int64_t i = 0; i < _masters->num_nodes(); i++)
  {
    for (std::int64_t j = 0; j < _masters->links(i).size(); j++)
    {
      // Only insert if master is in local range
      if ((_masters->links(i)[j] >= block_size * local_range[0])
          && (block_size * local_range[1] > _masters->links(i)[j]))
      {
        for (std::int64_t k = 0; k < _masters->links(i).size(); k++)
        {
          if (i != k)
          {
            // Check if other master is not locally owned
            if ((_masters->links(i)[k] < block_size * local_range[0])
                || (block_size * local_range[1] <= _masters->links(i)[k]))
            {
              const int other_master_as_int = _masters->links(i)[k];
              const std::div_t other_div
                  = std::div(other_master_as_int, block_size);
              const int other_index = other_div.quot;
              // Check if master has already been ghosted
              std::int32_t is_ghost = (new_ghosts == other_index).count() > 0;
              auto is_new_ghost
                  = std::find(additional_ghosts.begin(),
                              additional_ghosts.end(), other_index);

              if ((!is_ghost) && (is_new_ghost == additional_ghosts.end()))
                additional_ghosts.push_back(other_index);
            }
          }
        }
      }
    }
  }

  // Add ghosts for same master for different slave
  for (std::int64_t i = 0; i < _masters->num_nodes(); i++)
  {

    for (std::int64_t j = 0; j < _masters->links(i).size(); j++)
    {
      // Only insert if master is in local range
      if ((_masters->links(i)[j] >= block_size * local_range[0])
          && (block_size * local_range[1] > _masters->links(i)[j]))
      {
        // Check if master occurs multiple times
        std::int32_t occur
            = (_masters->array() == _masters->links(i)[j]).count();

        if (occur > 1)
        {
          const int master_as_int = _masters->links(i)[j];
          const std::div_t div = std::div(master_as_int, block_size);
          const int index = div.quot;
          Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_block(block_size);

          for (std::int64_t comp = 0; comp < block_size; comp++)
            master_block[comp] = block_size * index + comp;
          std::vector<Eigen::Index> occurs_at;
          // Find where other masters are in global array
          for (Eigen::Index k = 0; k < _masters->array().size(); k++)
          {
            if ((_masters->array()[i] == _masters->array()[k]) and (i != k))
            {
              occurs_at.push_back(k);
            }
          }
          // Find which slaves the master belongs
          for (std::int64_t o = 0; o < occurs_at.size(); o++)
          {
            for (std::int64_t k = 0; k < _masters->offsets().size() - 1; k++)
            {
              if (_masters->offsets()[k] <= occurs_at[o]
                  && occurs_at[o] < _masters->offsets()[k + 1])
              {
                for (std::int64_t l = 0; l < _masters->links(k).size(); l++)
                {
                  // Check if other master is not locally owned
                  if ((_masters->links(k)[l] < block_size * local_range[0])
                      || (block_size * local_range[1] <= _masters->links(k)[l]))
                  {
                    const int other_master_as_int = _masters->links(k)[l];
                    const std::div_t other_div
                        = std::div(other_master_as_int, block_size);
                    const int other_index = other_div.quot;
                    // Check if master has already been ghosted
                    std::int32_t is_ghost
                        = (new_ghosts == other_index).count() > 0;
                    auto is_new_ghost
                        = std::find(additional_ghosts.begin(),
                                    additional_ghosts.end(), other_index);

                    if ((!is_ghost)
                        && (is_new_ghost == additional_ghosts.end()))
                      additional_ghosts.push_back(other_index);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  timer3.stop();
  new_ghosts.conservativeResize(new_ghosts.size() + additional_ghosts.size());
  for (std::int64_t i = 0; i < additional_ghosts.size(); i++)
    new_ghosts(num_ghosts + i) = additional_ghosts[i];
  /// Get ghosts owners
  int mpi_size = -1;
  std::int32_t local_size = index_map->size_local();
  MPI_Comm_size(mesh.mpi_comm(), &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                mesh.mpi_comm());

  std::vector<std::int64_t> all_ranges(mpi_size + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   all_ranges.begin() + 1);

  // Compute rank of ghost owners
  std::vector<int> ghost_ranks(new_ghosts.size(), -1);
  for (int i = 0; i < new_ghosts.size(); ++i)
  {
    auto it
        = std::upper_bound(all_ranges.begin(), all_ranges.end(), new_ghosts[i]);
    const int p = std::distance(all_ranges.begin(), it) - 1;
    ghost_ranks[i] = p;
  }

  dolfinx::common::Timer timer4("MPC: Make new indexmap");
  std::shared_ptr<dolfinx::common::IndexMap> new_index_map
      = std::make_shared<dolfinx::common::IndexMap>(
          mesh.mpi_comm(), index_map->size_local(), new_ghosts, ghost_ranks,
          index_map->block_size());
  timer4.stop();

  return new_index_map;
}

/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern
MultiPointConstraint::create_sparsity_pattern(const dolfinx::fem::Form& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer(
      "MPC: Build sparsity pattern (with MPC additions)");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  /// Check that we are using the correct function-space in the bilinear
  /// form otherwise the index map will be wrong
  assert(a.function_space(0) == _function_space);
  assert(a.function_space(1) == _function_space);

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
  auto [unused_indexmap, _dofmap] = dolfinx::fem::DofMapBuilder::build(
      mesh.mpi_comm(), topology, layout, bs);
  _mpc_dofmap = std::make_shared<dolfinx::fem::DofMap>(
      old_dofmap->element_dof_layout, _index_map, _dofmap);

  std::array<std::int64_t, 2> local_range
      = _mpc_dofmap->index_map->local_range();
  std::int64_t local_size = local_range[1] - local_range[0];
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{_mpc_dofmap.get(), _mpc_dofmap.get()}};

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
    for (Eigen::Index j = 0; j < _cells_to_dofs[0]->links(i).size(); j++)
    {
      // Insert pattern for each master
      for (Eigen::Index k = 0;
           k < _masters_local->links(_cell_to_slave_index->links(i)[j]).size();
           k++)
      {
        // Loop over test and trial space
        for (std::size_t l = 0; l < 2; l++)
        {
          // Find dofs for full master block
          std::int32_t local_master
              = _masters_local->links(_cell_to_slave_index->links(i)[j])[k];
          const std::div_t div = std::div(local_master, block_size);
          const int index = div.quot;
          for (std::size_t comp = 0; comp < block_size; comp++)
            master_for_slave[l](comp) = block_size * index + comp;
        }
        // Add all values on cell (including slave), to get complete blocks
        pattern.insert(master_for_slave[0], cell_dof_lists[1]);
        pattern.insert(cell_dof_lists[0], master_for_slave[1]);
      }
      // Add pattern for master owned by other slave on same cell
      for (Eigen::Index k = j + 1; k < _cells_to_dofs[0]->links(i).size(); k++)
      {
        for (Eigen::Index l = 0;
             l
             < _masters_local->links(_cell_to_slave_index->links(i)[k]).size();
             l++)
        {
          std::int32_t other_local_master
              = _masters_local->links(_cell_to_slave_index->links(i)[k])[l];
          const std::div_t odiv = std::div(other_local_master, block_size);
          const int oindex = odiv.quot;
          for (std::size_t m = 0; m < 2; m++)
          {

            for (std::size_t comp = 0; comp < block_size; comp++)
              master_for_other_slave[m](comp) = block_size * oindex + comp;
          }
          pattern.insert(master_for_slave[0], master_for_other_slave[1]);
          pattern.insert(master_for_other_slave[0], master_for_slave[1]);
        }
      }
    }
  }

  // Add pattern for all local masters for the same slave
  for (std::int64_t i = 0; i < _masters_local->num_nodes(); i++)
  {
    for (std::int64_t j = 0; j < _masters_local->links(i).size(); j++)
    {
      if ((_masters->links(i)[j] >= block_size * local_range[0])
          && (block_size * local_range[1] > _masters->links(i)[j]))
      {
        std::int32_t local_master = _masters_local->links(i)[j];
        const std::div_t div = std::div(local_master, block_size);
        const int index = div.quot;
        Eigen::Array<PetscInt, Eigen::Dynamic, 1> local_master_dof(block_size);
        for (std::int64_t comp = 0; comp < block_size; comp++)
          local_master_dof[comp] = block_size * index + comp;
        for (std::int64_t k = j + 1; k < _masters_local->links(i).size(); k++)
        {
          // Map other master to local dof and add remainder of block
          std::int32_t other_local_master = _masters_local->links(i)[k];
          const std::div_t other_div = std::div(other_local_master, block_size);
          const int other_index = other_div.quot;
          Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_dof(
              block_size);

          for (std::int64_t comp = 0; comp < block_size; comp++)
            other_master_dof[comp] = block_size * other_index + comp;

          pattern.insert(local_master_dof, other_master_dof);
          pattern.insert(other_master_dof, local_master_dof);
        }
      }
    }
  }

  // Add pattern for same master for different slave
  for (std::int64_t i = 0; i < _masters->num_nodes(); i++)
  {
    for (std::int64_t j = 0; j < _masters->links(i).size(); j++)
    {
      // Only insert if master is in local range
      if ((_masters->links(i)[j] >= block_size * local_range[0])
          && (block_size * local_range[1] > _masters->links(i)[j]))
      {
        // Check if master occurs multiple times
        std::int32_t occur
            = (_masters->array() == _masters->links(i)[j]).count();
        if (occur > 1)
        {
          std::int32_t local_master = _masters_local->links(i)[j];
          const std::div_t div = std::div(local_master, block_size);
          const int index = div.quot;
          Eigen::Array<PetscInt, Eigen::Dynamic, 1> local_master_dof(
              block_size);
          for (std::int64_t comp = 0; comp < block_size; comp++)
            local_master_dof[comp] = block_size * index + comp;

          std::vector<Eigen::Index> occurs_at;
          // Find where other masters are in global array
          for (Eigen::Index k = 0; k < _masters->array().size(); k++)
          {
            if ((_masters->array()[i] == _masters->array()[k]) and (i != k))
            {
              occurs_at.push_back(k);
            }
          }
          // Find which slaves the master belongs
          for (std::int64_t o = 0; o < occurs_at.size(); o++)
          {
            for (std::int64_t k = 0; k < _masters->offsets().size() - 1; k++)
            {
              if (_masters->offsets()[k] <= occurs_at[o]
                  && occurs_at[o] < _masters->offsets()[k + 1])
              {
                // Add pattern for all other masters for the other occurrence
                // of the master
                for (std::int64_t l = 0; l < _masters_local->links(k).size();
                     l++)
                {
                  const std::div_t other_div
                      = std::div(_masters_local->links(k)[l], block_size);
                  const int other_index = other_div.quot;
                  Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_dof(
                      block_size);

                  for (std::int64_t comp = 0; comp < block_size; comp++)
                    other_master_dof[comp] = block_size * other_index + comp;
                  // Sparsity pattern insert is the local block each master
                  // is in
                  pattern.insert(local_master_dof, other_master_dof);
                  pattern.insert(other_master_dof, local_master_dof);
                }
              }
            }
          }
        }
      }
    }
  }
  return pattern;
}

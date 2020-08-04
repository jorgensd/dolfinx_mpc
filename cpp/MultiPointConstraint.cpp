// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
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
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coefficients,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets_master,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> master_owner_ranks)
    : _function_space(V), _index_map(), _mpc_dofmap(), _slaves(slaves),
      _masters(), _masters_local(), _slaves_local(),
      _coefficients(coefficients), _cells_to_dofs(2, 1), _cell_to_slave_index(),
      _slave_cells(), _master_cells(), _master_owner_ranks(), _master_num_occ()
{
  assert(masters.size() == master_owner_ranks.size());
  assert(slaves.size() == offsets_master.size() - 1);
  assert(offsets_master.tail(1)[0] == masters.size());

  LOG(INFO) << "Initializing MPC class";
  dolfinx::common::Timer timer("~MPC: Init: Total time");
  _masters = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
      masters, offsets_master);
  _master_owner_ranks
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          master_owner_ranks, offsets_master);

  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> dof_lists(2);
  dof_lists[0] = slaves;
  dof_lists[1] = masters;
  /// Locate all cells containing slaves and masters locally and create a
  /// cell_to_slave/master map
  auto cell_info = dolfinx_mpc::locate_cells_with_dofs(V, dof_lists);
  auto [q, cell_to_slave] = cell_info[0];
  auto [c_to_s, c_to_i] = cell_to_slave;
  _cells_to_dofs(0) = c_to_s;
  _slave_cells = q;

  auto [t, cell_to_master] = cell_info[1];
  auto [c_to_m, i_to_m] = cell_to_master;
  _cells_to_dofs(1) = c_to_m;
  _master_cells = t;

  _cell_to_slave_index = c_to_i;

  /// Generate MPC specific index map
  _index_map = generate_index_map();

  dolfinx::common::Timer timer_local(
      "~MPC: Init: Setup local indices per processor");
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
  timer_local.stop();
}

/// Generate MPC specific index map with the correct ghosting
std::shared_ptr<dolfinx::common::IndexMap>
MultiPointConstraint::generate_index_map()
{
  LOG(INFO) << "Generating MPC index map with additional ghosts";
  dolfinx::common::Timer timer("~MPC: Init: Indexmap Total");
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
  std::vector<std::int32_t> additional_ghost_ranks;

  int num_ghosts = new_ghosts.size();

  // Convert to list for fast search
  std::vector<std::int64_t> old_ghosts(new_ghosts.data(),
                                       new_ghosts.data() + num_ghosts);
  std::vector<std::int64_t> masters_std(_masters->array().data(),
                                        _masters->array().data()
                                            + _masters->array().size());
  // Offsets for slave and master
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_offsets
      = _cells_to_dofs[0]->offsets();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> master_cell_offsets
      = _cells_to_dofs[1]->offsets();
  const int mpi_rank = dolfinx::MPI::rank(mesh.mpi_comm());
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    // Loop over slaves in cell
    for (Eigen::Index j = 0; j < slave_offsets[i + 1] - slave_offsets[i]; j++)
    {
      const auto slave_index = _cell_to_slave_index->links(i)[j];
      // Loop over all masters for given slave
      for (Eigen::Index k = 0; k < _masters->links(slave_index).size(); k++)
      {
        if (_master_owner_ranks->links(slave_index)[k] != mpi_rank)
        {
          const int master_as_int = _masters->links(slave_index)[k];
          const std::div_t div = std::div(master_as_int, block_size);
          const int index = div.quot;
          auto it_old = std::find(old_ghosts.begin(), old_ghosts.end(), index);
          auto it = std::find(additional_ghosts.begin(),
                              additional_ghosts.end(), index);

          if ((it_old == old_ghosts.end()) && (it == additional_ghosts.end()))
          {
            // Ghost insert should be the global number of the block
            // containing the master
            additional_ghosts.push_back(index);
            additional_ghost_ranks.push_back(
                _master_owner_ranks->links(slave_index)[k]);
          }
        }
      }
    }
  }

  // Add ghosts for all other master for same slave
  for (std::int64_t i = 0; i < _masters->num_nodes(); i++)
  {
    for (std::int64_t j = 0; j < _masters->links(i).size(); j++)
    {
      // Only insert if master is in local range
      if (_master_owner_ranks->links(i)[j] == mpi_rank)
      {
        for (std::int64_t k = 0; k < _masters->links(i).size(); k++)
        {
          if ((j != k) && (_master_owner_ranks->links(i)[k] != mpi_rank))
          {
            const int other_master_as_int = _masters->links(i)[k];
            const std::div_t other_div
                = std::div(other_master_as_int, block_size);
            const int other_index = other_div.quot;

            // Check if master has already been ghosted
            auto is_old_ghost
                = std::find(old_ghosts.begin(), old_ghosts.end(), other_index);
            auto is_new_ghost = std::find(additional_ghosts.begin(),
                                          additional_ghosts.end(), other_index);

            if (is_old_ghost == old_ghosts.end()
                && is_new_ghost == additional_ghosts.end())
            {
              additional_ghosts.push_back(other_index);
              additional_ghost_ranks.push_back(
                  _master_owner_ranks->links(i)[k]);
            }
          }
        }
      }
    }
  }

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> new_ranks
      = index_map->ghost_owner_rank();

  new_ghosts.conservativeResize(new_ghosts.size() + additional_ghosts.size());
  for (std::int64_t i = 0; i < additional_ghosts.size(); i++)
    new_ghosts(num_ghosts + i) = additional_ghosts[i];

  // Append rank of new ghost owners
  std::vector<int> ghost_ranks(new_ghosts.size());
  for (std::int64_t i = 0; i < new_ranks.size(); i++)
    ghost_ranks[i] = new_ranks[i];

  for (int i = 0; i < additional_ghosts.size(); ++i)
  {
    ghost_ranks[num_ghosts + i] = additional_ghost_ranks[i];
  }

  dolfinx::common::Timer timer_im_init(
      "~MPC: Init: Indexmap Make new indexmap");
  std::shared_ptr<dolfinx::common::IndexMap> new_index_map
      = std::make_shared<dolfinx::common::IndexMap>(
          mesh.mpi_comm(), index_map->size_local(),
          dolfinx::MPI::compute_graph_edges(
              MPI_COMM_WORLD,
              std::set<int>(ghost_ranks.begin(), ghost_ranks.end())),
          new_ghosts, ghost_ranks, index_map->block_size());
  timer_im_init.stop();

  return new_index_map;
}

/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern MultiPointConstraint::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("~MPC: Sparsitypattern Total");
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

        // Add pattern for master owned by other slave on same cell
        for (Eigen::Index k = j + 1; k < _cells_to_dofs[0]->links(i).size();
             k++)
        {
          for (Eigen::Index l = 0;
               l < _masters_local->links(_cell_to_slave_index->links(i)[k])
                       .size();
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
  return pattern;
}

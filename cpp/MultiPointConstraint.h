// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>
#include <petscsys.h>
#include <petscvec.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace
{
//-----------------------------------------------------------------------------
/// Create a map from cell to a set of dofs
/// @param[in] The degrees of freedom (local to process)
/// @returns The map from cell index (local to process) to dofs (local to
/// process) in the cell
std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
create_cell_to_dofs_map(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                        const xtl::span<const std::int32_t>& dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();

  const std::int32_t local_size
      = dofmap.index_map->size_local() + dofmap.index_map->num_ghosts();
  const std::int32_t block_size = dofmap.index_map_bs();

  // Create dof -> cells map where only slave dofs have entries
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> cell_map;
  {
    std::vector<std::int32_t> num_slave_cells(local_size * block_size, 0);

    std::vector<std::int32_t> in_num_cells(local_size * block_size, 0);
    // Loop through all cells and count number of cells a dof occurs in
    for (std::size_t i = 0; i < num_cells; i++)
    {
      auto blocks = dofmap.cell_dofs(i);
      for (auto block : blocks)
        for (std::int32_t j = 0; j < block_size; j++)
          in_num_cells[block * block_size + j]++;
    }
    // Count only number of slave cells for dofs
    for (auto dof : dofs)
      num_slave_cells[dof] = in_num_cells[dof];

    std::vector<std::int32_t> insert_position(local_size * block_size, 0);
    std::vector<std::int32_t> cell_offsets(local_size * block_size + 1);
    cell_offsets[0] = 0;
    std::inclusive_scan(num_slave_cells.begin(), num_slave_cells.end(),
                        cell_offsets.begin() + 1);
    std::vector<std::int32_t> cell_data(cell_offsets.back());

    // Accumulate those cells whose contains a slave dof
    for (std::size_t i = 0; i < num_cells; i++)
    {
      auto blocks = dofmap.cell_dofs(i);
      for (auto block : blocks)
        for (std::int32_t j = 0; j < block_size; j++)
        {
          const std::int32_t dof = block * block_size + j;
          if (num_slave_cells[dof] > 0)
            cell_data[cell_offsets[dof] + insert_position[dof]++] = i;
        }
    }
    cell_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        cell_data, cell_offsets);
  }

  // Create inverse map (cells -> slave dofs)
  std::vector<std::int32_t> num_slaves(num_cells, 0);
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
  {
    auto cells = cell_map->links(i);
    for (auto cell : cells)
      num_slaves[cell]++;
  }
  std::vector<std::int32_t> insert_position(num_cells, 0);
  std::vector<std::int32_t> dof_offsets(num_cells + 1);
  dof_offsets[0] = 0;
  std::inclusive_scan(num_slaves.begin(), num_slaves.end(),
                      dof_offsets.begin() + 1);
  std::vector<std::int32_t> dof_data(dof_offsets.back());
  for (std::int32_t i = 0; i < cell_map->num_nodes(); i++)
  {
    auto cells = cell_map->links(i);
    for (auto cell : cells)
      dof_data[dof_offsets[cell] + insert_position[cell]++] = i;
  }
  return std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      dof_data, dof_offsets);
}
//-----------------------------------------------------------------------------

} // namespace

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint

{

public:
  /// Create contact constraint
  ///
  /// @param[in] V The function space
  /// @param[in] slaves List of local slave dofs
  /// @param[in] slaves_cells List of local slave cells
  /// @param[in] offsets Offsets for local slave cells
  /// @param[in] masters Array of all masters
  /// @param[in] coeffs Coefficients corresponding to each master
  /// @param[in] owners Owners for each master
  /// @param[in] offsets Offsets for masters
  MultiPointConstraint(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                       std::vector<std::int32_t> slaves,
                       std::vector<std::int64_t> masters, std::vector<T> coeffs,
                       std::vector<std::int32_t> owners,
                       std::vector<std::int32_t> offsets)
      : _V(V), _slaves(), _is_slave(), _cell_to_slaves_map(),
        _num_local_slaves(), _master_map(), _coeff_map(), _owner_map(),
        _dofmap()
  {
    // Create list indicating which dofs on the process are slaves
    const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
    const std::int32_t num_dofs_local
        = dofmap.index_map_bs()
          * (dofmap.index_map->size_local() + dofmap.index_map->num_ghosts());
    std::vector<std::int8_t> _slave_data(num_dofs_local, 0);
    for (auto dof : slaves)
      _slave_data[dof] = 1;
    _is_slave = std::move(_slave_data);

    // Create a map for cells owned by the process to the slaves
    _cell_to_slaves_map = create_cell_to_dofs_map(_V, slaves);

    // Create adjacency list with all local dofs, where the slave dofs maps to
    // its masters
    std::vector<std::int32_t> _num_masters(num_dofs_local);
    std::fill(_num_masters.begin(), _num_masters.end(), 0);
    for (std::int32_t i = 0; i < slaves.size(); i++)
      _num_masters[slaves[i]] = offsets[i + 1] - offsets[i];
    std::vector<std::int32_t> masters_offsets(num_dofs_local + 1);
    masters_offsets[0] = 0;
    std::inclusive_scan(_num_masters.begin(), _num_masters.end(),
                        masters_offsets.begin() + 1);

    // Reuse num masters as fill position array
    std::fill(_num_masters.begin(), _num_masters.end(), 0);
    std::vector<std::int64_t> _master_data(masters.size());
    std::vector<T> _coeff_data(masters.size());
    std::vector<std::int32_t> _owner_data(masters.size());
    /// Create adjacency lists spanning all local dofs mapping to master dofs,
    /// its owner and the corresponding coefficient
    for (std::int32_t i = 0; i < slaves.size(); i++)
    {
      for (std::int32_t j = 0; j < offsets[i + 1] - offsets[i]; j++)
      {
        _master_data[masters_offsets[slaves[i]] + _num_masters[slaves[i]]]
            = masters[offsets[i] + j];
        _coeff_data[masters_offsets[slaves[i]] + _num_masters[slaves[i]]]
            = coeffs[offsets[i] + j];
        _owner_data[masters_offsets[slaves[i]] + _num_masters[slaves[i]]]
            = owners[offsets[i] + j];
        _num_masters[slaves[i]]++;
      }
    }
    _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        _coeff_data, masters_offsets);
    _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        _owner_data, masters_offsets);

    // Create a vector containing all the slave dofs (sorted)
    std::vector<std::int32_t> sorted_slaves(slaves.size());
    std::int32_t c = 0;
    for (std::int32_t i = 0; i < _is_slave.size(); i++)
      if (_is_slave[i])
        sorted_slaves[c++] = i;
    _slaves = std::move(sorted_slaves);

    const std::int32_t num_local
        = dofmap.index_map_bs() * dofmap.index_map->size_local();
    auto it = std::lower_bound(_slaves.begin(), _slaves.end(), num_local);
    _num_local_slaves = std::distance(_slaves.begin(), it);
    // Create new index map with all masters
    convert_masters_to_local_index(_master_data, masters_offsets);
  }
  //-----------------------------------------------------------------------------
  /// Backsubstitute slave/master constraint for a given function
  void backsubstitution(xtl::span<T> vector)
  {
    for (auto slave : _slaves)
    {
      auto masters = _master_map->links(slave);
      auto coeffs = _coeff_map->links(slave);
      assert(masters.size() == coeffs.size());
      for (std::int32_t k = 0; k < masters.size(); ++k)
        vector[slave] += coeffs[k] * vector[masters[k]];
    }
  };

  /// Return map from cell to slaves contained in that cell
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  cell_to_slaves() const
  {
    return _cell_to_slaves_map;
  }
  /// Return map from slave to masters (local_index)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  masters() const
  {
    return _master_map;
  }

  /// Return map from slave to coefficients
  const std::shared_ptr<dolfinx::graph::AdjacencyList<T>>& coefficients() const
  {
    return _coeff_map;
  }

  /// Return map from slave to masters (global index)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  owners() const
  {
    return _owner_map;
  }

  /// Return of local dofs + num ghosts indicating if a dof is a slave
  const std::vector<std::int8_t>& is_slave() const { return _is_slave; }

  /// Return an array of all slave indices (sorted and local to process)
  const std::vector<std::int32_t>& slaves() const { return _slaves; }

  /// Return number of slaves owned by process
  const std::int32_t num_local_slaves() const { return _num_local_slaves; }

  /// Return constraint IndexMap
  std::shared_ptr<const dolfinx::common::IndexMap> index_map()
  {
    return _index_map;
  }

  std::shared_ptr<dolfinx::fem::DofMap> dofmap() { return _dofmap; }

  std::shared_ptr<const dolfinx::fem::FunctionSpace> org_space() { return _V; }

private:
  /// Given the global indices of the master dofs in the multi point constraint,
  /// create a new index map where all masters that where not previously ghosted
  /// are added as ghosts. This function also converts the global master indices
  /// @param[in] global_masters The list of master dofs (global index)
  /// @param[in] offsets The offsets of the masters (relative to the slave
  /// index)
  void convert_masters_to_local_index(std::vector<std::int64_t>& global_masters,
                                      std::vector<std::int32_t> offsets)
  {
    dolfinx::common::Timer timer("~MPC: Create new index map a");
    MPI_Comm comm = _V->mesh()->mpi_comm();
    const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
    std::shared_ptr<const dolfinx::common::IndexMap> index_map
        = dofmap.index_map;

    const std::int32_t& block_size = _V->dofmap()->index_map_bs();

    // Compute local master block index.
    std::vector<std::int32_t> local_blocks(global_masters.size());
    std::vector<std::int64_t> global_blocks(global_masters.size());
    std::vector<std::int32_t> remainders(global_masters.size());
    std::vector<std::int32_t>& owners = _owner_map->array();

    for (std::size_t i = 0; i < global_masters.size(); ++i)
    {
      const std::ldiv_t div
          = std::div(global_masters[i], std::int64_t(block_size));
      global_blocks[i] = div.quot;
      remainders[i] = div.rem;
    }

    int mpi_size = -1;
    MPI_Comm_size(comm, &mpi_size);
    if (mpi_size == 1)
    {
      _index_map = index_map;
    }
    else
    {
      // Map global master blocks to local blocks
      _V->dofmap()->index_map->global_to_local(global_blocks, local_blocks);

      // Check which local masters that are not on the process already
      std::vector<std::int64_t> new_ghosts;
      std::vector<std::int32_t> new_ranks;
      for (std::size_t i = 0; i < local_blocks.size(); i++)
      {
        // Check if master block already has a local index
        if (local_blocks[i] == -1)
        {
          // Check if master block has already been ghosted, which is the case
          // when we have multiple masters from the same block
          auto already_ghosted = std::find(new_ghosts.begin(), new_ghosts.end(),
                                           global_blocks[i]);
          if (already_ghosted == new_ghosts.end())
          {
            new_ghosts.push_back(global_blocks[i]);
            new_ranks.push_back(owners[i]);
          }
        }
      }

      // Append new ghosts (and corresponding rank) at the end of the old set of
      // ghosts originating from the old index map
      std::vector<std::int32_t> ranks = index_map->ghost_owner_rank();
      std::vector<std::int64_t> ghosts = index_map->ghosts();
      const std::int32_t num_ghosts = ghosts.size();
      ghosts.resize(num_ghosts + new_ghosts.size());
      ranks.resize(num_ghosts + new_ghosts.size());
      for (std::int64_t i = 0; i < new_ghosts.size(); i++)
      {
        ghosts[num_ghosts + i] = new_ghosts[i];
        ranks[num_ghosts + i] = new_ranks[i];
      }

      dolfinx::common::Timer timer_indexmap("MPC: Init indexmap");
      // Create new indexmap with ghosts for master blocks added
      _index_map = std::make_shared<dolfinx::common::IndexMap>(
          comm, index_map->size_local(),
          dolfinx::MPI::compute_graph_edges(
              MPI_COMM_WORLD, std::set<int>(ranks.begin(), ranks.end())),
          ghosts, ranks);
      timer_indexmap.stop();
    }

    // Extract information from the old dofmap to create a new one
    const dolfinx::fem::DofMap* old_dofmap = _V->dofmap().get();
    dolfinx::mesh::Topology topology = _V->mesh()->topology();
    dolfinx::fem::ElementDofLayout layout = *old_dofmap->element_dof_layout;

    // Create new dofmap based on the MPC index map
    // NOTE: This is a workaround to get the Adjacency list and block size
    // used to initialize the original dofmap
    auto [_, bs, dofmap_adj] = dolfinx::fem::build_dofmap_data(
        _V->mesh()->mpi_comm(), topology, layout,
        [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
        { return dolfinx::graph::scotch::compute_gps(g, 2).first; });

    // Permute dofmap as done in DOLFINx
    // If the element's DOF transformations are permutations, permute the DOF
    // numbering on each cell
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    if (element->needs_dof_permutations())
    {
      const int D = topology.dim();
      const int num_cells = topology.connectivity(D, 0)->num_nodes();
      topology.create_entity_permutations();
      const std::vector<std::uint32_t>& cell_info
          = topology.get_cell_permutation_info();

      const std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
          unpermute_dofs = element->get_dof_permutation_function(true, true);
      for (std::int32_t cell = 0; cell < num_cells; ++cell)
        unpermute_dofs(dofmap_adj.links(cell), cell_info[cell]);
    }

    // Create the new dofmap based on the extended index map
    _dofmap = std::make_shared<dolfinx::fem::DofMap>(
        old_dofmap->element_dof_layout, _index_map, bs, dofmap_adj, bs);

    // Compute the new local index of the master blocks
    std::vector<std::int32_t> new_local_blocks(global_blocks.size());
    _index_map->global_to_local(global_blocks, new_local_blocks);

    // Go from blocks to actual local dof
    for (std::size_t i = 0; i < new_local_blocks.size(); i++)
      new_local_blocks[i] = new_local_blocks[i] * block_size + remainders[i];

    // Create master map
    _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        new_local_blocks, offsets);
  }
  //-----------------------------------------------------------------------------

  // Original function space
  std::shared_ptr<const dolfinx::fem::FunctionSpace> _V;
  // Array including all slaves (local + ghosts)
  std::vector<std::int32_t> _slaves;
  std::vector<std::int8_t> _is_slave;

  // Map from slave cell to index in _slaves for a given slave cell
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_to_slaves_map;

  // Number of slaves owned by the process
  std::int32_t _num_local_slaves;
  // Map from slave (local to process) to masters (local to process)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _master_map;
  // Map from slave (local to process)to coefficients
  std::shared_ptr<dolfinx::graph::AdjacencyList<T>> _coeff_map;
  // Map from slave( local to process) to rank of process owning master
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _owner_map;
  // Index map including masters
  std::shared_ptr<const dolfinx::common::IndexMap> _index_map;
  // Dofmap including masters
  std::shared_ptr<dolfinx::fem::DofMap> _dofmap;
};
} // namespace dolfinx_mpc
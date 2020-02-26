// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN-X (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FormIntegrals.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx_mpc;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters,
    Eigen::Array<double, Eigen::Dynamic, 1> coefficients,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> offsets_master)
    : _function_space(V), _slaves(slaves), _masters(masters),
      _coefficients(coefficients), _offsets_master(offsets_master),
      _slave_cells(), _offsets_cell_to_slave(), _cell_to_slave(),
      _master_cells(), _offsets_cell_to_master(), _cell_to_master(),
      _glob_to_loc_ghosts(), _glob_master_to_loc_ghosts(), _index_map()
{
  auto [q, cell_to_slave] = dolfinx_mpc::locate_cells_with_dofs(V, slaves);
  auto [r, s] = cell_to_slave;
  auto [t, cell_to_master] = dolfinx_mpc::locate_cells_with_dofs(V, masters);
  auto [u, v] = cell_to_master;

  _slave_cells = q;
  _cell_to_slave = r;
  _offsets_cell_to_slave = s;
  _master_cells = t;
  _cell_to_master = u;
  _offsets_cell_to_master = v;

  _index_map = generate_index_map();
}

Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
MultiPointConstraint::slave_cells()
{
  return _slave_cells;
}

std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
          Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
MultiPointConstraint::cell_to_slave_mapping()
{
  return std::pair(_cell_to_slave, _offsets_cell_to_slave);
}

std::shared_ptr<dolfinx::common::IndexMap>
MultiPointConstraint::generate_index_map()
{
  const dolfinx::mesh::Mesh& mesh = *(_function_space->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_function_space->dofmap());

  // Get common::IndexMaps for each dimension
  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;
  int block_size = index_map->block_size();

  // Get old dofs that will be appended to
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> new_ghosts
      = index_map->ghosts();
  int num_ghosts = new_ghosts.size();
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    // Loop over slaves in cell
    for (Eigen::Index j = 0;
         j < _offsets_cell_to_slave[i + 1] - _offsets_cell_to_slave[i]; j++)
    {
      // Find index of slave in global array
      std::int64_t slave_index = 0; // Index in global slave array
      for (std::uint64_t counter = 0; counter < _slaves.size(); counter++)
        if (_slaves[counter] == _cell_to_slave[_offsets_cell_to_slave[i] + j])
          slave_index = counter;

      // Loop over all masters for given slave
      for (Eigen::Index k = 0;
           k < _offsets_master[slave_index + 1] - _offsets_master[slave_index];
           k++)
      {
        // Check if master is already owned by the processor by looping over
        // all owned cells
        bool master_on_proc = false;
        for (std::int64_t m = 0; m < unsigned(_master_cells.size()); m++)
        {
          for (Eigen::Index n = 0;
               n < _offsets_cell_to_master[m + 1] - _offsets_cell_to_master[m];
               n++)
          {
            if (_masters[_offsets_master[slave_index] + k]
                == _cell_to_master[_offsets_cell_to_master[m] + n])
            {
              master_on_proc = true;
            }
          }
        }
        // If master not on processor add it to the ghost array and
        // create a map from master to local index used in matrix creation
        if (!master_on_proc)
        {
          const int master_as_int = _masters[_offsets_master[slave_index] + k];
          const std::div_t div = std::div(master_as_int, block_size);
          const int index = div.quot;
          bool already_ghosted = false;
          for (std::int64_t gh = 0; gh < new_ghosts.size(); gh++)
          {
            if (index == new_ghosts[gh])
            {
              already_ghosted = true;
              _glob_to_loc_ghosts[_masters[_offsets_master[slave_index] + k]]
                  = gh + index_map->size_local();
            }
          }
          if (!already_ghosted)
          {
            // Ghost insert should be the global number of the block
            // containing the master
            new_ghosts.conservativeResize(new_ghosts.size() + 1);
            new_ghosts[num_ghosts] = index;
            _glob_to_loc_ghosts[_masters[_offsets_master[slave_index] + k]]
                = num_ghosts + index_map->size_local();
            num_ghosts++;
          }
        }
      }
    }
  }

  // Loop over master cells on local processor and add ghosts for all other
  // masters (This could probably be optimized, but thats for later)
  std::array<std::int64_t, 2> local_range = index_map->local_range();
  for (Eigen::Index i = 0; i < _masters.size(); ++i)
  {
    // Check if master is owned by processor
    if ((_masters[i] < block_size * local_range[0])
        || (block_size * local_range[1] <= _masters[i]))
    {
      // const std::int32_t index_block = _masters[i] / block_size;
      // If master is not owned, check if it is ghosted
      bool already_ghosted = false;
      const int master_as_int = _masters[i];
      const std::div_t div = std::div(master_as_int, block_size);
      const int index = div.quot;
      for (std::int64_t gh = 0; gh < new_ghosts.size(); gh++)
      {
        if (index == new_ghosts[gh])
        {
          already_ghosted = true;
          _glob_master_to_loc_ghosts[_masters[i]]
              = gh + index_map->size_local();
        }
      }
      if (!already_ghosted)
      {
        new_ghosts.conservativeResize(new_ghosts.size() + 1);
        new_ghosts[num_ghosts] = index;
        _glob_master_to_loc_ghosts[_masters[i]]
            = num_ghosts + index_map->size_local();
        num_ghosts++;
      }
    }
  }

  std::shared_ptr<dolfinx::common::IndexMap> new_index_map
      = std::make_shared<dolfinx::common::IndexMap>(
          mesh.mpi_comm(), index_map->size_local(), new_ghosts,
          index_map->block_size());
  return new_index_map;
}

dolfinx::la::SparsityPattern
MultiPointConstraint::create_sparsity_pattern(const dolfinx::fem::Form& a)
{

  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  /// Check that we are using the correct function-space in the bilinear form
  /// otherwise the index map will be wrong
  assert(a.function_space(0) == _function_space);
  assert(a.function_space(1) == _function_space);

  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  int block_size = _index_map->block_size();

  /// Create new dofmap using the MPC index-maps
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = _index_map;
  new_maps[1] = _index_map;
  const dolfinx::fem::DofMap* old_dofmap = a.function_space(0)->dofmap().get();
  const dolfinx::fem::DofMap new_dofmap = dolfinx::fem::DofMap(
      old_dofmap->element_dof_layout, _index_map, old_dofmap->dof_array());
  std::array<std::int64_t, 2> local_range = new_dofmap.index_map->local_range();
  const dolfinx::fem::DofMap* pointer;
  pointer = &new_dofmap;
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps = {{pointer, pointer}};

  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);

  // Add non-zeros for each slave cell to sparsity pattern.
  // For the i-th cell with a slave, all local entries has to be from the j-th
  // slave to the k-th master degree of freedom
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    // Loop over slaves in cell
    for (Eigen::Index j = 0;
         j < _offsets_cell_to_slave[i + 1] - _offsets_cell_to_slave[i]; j++)
    {
      // Figure out what the index of the current slave on the cell is in
      // _slaves
      std::int64_t slave_index = 0;
      for (std::uint64_t counter = 0; counter < _slaves.size(); counter++)
        if (_slaves[counter] == _cell_to_slave[_offsets_cell_to_slave[i] + j])
          slave_index = counter;

      // Insert pattern for each master
      for (Eigen::Index k = 0;
           k < _offsets_master[slave_index + 1] - _offsets_master[slave_index];
           k++)
      {
        /// Arrays replacing slave dof with master dof in sparsity pattern
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            master_for_slave;
        // Sparsity pattern needed for columns
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            slave_dof_neighbours;
        // Loop over test and trial space
        for (std::size_t l = 0; l < 2; l++)
        {
          auto cell_dof_list = dofmaps[l]->cell_dofs(_slave_cells[i]);

          master_for_slave[l].resize(block_size);

          // Replace slave dof with master dof (local insert)
          for (std::size_t m = 0; m < unsigned(cell_dof_list.size()); m++)
          {
            if (_slaves[slave_index]
                == unsigned(cell_dof_list[m] + block_size * local_range[0]))
            {
              // Check if master is a ghost
              if (_glob_to_loc_ghosts.find(
                      _masters[_offsets_master[slave_index] + k])
                  != _glob_to_loc_ghosts.end())
              {
                // Local block index (larger than local_size)
                std::int64_t block_index
                    = _glob_to_loc_ghosts[_masters[_offsets_master[slave_index]
                                                   + k]];
                for (std::size_t comp = 0; comp < block_size; comp++)
                  master_for_slave[l](comp) = block_size * block_index + comp;
              }
              else
              {
                const int master_int
                    = _masters[_offsets_master[slave_index] + k];
                const std::div_t div = std::div(master_int, block_size);
                const int index = div.quot;
                /// Add non zeros for full block
                for (std::size_t comp = 0; comp < block_size; comp++)
                  master_for_slave[l](comp)
                      = block_size * index + comp - block_size * local_range[0];
              }
            }
          }
          // Add all values on cell (including slave), to get complete blocks
          slave_dof_neighbours[l] = cell_dof_list;
        }
        pattern.insert(master_for_slave[0], slave_dof_neighbours[1]);
        pattern.insert(slave_dof_neighbours[0], master_for_slave[1]);
      }
    }
  }
  // Loop over local master cells
  for (std::int64_t i = 0; i < unsigned(_master_cells.size()); i++)
  {
    for (Eigen::Index j = 0;
         j < _offsets_cell_to_master[i + 1] - _offsets_cell_to_master[i]; j++)
    {
      // Check if dof of owned cell is a local cell
      Eigen::Array<PetscInt, Eigen::Dynamic, 1> local_master_dof(block_size);
      if ((_cell_to_master[_offsets_cell_to_master[i] + j]
           < block_size * local_range[0])
          || (block_size * local_range[1]
              <= _cell_to_master[_offsets_cell_to_master[i] + j]))
      {
        // Local block index (larger than local_size)
        std::int64_t block_index
            = _glob_to_loc_ghosts[_cell_to_master[_offsets_cell_to_master[i]
                                                  + j]];
        for (std::size_t comp = 0; comp < block_size; comp++)
          local_master_dof[comp] = block_size * block_index + comp;
      }
      else
      {
        const int master_int = _cell_to_master[_offsets_cell_to_master[i] + j];
        const std::div_t div = std::div(master_int, block_size);
        const int index = div.quot;
        /// Add non zeros for full block
        for (std::size_t comp = 0; comp < block_size; comp++)
          local_master_dof[comp]
              = block_size * index + comp - block_size * local_range[0];
      }

      Eigen::Array<PetscInt, Eigen::Dynamic, 1> other_master_dof(block_size);
      for (Eigen::Index k = 0; k < _masters.size(); k++)
      {
        // If not on processor add ghost-index, else add local number
        if (_masters[k] != _cell_to_master[_offsets_cell_to_master[i] + j])
        {
          if ((_masters[k] < block_size * local_range[0])
              || (block_size * local_range[1] <= _masters[k]))
          {
            // Local block index (larger than local_size)
            std::int64_t block_index = _glob_to_loc_ghosts[_masters[k]];
            for (std::size_t comp = 0; comp < block_size; comp++)
              other_master_dof[comp] = block_size * block_index + comp;
          }
          else
          {
            const int master_int = _masters[k];
            const std::div_t div = std::div(master_int, block_size);
            const int index = div.quot;
            /// Add non zeros for full block
            for (std::size_t comp = 0; comp < block_size; comp++)
              other_master_dof[comp]
                  = block_size * index + comp - block_size * local_range[0];
          }
          pattern.insert(local_master_dof, other_master_dof);
          pattern.insert(other_master_dof, local_master_dof);
        }
      }
    }
  }
  return pattern;
}

Eigen::Array<std::int64_t, Eigen::Dynamic, 1> MultiPointConstraint::slaves()
{
  return _slaves;
}

std::shared_ptr<dolfinx::common::IndexMap> MultiPointConstraint::index_map()
{
  return _index_map;
}

std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
          Eigen::Array<double, Eigen::Dynamic, 1>>
MultiPointConstraint::masters_and_coefficients()
{
  return std::pair(_masters, _coefficients);
}

/// Return master offset data
Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
MultiPointConstraint::master_offsets()
{
  return _offsets_master;
}

/// Return the global to local mapping of a master coefficient it it is not
/// on this processor
std::unordered_map<int, int> MultiPointConstraint::glob_to_loc_ghosts()
{
  return _glob_to_loc_ghosts;
}

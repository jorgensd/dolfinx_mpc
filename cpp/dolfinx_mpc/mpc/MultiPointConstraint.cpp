// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN-X (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/MeshIterator.h>

using namespace dolfinx_mpc;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    std::vector<std::int64_t> slaves, std::vector<std::int64_t> masters,
    std::vector<double> coefficients, std::vector<std::int64_t> offsets)
    : _function_space(V), _slaves(slaves), _masters(masters),
      _coefficients(coefficients), _offsets(offsets), _slave_cells(),
      _normal_cells(), _offsets_cell_to_slave(), _cell_to_slave(),
      _master_cells(), _offsets_cell_to_master(), _cell_to_master(),
      _glob_to_loc_ghosts() {
  find_slave_cells();
}

/// Get the master nodes for the i-th slave
std::vector<std::int64_t> MultiPointConstraint::masters(std::int64_t i) {
  assert(i < unsigned(_masters.size()));
  std::vector<std::int64_t> owners;
  // Check if this is the final entry or beyond
  for (std::int64_t j = _offsets[i]; j < _offsets[i + 1]; j++) {
    owners.push_back(_masters[j]);
  }
  return owners;
}

/// Get the master nodes for the i-th slave
std::vector<double> MultiPointConstraint::coefficients(std::int64_t i) {
  assert(i < unsigned(_coefficients.size()));
  std::vector<double> coeffs;
  for (std::int64_t j = _offsets[i]; j < _offsets[i + 1]; j++) {
    coeffs.push_back(_coefficients[j]);
  }
  return coeffs;
}

std::vector<std::int64_t> MultiPointConstraint::slave_cells() {
  return _slave_cells;
}

/// Partition cells into cells containing slave dofs and cells with no cell
void MultiPointConstraint::find_slave_cells() {
  const dolfinx::mesh::Mesh &mesh = *(_function_space->mesh());
  const dolfinx::fem::DofMap &dofmap = *(_function_space->dofmap());
  const std::vector<int64_t> &global_indices =
      mesh.topology().global_indices(mesh.topology().dim());

  // Categorise cells as normal cells or master cells or slave cells,
  // which can later be used in the custom assembly
  std::int64_t j = 0;
  std::int64_t k = 0;
  _offsets_cell_to_slave.push_back(j);
  _offsets_cell_to_master.push_back(k);
  std::array<std::int64_t, 2> local_range = dofmap.index_map->local_range();

  for (auto &cell : dolfinx::mesh::MeshRange(mesh, mesh.topology().dim())) {
    const int cell_index = cell.index();
    auto dofs = dofmap.cell_dofs(cell_index);
    bool slave_cell = false;
    bool master_cell = false;
    for (Eigen::Index i = 0; i < dofs.size(); ++i) {
      for (auto slave : _slaves) {
        if ((local_range[0] <= slave) && (slave < local_range[1]) &&
            (unsigned(dofs[i] + dofmap.index_map->local_range()[0]) == slave)) {
          _cell_to_slave.push_back(slave);
          j++;
          slave_cell = true;
        }
      }
      for (auto master : _masters) {
        if ((local_range[0] <= master) && (master < local_range[1]) &&
            (unsigned(dofs[i] + dofmap.index_map->local_range()[0]) ==
             master)) {
          _cell_to_master.push_back(master);
          k++;
          master_cell = true;
        }
      }
    }
    if (slave_cell) {
      _slave_cells.push_back(cell_index);
      _offsets_cell_to_slave.push_back(j);
    }
    if (master_cell) {
      _master_cells.push_back(master_cell);
      _offsets_cell_to_master.push_back(k);
    }
    if (!slave_cell && !master_cell) {
      _normal_cells.push_back(cell_index);
    }
  }
}

std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
MultiPointConstraint::cell_to_slave_mapping() {
  return std::pair(_cell_to_slave, _offsets_cell_to_slave);
}

// Append to existing sparsity pattern
dolfinx::la::PETScMatrix
MultiPointConstraint::generate_petsc_matrix(const dolfinx::fem::Form &a) {

  if (a.rank() != 2) {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  // Get dof maps
  std::array<const dolfinx::fem::DofMap *, 2> dofmaps = {
      {a.function_space(0)->dofmap().get(),
       a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh &mesh = *(a.mesh());

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps = {
      {dofmaps[0]->index_map, dofmaps[1]->index_map}};

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts0;
  if (index_maps[0]->num_ghosts() > 0) {
    ghosts0 = index_maps[0]->ghosts();
  }
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> new_ghosts0;
  for (int i = 0; i < ghosts0.size(); ++i) {
    new_ghosts0.conservativeResize(new_ghosts0.size() + 1);
    new_ghosts0[i] = ghosts0[i];
  }
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts1;
  if (index_maps[0]->num_ghosts() > 0) {
    ghosts1 = index_maps[1]->ghosts();
  }
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> new_ghosts1;
  for (int i = 0; i < ghosts1.size(); ++i) {
    new_ghosts1.conservativeResize(new_ghosts1.size() + 1);
    new_ghosts1[i] = ghosts1[i];
  }
  // Loop over slave cells on local processor
  int num_ghosts0 = ghosts0.size();
  int num_ghosts1 = ghosts1.size();

  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++) {
    std::vector<std::int64_t> slaves_i(
        _cell_to_slave.begin() + _offsets_cell_to_slave[i],
        _cell_to_slave.begin() + _offsets_cell_to_slave[i + 1]);
    // Loop over slaves in cell
    for (auto slave_dof : slaves_i) {
      std::int64_t slave_index = 0; // Index in slave array
      // Find place of slave in global setting to obtain corresponding master
      // dofs
      for (std::uint64_t counter = 0; counter < _slaves.size(); counter++) {
        if (_slaves[counter] == slave_dof) {
          slave_index = counter;
        }
      }

      std::vector<std::int64_t> masters_i(
          _masters.begin() + _offsets[slave_index],
          _masters.begin() + _offsets[slave_index + 1]);
      for (auto master : masters_i) {
        // Loop over all local master cells to determine if master is in a
        // local cell
        bool master_on_proc = false;
        for (std::int64_t m = 0; m < unsigned(_master_cells.size()); m++) {
          std::vector<std::int64_t> masters_on_cell(
              _cell_to_master.begin() + _offsets_cell_to_master[m],
              _cell_to_master.begin() + _offsets_cell_to_master[m + 1]);
          if (std::find(masters_on_cell.begin(), masters_on_cell.end(),
                        master) != masters_on_cell.end()) {
            master_on_proc = true;
          }
        }
        if (!master_on_proc) {
          new_ghosts0.conservativeResize(new_ghosts0.size() + 1);
          new_ghosts0[num_ghosts0] = master;
          new_ghosts1.conservativeResize(new_ghosts1.size() + 1);
          new_ghosts1[num_ghosts1] = master;
          _glob_to_loc_ghosts[master] =
              num_ghosts0 + index_maps[0]->size_local();
          num_ghosts0 = num_ghosts0 + 1;
          num_ghosts1 = num_ghosts1 + 1;
        }
      }
    }
  }

  // Loop over master cells on local processor and add ghosts for all other
  // masters (This could probably be optimized, but thats for later)
  std::unordered_map<int, int> glob_master_to_loc_ghosts;
  std::array<std::int64_t, 2> local_range =
      dofmaps[0]->index_map->local_range();
  for (auto global_master : _masters) {
    if ((global_master < local_range[0]) || (local_range[1] < global_master)) {
      bool already_ghosted = false;
      for (std::int64_t gh = 0; gh < new_ghosts0.size(); gh++) {
        if (global_master == new_ghosts0[gh]) {

          glob_master_to_loc_ghosts[global_master] = gh;
          already_ghosted = true;
        }
      }
      if (!already_ghosted) {
        new_ghosts0.conservativeResize(new_ghosts0.size() + 1);
        new_ghosts0[num_ghosts0] = global_master;
        new_ghosts1.conservativeResize(new_ghosts1.size() + 1);
        new_ghosts1[num_ghosts1] = global_master;
        glob_master_to_loc_ghosts[global_master] =
            num_ghosts0 + index_maps[0]->size_local();
        num_ghosts0 = num_ghosts0 + 1;
        num_ghosts1 = num_ghosts1 + 1;
      }
    }
  }

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = std::make_shared<dolfinx::common::IndexMap>(
      mesh.mpi_comm(), index_maps[0]->size_local(), new_ghosts0,
      index_maps[0]->block_size);
  new_maps[1] = std::make_shared<dolfinx::common::IndexMap>(
      mesh.mpi_comm(), index_maps[1]->size_local(), new_ghosts1,
      index_maps[1]->block_size);
  // create and build sparsity pattern
  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps);
  // dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), index_maps);

  if (a.integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::cell) > 0)
    dolfinx::fem::SparsityPatternBuilder::cells(pattern, mesh,
                                                {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::interior_facet) > 0)
    dolfinx::fem::SparsityPatternBuilder::interior_facets(
        pattern, mesh, {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::exterior_facet) > 0)
    dolfinx::fem::SparsityPatternBuilder::exterior_facets(
        pattern, mesh, {{dofmaps[0], dofmaps[1]}});
  // pattern.info_statistics();

  // Loop over slave cells
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++) {
    std::vector<std::int64_t> slaves_i(
        _cell_to_slave.begin() + _offsets_cell_to_slave[i],
        _cell_to_slave.begin() + _offsets_cell_to_slave[i + 1]);
    // Loop over slaves in cell
    for (auto slave_dof : slaves_i) {
      std::int64_t slave_index = 0; // Index in slave array
      // FIXME: Move this somewhere else as there should exist a map for
      // this
      for (std::uint64_t counter = 0; counter < _slaves.size(); counter++) {
        if (_slaves[counter] == slave_dof) {
          slave_index = counter;
        }
      }
      std::vector<std::int64_t> masters_i(
          _masters.begin() + _offsets[slave_index],
          _masters.begin() + _offsets[slave_index + 1]);
      // Insert pattern for each master
      for (auto master : masters_i) {
        // New sparsity pattern arrays
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            new_master_dofs;
        // Sparsity pattern needed for columns
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            master_slave_dofs;
        // Loop over test and trial space
        for (std::size_t j = 0; j < 2; j++) {
          auto cell_dof_list = dofmaps[j]->cell_dofs(_slave_cells[i]);

          new_master_dofs[j].resize(1);

          // Replace slave dof with master dof (local insert)
          int l = 0;
          for (std::size_t k = 0; k < unsigned(cell_dof_list.size()); ++k) {
            if (_slaves[slave_index] ==
                unsigned(cell_dof_list[k] + local_range[0])) {
              // Check if master is a ghost
              if (_glob_to_loc_ghosts.find(master) != _glob_to_loc_ghosts.end())
                new_master_dofs[j](0) = _glob_to_loc_ghosts[master];
              else
                new_master_dofs[j](0) = master - local_range[0];
            } else {
              master_slave_dofs[j].conservativeResize(
                  master_slave_dofs[j].size() + 1);
              master_slave_dofs[j](l) = cell_dof_list(k);
              l = l + 1;
            }
          }
        }

        pattern.insert_local(new_master_dofs[0], master_slave_dofs[1]);
        pattern.insert_local(master_slave_dofs[0], new_master_dofs[1]);
      }
    }
  }

  // Loop over local master cells
  for (std::int64_t i = 0; i < unsigned(_master_cells.size()); i++) {
    std::vector<std::int64_t> masters_i(
        _cell_to_master.begin() + _offsets_cell_to_master[i],
        _cell_to_master.begin() + _offsets_cell_to_master[i + 1]);
    for (auto master : masters_i) {
      // Check if dof of owned cell is a local cell
      Eigen::Array<PetscInt, 1, 1> local_master_dof(1);
      if ((master < local_range[0]) || (local_range[1] <= master)) {
        local_master_dof[0] = glob_master_to_loc_ghosts[master];
      } else {
        local_master_dof[0] = master - local_range[0];
      }
      Eigen::Array<PetscInt, 1, 1> other_master_dof;
      for (auto other_master : _masters) {
        // If not on processor add ghost-index, else add local number
        if (other_master != master) {
          if ((other_master < local_range[0]) ||
              (local_range[1] <= other_master)) {
            other_master_dof[0] = glob_master_to_loc_ghosts[other_master];
          } else {
            other_master_dof[0] = other_master - local_range[0];
          }
          pattern.insert_local(local_master_dof, other_master_dof);
          pattern.insert_local(other_master_dof, local_master_dof);
        }
      }
    }
  }

  // pattern.info_statistics();
  pattern.assemble();
  dolfinx::la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);

  return A;
}

std::vector<std::int64_t> MultiPointConstraint::slaves() { return _slaves; }

std::pair<std::vector<std::int64_t>, std::vector<double>>
MultiPointConstraint::masters_and_coefficients() {
  return std::pair(_masters, _coefficients);
}

/// Return master offset data
std::vector<std::int64_t> MultiPointConstraint::master_offsets() {
  return _offsets;
}

/// Return the global to local mapping of a master coefficient it it is not on
/// this processor
std::unordered_map<int, int> MultiPointConstraint::glob_to_loc_ghosts() {
  return _glob_to_loc_ghosts;
}

// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <petscsys.h>

namespace dolfinx_mpc
{

class ContactConstraint
{

public:
  /// Create contact constraint
  ///
  /// @param[in] V The function space
  /// @param[in] slaves List of local slave dofs
  /// @param[in] slaves_cells List of local slave cells
  /// @param[in] offsets Offsets for local slave cells

  ContactConstraint(std::shared_ptr<const dolfinx::function::FunctionSpace> V,
                    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_slaves,
                    std::int32_t num_local_slaves);

  /// Return the array of master dofs and corresponding coefficients
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves()
  {
    return _slaves->array();
  };

  /// Return point to array of all unique local cells containing a slave
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_cells()
  {
    return _slave_cells->array();
  };

  /// Return map from slave to cells containing that slave
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> slave_to_cells()
  {
    return _slave_to_cells_map;
  }

  /// Return map from cell to slaves contained in that cell
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> cell_to_slaves()
  {
    return _cell_to_slaves_map;
  }
  /// Return map from slave to masters (global index)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> masters_local()
  {
    return _master_local_map;
  }

  /// Return map from slave to masters (global index)
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coefficients()
  {
    return _coeff_map->array();
  }

  /// Return map from slave to masters (global index)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> owners()
  {
    return _owner_map;
  }

  /// Return number of local slaves
  std::int32_t num_local_slaves() { return _num_local_slaves; }

  /// Return constraint IndexMap
  std::shared_ptr<dolfinx::common::IndexMap> index_map() { return _index_map; }

  /// Return shared indices for the function space of the Contact constraint
  // std::map<std::int32_t, std::set<int>> compute_shared_indices()
  // {
  //   return _V->dofmap()->index_map->compute_shared_indices();
  // }

  /// Create map from cell to slaves and its inverse
  /// @param[in] The local degrees of freedom that will be used to compute
  /// the cell to dof map
  std::pair<
      std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
      std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
  create_cell_maps(Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs);

  /// Add masters for the slaves in the contact constraint.
  /// Identifies which masters are non-local, and creates a new index-map where
  /// These entries are added
  /// @param[in] masters Array of all masters
  /// @param[in] coeffs Coefficients corresponding to each master
  /// @param[in] owners Owners for each master
  /// @param[in] offsets Offsets for masters
  void add_masters(Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
                   Eigen::Array<PetscScalar, Eigen::Dynamic, 1>,
                   Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                   Eigen::Array<std::int32_t, Eigen::Dynamic, 1>);
  /// Helper function for creating new index map
  /// including all masters as ghosts and replacing
  /// _master_block_map and _master_local_map
  void create_new_index_map();

  /// Add sparsity pattern for multi-point constraints to existing
  /// sparsity pattern
  /// @param[in] a bi-linear form for the current variational problem
  /// (The one used to generate input sparsity-pattern).
  dolfinx::la::SparsityPattern
  create_sparsity_pattern(const dolfinx::fem::Form<PetscScalar>& a);

  std::shared_ptr<dolfinx::fem::DofMap> dofmap() { return _dofmap; }

private:
  // Original function space
  std::shared_ptr<const dolfinx::function::FunctionSpace> _V;
  // Array including all slaves (local + ghosts)
  std::shared_ptr<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> _slaves;
  // Array for all cells containing slaves (local + ghosts)
  std::shared_ptr<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> _slave_cells;
  // Map from slave cell to index in _slaves for a given slave cell
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_to_slaves_map;
  // Map from slave (local+ghosts) to cells it is in
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _slave_to_cells_map;
  // Number of local slaves
  std::int32_t _num_local_slaves;
  // Map from local slave to masters (global index)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> _master_map;
  // Map from local slave to coefficients
  std::shared_ptr<dolfinx::graph::AdjacencyList<PetscScalar>> _coeff_map;
  // Map from local slave to rank of process owning master
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _owner_map;
  // Map from local slave to master (local index)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_local_map;
  // Map from local slave to master block (local)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_block_map;
  // Index map including masters
  std::shared_ptr<dolfinx::common::IndexMap> _index_map;
  // Dofmap including masters
  std::shared_ptr<dolfinx::fem::DofMap> _dofmap;
};
} // namespace dolfinx_mpc
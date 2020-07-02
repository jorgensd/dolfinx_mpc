// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx_mpc
{
/// This class provides the interface for setting multi-point
/// constraints.
///
///   u_i = u_j,
///
/// where u_i and u_j denotes the i-th and j-th global degree of freedom
/// in the corresponding function space. A MultiPointBC is specified by
/// the function space (trial space), and a vector of pairs, connecting
/// master and slave nodes with a linear dependency.

class MultiPointConstraint
{

public:
  /// Create multi-point constraint
  ///
  /// @param[in] V The functionspace on which the multipoint constraint
  /// condition is applied
  /// @param[in] slaves List of slave nodes
  /// @param[in] masters List of master nodes, listed such as all the
  /// master nodes for the first coefficients comes first, then the
  /// second slave node etc.
  /// @param[in] coefficients List of coefficients corresponding to the
  /// it-h master node
  /// @param[in] offsets Gives the starting point for the i-th slave
  /// node in the master and coefficients list.
  MultiPointConstraint(
      std::shared_ptr<const dolfinx::function::FunctionSpace> V,
      Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves,
      Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters,
      Eigen::Array<double, Eigen::Dynamic, 1> coefficients,
      Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets,
      Eigen::Array<std::int32_t, Eigen::Dynamic, 1> master_owner_ranks);

  /// Add sparsity pattern for multi-point constraints to existing
  /// sparsity pattern
  /// @param[in] a bi-linear form for the current variational problem
  /// (The one used to generate input sparsity-pattern).
  dolfinx::la::SparsityPattern
  create_sparsity_pattern(const dolfinx::fem::Form& a);

  /// Generate indexmap including MPC ghosts
  std::shared_ptr<dolfinx::common::IndexMap> generate_index_map();

  /// Return array of slave coefficients
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves() { return _slaves; };

  /// Return array of slave coefficients
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves_local()
  {
    return _slaves_local;
  };

  /// Return local_indices of master coefficients
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> masters_local()
  {
    return _masters_local;
  };

  // Local indices of cells containing slave coefficients
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slave_cells()
  {
    return _slave_cells;
  };

  /// Return the index_map for the test and trial space
  std::shared_ptr<dolfinx::common::IndexMap> index_map() { return _index_map; };

  /// Return the array of master dofs and corresponding coefficients
  Eigen::Array<double, Eigen::Dynamic, 1> coefficients()
  {
    return _coefficients;
  };

  /// Return map from cell with slaves to the slave degrees of freedom
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>
  slave_cell_to_dofs()
  {
    return _cells_to_dofs[0];
  };

  // Return the rank of the owner of the master dof
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  master_owner_ranks()
  {
    return _master_owner_ranks;
  }

  /// Return dofmap with MPC ghost values.
  std::shared_ptr<dolfinx::fem::DofMap> mpc_dofmap() { return _mpc_dofmap; };

private:
  std::shared_ptr<const dolfinx::function::FunctionSpace> _function_space;
  std::shared_ptr<dolfinx::common::IndexMap> _index_map;
  std::shared_ptr<dolfinx::fem::DofMap> _mpc_dofmap;

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _slaves;

  // AdjacencyList for master and local master
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> _masters;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _masters_local;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _slaves_local;

  Eigen::Array<double, Eigen::Dynamic, 1> _coefficients;

  // AdjacencyLists for slave cells and master cells
  Eigen::Array<std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>,
               Eigen::Dynamic, 1>
      _cells_to_dofs;
  // Cell to _slaves index (will have same offsets as _cells to dofs)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>
      _cell_to_slave_index;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _slave_cells;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _master_cells;

  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_owner_ranks;
};

} // namespace dolfinx_mpc

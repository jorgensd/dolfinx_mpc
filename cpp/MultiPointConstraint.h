// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN-X MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/common/IndexMap.h>
#include <Eigen/Dense>
#include "utils.h"

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
  MultiPointConstraint(std::shared_ptr<const dolfinx::function::FunctionSpace> V,
                       Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves, Eigen::Array<std::int64_t, Eigen::Dynamic, 1> masters,
                       Eigen::Array<double, Eigen::Dynamic, 1> coefficients, Eigen::Array<std::int64_t, Eigen::Dynamic, 1> offsets);

  /// Add sparsity pattern for multi-point constraints to existing
  /// sparsity pattern
  /// @param[in] a bi-linear form for the current variational problem
  /// (The one used to generate input sparsity-pattern).
  dolfinx::la::SparsityPattern create_sparsity_pattern(const dolfinx::fem::Form &a);

  /// Generate indexmap including MPC ghosts
  std::shared_ptr<dolfinx::common::IndexMap> generate_index_map();

  /// Return array of slave coefficients
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slaves();

  // Local indices of cells containing slave coefficients
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> slave_cells();

  /// Return the index_map for the test and trial space
  std::shared_ptr<dolfinx::common::IndexMap> index_map();

  /// Return master offset data
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> master_offsets();

  /// Return the array of master dofs and corresponding coefficients
  std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>, Eigen::Array<double, Eigen::Dynamic, 1>>
  masters_and_coefficients();

  /// Return map from cell with slaves to the dof numbers
  std::pair< Eigen::Array<std::int64_t, Eigen::Dynamic, 1>, Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
  cell_to_slave_mapping();

  /// Return the global to local mapping of a master coefficient it it
  /// is not on this processor
  std::unordered_map<int, int> glob_to_loc_ghosts();

  /// Return dofmap with MPC ghost values.
  std::shared_ptr<dolfinx::fem::DofMap> mpc_dofmap()
	{return _mpc_dofmap;};

private:
  std::shared_ptr<const dolfinx::function::FunctionSpace> _function_space;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _slaves;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _masters;
  Eigen::Array<double, Eigen::Dynamic, 1> _coefficients;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _offsets_master;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _slave_cells;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _offsets_cell_to_slave;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _cell_to_slave;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _master_cells;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _offsets_cell_to_master;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _cell_to_master;
  std::unordered_map<int, int> _glob_to_loc_ghosts;
  std::unordered_map<int, int> _glob_master_to_loc_ghosts;
  std::shared_ptr<dolfinx::common::IndexMap> _index_map;
  std::shared_ptr<dolfinx::fem::DofMap> _mpc_dofmap;
};



} // namespace dolfinx_mpc

// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN-X (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/common/IndexMap.h>

namespace dolfinx_mpc
{
  /// This class provides the interface for setting multi-point constraints.
  ///
  ///   u_i = u_j,
  ///
  /// where u_i and u_j denotes the i-th and j-th global degree of freedom in the corresponding
  /// function space.
  /// A MultiPointBC is specified by the function space (trial space), and a vector of pairs, connecting
  /// master and slave nodes with a linear dependency.

  class MultiPointConstraint
  {

  public:
  /// Create multi-point constraint
  ///
  /// @param[in] V The functionspace on which the multipoint constraint
  /// condition is applied
  /// @param[in] slaves List of slave nodes
  /// @param[in] masters List of master nodes, listed such as all the master nodes for the first coefficients
  /// comes first, then the second slave node etc.
  /// @param[in] coefficients List of coefficients corresponding to the it-h master node
  /// @param[in] offsets Gives the starting point for the i-th slave node in the master and coefficients list.
  MultiPointConstraint(std::shared_ptr<const dolfinx::function::FunctionSpace> V,
						 std::vector<std::int64_t> slaves, std::vector<std::int64_t> masters, std::vector<double> coefficients, std::vector<std::int64_t> offsets);


	// Masters for the i-th slave node
  std::vector<std::int64_t> masters(std::int64_t i);

	// Master coefficients for the i-th slave node
  std::vector<double> coefficients(std::int64_t i);

  // Local indices of cells containing slave coefficients
  std::vector<std::int64_t> slave_cells();


  /// Return an array containing the local index of all slave cells
  void find_slave_cells();


  /// Add sparsity pattern for multi-point constraints to existing sparsity pattern
  /// @param[in] a       bi-linear form for the current variational problem (The one used to generate input sparsity-pattern).
  /// @param[in] pattern Existing sparsity pattern.
  dolfinx::la::PETScMatrix generate_petsc_matrix(const dolfinx::fem::Form& a);

  /// Return array of slave coefficients
  std::vector<std::int64_t> slaves();

  /// Return sparsity pattern for modified matrix
  std::shared_ptr<dolfinx::la::SparsityPattern> sparsity_pattern();

  /// Return master offset data
  std::vector<std::int64_t> master_offsets();

  /// Return the array of master dofs and corresponding coefficients
  std::pair<std::vector<std::int64_t>, std::vector<double>>
    masters_and_coefficients();

  /// Return map from cell with slaves to the dof numbers
  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
	cell_to_slave_mapping();

  /// Return the global to local mapping of a master coefficient it it is not on this processor
  std::unordered_map<int, int> glob_to_loc_ghosts();

  private:
	std::shared_ptr<const dolfinx::function::FunctionSpace> _function_space;
	std::vector<std::int64_t> _slaves;
    std::vector<std::int64_t> _masters;
	std::vector<double> _coefficients;
	std::vector<std::int64_t> _offsets;
	std::vector<std::int64_t> _slave_cells;
	std::vector<std::int64_t> _normal_cells;
	std::vector<std::int64_t> _offsets_cell_to_slave;
	std::vector<std::int64_t> _cell_to_slave;
	std::vector<std::int64_t> _master_cells;
	std::vector<std::int64_t> _offsets_cell_to_master;
	std::vector<std::int64_t> _cell_to_master;
	std::unordered_map<int, int> _glob_to_loc_ghosts;
	std::shared_ptr<dolfinx::la::SparsityPattern> _pattern;
   };

}

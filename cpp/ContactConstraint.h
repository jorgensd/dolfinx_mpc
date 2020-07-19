// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
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
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves() { return _slaves; };

  /// Return point to array of all unique local cells containing a slave
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_cells()
  {
    return _slave_cells;
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
  std::shared_ptr<dolfinx::graph::AdjacencyList<PetscScalar>> coefficients()
  {
    return _coeff_map;
  }

  /// Return map from slave to masters (global index)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> owners()
  {
    return _owner_map;
  }

  /// Return number of local slaves
  std::int32_t num_local_slaves() { return _num_local_slaves; }

  /// Return shared indices for the function space of the Contact constraint

  std::map<std::int32_t, std::set<int>> compute_shared_indices()
  {
    return _V->dofmap()->index_map->compute_shared_indices();
  }

  /// Create map from cell to slaves and its inverse
  /// @param[in] The local degrees of freedom that will be used to compute
  /// the cell to dof map
  std::pair<
      std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
      std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
  create_cell_maps(Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs);

  void add_masters(Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
                   Eigen::Array<PetscScalar, Eigen::Dynamic, 1>,
                   Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                   Eigen::Array<std::int32_t, Eigen::Dynamic, 1>);

private:
  std::shared_ptr<const dolfinx::function::FunctionSpace> _V;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _slaves;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _slave_cells;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_to_slaves_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _slave_to_cells_map;
  std::int32_t _num_local_slaves;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>> _master_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<PetscScalar>> _coeff_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _owner_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_local_map;
};
} // namespace dolfinx_mpc
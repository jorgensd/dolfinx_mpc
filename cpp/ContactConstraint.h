// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx_mpc
{

class ContactConstraint
{

public:
  /// Create contact constraint
  ///
  /// @param[in] slaves_cells List of local slave cells
  /// @param[in] offsets Offsets for local slave cells

  ContactConstraint(Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_slaves,
                    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves_cells,
                    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets);

  /// Return the array of master dofs and corresponding coefficients
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves() { return _slaves; };

  /// Return map from slave to cells containing that slave
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> slave_cells()
  {
    return _slave_cells;
  };

private:
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _slaves;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _slave_cells;
};
} // namespace dolfinx_mpc
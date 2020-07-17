// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ContactConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/graph/AdjacencyList.h>
#include <iostream>

using namespace dolfinx_mpc;

ContactConstraint::ContactConstraint(
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_cells,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets)
    : _slaves(slaves), _slave_cells()
{
  _slave_cells = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      slave_cells, offsets);
}
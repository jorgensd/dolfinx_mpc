// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "mpc_helpers.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>

namespace dolfinx_mpc
{
template <typename T, std::floating_point U>
class MultiPointConstraint

{

public:
  /// Create contact constraint
  ///
  /// @param[in] V The function space
  /// @param[in] slaves List of local slave dofs
  /// @param[in] masters Array of all masters
  /// @param[in] coeffs Coefficients corresponding to each master
  /// @param[in] owners Owners for each master
  /// @param[in] offsets Offsets for masters
  /// @tparam The floating type of the mesh
  MultiPointConstraint(std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
                       std::span<const std::int32_t> slaves,
                       std::span<const std::int64_t> masters,
                       std::span<const T> coeffs,
                       std::span<const std::int32_t> owners,
                       std::span<const std::int32_t> offsets)
      : _slaves(), _is_slave(), _cell_to_slaves_map(), _num_local_slaves(),
        _master_map(), _coeff_map(), _owner_map(), _mpc_constants(), _V()
  {
    assert(slaves.size() == offsets.size() - 1);
    assert(masters.size() == coeffs.size());
    assert(coeffs.size() == owners.size());
    assert(offsets.back() == owners.size());

    // Create list indicating which dofs on the process are slaves
    const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
    const std::int32_t num_dofs_local
        = dofmap.index_map_bs()
          * (dofmap.index_map->size_local() + dofmap.index_map->num_ghosts());
    std::vector<std::int8_t> _slave_data(num_dofs_local, 0);
    _mpc_constants = std::vector<T>(num_dofs_local, 0);
    for (auto dof : slaves)
    {
      _slave_data[dof] = 1;
      // FIXME: Add input vector for this data
      _mpc_constants[dof] = 1;
    }
    _is_slave = std::move(_slave_data);

    // Create a map for cells owned by the process to the slaves
    _cell_to_slaves_map = create_cell_to_dofs_map(*V, slaves);

    // Create adjacency list with all local dofs, where the slave dofs maps to
    // its masters
    std::vector<std::int32_t> _num_masters(num_dofs_local);
    std::ranges::fill(_num_masters, 0);
    for (std::int32_t i = 0; i < slaves.size(); i++)
      _num_masters[slaves[i]] = offsets[i + 1] - offsets[i];
    std::vector<std::int32_t> masters_offsets(num_dofs_local + 1);
    masters_offsets[0] = 0;
    std::inclusive_scan(_num_masters.begin(), _num_masters.end(),
                        masters_offsets.begin() + 1);

    // Reuse num masters as fill position array
    std::ranges::fill(_num_masters, 0);
    std::vector<std::int64_t> _master_data(masters.size());
    std::vector<T> _coeff_data(masters.size());
    std::vector<std::int32_t> _owner_data(masters.size());
    /// Create adjacency lists spanning all local dofs mapping to master dofs,
    /// its owner and the corresponding coefficient
    for (std::size_t i = 0; i < slaves.size(); i++)
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
    for (std::size_t i = 0; i < _is_slave.size(); i++)
      if (_is_slave[i])
        sorted_slaves[c++] = i;
    _slaves = std::move(sorted_slaves);

    const std::int32_t num_local
        = dofmap.index_map_bs() * dofmap.index_map->size_local();
    auto it = std::ranges::lower_bound(_slaves, num_local);
    _num_local_slaves = std::distance(_slaves.begin(), it);

    // Create new function space with extended index map
    _V = std::make_shared<const dolfinx::fem::FunctionSpace<U>>(
        create_extended_functionspace(*V, _master_data, _owner_data));

    // Map global masters to local index in extended function space
    std::vector<std::int32_t> masters_local
        = map_dofs_global_to_local<U>(*_V, _master_data);
    _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        masters_local, masters_offsets);
  }
  //-----------------------------------------------------------------------------
  /// Backsubstitute slave/master constraint for a given function
  void backsubstitution(std::span<T> vector)
  {
    for (auto slave : _slaves)
    {
      auto masters = _master_map->links(slave);
      auto coeffs = _coeff_map->links(slave);
      assert(masters.size() == coeffs.size());
      for (std::size_t k = 0; k < masters.size(); ++k)
      {
        vector[slave]
            += coeffs[k] * vector[masters[k]]; //+ _mpc_constants[slave];
        std::cout << slave << " " << masters[k] << " " << vector[slave] << " "
                  << coeffs[k] << " " << vector[masters[k]] << "\n";
      }
    }
  };

  /// Homogenize slave DoFs (particularly useful for nonlinear problems)
  void homogenize(std::span<T> vector) const
  {
    for (auto slave : _slaves)
      vector[slave] = 0.0;
  };

  /// Return map from cell to slaves contained in that cell
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  cell_to_slaves() const
  {
    return _cell_to_slaves_map;
  }
  /// Return map from slave to masters (local_index)
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  masters() const
  {
    return _master_map;
  }

  /// Return map from slave to coefficients
  std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients() const
  {
    return _coeff_map;
  }

  /// Return map from slave to masters (global index)
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  owners() const
  {
    return _owner_map;
  }

  /// Return of local dofs + num ghosts indicating if a dof is a slave
  std::span<const std::int8_t> is_slave() const
  {
    return std::span<const std::int8_t>(_is_slave);
  }

  /// Return the constant values for the constraint
  const std::vector<T>& constant_values() const { return _mpc_constants; }

  /// Return an array of all slave indices (sorted and local to process)
  const std::vector<std::int32_t>& slaves() const { return _slaves; }

  /// Return number of slaves owned by process
  const std::int32_t num_local_slaves() const { return _num_local_slaves; }

  /// Return the MPC FunctionSpace
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> function_space() const
  {
    return _V;
  }

private:
  // MPC function space
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> _V;

  // Array including all slaves (local + ghosts)
  std::vector<std::int32_t> _slaves;
  std::vector<std::int8_t> _is_slave;

  std::vector<T> _mpc_constants;

  // Map from slave cell to index in _slaves for a given slave cell
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_to_slaves_map;

  // Number of slaves owned by the process
  std::int32_t _num_local_slaves;
  // Map from slave (local to process) to masters (local to process)
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_map;
  // Map from slave (local to process)to coefficients
  std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> _coeff_map;
  // Map from slave( local to process) to rank of process owning master
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> _owner_map;
};
} // namespace dolfinx_mpc
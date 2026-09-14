// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "mpc_helpers.h"
#include <algorithm>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <format>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

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
  /// @param[in] rhs_coeffs Inhomogeneity @f$g@f$ of the constraint, i.e.
  /// @f$u_s = \sum_j c_j u_{m_j} + g_s@f$, for all dofs local to the process
  /// (owned and ghost). Pass an empty span for a homogeneous constraint.
  /// @param[in] bcs Dirichlet conditions on the input space. A master that is
  /// constrained by one of these is removed from the master list of its slave
  /// and its contribution folded into the constraint offset.
  /// @tparam The floating type of the mesh
  MultiPointConstraint(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
      std::span<const std::int32_t> slaves,
      std::span<const std::int64_t> masters, std::span<const T> coeffs,
      std::span<const std::int32_t> owners,
      std::span<const std::int32_t> offsets, std::span<const T> rhs_coeffs = {},
      const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
          bcs = {})
      : _slaves(), _is_slave(), _cell_to_slaves_map(), _num_local_slaves(),
        _master_map(), _coeff_map(), _owner_map(), _mpc_constants(),
        _rhs_coeffs(), _bcs(bcs), _bc_master_map(), _bc_coeff_map(), _V()
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
    if (!rhs_coeffs.empty()
        and rhs_coeffs.size() != static_cast<std::size_t>(num_dofs_local))
    {
      throw std::invalid_argument(
          std::format("rhs_coeffs has {} entries, expected {} (one per dof "
                      "local to the process, owned and ghost)",
                      rhs_coeffs.size(), num_dofs_local));
    }
    _mpc_constants = std::vector<T>(num_dofs_local, 0);
    _rhs_coeffs = std::vector<T>(num_dofs_local, 0);
    if (!rhs_coeffs.empty())
      std::ranges::copy(rhs_coeffs, _rhs_coeffs.begin());
    std::vector<std::int8_t> _slave_data(num_dofs_local, 0);
    for (auto dof : slaves)
      _slave_data[dof] = 1;
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
    _num_local_slaves = std::ranges::distance(_slaves.begin(), it);

    // Create new function space with extended index map
    _V = std::make_shared<const dolfinx::fem::FunctionSpace<U>>(
        create_extended_functionspace(*V, _master_data, _owner_data));

    // Map global masters to local index in extended function space
    std::vector<std::int32_t> masters_local
        = map_dofs_global_to_local<U>(*_V, _master_data);

    // Split masters into those constrained by a Dirichlet condition, whose
    // contribution is folded into the constraint offset, and those that remain
    std::vector<std::int8_t> bc_marker = gather_bc_markers();

    // A dof cannot be prescribed twice. Checking only the slaves keeps this
    // O(num_slaves), and the same condition on masters is what the fold below
    // resolves rather than rejects. The offending dof need not exist on every
    // process, so the verdict is reduced before throwing: an error raised on
    // some processes only would leave the rest waiting in a collective.
    if (!bc_marker.empty())
    {
      int local = std::ranges::any_of(_slaves, [&bc_marker](std::int32_t slave)
                                      { return bc_marker[slave] != 0; })
                      ? 1
                      : 0;
      int global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LOR, V->mesh()->comm());
      if (global != 0)
      {
        throw std::invalid_argument(
            "A dof is both a slave of the multi point constraint and "
            "constrained by a Dirichlet condition. Exclude it from one of the "
            "two.");
      }
    }

    std::vector<std::int32_t> keep_masters, keep_owners, keep_offsets(1, 0);
    std::vector<T> keep_coeffs;
    std::vector<std::int32_t> bc_masters, bc_offsets(1, 0);
    std::vector<T> bc_coeffs;
    keep_masters.reserve(masters_local.size());
    keep_owners.reserve(masters_local.size());
    keep_coeffs.reserve(masters_local.size());
    keep_offsets.reserve(num_dofs_local + 1);
    bc_offsets.reserve(num_dofs_local + 1);
    for (std::int32_t dof = 0; dof < num_dofs_local; ++dof)
    {
      for (std::int32_t j = masters_offsets[dof]; j < masters_offsets[dof + 1];
           ++j)
      {
        if (!bc_marker.empty() and bc_marker[masters_local[j]])
        {
          bc_masters.push_back(masters_local[j]);
          bc_coeffs.push_back(_coeff_data[j]);
        }
        else
        {
          keep_masters.push_back(masters_local[j]);
          keep_coeffs.push_back(_coeff_data[j]);
          keep_owners.push_back(_owner_data[j]);
        }
      }
      keep_offsets.push_back((std::int32_t)keep_masters.size());
      bc_offsets.push_back((std::int32_t)bc_masters.size());
    }

    _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        keep_masters, keep_offsets);
    _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        keep_coeffs, keep_offsets);
    _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        keep_owners, keep_offsets);
    _bc_master_map
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            bc_masters, bc_offsets);
    _bc_coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        bc_coeffs, bc_offsets);

    update_constants();
  }
  //-----------------------------------------------------------------------------
  /// @brief Backsubstitute slave/master constraint for a given function
  ///
  /// Computes @f$u_s = \sum_k c_k u_{m_k} + g_s@f$ for every slave @f$s@f$.
  void backsubstitution(std::span<T> vector)
  {
    for (auto slave : _slaves)
    {
      // Initialise with the constraint offset, then accumulate masters
      vector[slave] = _mpc_constants[slave];
      auto masters = _master_map->links(slave);
      auto coeffs = _coeff_map->links(slave);
      assert(masters.size() == coeffs.size());
      for (std::size_t k = 0; k < masters.size(); ++k)
        vector[slave] += coeffs[k] * vector[masters[k]];
    }
  };

  /// @brief Recompute the constraint offsets from the current Dirichlet data.
  ///
  /// The offset of a slave is the user supplied inhomogeneity plus the
  /// contribution of every master that was eliminated because it is
  /// constrained by a Dirichlet condition. As `dolfinx::fem::DirichletBC::set`
  /// reads the current value of the underlying function, calling this picks up
  /// any change to time dependent boundary data.
  ///
  /// @note Collective. Must be called by every process, and again whenever the
  /// values of the Dirichlet conditions supplied at construction change.
  void update_constants()
  {
    // The offset is only defined for slaves. Zeroing it elsewhere keeps
    // `constant_values` equal to the g of `x = K x_red + g`, so that a value
    // supplied for an unconstrained dof cannot silently perturb the lifting.
    for (std::size_t i = 0; i < _mpc_constants.size(); ++i)
      _mpc_constants[i] = _is_slave[i] ? _rhs_coeffs[i] : T(0);

    if (!_bcs.empty())
    {
      const std::vector<T> g = gather_bc_values();
      for (auto slave : _slaves)
      {
        auto masters = _bc_master_map->links(slave);
        auto coeffs = _bc_coeff_map->links(slave);
        assert(masters.size() == coeffs.size());
        for (std::size_t k = 0; k < masters.size(); ++k)
          _mpc_constants[slave] += coeffs[k] * g[masters[k]];
      }
    }

    // Reduce whether any process carries a non-zero offset, so that callers
    // can skip the lifting pass on every process or on none. The pass reaches
    // collectives, so the decision must not be taken rank-locally.
    int local
        = std::ranges::any_of(_mpc_constants, [](T v) { return v != T(0); })
              ? 1
              : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LOR, _V->mesh()->comm());
    _has_inhomogeneity = global != 0;
  };

  /// @brief Replace the user supplied inhomogeneity @f$g@f$.
  ///
  /// Does not recompute the offsets; call `update_constants` afterwards.
  /// @param[in] rhs_coeffs Inhomogeneity for all dofs local to the process
  void set_rhs_coeffs(std::span<const T> rhs_coeffs)
  {
    if (rhs_coeffs.size() != _rhs_coeffs.size())
    {
      throw std::invalid_argument(
          std::format("rhs_coeffs has {} entries, expected {}",
                      rhs_coeffs.size(), _rhs_coeffs.size()));
    }
    std::ranges::copy(rhs_coeffs, _rhs_coeffs.begin());
  }

  /// @brief Whether any process carries a non-zero constraint offset.
  ///
  /// The value is globally reduced, so it is identical on every process.
  bool has_inhomogeneity() const { return _has_inhomogeneity; }

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

  /// Return map from slave to the masters eliminated by a Dirichlet condition
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  bc_masters() const
  {
    return _bc_master_map;
  }

  /// Return map from slave to the coefficients of the eliminated masters
  std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>
  bc_coefficients() const
  {
    return _bc_coeff_map;
  }

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
  /// @brief Gather Dirichlet values on the extended function space.
  ///
  /// Owned dofs keep their local index in the extended space, so the owned
  /// block is filled directly and a forward scatter supplies the values of
  /// masters owned by another process.
  std::vector<T> gather_bc_values() const
  {
    const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
    const int bs = dofmap.index_map_bs();
    const std::int32_t num_owned = bs * dofmap.index_map->size_local();

    std::vector<T> local(num_owned + bs * dofmap.index_map->num_ghosts(), 0);
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : _bcs)
      bc->set(local, std::nullopt, 1);

    dolfinx::la::Vector<T> g(dofmap.index_map, bs);
    std::ranges::copy_n(local.begin(), num_owned, g.array().begin());
    g.scatter_fwd();
    return std::vector<T>(g.array().begin(), g.array().end());
  }

  /// @brief Gather Dirichlet markers on the extended function space.
  ///
  /// Returns an empty vector when no Dirichlet conditions were supplied.
  std::vector<std::int8_t> gather_bc_markers() const
  {
    if (_bcs.empty())
      return {};

    const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
    const int bs = dofmap.index_map_bs();
    const std::int32_t num_owned = bs * dofmap.index_map->size_local();

    std::vector<std::int8_t> local(
        num_owned + bs * dofmap.index_map->num_ghosts(), 0);
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : _bcs)
      bc->mark_dofs(local);

    dolfinx::la::Vector<std::int8_t> marker(dofmap.index_map, bs);
    std::ranges::copy_n(local.begin(), num_owned, marker.array().begin());
    marker.scatter_fwd();
    return std::vector<std::int8_t>(marker.array().begin(),
                                    marker.array().end());
  }

  // MPC function space
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> _V;

  // Array including all slaves (local + ghosts)
  std::vector<std::int32_t> _slaves;
  std::vector<std::int8_t> _is_slave;

  // Constraint offset g (derived: user input plus eliminated Dirichlet masters)
  std::vector<T> _mpc_constants;

  // User supplied inhomogeneity, as given at construction
  std::vector<T> _rhs_coeffs;

  // Globally reduced marker for a non-zero constraint offset
  bool _has_inhomogeneity = false;

  // Dirichlet conditions whose masters have been eliminated
  std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> _bcs;

  // Map from slave (local to process) to the masters eliminated because they
  // are constrained by a Dirichlet condition, and their coefficients
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _bc_master_map;
  std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> _bc_coeff_map;

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
// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "mpc_helpers.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
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
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dolfinx_mpc
{
template <typename T, std::floating_point U>
class MultiPointConstraint;

/// @brief The constraint relations of one function space, as non-owning views.
///
/// See the constructor of `MultiPointConstraint` for the meaning of each field.
template <typename T>
struct mpc_block_view
{
  std::span<const std::int32_t> slaves;
  std::span<const std::int64_t> masters;
  std::span<const T> coeffs;
  std::span<const std::int32_t> owners;
  std::span<const std::int32_t> offsets;
  std::span<const T> rhs_coeffs;
  std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> bcs;
};

/// @brief Create the multi point constraints of several function spaces
/// together.
///
/// Entry `k` of the result constrains `V[k]` with the relations in `data[k]`.
/// The call is collective. The checks are reduced once for all blocks, so that
/// either every process throws or none does.
///
/// @param[in] V Function space of each block. The meshes must be defined on
/// congruent communicators.
/// @param[in] data Constraint of each block.
/// @param[in] filter See `MultiPointConstraint`; applied to every block.
/// @throws std::invalid_argument On a bad argument, identically on every
/// process.
template <typename T, std::floating_point U>
std::vector<std::shared_ptr<MultiPointConstraint<T, U>>>
create_multipointconstraints(
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    const std::vector<mpc_block_view<T>>& data,
    std::optional<U> filter = std::nullopt);

template <typename T, std::floating_point U>
class MultiPointConstraint

{

public:
  /// Create a multi point constraint on a single function space
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
  /// @param[in] filter If set, discard master @f$m_j@f$ of slave @f$s@f$ when
  /// @f$|c_{sj}| < \mathrm{filter}\cdot\max_k|c_{sk}|@f$, the maximum being
  /// over the masters of that same slave. A negligible coefficient contributes
  /// nothing to the constraint, but still costs a ghost, a row of the sparsity
  /// pattern and an entry in every element matrix modification. Unset (the
  /// default) keeps every master supplied.
  /// @note Filtering changes the constraint that is enforced, by exactly the
  /// terms dropped. It is local and performs no communication.
  /// @tparam The floating type of the mesh
  MultiPointConstraint(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
      std::span<const std::int32_t> slaves,
      std::span<const std::int64_t> masters, std::span<const T> coeffs,
      std::span<const std::int32_t> owners,
      std::span<const std::int32_t> offsets, std::span<const T> rhs_coeffs = {},
      const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
          bcs = {},
      std::optional<U> filter = std::nullopt)
      : MultiPointConstraint(
            std::move(*create_multipointconstraints<T, U>(
                           {V},
                           {mpc_block_view<T>{slaves, masters, coeffs, owners,
                                              offsets, rhs_coeffs, bcs}},
                           filter)
                           .front()))
  {
  }
  //-----------------------------------------------------------------------------
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
  }

  void update_constants()
  {
    if (!_has_inhomogeneity)
      return;

    // Bulk zero is faster even if we only have a few slaves, to avoid
    // branching.
    std::ranges::fill(_mpc_constants, T(0));

    // If no BCs, OR if the BCs don't constrain any master DoFs,
    // use the user supplied inhomogeneity `g`.
    if (_bcs.empty())
    {
      for (auto slave : _slaves)
        _mpc_constants[slave] = _rhs_coeffs[slave];
    }
    else
    {
      // Collective gathering of BC values
      const std::vector<T> g = gather_bc_values();

      // Local optimization; if this specific rank has no BC-constrained
      // masters, skip the graph lookups and just apply the rhs_coeffs.
      if (_bc_master_map->offsets().back() == 0)
      {
        for (auto slave : _slaves)
          _mpc_constants[slave] = _rhs_coeffs[slave];
      }
      else
      {
        // Compute g + c_i g_i for every master i that is constrained by a
        // Dirichlet condition
        for (auto slave : _slaves)
        {
          T val = _rhs_coeffs[slave];
          auto masters = _bc_master_map->links(slave);
          auto coeffs = _bc_coeff_map->links(slave);
          assert(masters.size() == coeffs.size());

          for (std::size_t k = 0; k < masters.size(); ++k)
            val += coeffs[k] * g[masters[k]];

          _mpc_constants[slave] = val;
        }
      }
    }
  }

  /// @brief Replace the user supplied inhomogeneity @f$g@f$.
  ///
  /// Does not recompute the offsets; call `update_constants` afterwards.
  /// Throws an error if the constraint was originally created as homogeneous.
  /// @param[in] rhs_coeffs Inhomogeneity for all dofs local to the process
  void set_rhs_coeffs(std::span<const T> rhs_coeffs)
  {
    if (!_has_inhomogeneity)
    {
      throw std::logic_error(
          "Cannot set rhs_coeffs: the multi-point constraint was created "
          "as homogeneous. You must supply an initial rhs_coeffs at creation "
          "to update it later.");
    }
    if (rhs_coeffs.size() != _rhs_coeffs.size())
    {
      throw std::invalid_argument(
          std::format("rhs_coeffs has {} entries, expected {}",
                      rhs_coeffs.size(), _rhs_coeffs.size()));
    }
    std::ranges::copy(rhs_coeffs, _rhs_coeffs.begin());
  }

  /// @brief Coefficients of all masters per local dof, including masters
  /// eliminated by a Dirichlet condition, in the order supplied at
  /// construction.
  ///
  /// This is the layout taken by `update_coefficients`.
  /// @return (coefficients, offsets), where the coefficients of dof `i` are
  /// `coefficients[offsets[i]:offsets[i+1]]`
  std::pair<std::vector<T>, std::vector<std::int32_t>> all_coefficients() const
  {
    if (_all_to_split.empty())
      return {_coeff_map->array(), _coeff_map->offsets()};

    const std::vector<T>& keep = _coeff_map->array();
    const std::vector<T>& bc = _bc_coeff_map->array();
    std::vector<T> coeffs(_all_to_split.size());
    std::ranges::transform(_all_to_split, coeffs.begin(),
                           [&keep, &bc](std::int32_t k)
                           { return k >= 0 ? keep[k] : bc[-k - 1]; });
    return {std::move(coeffs), _all_offsets};
  }

  /// @brief Masters (local index in the MPC function space) in the layout of
  /// `all_coefficients`.
  std::vector<std::int32_t> all_masters() const
  {
    if (_all_to_split.empty())
      return _master_map->array();

    const std::vector<std::int32_t>& keep = _master_map->array();
    const std::vector<std::int32_t>& bc = _bc_master_map->array();
    std::vector<std::int32_t> masters(_all_to_split.size());
    std::ranges::transform(_all_to_split, masters.begin(),
                           [&keep, &bc](std::int32_t k)
                           { return k >= 0 ? keep[k] : bc[-k - 1]; });
    return masters;
  }

  /// @brief Replace the coefficient of every master, including masters
  /// eliminated by a Dirichlet condition, and recompute the constraint
  /// offsets.
  ///
  /// The masters are fixed at construction; a master dropped by `tol` or
  /// `filter` cannot be given a coefficient. Coefficients are updated in
  /// place.
  /// @param[in] coeffs New coefficients in the layout of `all_coefficients`,
  /// for all dofs local to the process (owned and ghost)
  /// @note Collective if the constraint has an inhomogeneity and Dirichlet
  /// conditions.
  void update_coefficients(std::span<const T> coeffs)
  {
    std::vector<T>& keep = _coeff_map->array();
    if (_all_to_split.empty())
    {
      if (coeffs.size() != keep.size())
      {
        throw std::invalid_argument(std::format(
            "coeffs has {} entries, expected {}", coeffs.size(), keep.size()));
      }
      std::ranges::copy(coeffs, keep.begin());
    }
    else
    {
      if (coeffs.size() != _all_to_split.size())
      {
        throw std::invalid_argument(
            std::format("coeffs has {} entries, expected {}", coeffs.size(),
                        _all_to_split.size()));
      }
      std::vector<T>& bc = _bc_coeff_map->array();
      for (std::size_t j = 0; j < coeffs.size(); ++j)
      {
        const std::int32_t k = _all_to_split[j];
        if (k >= 0)
          keep[k] = coeffs[j];
        else
          bc[-k - 1] = coeffs[j];
      }
    }
    update_constants();
  }

  /// @brief Multiply the coefficients of every master of slave @f$s@f$ by
  /// `factors[s]`, and recompute the constraint offsets.
  ///
  /// Applies to masters eliminated by a Dirichlet condition as well. The user
  /// supplied inhomogeneity is not scaled. Repeated calls compound.
  /// @param[in] factors Factor for every dof local to the process (owned and
  /// ghost). Only entries of slaves are read.
  /// @note Collective if the constraint has an inhomogeneity and Dirichlet
  /// conditions.
  void scale_coefficients(std::span<const T> factors)
  {
    if (factors.size() != _is_slave.size())
    {
      throw std::invalid_argument(
          std::format("factors has {} entries, expected {} (one per dof local "
                      "to the process, owned and ghost)",
                      factors.size(), _is_slave.size()));
    }
    for (std::int32_t slave : _slaves)
    {
      for (T& c : _coeff_map->links(slave))
        c *= factors[slave];
      for (T& c : _bc_coeff_map->links(slave))
        c *= factors[slave];
    }
    update_constants();
  }

  /// @brief Whether any process carries a non-zero constraint offset.
  ///
  /// The value is globally reduced at construction, so it is identical on every
  /// process.
  bool has_inhomogeneity() const { return _has_inhomogeneity; }

  /// Homogenize slave DoFs (particularly useful for  nonlinear problems)
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
  template <typename T2, std::floating_point U2>
  friend std::vector<std::shared_ptr<MultiPointConstraint<T2, U2>>>
  create_multipointconstraints(
      const std::vector<
          std::shared_ptr<const dolfinx::fem::FunctionSpace<U2>>>&,
      const std::vector<mpc_block_view<T2>>&, std::optional<U2>);

  /// Empty constraint, filled in by `create_multipointconstraints`
  MultiPointConstraint() = default;

  /// Verdicts of the checks that need the whole communicator. Each is local
  /// here and reduced by `create_multipointconstraints`.
  struct checks
  {
    int slave_is_bc = 0;
    int master_is_slave = 0;
    int unmapped_master = 0;
    int inhomogeneous = 0;
  };

  /// Build the constraint of one function space, without the collective
  /// verdicts: those are returned for `create_multipointconstraints` to
  /// reduce. The constraint is not usable until it has done so.
  checks
  init(const dolfinx::fem::FunctionSpace<U>& V,
       std::span<const std::int32_t> slaves,
       std::span<const std::int64_t> masters, std::span<const T> coeffs,
       std::span<const std::int32_t> owners,
       std::span<const std::int32_t> offsets, std::span<const T> rhs_coeffs,
       std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> bcs)
  {
    _bcs = std::move(bcs);
    // Create list indicating which dofs on the process are slaves
    const dolfinx::fem::DofMap& dofmap = *(V.dofmap());
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

    // Create a map from every cell of the mesh, ghosts included, to the
    // slaves it contains
    _cell_to_slaves_map = create_cell_to_dofs_map(V, slaves);

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
        create_extended_functionspace(V, _master_data, _owner_data));

    // Map global masters to local index in extended function space
    std::vector<std::int32_t> masters_local
        = map_dofs_global_to_local<U>(*_V, _master_data);

    // Every master must have a local index in the extended space, or it has
    // been mapped through the index map of the wrong function space
    checks flags;
    flags.unmapped_master
        = std::ranges::any_of(masters_local,
                              [](std::int32_t m) { return m < 0; })
              ? 1
              : 0;

    // A master that is itself a slave is not resolved by `backsubstitution`,
    // which makes a single pass. Slaves are known on the owner of a dof, so
    // forward the marker to the ghosts of the extended map.
    if (!flags.unmapped_master)
    {
      dolfinx::la::Vector<std::int8_t> slave_marker(
          _V->dofmap()->index_map, _V->dofmap()->index_map_bs());
      std::ranges::copy(_is_slave, slave_marker.array().begin());
      slave_marker.scatter_fwd();
      std::span<const std::int8_t> marked = slave_marker.array();
      flags.master_is_slave
          = std::ranges::any_of(masters_local, [marked](std::int32_t m)
                                { return marked[m] != 0; })
                ? 1
                : 0;
    }

    // Split masters into those constrained by a Dirichlet condition, whose
    // contribution is folded into the constraint offset, and those that remain
    std::vector<std::int8_t> bc_marker = gather_bc_markers();

    // Prevent double-constrained DoFs. Checking slaves only keeps this
    // O(num_slaves) (Dirichlet conditions on masters aren't errors; their
    // values are substituted into the equations later). The verdict is local;
    // the factory reduces it before anything is thrown.
    if (!bc_marker.empty())
    {
      flags.slave_is_bc
          = std::ranges::any_of(
                _slaves,
                [&bc_marker](std::int32_t slave)
                {
                  assert(slave < static_cast<std::int32_t>(bc_marker.size()));
                  return bc_marker[slave] != 0;
                })
                ? 1
                : 0;
    }

    // Transfer masters that are constrained by a Dirichlet condition to
    // separate adjacency lists, and keep the rest in the original adjacency
    // lists
    std::vector<std::int32_t> keep_masters, keep_owners, keep_offsets;
    std::vector<T> keep_coeffs;
    std::vector<std::int32_t> bc_masters, bc_offsets;
    std::vector<T> bc_coeffs;

    if (bc_marker.empty())
    {
      keep_masters = std::move(masters_local);
      keep_coeffs = std::move(_coeff_data);
      keep_owners = std::move(_owner_data);
      keep_offsets = std::move(masters_offsets);
      bc_offsets = std::vector<std::int32_t>(num_dofs_local + 1, 0);
    }
    else
    {
      // FIX 3: Moved reserves here so we don't allocate memory we throw away
      keep_masters.reserve(masters_local.size());
      keep_owners.reserve(masters_local.size());
      keep_coeffs.reserve(masters_local.size());
      keep_offsets.reserve(num_dofs_local + 1);
      bc_offsets.reserve(num_dofs_local + 1);

      // FIX 2: Initialize with a single 0 here, instead of (1, 0) in the
      // constructor
      keep_offsets.push_back(0);
      bc_offsets.push_back(0);

      _all_to_split.reserve(masters_local.size());
      for (std::int32_t dof = 0; dof < num_dofs_local; ++dof)
      {
        const std::int32_t start = masters_offsets[dof];
        const std::int32_t end = masters_offsets[dof + 1];

        for (std::int32_t j = start; j < end; ++j)
        {
          const auto master = masters_local[j];

          if (bc_marker[master])
          {
            // Negative to avoid duplicate storage of the indices
            // If k=_all_to_split[j]<0 then its coefficient is
            // stored at -k-1 in _bc_coeff_map
            _all_to_split.push_back(
                -static_cast<std::int32_t>(bc_masters.size()) - 1);
            bc_masters.push_back(master);
            bc_coeffs.push_back(_coeff_data[j]);
          }
          else
          {
            // Coeff stored as k=_all_to_split[j]>=0 in _coeff_map->array()
            _all_to_split.push_back(
                static_cast<std::int32_t>(keep_masters.size()));
            keep_masters.push_back(master);
            keep_coeffs.push_back(_coeff_data[j]);
            keep_owners.push_back(_owner_data[j]);
          }
        }
        keep_offsets.push_back(static_cast<std::int32_t>(keep_masters.size()));
        bc_offsets.push_back(static_cast<std::int32_t>(bc_masters.size()));
      }
      if (bc_masters.empty())
        _all_to_split.clear();
      else
        _all_offsets = std::move(masters_offsets);
    }
    // Whether a master was eliminated by a Dirichlet condition has to be read
    // before bc_coeffs is moved from below.
    const bool eliminated_bc_masters = !bc_coeffs.empty();

    // AdjacencyList takes (U&& data, V&& offsets) by forwarding reference, so
    // an lvalue is copied. Move instead: each array is large, and the copies
    // would double peak memory right before the extended index map is built.
    // Each offsets array is shared by several lists, so only its last use
    // moves.
    _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        std::move(keep_masters), keep_offsets);
    _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        std::move(keep_coeffs), keep_offsets);
    _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        std::move(keep_owners), std::move(keep_offsets));
    _bc_master_map
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(bc_masters), bc_offsets);
    _bc_coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        std::move(bc_coeffs), std::move(bc_offsets));

    // Whether this process carries an inhomogeneity: one supplied by the user,
    // or the contribution of a master eliminated by a Dirichlet condition. The
    // factory reduces it, so that every process agrees.
    flags.inhomogeneous
        = (!rhs_coeffs.empty() or eliminated_bc_masters) ? 1 : 0;
    return flags;
  }

  /// Record the globally reduced inhomogeneity, and compute the offsets
  void finalize_offsets(bool has_inhomogeneity)
  {
    _has_inhomogeneity = has_inhomogeneity;
    update_constants();
  }

  /// @brief Gather Dirichlet values on the extended function space.
  ///
  /// Owned dofs keep their local index in the extended space, so the owned
  /// block is filled directly and a forward scatter supplies the values of
  /// masters owned by another process.
  std::vector<T> gather_bc_values() const
  {
    const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
    const int bs = dofmap.index_map_bs();
    dolfinx::la::Vector<T> g(dofmap.index_map, bs);
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : _bcs)
      bc->set(g.array(), std::nullopt, 1);
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

    dolfinx::la::Vector<std::int8_t> marker(dofmap.index_map, bs);
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : _bcs)
      bc->mark_dofs(marker.array());
    marker.scatter_fwd();
    return std::vector<std::int8_t>(marker.array().begin(),
                                    marker.array().end());
  }

  // MPC function space
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> _V;

  // Array including all slaves (local + ghosts)
  std::vector<std::int32_t> _slaves;
  std::vector<std::int8_t> _is_slave;

  // Constraint offset g (derived: user input plus eliminated Dirichlet
  // masters)
  std::vector<T> _mpc_constants;

  // User supplied inhomogeneity, as given at construction
  std::vector<T> _rhs_coeffs;

  // Marker for a non-zero constraint offset
  bool _has_inhomogeneity = false;

  // Dirichlet conditions whose masters have been eliminated
  std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> _bcs;

  // Map from slave (local to process) to the masters eliminated because they
  // are constrained by a Dirichlet condition, and their coefficients
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _bc_master_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<T>> _bc_coeff_map;

  // Per local dof, all masters (kept and eliminated) in the order supplied at
  // construction. Entry j of that layout is stored at _all_to_split[j] in
  // _coeff_map if non-negative, else at -_all_to_split[j]-1 in _bc_coeff_map.
  // Both are empty when no master was split off, as the layout then is that
  // of _coeff_map.
  std::vector<std::int32_t> _all_offsets;
  std::vector<std::int32_t> _all_to_split;

  // Map from slave cell to index in _slaves for a given slave cell
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_to_slaves_map;

  // Number of slaves owned by the process
  std::int32_t _num_local_slaves = 0;
  // Map from slave (local to process) to masters (local to process)
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _master_map;
  // Map from slave (local to process) to coefficients. Non-const so that
  // coefficients can be updated in place, keeping views into them valid.
  std::shared_ptr<dolfinx::graph::AdjacencyList<T>> _coeff_map;
  // Map from slave( local to process) to rank of process owning master
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> _owner_map;
};

template <typename T, std::floating_point U>
std::vector<std::shared_ptr<MultiPointConstraint<T, U>>>
create_multipointconstraints(
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    const std::vector<mpc_block_view<T>>& data, std::optional<U> filter)
{
  const std::size_t nb = V.size();
  if (nb == 0 or data.size() != nb)
  {
    throw std::invalid_argument(
        std::format("Expected one constraint per function space, got {} "
                    "spaces and {} constraints.",
                    nb, data.size()));
  }

  // Everything up to the first collective call is local and depends only on
  // replicated information, so a throw is the same on every process. Meshes
  // duplicate their communicator, so equality of handles is too strict, while
  // MPI_SIMILAR would permute the ranks that `owners` refers to.
  for (std::size_t k = 1; k < nb; ++k)
  {
    int result = MPI_UNEQUAL;
    MPI_Comm_compare(V[0]->mesh()->comm(), V[k]->mesh()->comm(), &result);
    if (result != MPI_IDENT and result != MPI_CONGRUENT)
    {
      throw std::invalid_argument(
          std::format("The meshes of function space 0 and {} are on "
                      "communicators that differ in size or rank order.",
                      k));
    }
  }

  if (filter.has_value() and (!(*filter >= 0) or !std::isfinite(*filter)))
  {
    throw std::invalid_argument(
        std::format("filter must be finite and non-negative, got {}", *filter));
  }

  for (std::size_t k = 0; k < nb; ++k)
  {
    // Every condition must be defined on the block's space, or on a subspace
    // of it. A condition on another block would be marked into an array sized
    // for this one.
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc :
         data[k].bcs)
    {
      if (!bc or !bc->function_space())
        throw std::invalid_argument("Dirichlet condition is empty.");
      if (!V[k]->contains(*bc->function_space()))
      {
        throw std::invalid_argument(std::format(
            "A Dirichlet condition given to the multi point constraint is not "
            "defined on the constraint's function space (block {}), nor on a "
            "subspace of it. Pass each block only the conditions that belong "
            "to it.",
            k));
      }
    }
  }

  std::vector<std::shared_ptr<MultiPointConstraint<T, U>>> mpcs;
  mpcs.reserve(nb);
  std::vector<int> local_flags;
  local_flags.reserve(4 * nb);
  for (std::size_t k = 0; k < nb; ++k)
  {
    const mpc_block_view<T>& d = data[k];
    assert(d.slaves.size() == d.offsets.size() - 1);
    assert(d.masters.size() == d.coeffs.size());
    assert(d.coeffs.size() == d.owners.size());
    assert(d.offsets.back() == d.owners.size());

    std::span<const std::int64_t> masters = d.masters;
    std::span<const T> coeffs = d.coeffs;
    std::span<const std::int32_t> owners = d.owners;
    std::span<const std::int32_t> offsets = d.offsets;

    // Storage for the filtered constraint, kept alive for as long as the spans
    // below point into it.
    std::vector<std::int64_t> kept_masters;
    std::vector<T> kept_coeffs;
    std::vector<std::int32_t> kept_owners, kept_offsets;
    if (filter.has_value())
    {
      kept_masters.reserve(masters.size());
      kept_coeffs.reserve(coeffs.size());
      kept_owners.reserve(owners.size());
      kept_offsets.reserve(offsets.size());
      kept_offsets.push_back(0);
      for (std::size_t i = 0; i < d.slaves.size(); ++i)
      {
        const std::int32_t begin = offsets[i];
        const std::int32_t end = offsets[i + 1];
        U max_coeff = 0;
        for (std::int32_t j = begin; j < end; ++j)
          max_coeff = std::max(max_coeff, static_cast<U>(std::abs(coeffs[j])));
        // An all-zero row keeps nothing: the relation is then u_s = g_s
        const U threshold = *filter * max_coeff;
        for (std::int32_t j = begin; j < end; ++j)
        {
          if (max_coeff > 0
              and static_cast<U>(std::abs(coeffs[j])) >= threshold)
          {
            kept_masters.push_back(masters[j]);
            kept_coeffs.push_back(coeffs[j]);
            kept_owners.push_back(owners[j]);
          }
        }
        kept_offsets.push_back(static_cast<std::int32_t>(kept_masters.size()));
      }
      spdlog::debug("MPC filter {}: kept {} of {} masters", *filter,
                    kept_masters.size(), masters.size());
      masters = kept_masters;
      coeffs = kept_coeffs;
      owners = kept_owners;
      offsets = kept_offsets;
    }

    // Never skip a block, even one without masters on this process: creating
    // the extended index map is collective, and so is the reduction below
    mpcs.push_back(std::shared_ptr<MultiPointConstraint<T, U>>(
        new MultiPointConstraint<T, U>()));
    const typename MultiPointConstraint<T, U>::checks flags = mpcs.back()->init(
        *V[k], d.slaves, masters, coeffs, owners, offsets, d.rhs_coeffs, d.bcs);
    local_flags.insert(local_flags.end(),
                       {flags.slave_is_bc, flags.master_is_slave,
                        flags.unmapped_master, flags.inhomogeneous});
  }

  // One reduction for every verdict of every block
  std::vector<int> global_flags(local_flags.size());
  MPI_Allreduce(local_flags.data(), global_flags.data(),
                static_cast<int>(local_flags.size()), MPI_INT, MPI_MAX,
                V[0]->mesh()->comm());
  for (std::size_t k = 0; k < nb; ++k)
  {
    if (global_flags[4 * k] != 0)
    {
      throw std::invalid_argument(std::format(
          "A dof is both a slave of the multi point constraint and "
          "constrained by a Dirichlet condition (block {}). Exclude it from "
          "one of the two.",
          k));
    }
    if (global_flags[4 * k + 1] != 0)
    {
      throw std::invalid_argument(std::format(
          "A master of the multi point constraint (block {}) is also a "
          "slave. Constraints cannot be chained: express the slave in terms "
          "of masters that are not constrained.",
          k));
    }
    if (global_flags[4 * k + 2] != 0)
    {
      throw std::invalid_argument(std::format(
          "A master of the multi point constraint (block {}) has no local "
          "index in the extended function space.",
          k));
    }
  }
  for (std::size_t k = 0; k < nb; ++k)
    mpcs[k]->finalize_offsets(global_flags[4 * k + 3] != 0);
  return mpcs;
}
} // namespace dolfinx_mpc

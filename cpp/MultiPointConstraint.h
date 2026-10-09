// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "ConstraintChains.h"
#include "mpc_helpers.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
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
  /// Block of each master, parallel to `masters`: the index in the list of
  /// function spaces whose global numbering `masters` uses. Empty if every
  /// master is in the block of the slaves.
  std::span<const std::int32_t> master_blocks = {};
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
/// @param[in] resolve_chains If true, a master that is itself a slave, in any
/// block, is replaced by the masters of its slave row, until no master is a
/// slave (see `resolve_chains`). If false, such a master is an error.
/// @param[in] round_limit The largest number of rounds of substitution, which
/// is the length of the longest chain. Unset is the global number of slaves.
/// A cycle, which never resolves, raises once the limit is reached.
/// @throws std::invalid_argument On a bad argument, identically on every
/// process.
template <typename T, std::floating_point U>
std::vector<std::shared_ptr<MultiPointConstraint<T, U>>>
create_multipointconstraints(
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    const std::vector<mpc_block_view<T>>& data,
    std::optional<U> filter = std::nullopt, bool resolve_chains = false,
    std::optional<std::int64_t> round_limit = std::nullopt);

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
  /// @note Only for a constraint whose masters are all in its own block; see
  /// the overload taking a vector per block otherwise.
  void backsubstitution(std::span<T> vector)
  {
    if (_cross_block)
    {
      throw std::invalid_argument(
          "The constraint has masters in another block. Pass the vector of "
          "every block to backsubstitution.");
    }
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

  /// @brief Backsubstitute the constraint, with masters in any block.
  ///
  /// Computes @f$u_s = \sum_k c_k u_{m_k} + g_s@f$ for every slave @f$s@f$,
  /// reading each master from the vector of its block.
  /// @param[in,out] vectors The vector of every block, owned and ghost
  /// entries, in the order of the function spaces given to
  /// `create_multipointconstraints`. The ghosts of the blocks holding masters
  /// must be up to date; only the vector of this constraint's block is
  /// changed.
  void backsubstitution(const std::vector<std::span<T>>& vectors)
  {
    if (vectors.size() != _V_all.size())
    {
      throw std::invalid_argument(
          std::format("Expected a vector for each of the {} blocks, got {}.",
                      _V_all.size(), vectors.size()));
    }
    std::span<T> vector = vectors[_block];
    const std::vector<std::int32_t>& offsets = _master_map->offsets();
    for (auto slave : _slaves)
    {
      vector[slave] = _mpc_constants[slave];
      auto masters = _master_map->links(slave);
      auto coeffs = _coeff_map->links(slave);
      for (std::size_t k = 0; k < masters.size(); ++k)
      {
        const std::int32_t block = _master_blocks[offsets[slave] + k];
        vector[slave] += coeffs[k] * vectors[block][masters[k]];
      }
    }
  }

  /// @brief Recompute the constraint offsets from the current Dirichlet data.
  /// @note Collective if a master is eliminated by a Dirichlet condition on
  /// any process.
  void update_constants()
  {
    if (!_has_inhomogeneity)
      return;

    // Bulk zero is faster even if we only have a few slaves, to avoid
    // branching.
    std::ranges::fill(_mpc_constants, T(0));

    // The values of the Dirichlet conditions of every block that holds an
    // eliminated master. Which blocks those are was reduced at construction,
    // so every process gathers the same ones.
    std::vector<std::vector<T>> g(_V_all.size());
    for (std::size_t j = 0; j < _V_all.size(); ++j)
      if (_bc_blocks_used[j])
        g[j] = gather_bc_values(*_V_all[j], _bcs_all[j]);

    // The user offsets of the slaves substituted into the rows by chain
    // resolution, in the blocks that hold one (reduced at construction)
    std::vector<std::vector<T>> r(_V_all.size());
    for (std::size_t j = 0; j < _V_all.size() and _chains; ++j)
      if (_chain_rhs_blocks_used[j])
        r[j] = gather_rhs_values(*_V_all[j], _chains->rhs[j]);

    // g_s + c_i g_i for every master i that is constrained by a Dirichlet
    // condition
    const std::vector<std::int32_t>& offsets = _bc_master_map->offsets();
    for (auto slave : _slaves)
    {
      T val = _rhs_coeffs[slave];
      auto masters = _bc_master_map->links(slave);
      auto coeffs = _bc_coeff_map->links(slave);
      assert(masters.size() == coeffs.size());
      for (std::size_t k = 0; k < masters.size(); ++k)
      {
        const std::int32_t block = _bc_master_blocks[offsets[slave] + k];
        val += coeffs[k] * g[block][masters[k]];
      }
      if (!_chain_rhs_offsets.empty())
      {
        for (std::int32_t k = _chain_rhs_offsets[slave];
             k < _chain_rhs_offsets[slave + 1]; ++k)
        {
          val += (*_chain_rhs_coeffs)[k]
                 * r[_chain_rhs_blocks[k]][_chain_rhs_dofs[k]];
        }
      }
      _mpc_constants[slave] = val;
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
    if (_chains)
      _chains->rhs[_block] = _rhs_coeffs;
  }

  /// @brief Coefficients of all masters per local dof, including masters
  /// eliminated by a Dirichlet condition, in the order supplied at
  /// construction.
  ///
  /// This is the layout taken by `update_coefficients`. With chains resolved,
  /// these are the rows supplied, before substitution.
  /// @return (coefficients, offsets), where the coefficients of dof `i` are
  /// `coefficients[offsets[i]:offsets[i+1]]`
  std::pair<std::vector<T>, std::vector<std::int32_t>> all_coefficients() const
  {
    if (_chains)
    {
      const constraint_rows<T>& gen = _chains->generators[_block];
      return {gen.coeffs, gen.offsets};
    }
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
    if (_chains)
      return _generator_masters;
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

  /// @brief Block of each master in the layout of `all_coefficients`.
  std::vector<std::int32_t> all_master_blocks() const
  {
    if (_chains)
      return _chains->generators[_block].blocks;
    if (_all_to_split.empty())
      return _master_blocks;

    std::vector<std::int32_t> blocks(_all_to_split.size());
    std::ranges::transform(
        _all_to_split, blocks.begin(), [this](std::int32_t k)
        { return k >= 0 ? _master_blocks[k] : _bc_master_blocks[-k - 1]; });
    return blocks;
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
  /// conditions, or if chains were resolved. The chains are then substituted
  /// again, which updates the rows of every block finalized together with
  /// this one; call `update_constants` on the others.
  void update_coefficients(std::span<const T> coeffs)
  {
    if (_chains)
    {
      std::vector<T>& gen = _chains->generators[_block].coeffs;
      if (coeffs.size() != gen.size())
      {
        throw std::invalid_argument(std::format(
            "coeffs has {} entries, expected {}", coeffs.size(), gen.size()));
      }
      std::ranges::copy(coeffs, gen.begin());
      _chains->resolve();
      update_constants();
      return;
    }
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
  /// conditions, or if chains were resolved, as in `update_coefficients`.
  void scale_coefficients(std::span<const T> factors)
  {
    if (factors.size() != _is_slave.size())
    {
      throw std::invalid_argument(
          std::format("factors has {} entries, expected {} (one per dof local "
                      "to the process, owned and ghost)",
                      factors.size(), _is_slave.size()));
    }
    if (_chains)
    {
      constraint_rows<T>& gen = _chains->generators[_block];
      for (std::int32_t slave : _slaves)
        for (std::int32_t j = gen.offsets[slave]; j < gen.offsets[slave + 1];
             ++j)
          gen.coeffs[j] *= factors[slave];
      _chains->resolve();
      update_constants();
      return;
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

  /// @brief The extended function space of every block created together with
  /// this constraint, in the order given to `create_multipointconstraints`.
  const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>&
  function_spaces() const
  {
    return _V_all;
  }

  /// The index of this constraint's block among `function_spaces()`
  int block() const { return _block; }

  /// @brief Block of each master, parallel to `masters()->array()`. A master's
  /// local index is in the extended space of its block.
  std::span<const std::int32_t> master_blocks() const { return _master_blocks; }

  /// Block of each eliminated master, parallel to `bc_masters()->array()`
  std::span<const std::int32_t> bc_master_blocks() const
  {
    return _bc_master_blocks;
  }

  /// @brief Whether a master on any process is in another block than the
  /// slaves. Identical on every process.
  bool has_cross_block_masters() const { return _cross_block; }

private:
  template <typename T2, std::floating_point U2>
  friend std::vector<std::shared_ptr<MultiPointConstraint<T2, U2>>>
  create_multipointconstraints(
      const std::vector<
          std::shared_ptr<const dolfinx::fem::FunctionSpace<U2>>>&,
      const std::vector<mpc_block_view<T2>>&, std::optional<U2>, bool,
      std::optional<std::int64_t>);

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
    int cross_block = 0;
  };

  /// The masters of the local dofs, ordered as in `_master_map`, between the
  /// stages of `create_multipointconstraints`. `global` is in the numbering
  /// of the master's own block.
  using pending_masters = constraint_rows<T>;

  /// @brief First stage of construction: the slaves of this block, and its
  /// masters in global numbering. Local; the masters cannot be given local
  /// indices until the extended space of every block exists.
  pending_masters prepare(const dolfinx::fem::FunctionSpace<U>& V, int block,
                          std::span<const std::int32_t> slaves,
                          std::span<const std::int64_t> masters,
                          std::span<const T> coeffs,
                          std::span<const std::int32_t> owners,
                          std::span<const std::int32_t> offsets,
                          std::span<const std::int32_t> master_blocks,
                          std::span<const T> rhs_coeffs)
  {
    _block = block;
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

    // Offsets over all local dofs, a slave mapping to its masters
    std::vector<std::int32_t> num_masters(num_dofs_local, 0);
    for (std::size_t i = 0; i < slaves.size(); i++)
      num_masters[slaves[i]] = offsets[i + 1] - offsets[i];
    pending_masters pending;
    pending.offsets.resize(num_dofs_local + 1);
    pending.offsets[0] = 0;
    std::inclusive_scan(num_masters.begin(), num_masters.end(),
                        pending.offsets.begin() + 1);

    // Reuse num masters as fill position array
    std::ranges::fill(num_masters, 0);
    pending.global.resize(masters.size());
    pending.coeffs.resize(masters.size());
    pending.owners.resize(masters.size());
    pending.blocks.resize(masters.size());
    for (std::size_t i = 0; i < slaves.size(); i++)
    {
      for (std::int32_t j = 0; j < offsets[i + 1] - offsets[i]; j++)
      {
        const std::int32_t pos
            = pending.offsets[slaves[i]] + num_masters[slaves[i]]++;
        pending.global[pos] = masters[offsets[i] + j];
        pending.coeffs[pos] = coeffs[offsets[i] + j];
        pending.owners[pos] = owners[offsets[i] + j];
        pending.blocks[pos]
            = master_blocks.empty() ? block : master_blocks[offsets[i] + j];
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
    return pending;
  }

  /// @brief Last stage of construction: give every master a local index in
  /// the extended space of its block, and split off those eliminated by a
  /// Dirichlet condition. Local, apart from nothing; the collective verdicts
  /// are returned for `create_multipointconstraints` to reduce.
  /// @param[in] V_all Extended space of every block
  /// @param[in] bcs_all Dirichlet conditions of every block
  /// @param[in] bc_markers Dirichlet markers of every block on its extended
  /// space, empty for a block without conditions
  /// @param[in] slave_markers Slave markers of every block on its extended
  /// space, ghosts included
  /// @param[in] pending The masters from `prepare`
  /// @param[in] has_rhs_coeffs Whether an inhomogeneity was supplied
  checks complete(
      std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>> V_all,
      std::vector<
          std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>
          bcs_all,
      const std::vector<std::vector<std::int8_t>>& bc_markers,
      const std::vector<std::vector<std::int8_t>>& slave_markers,
      pending_masters&& pending, bool has_rhs_coeffs)
  {
    _V_all = std::move(V_all);
    _bcs_all = std::move(bcs_all);
    _V = _V_all[_block];
    const std::int32_t num_dofs_local = _is_slave.size();
    checks flags;

    // Map each master to its local index in the extended space of its block.
    // Mapping through any other block would give a valid but wrong index,
    // as the global numberings of independent spaces overlap.
    std::vector<std::int32_t> masters_local(pending.global.size(), -1);
    for (std::size_t j = 0; j < _V_all.size(); ++j)
    {
      std::vector<std::int64_t> global;
      std::vector<std::size_t> position;
      for (std::size_t i = 0; i < pending.global.size(); ++i)
      {
        if (pending.blocks[i] == static_cast<std::int32_t>(j))
        {
          global.push_back(pending.global[i]);
          position.push_back(i);
        }
      }
      if (global.empty())
        continue;
      std::vector<std::int32_t> local
          = map_dofs_global_to_local<U>(*_V_all[j], global);
      for (std::size_t i = 0; i < position.size(); ++i)
        masters_local[position[i]] = local[i];
    }
    flags.unmapped_master
        = std::ranges::any_of(masters_local,
                              [](std::int32_t m) { return m < 0; })
              ? 1
              : 0;
    flags.cross_block
        = std::ranges::any_of(pending.blocks,
                              [this](std::int32_t b) { return b != _block; })
              ? 1
              : 0;

    // A master that is itself a slave is not resolved by `backsubstitution`,
    // which makes a single pass. The markers were forwarded to the ghosts, so
    // a master owned elsewhere is checked against its owner's slaves.
    if (!flags.unmapped_master)
    {
      for (std::size_t i = 0; i < masters_local.size(); ++i)
      {
        if (slave_markers[pending.blocks[i]][masters_local[i]] != 0)
        {
          flags.master_is_slave = 1;
          break;
        }
      }
    }

    // Prevent double-constrained DoFs. Checking slaves only keeps this
    // O(num_slaves) (Dirichlet conditions on masters aren't errors; their
    // values are substituted into the equations later). The verdict is local;
    // the factory reduces it before anything is thrown.
    const std::vector<std::int8_t>& own_bc_marker = bc_markers[_block];
    if (!own_bc_marker.empty())
    {
      flags.slave_is_bc
          = std::ranges::any_of(_slaves,
                                [&own_bc_marker](std::int32_t slave)
                                {
                                  assert(slave < static_cast<std::int32_t>(
                                             own_bc_marker.size()));
                                  return own_bc_marker[slave] != 0;
                                })
                ? 1
                : 0;
    }

    // Transfer masters that are constrained by a Dirichlet condition of their
    // block to separate adjacency lists, and keep the rest in the original
    // ones
    auto is_bc_master
        = [&bc_markers, &masters_local, &pending, &flags](std::size_t i)
    {
      const std::vector<std::int8_t>& marker = bc_markers[pending.blocks[i]];
      return !flags.unmapped_master and !marker.empty()
             and marker[masters_local[i]] != 0;
    };
    std::vector<std::int32_t> keep_masters, keep_owners, keep_offsets,
        keep_blocks;
    std::vector<T> keep_coeffs;
    std::vector<std::int32_t> bc_masters, bc_offsets, bc_blocks;
    std::vector<T> bc_coeffs;
    keep_masters.reserve(masters_local.size());
    keep_owners.reserve(masters_local.size());
    keep_coeffs.reserve(masters_local.size());
    keep_blocks.reserve(masters_local.size());
    keep_offsets.reserve(num_dofs_local + 1);
    bc_offsets.reserve(num_dofs_local + 1);
    keep_offsets.push_back(0);
    bc_offsets.push_back(0);
    _all_to_split.reserve(masters_local.size());
    for (std::int32_t dof = 0; dof < num_dofs_local; ++dof)
    {
      for (std::int32_t j = pending.offsets[dof]; j < pending.offsets[dof + 1];
           ++j)
      {
        if (is_bc_master(j))
        {
          // Negative to avoid duplicate storage of the indices
          // If k=_all_to_split[j]<0 then its coefficient is
          // stored at -k-1 in _bc_coeff_map
          _all_to_split.push_back(-static_cast<std::int32_t>(bc_masters.size())
                                  - 1);
          bc_masters.push_back(masters_local[j]);
          bc_coeffs.push_back(pending.coeffs[j]);
          bc_blocks.push_back(pending.blocks[j]);
        }
        else
        {
          // Coeff stored as k=_all_to_split[j]>=0 in _coeff_map->array()
          _all_to_split.push_back(
              static_cast<std::int32_t>(keep_masters.size()));
          keep_masters.push_back(masters_local[j]);
          keep_coeffs.push_back(pending.coeffs[j]);
          keep_owners.push_back(pending.owners[j]);
          keep_blocks.push_back(pending.blocks[j]);
        }
      }
      keep_offsets.push_back(static_cast<std::int32_t>(keep_masters.size()));
      bc_offsets.push_back(static_cast<std::int32_t>(bc_masters.size()));
    }
    if (bc_masters.empty())
      _all_to_split.clear();
    else
      _all_offsets = std::move(pending.offsets);

    // The blocks holding an eliminated master, reduced by the factory
    _bc_blocks_used.assign(_V_all.size(), 0);
    for (std::int32_t b : bc_blocks)
      _bc_blocks_used[b] = 1;

    // Whether a master was eliminated by a Dirichlet condition has to be read
    // before bc_coeffs is moved from below.
    const bool eliminated_bc_masters = !bc_coeffs.empty();

    // AdjacencyList takes (U&& data, V&& offsets) by forwarding reference, so
    // an lvalue is copied. Move instead: each array is large. Each offsets
    // array is shared by several lists, so only its last use moves.
    _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        std::move(keep_masters), keep_offsets);
    _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        std::move(keep_coeffs), keep_offsets);
    _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
        std::move(keep_owners), std::move(keep_offsets));
    _master_blocks = std::move(keep_blocks);
    _bc_master_map
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(bc_masters), bc_offsets);
    _bc_coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<T>>(
        std::move(bc_coeffs), std::move(bc_offsets));
    _bc_master_blocks = std::move(bc_blocks);

    // Whether this process carries an inhomogeneity: one supplied by the user,
    // or the contribution of a master eliminated by a Dirichlet condition. The
    // factory reduces it, so that every process agrees.
    flags.inhomogeneous = (has_rhs_coeffs or eliminated_bc_masters) ? 1 : 0;
    return flags;
  }

  /// @brief Record the chains resolved for this block: the local index of
  /// every master supplied and of every offset term, and share the
  /// coefficient arrays with `chains`. Local.
  /// @param[in] chains The chains of every block
  /// @param[in] rows The resolved rows of this block
  /// @param[in] has_rhs Whether each block has a user offset
  /// @return Whether a dof has no local index, and whether an offset term is a
  /// user offset
  std::pair<int, int> complete_chains(std::shared_ptr<chain_group<T>> chains,
                                      const constraint_rows<T>& rows,
                                      const std::vector<int>& has_rhs)
  {
    _chains = std::move(chains);
    _chains->coeffs[_block] = _coeff_map;
    _chains->bc_coeffs[_block] = _bc_coeff_map;
    _chains->resolved_to_split[_block] = _all_to_split;
    _chain_rhs_coeffs = std::make_shared<std::vector<T>>(rows.rhs_coeffs);
    _chains->rhs_coeffs[_block] = _chain_rhs_coeffs;
    _chain_rhs_offsets = rows.rhs_offsets;
    _chain_rhs_blocks = rows.rhs_blocks;

    // Local index of each dof in the extended space of its block
    auto to_local = [this](std::span<const std::int64_t> global,
                           std::span<const std::int32_t> blocks)
    {
      std::vector<std::int32_t> local(global.size(), -1);
      for (std::size_t j = 0; j < _V_all.size(); ++j)
      {
        std::vector<std::int64_t> g;
        std::vector<std::size_t> position;
        for (std::size_t i = 0; i < global.size(); ++i)
        {
          if (blocks[i] == static_cast<std::int32_t>(j))
          {
            g.push_back(global[i]);
            position.push_back(i);
          }
        }
        if (g.empty())
          continue;
        std::vector<std::int32_t> l
            = map_dofs_global_to_local<U>(*_V_all[j], g);
        for (std::size_t i = 0; i < position.size(); ++i)
          local[position[i]] = l[i];
      }
      return local;
    };
    const constraint_rows<T>& gen = _chains->generators[_block];
    _generator_masters = to_local(gen.global, gen.blocks);
    _chain_rhs_dofs = to_local(rows.rhs_global, rows.rhs_blocks);

    _chain_rhs_blocks_used.assign(_V_all.size(), 0);
    for (std::int32_t b : _chain_rhs_blocks)
      _chain_rhs_blocks_used[b] = 1;
    auto unmapped = [](std::int32_t l) { return l < 0; };
    const int is_unmapped
        = std::ranges::any_of(_generator_masters, unmapped)
                  or std::ranges::any_of(_chain_rhs_dofs, unmapped)
              ? 1
              : 0;
    const int user_offset
        = std::ranges::any_of(_chain_rhs_blocks, [&has_rhs](std::int32_t b)
                              { return has_rhs[b] != 0; })
              ? 1
              : 0;
    return {is_unmapped, user_offset};
  }

  /// Record the globally reduced verdicts, and compute the offsets
  void finalize_offsets(bool has_inhomogeneity, bool cross_block,
                        std::span<const int> bc_blocks_used,
                        std::span<const int> chain_rhs_blocks_used)
  {
    _has_inhomogeneity = has_inhomogeneity;
    _cross_block = cross_block;
    std::ranges::copy(bc_blocks_used, _bc_blocks_used.begin());
    if (_chains)
      std::ranges::copy(chain_rhs_blocks_used, _chain_rhs_blocks_used.begin());
    update_constants();
  }

  /// @brief Gather a user offset on an extended function space.
  ///
  /// The owned entries are copied, and a forward scatter supplies the ghosts.
  /// @param[in] V The extended space
  /// @param[in] rhs The offset of the owned and ghost dofs of the original
  /// space, or empty for zero
  static std::vector<T>
  gather_rhs_values(const dolfinx::fem::FunctionSpace<U>& V,
                    std::span<const T> rhs)
  {
    const dolfinx::fem::DofMap& dofmap = *V.dofmap();
    dolfinx::la::Vector<T> r(dofmap.index_map, dofmap.index_map_bs());
    if (!rhs.empty())
    {
      const std::size_t num_owned
          = dofmap.index_map->size_local() * dofmap.index_map_bs();
      std::copy_n(rhs.begin(), num_owned, r.array().begin());
    }
    r.scatter_fwd();
    return std::vector<T>(r.array().begin(), r.array().end());
  }

  /// @brief Gather Dirichlet values on an extended function space.
  ///
  /// Owned dofs keep their local index in the extended space, so the owned
  /// block is filled directly and a forward scatter supplies the values of
  /// masters owned by another process.
  static std::vector<T> gather_bc_values(
      const dolfinx::fem::FunctionSpace<U>& V,
      const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
          bcs)
  {
    const dolfinx::fem::DofMap& dofmap = *V.dofmap();
    dolfinx::la::Vector<T> g(dofmap.index_map, dofmap.index_map_bs());
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : bcs)
      bc->set(g.array(), std::nullopt, 1);
    g.scatter_fwd();
    return std::vector<T>(g.array().begin(), g.array().end());
  }

  /// @brief Gather Dirichlet markers on an extended function space.
  ///
  /// Returns an empty vector when no Dirichlet conditions were supplied.
  static std::vector<std::int8_t> gather_bc_markers(
      const dolfinx::fem::FunctionSpace<U>& V,
      const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
          bcs)
  {
    if (bcs.empty())
      return {};
    const dolfinx::fem::DofMap& dofmap = *V.dofmap();
    dolfinx::la::Vector<std::int8_t> marker(dofmap.index_map,
                                            dofmap.index_map_bs());
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : bcs)
      bc->mark_dofs(marker.array());
    marker.scatter_fwd();
    return std::vector<std::int8_t>(marker.array().begin(),
                                    marker.array().end());
  }

  /// @brief Forward this block's slave marker to the ghosts of an extended
  /// space of this block. Collective.
  std::vector<std::int8_t>
  gather_slave_marker(const dolfinx::fem::FunctionSpace<U>& V_ext) const
  {
    dolfinx::la::Vector<std::int8_t> marker(V_ext.dofmap()->index_map,
                                            V_ext.dofmap()->index_map_bs());
    std::ranges::copy(_is_slave, marker.array().begin());
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

  // Extended space and Dirichlet conditions of every block created together
  // with this constraint, and the index of this one
  std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>> _V_all;
  std::vector<std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>
      _bcs_all;
  int _block = 0;

  // Whether any process has a master in another block (reduced)
  bool _cross_block = false;

  // Per block, whether any process has an eliminated master in it (reduced)
  std::vector<int> _bc_blocks_used;

  // Block of each master, parallel to the data of _master_map and
  // _bc_master_map
  std::vector<std::int32_t> _master_blocks;
  std::vector<std::int32_t> _bc_master_blocks;

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

  // The chains of the constraints finalized together, if resolved. Then the
  // rows supplied are kept there, `all_coefficients` and the updates use
  // them, and every update substitutes the chains again.
  std::shared_ptr<chain_group<T>> _chains;
  // Local index (in the extended space of its block) of every master supplied,
  // in the layout of `all_coefficients`
  std::vector<std::int32_t> _generator_masters;
  // Per local dof, the offset terms of the resolved rows: the slave whose
  // user offset contributes (local index in the extended space of its block),
  // its block and coefficient. Empty without chains.
  std::vector<std::int32_t> _chain_rhs_offsets;
  std::vector<std::int32_t> _chain_rhs_dofs;
  std::vector<std::int32_t> _chain_rhs_blocks;
  std::shared_ptr<std::vector<T>> _chain_rhs_coeffs;
  // Per block, whether any process has an offset term in it (reduced)
  std::vector<int> _chain_rhs_blocks_used;
};

template <typename T, std::floating_point U>
std::vector<std::shared_ptr<MultiPointConstraint<T, U>>>
create_multipointconstraints(
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    const std::vector<mpc_block_view<T>>& data, std::optional<U> filter,
    bool resolve_chains, std::optional<std::int64_t> round_limit)
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
  if (round_limit.has_value() and *round_limit < 0)
  {
    throw std::invalid_argument(
        std::format("round_limit must be non-negative, got {}", *round_limit));
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

  // Stage 1, local: the slaves of each block and its masters in global
  // numbering
  std::vector<std::shared_ptr<MultiPointConstraint<T, U>>> mpcs;
  mpcs.reserve(nb);
  std::vector<typename MultiPointConstraint<T, U>::pending_masters> pending;
  pending.reserve(nb);
  std::vector<int> duplicate_slave(nb, 0), bad_master_block(nb, 0);
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
    std::span<const std::int32_t> master_blocks = d.master_blocks;

    // The block of each master must be one of the blocks. The arrays are local
    // to this process, so a bad one is reported with the other verdicts rather
    // than thrown here, where another process may be entering a collective.
    bad_master_block[k]
        = (!master_blocks.empty() and master_blocks.size() != masters.size())
                  or std::ranges::any_of(
                      master_blocks, [nb](std::int32_t b)
                      { return b < 0 or b >= static_cast<std::int32_t>(nb); })
              ? 1
              : 0;

    // Storage for the filtered constraint, kept alive for as long as the spans
    // below point into it.
    std::vector<std::int64_t> kept_masters;
    std::vector<T> kept_coeffs;
    std::vector<std::int32_t> kept_owners, kept_offsets, kept_blocks;
    if (filter.has_value())
    {
      kept_masters.reserve(masters.size());
      kept_coeffs.reserve(coeffs.size());
      kept_owners.reserve(owners.size());
      kept_blocks.reserve(master_blocks.size());
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
            if (!master_blocks.empty())
              kept_blocks.push_back(master_blocks[j]);
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
      master_blocks = kept_blocks;
    }

    // A slave constrained twice, as when two periodic conditions share a
    // corner, has no meaning and would corrupt the offsets below. Build the
    // block empty instead of throwing, so that this process still takes part
    // in the collective calls, and report it with the other verdicts. The same
    // goes for a bad master block.
    std::vector<std::int32_t> sorted_slaves(d.slaves.begin(), d.slaves.end());
    std::ranges::sort(sorted_slaves);
    duplicate_slave[k]
        = std::ranges::adjacent_find(sorted_slaves) != sorted_slaves.end() ? 1
                                                                           : 0;
    static constexpr std::int32_t no_offsets[1] = {0};
    std::span<const std::int32_t> slaves = d.slaves;
    if (duplicate_slave[k] or bad_master_block[k])
    {
      slaves = {};
      masters = {};
      coeffs = {};
      owners = {};
      master_blocks = {};
      offsets = std::span<const std::int32_t>(no_offsets, 1);
    }

    mpcs.push_back(std::shared_ptr<MultiPointConstraint<T, U>>(
        new MultiPointConstraint<T, U>()));
    pending.push_back(mpcs.back()->prepare(*V[k], static_cast<int>(k), slaves,
                                           masters, coeffs, owners, offsets,
                                           master_blocks, d.rhs_coeffs));
  }

  // Stage 1b, collective and only if asked: substitute the chained masters.
  // The rows supplied are kept, for the updates to substitute them again.
  MPI_Comm comm = V[0]->mesh()->comm();
  std::shared_ptr<chain_group<T>> chains;
  if (resolve_chains)
  {
    chains = std::make_shared<chain_group<T>>(comm);
    chains->generators = pending;
    chains->coeffs.resize(nb);
    chains->bc_coeffs.resize(nb);
    chains->resolved_to_split.resize(nb);
    chains->rhs_coeffs.resize(nb);
    std::int64_t num_slaves = 0;
    for (std::size_t k = 0; k < nb; ++k)
    {
      chains->is_slave.push_back(mpcs[k]->_is_slave);
      const dolfinx::common::IndexMap& imap = *V[k]->dofmap()->index_map;
      const int bs = V[k]->dofmap()->index_map_bs();
      chains->owned.push_back(
          {imap.local_range()[0] * bs, imap.local_range()[1] * bs});
      chains->rhs.emplace_back(data[k].rhs_coeffs.begin(),
                               data[k].rhs_coeffs.end());
      num_slaves += mpcs[k]->_num_local_slaves;
    }
    // A chain without cycles has at most one round per slave
    if (round_limit.has_value())
      chains->round_limit = *round_limit;
    else
    {
      MPI_Allreduce(&num_slaves, &chains->round_limit, 1, MPI_INT64_T, MPI_SUM,
                    comm);
    }
    std::vector<std::span<const std::int8_t>> markers(chains->is_slave.begin(),
                                                      chains->is_slave.end());
    std::vector<constraint_rows<T>> resolved = dolfinx_mpc::resolve_chains<T>(
        comm, chains->generators, markers, chains->owned, chains->round_limit);
    for (std::size_t k = 0; k < nb; ++k)
      pending[k] = std::move(resolved[k]);
  }

  // Stage 2, collective and in block order: the extended space of each block,
  // with every master that lives in it, whichever block its slave is in. With
  // chains, the masters supplied and the slaves of the offset terms as well.
  // Never skip a block, even one without masters on this process.
  std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>> V_ext;
  V_ext.reserve(nb);
  for (std::size_t j = 0; j < nb; ++j)
  {
    std::vector<std::int64_t> global;
    std::vector<std::int32_t> owners;
    auto add = [j, &global, &owners](std::span<const std::int64_t> g,
                                     std::span<const std::int32_t> o,
                                     std::span<const std::int32_t> b)
    {
      for (std::size_t i = 0; i < g.size(); ++i)
      {
        if (b[i] == static_cast<std::int32_t>(j))
        {
          global.push_back(g[i]);
          owners.push_back(o[i]);
        }
      }
    };
    for (const auto& p : pending)
    {
      add(p.global, p.owners, p.blocks);
      add(p.rhs_global, p.rhs_owners, p.rhs_blocks);
    }
    if (chains)
      for (const constraint_rows<T>& p : chains->generators)
        add(p.global, p.owners, p.blocks);
    V_ext.push_back(std::make_shared<const dolfinx::fem::FunctionSpace<U>>(
        create_extended_functionspace(*V[j], global, owners)));
  }

  // The Dirichlet and slave markers of every block, forwarded to the ghosts of
  // its extended space. Collective, in block order.
  std::vector<std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>
      bcs_all(nb);
  std::vector<std::vector<std::int8_t>> bc_markers(nb), slave_markers(nb);
  for (std::size_t j = 0; j < nb; ++j)
  {
    bcs_all[j] = data[j].bcs;
    bc_markers[j]
        = MultiPointConstraint<T, U>::gather_bc_markers(*V_ext[j], bcs_all[j]);
    slave_markers[j] = mpcs[j]->gather_slave_marker(*V_ext[j]);
  }

  // Stage 3, local: local indices of the masters, and the verdicts. Per block:
  // duplicate slave, slave is a Dirichlet dof, master is a slave, master
  // unmapped, inhomogeneous, cross-block masters, bad master block, then one
  // flag per block for whether an eliminated master lives there.
  // With chains, one more flag per block for whether an offset term lives
  // there.
  const std::size_t stride = 7 + 2 * nb;
  std::vector<int> local_flags(stride * nb, 0);
  std::vector<int> has_rhs(nb);
  for (std::size_t k = 0; k < nb; ++k)
    has_rhs[k] = data[k].rhs_coeffs.empty() ? 0 : 1;
  for (std::size_t k = 0; k < nb; ++k)
  {
    // The offset terms are not part of the masters, and are kept for chains
    constraint_rows<T> offset_terms;
    if (chains)
    {
      offset_terms.rhs_offsets = std::move(pending[k].rhs_offsets);
      offset_terms.rhs_global = std::move(pending[k].rhs_global);
      offset_terms.rhs_coeffs = std::move(pending[k].rhs_coeffs);
      offset_terms.rhs_blocks = std::move(pending[k].rhs_blocks);
    }
    typename MultiPointConstraint<T, U>::checks flags
        = mpcs[k]->complete(V_ext, bcs_all, bc_markers, slave_markers,
                            std::move(pending[k]), has_rhs[k] != 0);
    int* f = local_flags.data() + stride * k;
    if (chains)
    {
      const auto [unmapped, user_offset]
          = mpcs[k]->complete_chains(chains, offset_terms, has_rhs);
      flags.unmapped_master = std::max(flags.unmapped_master, unmapped);
      flags.inhomogeneous = std::max(flags.inhomogeneous, user_offset);
      std::ranges::copy(mpcs[k]->_chain_rhs_blocks_used, f + 7 + nb);
    }
    f[0] = duplicate_slave[k];
    f[1] = flags.slave_is_bc;
    f[2] = flags.master_is_slave;
    f[3] = flags.unmapped_master;
    f[4] = flags.inhomogeneous;
    f[5] = flags.cross_block;
    f[6] = bad_master_block[k];
    std::ranges::copy(mpcs[k]->_bc_blocks_used, f + 7);
  }

  // One reduction for every verdict of every block
  std::vector<int> global_flags(local_flags.size());
  MPI_Allreduce(local_flags.data(), global_flags.data(),
                static_cast<int>(local_flags.size()), MPI_INT, MPI_MAX, comm);
  for (std::size_t k = 0; k < nb; ++k)
  {
    const int* f = global_flags.data() + stride * k;
    if (f[6] != 0)
    {
      throw std::invalid_argument(std::format(
          "Block {}: the master blocks must hold one entry per master, each "
          "one of the {} blocks.",
          k, nb));
    }
    if (f[0] != 0)
    {
      throw std::invalid_argument(std::format(
          "A dof is the slave of more than one constraint (block {}). Give "
          "each slave one relation, for instance with a single indicator for "
          "a domain that is periodic in several directions.",
          k));
    }
    if (f[1] != 0)
    {
      throw std::invalid_argument(std::format(
          "A dof is both a slave of the multi point constraint and "
          "constrained by a Dirichlet condition (block {}). Exclude it from "
          "one of the two.",
          k));
    }
    if (f[2] != 0)
    {
      throw std::invalid_argument(std::format(
          "A master of the multi point constraint (block {}) is also a "
          "slave. Finalize with resolve_chains=True to substitute the chain, "
          "or express the slave in terms of masters that are not "
          "constrained.",
          k));
    }
    if (f[3] != 0)
    {
      throw std::invalid_argument(std::format(
          "A master of the multi point constraint (block {}) has no local "
          "index in the extended function space of its block.",
          k));
    }
  }
  for (std::size_t k = 0; k < nb; ++k)
  {
    const int* f = global_flags.data() + stride * k;
    mpcs[k]->finalize_offsets(f[4] != 0, f[5] != 0,
                              std::span<const int>(f + 7, nb),
                              std::span<const int>(f + 7 + nb, nb));
  }
  return mpcs;
}
} // namespace dolfinx_mpc

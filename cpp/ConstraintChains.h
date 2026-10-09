// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "mpi_utils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <map>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace impl
{
/// A term of a constraint row
template <typename T>
struct row_term
{
  std::int32_t block;
  std::int64_t global;
  std::int32_t owner;
  T coeff;
};

/// @brief Order terms by (block, global dof), summing the coefficients of
/// equal dofs in the order they were given, so that the result does not
/// depend on the partition.
template <typename T>
void merge_terms(std::vector<row_term<T>>& terms)
{
  std::ranges::stable_sort(terms, {}, [](const row_term<T>& t)
                           { return std::pair(t.block, t.global); });
  std::size_t n = 0;
  for (std::size_t i = 0; i < terms.size(); ++i)
  {
    if (n > 0 and terms[n - 1].block == terms[i].block
        and terms[n - 1].global == terms[i].global)
    {
      terms[n - 1].coeff += terms[i].coeff;
    }
    else
      terms[n++] = terms[i];
  }
  terms.resize(n);
}
} // namespace impl

namespace dolfinx_mpc
{
/// @brief The masters of every local dof (owned and ghost) of one block, in
/// global numbering.
///
/// Dof `d` has the terms `[offsets[d], offsets[d + 1])`: the dof
/// `global[i]` of block `blocks[i]`, owned by process `owners[i]`, with
/// coefficient `coeffs[i]`. The offset terms `rhs_*`, laid out by
/// `rhs_offsets`, are the slaves substituted into the row by
/// `resolve_chains`: offset term `i` adds `rhs_coeffs[i]` times the offset
/// @f$g@f$ supplied for slave `rhs_global[i]` of block `rhs_blocks[i]`. They
/// are empty unless chains were resolved.
template <typename T>
struct constraint_rows
{
  std::vector<std::int32_t> offsets;
  std::vector<std::int64_t> global;
  std::vector<T> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> blocks;
  std::vector<std::int32_t> rhs_offsets;
  std::vector<std::int64_t> rhs_global;
  std::vector<T> rhs_coeffs;
  std::vector<std::int32_t> rhs_owners;
  std::vector<std::int32_t> rhs_blocks;
};

/// @brief Substitute the row of every slave wherever it is a master, until no
/// master is a slave.
///
/// In a row @f$u_s = \sum_m c_{sm} u_m + g_s@f$, a master @f$m@f$ with row
/// @f$u_m = \sum_n c_{mn} u_n + g_m@f$ becomes @f$\sum_n c_{sm} c_{mn} u_n@f$,
/// and @f$c_{sm}@f$ times @f$g_m@f$ is recorded as an offset term. A round
/// substitutes every chained master once, with the rows of `generators`, so a
/// chain of `L` slaves after the first takes `L` rounds. A substituted row is
/// ordered by (block, global dof), with repeated masters merged; the other
/// rows are returned as given.
///
/// @param[in] comm The communicator of every block
/// @param[in] generators The rows of every block, without offset terms
/// @param[in] is_slave The slave marker of every block, owned and ghost dofs
/// @param[in] owned The global range of the dofs owned by this process, in
/// every block
/// @param[in] round_limit The largest number of rounds
/// @return The resolved rows of every block
/// @throws std::invalid_argument On every process, if a master is still a
/// slave after `round_limit` rounds: the constraints have a cycle, or a chain
/// longer than the limit.
/// @note Collective. Every round makes two neighbourhood exchanges and one
/// reduction.
template <typename T>
std::vector<constraint_rows<T>>
resolve_chains(MPI_Comm comm, const std::vector<constraint_rows<T>>& generators,
               const std::vector<std::span<const std::int8_t>>& is_slave,
               const std::vector<std::array<std::int64_t, 2>>& owned,
               std::int64_t round_limit)
{
  const int rank = dolfinx::MPI::rank(comm);
  const std::size_t nb = generators.size();

  using key_t = std::pair<std::int32_t, std::int64_t>;
  auto local_index = [&owned](const key_t& k) -> std::int64_t
  {
    const std::int64_t l = k.second - owned[k.first][0];
    return (l >= 0 and k.second < owned[k.first][1]) ? l : -1;
  };

  // The generator row of every slave met so far, and the dofs known not to be
  // slaves. Generator rows are fixed, so an answer is never asked again.
  std::map<key_t, std::vector<impl::row_term<T>>> slave_rows;
  std::map<key_t, bool> known;

  auto generator_row = [&generators](std::int32_t block, std::int32_t dof)
  {
    const constraint_rows<T>& g = generators[block];
    std::vector<impl::row_term<T>> row;
    for (std::int32_t j = g.offsets[dof]; j < g.offsets[dof + 1]; ++j)
      row.push_back({g.blocks[j], g.global[j], g.owners[j], g.coeffs[j]});
    return row;
  };

  std::vector<constraint_rows<T>> rows = generators;
  for (constraint_rows<T>& r : rows)
    r.rhs_offsets.assign(r.offsets.size(), 0);

  for (std::int64_t round = 0;; ++round)
  {
    // Ask the owner of every master not met before whether it is a slave
    std::vector<int> dest;
    std::vector<std::int64_t> query;
    for (const constraint_rows<T>& r : rows)
    {
      for (std::size_t i = 0; i < r.global.size(); ++i)
      {
        const key_t k(r.blocks[i], r.global[i]);
        if (known.contains(k))
          continue;
        if (r.owners[i] == rank)
        {
          const std::int64_t l = local_index(k);
          const bool slave = l >= 0 and is_slave[k.first][l] != 0;
          known[k] = slave;
          if (slave)
            slave_rows[k]
                = generator_row(k.first, static_cast<std::int32_t>(l));
        }
        else
        {
          // Recorded as pending, so that it is asked only once
          known[k] = false;
          dest.push_back(r.owners[i]);
          query.insert(query.end(), {k.first, k.second});
        }
      }
    }
    const auto [recv_query, recv_unused, source]
        = impl::send_rows<std::int64_t, T>(comm, dest, query, 2, {}, 0);

    // Answer with the generator row of each slave: [block, dof, kind, master
    // block, master, owner], coefficient. Kind 0 is a master of the slave, 1
    // marks a slave (even one without masters), 2 a dof that is not a slave.
    std::vector<int> reply_dest;
    std::vector<std::int64_t> reply;
    std::vector<T> reply_coeffs;
    for (std::size_t j = 0; j < source.size(); ++j)
    {
      const key_t k(static_cast<std::int32_t>(recv_query[2 * j]),
                    recv_query[2 * j + 1]);
      const std::int64_t l = local_index(k);
      const bool slave = l >= 0 and is_slave[k.first][l] != 0;
      reply_dest.push_back(source[j]);
      reply.insert(reply.end(), {k.first, k.second, slave ? 1 : 2, 0, 0, 0});
      reply_coeffs.push_back(T(0));
      if (!slave)
        continue;
      for (const impl::row_term<T>& t :
           generator_row(k.first, static_cast<std::int32_t>(l)))
      {
        reply_dest.push_back(source[j]);
        reply.insert(reply.end(), {k.first, k.second, 0, t.block, t.global,
                                   static_cast<std::int64_t>(t.owner)});
        reply_coeffs.push_back(t.coeff);
      }
    }
    const auto [answers, answer_coeffs, answer_source]
        = impl::send_rows<std::int64_t, T>(comm, reply_dest, reply, 6,
                                           reply_coeffs, 1);
    for (std::size_t j = 0; j < answer_source.size(); ++j)
    {
      const std::int64_t* a = answers.data() + 6 * j;
      const key_t k(static_cast<std::int32_t>(a[0]), a[1]);
      if (a[2] == 1)
      {
        known[k] = true;
        slave_rows[k];
      }
      else if (a[2] == 0)
      {
        slave_rows[k].push_back({static_cast<std::int32_t>(a[3]), a[4],
                                 static_cast<std::int32_t>(a[5]),
                                 answer_coeffs[j]});
      }
    }

    // Whether a master is still a slave anywhere
    std::int64_t num_chained = 0;
    for (const constraint_rows<T>& r : rows)
      for (std::size_t i = 0; i < r.global.size(); ++i)
        num_chained += known.at({r.blocks[i], r.global[i]}) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &num_chained, 1, MPI_INT64_T, MPI_SUM, comm);
    if (num_chained == 0)
      return rows;
    if (round == round_limit)
    {
      throw std::invalid_argument(std::format(
          "A master of the multi point constraint is still a slave after {} "
          "rounds of substitution. The constraints contain a cycle, or a chain "
          "longer than round_limit={}.",
          round, round_limit));
    }

    // Substitute the chained masters of every row
    for (std::size_t b = 0; b < nb; ++b)
    {
      const constraint_rows<T>& r = rows[b];
      constraint_rows<T> out;
      const std::size_t num_dofs = r.offsets.size() - 1;
      out.offsets.reserve(num_dofs + 1);
      out.rhs_offsets.reserve(num_dofs + 1);
      out.offsets.push_back(0);
      out.rhs_offsets.push_back(0);
      std::vector<impl::row_term<T>> terms, rhs_terms;
      for (std::size_t d = 0; d < num_dofs; ++d)
      {
        terms.clear();
        rhs_terms.clear();
        for (std::int32_t j = r.rhs_offsets[d]; j < r.rhs_offsets[d + 1]; ++j)
        {
          rhs_terms.push_back({r.rhs_blocks[j], r.rhs_global[j],
                               r.rhs_owners[j], r.rhs_coeffs[j]});
        }
        bool chained = false;
        for (std::int32_t j = r.offsets[d]; j < r.offsets[d + 1]; ++j)
        {
          const key_t k(r.blocks[j], r.global[j]);
          if (!known.at(k))
          {
            terms.push_back(
                {r.blocks[j], r.global[j], r.owners[j], r.coeffs[j]});
            continue;
          }
          chained = true;
          for (const impl::row_term<T>& t : slave_rows.at(k))
            terms.push_back(
                {t.block, t.global, t.owner, r.coeffs[j] * t.coeff});
          rhs_terms.push_back(
              {r.blocks[j], r.global[j], r.owners[j], r.coeffs[j]});
        }
        if (chained)
        {
          impl::merge_terms(terms);
          impl::merge_terms(rhs_terms);
        }
        for (const impl::row_term<T>& t : terms)
        {
          out.global.push_back(t.global);
          out.coeffs.push_back(t.coeff);
          out.owners.push_back(t.owner);
          out.blocks.push_back(t.block);
        }
        for (const impl::row_term<T>& t : rhs_terms)
        {
          out.rhs_global.push_back(t.global);
          out.rhs_coeffs.push_back(t.coeff);
          out.rhs_owners.push_back(t.owner);
          out.rhs_blocks.push_back(t.block);
        }
        out.offsets.push_back(static_cast<std::int32_t>(out.global.size()));
        out.rhs_offsets.push_back(
            static_cast<std::int32_t>(out.rhs_global.size()));
      }
      rows[b] = std::move(out);
    }
  }
}

/// @brief The chains of the constraints finalized together, kept to substitute
/// them again when the coefficients of a generator row change.
///
/// Shared by the constraints of every block. It writes the resolved
/// coefficients into the coefficient arrays of every block, which those
/// constraints share with it, so that a change of one block reaches the rows
/// of the others that chain through it.
template <typename T>
struct chain_group
{
  /// @param[in] comm Communicator of the blocks, duplicated
  explicit chain_group(MPI_Comm comm) : comm(comm) {}

  /// Communicator of the blocks
  dolfinx::MPI::Comm comm;
  /// The rows supplied for every block, in global numbering
  std::vector<constraint_rows<T>> generators;
  /// The slave marker of every block, owned and ghost dofs
  std::vector<std::vector<std::int8_t>> is_slave;
  /// The global range of the owned dofs of every block
  std::vector<std::array<std::int64_t, 2>> owned;
  /// The largest number of rounds
  std::int64_t round_limit = 0;
  /// The user offset of every block, owned and ghost dofs, empty for none
  std::vector<std::vector<T>> rhs;

  /// Per block, the coefficients of the masters kept and of those eliminated
  /// by a Dirichlet condition, and the position of each resolved term in
  /// them: `k >= 0` in `coeffs`, else `-k - 1` in `bc_coeffs`. Empty if all
  /// are kept, in the resolved order.
  std::vector<std::shared_ptr<dolfinx::graph::AdjacencyList<T>>> coeffs;
  std::vector<std::shared_ptr<dolfinx::graph::AdjacencyList<T>>> bc_coeffs;
  std::vector<std::vector<std::int32_t>> resolved_to_split;
  /// Per block, the coefficients of the offset terms
  std::vector<std::shared_ptr<std::vector<T>>> rhs_coeffs;

  /// @brief Substitute the chains again, from the current generator rows,
  /// and write the coefficients of every block.
  /// @note Collective.
  void resolve()
  {
    std::vector<std::span<const std::int8_t>> markers(is_slave.begin(),
                                                      is_slave.end());
    const std::vector<constraint_rows<T>> rows = resolve_chains<T>(
        comm.comm(), generators, markers, owned, round_limit);
    for (std::size_t b = 0; b < rows.size(); ++b)
    {
      std::vector<T>& keep = coeffs[b]->array();
      if (resolved_to_split[b].empty())
      {
        assert(keep.size() == rows[b].coeffs.size());
        std::ranges::copy(rows[b].coeffs, keep.begin());
      }
      else
      {
        assert(resolved_to_split[b].size() == rows[b].coeffs.size());
        std::vector<T>& bc = bc_coeffs[b]->array();
        for (std::size_t j = 0; j < rows[b].coeffs.size(); ++j)
        {
          const std::int32_t k = resolved_to_split[b][j];
          if (k >= 0)
            keep[k] = rows[b].coeffs[j];
          else
            bc[-k - 1] = rows[b].coeffs[j];
        }
      }
      assert(rhs_coeffs[b]->size() == rows[b].rhs_coeffs.size());
      std::ranges::copy(rows[b].rhs_coeffs, rhs_coeffs[b]->begin());
    }
  }
};
} // namespace dolfinx_mpc

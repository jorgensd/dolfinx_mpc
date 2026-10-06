// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "utils.h"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <iterator>
#include <limits>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace impl
{
/// @brief Send row `i` of `rows` (`n` entries per row) and of `values` (`m`
/// entries per row) to process `dest[i]`, in one neighbourhood exchange.
/// @return The received rows and values, grouped by source process in
/// ascending order, and the source process of each received row.
/// @note Collective.
template <typename I, typename R>
std::tuple<std::vector<I>, std::vector<R>, std::vector<int>>
send_rows(MPI_Comm comm, std::span<const int> dest, std::span<const I> rows,
          int n, std::span<const R> values, int m)
{
  std::vector<std::size_t> order(dest.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, {},
                           [&dest](std::size_t i) { return dest[i]; });

  std::vector<int> ranks;
  std::vector<int> send_counts;
  std::vector<I> send_rows;
  std::vector<R> send_values;
  send_rows.reserve(rows.size());
  send_values.reserve(values.size());
  for (std::size_t i : order)
  {
    if (ranks.empty() or ranks.back() != dest[i])
    {
      ranks.push_back(dest[i]);
      send_counts.push_back(0);
    }
    ++send_counts.back();
    send_rows.insert(send_rows.end(), std::next(rows.begin(), n * i),
                     std::next(rows.begin(), n * (i + 1)));
    send_values.insert(send_values.end(), std::next(values.begin(), m * i),
                       std::next(values.begin(), m * (i + 1)));
  }
  // Some MPI implementations require non-null pointers for empty arrays
  ranks.reserve(1);
  send_counts.reserve(1);

  // The consensus returns the sources in no particular order
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, ranks);
  std::ranges::sort(src);
  MPI_Comm neighbors;
  MPI_Dist_graph_create_adjacent(comm, static_cast<int>(src.size()), src.data(),
                                 MPI_UNWEIGHTED, static_cast<int>(ranks.size()),
                                 ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &neighbors);
  std::vector<int> recv_counts(src.size());
  recv_counts.reserve(1);
  MPI_Neighbor_alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                        MPI_INT, neighbors);

  auto exchange = [&neighbors, &send_counts, &recv_counts]<typename V>(
                      const std::vector<V>& data, int stride) -> std::vector<V>
  {
    auto scaled = [stride](const std::vector<int>& counts)
    {
      std::pair<std::vector<int>, std::vector<int>> cd;
      cd.first.reserve(std::max<std::size_t>(counts.size(), 1));
      cd.second.assign(counts.size() + 1, 0);
      for (std::size_t i = 0; i < counts.size(); ++i)
      {
        cd.first.push_back(stride * counts[i]);
        cd.second[i + 1] = cd.second[i] + stride * counts[i];
      }
      return cd;
    };
    const auto [sc, sd] = scaled(send_counts);
    const auto [rc, rd] = scaled(recv_counts);
    std::vector<V> recv(rd.back());
    MPI_Neighbor_alltoallv(data.data(), sc.data(), sd.data(),
                           dolfinx::MPI::mpi_t<V>, recv.data(), rc.data(),
                           rd.data(), dolfinx::MPI::mpi_t<V>, neighbors);
    return recv;
  };
  std::vector<I> recv_rows = exchange(send_rows, n);
  std::vector<R> recv_values = exchange(send_values, m);
  MPI_Comm_free(&neighbors);

  std::vector<int> source;
  for (std::size_t s = 0; s < src.size(); ++s)
    source.insert(source.end(), recv_counts[s], src[s]);
  return {std::move(recv_rows), std::move(recv_values), std::move(source)};
}
} // namespace impl

namespace dolfinx_mpc
{
/// @brief The basis functions of a function space at a set of points, each
/// evaluated in a cell containing its point.
///
/// Row `i` holds point `i`: the `num_dofs` basis functions of its cell, and
/// for basis function `j` and component `b` the dof `j * bs + b`.
template <std::floating_point U>
struct point_basis
{
  /// Number of basis functions per point (dofs of a cell, not unrolled)
  int num_dofs = 0;
  /// Block size of the dofs
  int bs = 1;
  /// Whether a cell containing the point was found
  std::vector<std::int8_t> found;
  /// Global index of each dof, shape `(num_points, num_dofs * bs)`
  std::vector<std::int64_t> dofs;
  /// Process owning each dof, shape `(num_points, num_dofs * bs)`
  std::vector<std::int32_t> owners;
  /// Value of each basis function, shape `(num_points, num_dofs)`
  std::vector<U> values;
};

/// @brief Evaluate the basis functions of `V` at points in given cells.
///
/// The dofs are numbered in `parent`, of which `V` may be a collapsed
/// subspace.
///
/// @param[in] V The space, with a scalar-valued (possibly blocked) element
/// @param[in] points The points, shape `(num_points, 3)`, row major
/// @param[in] cells A cell of `V`'s mesh containing each point, or -1 to
/// skip the point
/// @param[in] to_parent The dof in `parent` of each (unrolled) local dof of
/// `V`. Empty if `parent` is `V`.
/// @param[in] parent The space numbering the dofs
/// @param[in] tol Tolerance of the pull-back on non-affine cells
/// @param[in] num_threads The number of threads to use
template <std::floating_point U>
point_basis<U>
evaluate_basis_in_cells(const dolfinx::fem::FunctionSpace<U>& V,
                        std::span<const U> points,
                        std::span<const std::int32_t> cells,
                        std::span<const std::int32_t> to_parent,
                        const dolfinx::fem::FunctionSpace<U>& parent, U tol,
                        std::size_t num_threads)
{
  auto [basis, shape]
      = evaluate_basis_functions<U>(V, points, cells, tol, num_threads);
  if (shape[2] != 1)
  {
    throw std::invalid_argument(
        "Constraints from point evaluation need a scalar-valued element, "
        "possibly blocked");
  }
  const std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  assert(dofmap->bs() == dofmap->index_map_bs());

  point_basis<U> out;
  out.num_dofs = static_cast<int>(dofmap->map().extent(1));
  out.bs = dofmap->index_map_bs();
  const int bs = out.bs;
  const std::size_t width = out.num_dofs * bs;
  out.found.assign(cells.size(), 0);
  out.dofs.assign(cells.size() * width, -1);
  out.owners.assign(cells.size() * width, -1);
  out.values.assign(cells.size() * out.num_dofs, 0);

  const std::shared_ptr<const dolfinx::common::IndexMap> pmap
      = parent.dofmap()->index_map;
  const int pbs = parent.dofmap()->index_map_bs();
  const std::int32_t size_local = pmap->size_local();
  std::span<const int> ghost_owners = pmap->owners();
  const int rank = dolfinx::MPI::rank(pmap->comm());

  // Parent blocks of all dofs, mapped to global indices in one call
  std::vector<std::int32_t> blocks;
  std::vector<std::int32_t> rems;
  blocks.reserve(cells.size() * width);
  rems.reserve(cells.size() * width);
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    if (cells[i] < 0)
      continue;
    out.found[i] = 1;
    std::copy_n(std::next(basis.begin(), i * shape[1]), out.num_dofs,
                std::next(out.values.begin(), i * out.num_dofs));
    for (std::int32_t block : dofmap->cell_dofs(cells[i]))
    {
      for (int b = 0; b < bs; ++b)
      {
        const std::int32_t dof = block * bs + b;
        const std::int32_t pdof = to_parent.empty() ? dof : to_parent[dof];
        blocks.push_back(pdof / pbs);
        rems.push_back(pdof % pbs);
      }
    }
  }
  std::vector<std::int64_t> global(blocks.size());
  pmap->local_to_global(blocks, global);
  std::size_t pos = 0;
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    if (!out.found[i])
      continue;
    for (std::size_t k = 0; k < width; ++k, ++pos)
    {
      out.dofs[i * width + k] = global[pos] * pbs + rems[pos];
      out.owners[i * width + k] = blocks[pos] < size_local
                                      ? rank
                                      : ghost_owners[blocks[pos] - size_local];
    }
  }
  return out;
}

/// @brief Evaluate the basis functions of `V` at points, each in a cell
/// containing it, searched among `cells` on every process.
///
/// A point is evaluated in a local cell if there is one, else in a cell of
/// the lowest-ranked process that has one. The dofs are numbered in
/// `parent`, of which `V` may be a collapsed subspace.
///
/// @param[in] V The space, with a scalar-valued (possibly blocked) element
/// @param[in] cells The cells (local to the process) to search
/// @param[in] points The points, shape `(num_points, 3)`, row major
/// @param[in] padding Padding of the bounding boxes of the cells
/// @param[in] eps2 Largest squared distance from a point to its cell. The
/// pull-back on non-affine cells stops at `max(eps2, 500 eps)`.
/// @param[in] to_parent The dof in `parent` of each (unrolled) local dof of
/// `V`. Empty if `parent` is `V`.
/// @param[in] parent The space numbering the dofs
/// @param[in] num_threads The number of threads to use
/// @note Collective over the communicator of `V`'s mesh.
template <std::floating_point U>
point_basis<U> evaluate_basis_at_points(
    const dolfinx::fem::FunctionSpace<U>& V,
    std::span<const std::int32_t> cells, std::span<const U> points, U padding,
    U eps2, std::span<const std::int32_t> to_parent,
    const dolfinx::fem::FunctionSpace<U>& parent, std::size_t num_threads)
{
  const dolfinx::mesh::Mesh<U>& mesh = *V.mesh();
  MPI_Comm comm = mesh.comm();
  const int rank = dolfinx::MPI::rank(comm);
  const dolfinx::geometry::BoundingBoxTree<U> tree(mesh, mesh.topology()->dim(),
                                                   padding, cells);
  const dolfinx::geometry::BoundingBoxTree<U> process_tree
      = tree.create_global_tree(comm);

  // The pull-back stops when its Newton step, in reference coordinates, is
  // below this. Rounding keeps the step above a multiple of the machine
  // epsilon, which a squared distance such as eps2 = 1e-20 is below.
  const U pull_back_tol
      = std::max(eps2, 500 * std::numeric_limits<U>::epsilon());

  const std::vector<std::int32_t> local
      = find_local_collisions<U>(mesh, tree, points, eps2);
  point_basis<U> out = evaluate_basis_in_cells<U>(
      V, points, local, to_parent, parent, pull_back_tol, num_threads);

  // Ask the processes whose cells' bounding box holds a point not found here
  const std::size_t num_missing = std::ranges::count(out.found, 0);
  std::vector<std::int32_t> missing;
  missing.reserve(num_missing);
  std::vector<U> missing_x;
  missing_x.reserve(3 * num_missing);
  for (std::size_t i = 0; i < out.found.size(); ++i)
  {
    if (!out.found[i])
    {
      missing.push_back(static_cast<std::int32_t>(i));
      missing_x.insert(missing_x.end(), std::next(points.begin(), 3 * i),
                       std::next(points.begin(), 3 * (i + 1)));
    }
  }
  const dolfinx::graph::AdjacencyList<std::int32_t> candidates
      = dolfinx::geometry::compute_collisions<U>(process_tree, missing_x);
  // At most one query per candidate process of each point
  const std::size_t max_queries = candidates.array().size();
  std::vector<int> dest;
  dest.reserve(max_queries);
  std::vector<std::int64_t> query;
  query.reserve(max_queries);
  std::vector<U> query_x;
  query_x.reserve(3 * max_queries);
  for (std::size_t i = 0; i < missing.size(); ++i)
  {
    for (std::int32_t p : candidates.links(i))
    {
      if (p == rank)
        continue;
      dest.push_back(p);
      query.push_back(missing[i]);
      query_x.insert(query_x.end(), std::next(missing_x.begin(), 3 * i),
                     std::next(missing_x.begin(), 3 * (i + 1)));
    }
  }
  const auto [recv_query, recv_x, source]
      = impl::send_rows<std::int64_t, U>(comm, dest, query, 1, query_x, 3);

  // Answer with the basis in a local cell: [point, dofs, owners], values
  const std::vector<std::int32_t> recv_cells
      = find_local_collisions<U>(mesh, tree, recv_x, eps2);
  const point_basis<U> remote = evaluate_basis_in_cells<U>(
      V, recv_x, recv_cells, to_parent, parent, pull_back_tol, num_threads);
  const int width = out.num_dofs * out.bs;
  // At most one reply per query received
  std::vector<int> reply_dest;
  reply_dest.reserve(recv_query.size());
  std::vector<std::int64_t> reply;
  reply.reserve(recv_query.size() * (1 + 2 * width));
  std::vector<U> reply_values;
  reply_values.reserve(recv_query.size() * out.num_dofs);
  for (std::size_t j = 0; j < recv_query.size(); ++j)
  {
    if (!remote.found[j])
      continue;
    reply_dest.push_back(source[j]);
    reply.push_back(recv_query[j]);
    reply.insert(reply.end(), std::next(remote.dofs.begin(), j * width),
                 std::next(remote.dofs.begin(), (j + 1) * width));
    reply.insert(reply.end(), std::next(remote.owners.begin(), j * width),
                 std::next(remote.owners.begin(), (j + 1) * width));
    reply_values.insert(
        reply_values.end(), std::next(remote.values.begin(), j * out.num_dofs),
        std::next(remote.values.begin(), (j + 1) * out.num_dofs));
  }
  const auto [answers, answer_values, answer_source]
      = impl::send_rows<std::int64_t, U>(comm, reply_dest, reply, 1 + 2 * width,
                                         reply_values, out.num_dofs);

  // Answers come by ascending source: keep the first for each point
  for (std::size_t r = 0; r < answer_source.size(); ++r)
  {
    auto row = std::next(answers.begin(), r * (1 + 2 * width));
    const std::size_t i = *row;
    if (out.found[i])
      continue;
    out.found[i] = 1;
    std::copy_n(std::next(row, 1), width,
                std::next(out.dofs.begin(), i * width));
    std::copy_n(std::next(row, 1 + width), width,
                std::next(out.owners.begin(), i * width));
    std::copy_n(std::next(answer_values.begin(), r * out.num_dofs),
                out.num_dofs, std::next(out.values.begin(), i * out.num_dofs));
  }
  return out;
}
} // namespace dolfinx_mpc

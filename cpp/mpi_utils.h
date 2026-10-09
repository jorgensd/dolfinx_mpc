// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfinx_mpc
{
/// @brief Create a MPI-communicator from owners in the index map to the
/// processes with ghosts
///
/// @param[in] map The index map
/// @returns The mpi communicator
MPI_Comm create_owner_to_ghost_comm(const dolfinx::common::IndexMap& map);

std::pair<std::vector<int>, std::vector<int>>
compute_neighborhood(const MPI_Comm& comm);

} // namespace dolfinx_mpc

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

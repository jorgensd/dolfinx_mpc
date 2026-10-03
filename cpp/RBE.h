// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <format>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace dolfinx_mpc
{

/// @brief Global block, owning rank and coordinate of spiders of a
/// spider mesh, found through a post office.
///
/// A spider is a point of a point mesh, named by its input index: the
/// original cell index of the point. The owner of each point's dofs in `W`
/// sends the point's block and the dofs' coordinate to the post office
/// `dolfinx::MPI::index_owner` of the input index, which records the sender
/// as the owner. The rows a
/// process needs are then fetched from the post offices with
/// `dolfinx::MPI::distribute_from_postoffice`. No process holds more than
/// its share of the spiders.
///
/// @param[in] W Space on the point mesh, one block of dofs per point.
/// @param[in] spiders Input indices of the spiders required. May repeat.
/// @return (0) Global block in `W`, (1) owning rank and (2) coordinate
/// (row-major, shape `(spiders.size(), 3)`) of each spider in `spiders`.
/// @note Collective.
template <std::floating_point U>
std::tuple<std::vector<std::int64_t>, std::vector<std::int32_t>, std::vector<U>>
locate_spiders(const dolfinx::fem::FunctionSpace<U>& W,
               std::span<const std::int64_t> spiders)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = W.mesh();
  MPI_Comm comm = mesh->comm();
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  std::shared_ptr<const dolfinx::mesh::Topology> topology = mesh->topology();
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = topology->index_map(topology->dim());
  const std::int32_t num_cells = cell_map->size_local();
  const std::int64_t K = cell_map->size_global();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = W.dofmap();
  if (dofmap->element_dof_layout().num_dofs() != 1)
  {
    throw std::invalid_argument(
        "The space on the spider mesh must have one block of dofs per point");
  }
  std::span<const std::int64_t> original
      = topology->original_cell_index.front();
  std::shared_ptr<const dolfinx::common::IndexMap> imap = dofmap->index_map;
  const std::vector<U> x = W.tabulate_dof_coordinates(false);

  // Publish: each owned point sends (input index, block) and its coordinate
  // to its post office, this process included, in one neighbourhood exchange.
  // The source of a row is the owner of the point.
  std::vector<std::int32_t> published;
  std::vector<int> office;
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::int32_t dof = dofmap->cell_dofs(c).front();
    if (dof < imap->size_local())
    {
      published.push_back(c);
      office.push_back(dolfinx::MPI::index_owner(size, original[c], K));
    }
  }
  std::vector<std::size_t> order(published.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, {},
                           [&office](std::size_t i) { return office[i]; });

  std::vector<int> dest;
  std::vector<int> send_counts;
  std::vector<std::int64_t> send_rows;
  std::vector<U> send_x;
  for (std::size_t i : order)
  {
    const std::int32_t dof = dofmap->cell_dofs(published[i]).front();
    if (dest.empty() or dest.back() != office[i])
    {
      dest.push_back(office[i]);
      send_counts.push_back(0);
    }
    ++send_counts.back();
    send_rows.push_back(original[published[i]]);
    send_rows.push_back(imap->local_range()[0] + dof);
    send_x.insert(send_x.end(), std::next(x.begin(), 3 * dof),
                  std::next(x.begin(), 3 * (dof + 1)));
  }
  // Some MPI implementations require non-null pointers for empty arrays
  dest.reserve(1);
  send_counts.reserve(1);

  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  MPI_Comm neighbors;
  MPI_Dist_graph_create_adjacent(comm, static_cast<int>(src.size()), src.data(),
                                 MPI_UNWEIGHTED, static_cast<int>(dest.size()),
                                 dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &neighbors);
  std::vector<int> recv_counts(src.size());
  recv_counts.reserve(1);
  MPI_Neighbor_alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                        MPI_INT, neighbors);

  // Displacements, in rows
  std::vector<int> send_disp(send_counts.size() + 1, 0);
  std::partial_sum(send_counts.begin(), send_counts.end(),
                   std::next(send_disp.begin()));
  std::vector<int> recv_disp(recv_counts.size() + 1, 0);
  std::partial_sum(recv_counts.begin(), recv_counts.end(),
                   std::next(recv_disp.begin()));
  auto exchange = [&neighbors, &send_counts, &send_disp, &recv_counts,
                   &recv_disp]<typename V>(const std::vector<V>& data,
                                           int stride) -> std::vector<V>
  {
    auto scale = [stride](const std::vector<int>& v)
    {
      std::vector<int> scaled(v.size());
      std::ranges::transform(v, scaled.begin(),
                             [stride](int n) { return stride * n; });
      scaled.reserve(1);
      return scaled;
    };
    const std::vector<int> sc = scale(send_counts), sd = scale(send_disp),
                           rc = scale(recv_counts), rd = scale(recv_disp);
    std::vector<V> recv(rd.back());
    MPI_Neighbor_alltoallv(data.data(), sc.data(), sd.data(),
                           dolfinx::MPI::mpi_t<V>, recv.data(), rc.data(),
                           rd.data(), dolfinx::MPI::mpi_t<V>, neighbors);
    return recv;
  };
  const std::vector<std::int64_t> recv_rows = exchange(send_rows, 2);
  const std::vector<U> recv_x = exchange(send_x, 3);
  MPI_Comm_free(&neighbors);

  // The directory of this post office: (block, owner) and coordinate per row
  const std::array<std::int64_t, 2> range
      = dolfinx::common::local_range(rank, K, size);
  const std::int64_t num_rows = range[1] - range[0];
  std::vector<std::int64_t> directory(2 * num_rows, -1);
  std::vector<U> directory_x(3 * num_rows, 0);
  for (std::size_t s = 0; s < src.size(); ++s)
  {
    for (int i = recv_disp[s]; i < recv_disp[s + 1]; ++i)
    {
      const std::int64_t row = recv_rows[2 * i] - range[0];
      directory[2 * row] = recv_rows[2 * i + 1];
      directory[2 * row + 1] = src[s];
      std::copy_n(std::next(recv_x.begin(), 3 * i), 3,
                  std::next(directory_x.begin(), 3 * row));
    }
  }

  // One reduction for both checks, so that every process throws or none
  std::array<int, 2> failed
      = {std::ranges::any_of(spiders,
                             [K](std::int64_t k) { return k < 0 or k >= K; }),
         std::ranges::any_of(directory, [](std::int64_t v) { return v < 0; })};
  MPI_Allreduce(MPI_IN_PLACE, failed.data(), 2, MPI_INT, MPI_MAX, comm);
  if (failed[0])
  {
    throw std::out_of_range(std::format(
        "A spider index is outside the {} points of the spider mesh", K));
  }
  if (failed[1])
  {
    throw std::runtime_error(
        "A point of the spider mesh has no owned dofs in the space on it");
  }

  // Query: each post office holds a contiguous block of rows, as
  // distribute_from_postoffice expects
  const std::vector<std::int64_t> rows
      = dolfinx::MPI::distribute_from_postoffice(comm, spiders, directory,
                                                 {K, 2}, range[0]);
  std::vector<U> coordinates = dolfinx::MPI::distribute_from_postoffice(
      comm, spiders, directory_x, {K, 3}, range[0]);
  std::vector<std::int64_t> blocks(spiders.size());
  std::vector<std::int32_t> owners(spiders.size());
  for (std::size_t i = 0; i < spiders.size(); ++i)
  {
    blocks[i] = rows[2 * i];
    owners[i] = static_cast<std::int32_t>(rows[2 * i + 1]);
  }
  return {std::move(blocks), std::move(owners), std::move(coordinates)};
}

} // namespace dolfinx_mpc

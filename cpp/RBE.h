// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <format>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace impl
{
/// @brief Check the spaces of an RBE2 constraint.
/// @return (geometric dimension, block size of `W`, whether `W` has
/// rotations)
template <std::floating_point U>
std::tuple<int, int, bool>
check_rbe2_spaces(const dolfinx::fem::FunctionSpace<U>& V,
                  const dolfinx::fem::FunctionSpace<U>& W)
{
  const int gdim = V.mesh()->geometry().dim();
  const int bs = V.dofmap()->index_map_bs();
  if (bs != gdim)
  {
    throw std::invalid_argument(std::format(
        "The tied space must have one component per dimension ({}), it has {}",
        gdim, bs));
  }
  if (W.mesh()->topology()->cell_type() != dolfinx::mesh::CellType::point)
    throw std::invalid_argument(
        "The body of a spider must be a space on a point mesh");
  const int num_body = W.dofmap()->index_map_bs();
  const int num_rigid = gdim == 3 ? 6 : 3;
  if (num_body != gdim and num_body != num_rigid)
  {
    throw std::invalid_argument(
        std::format("The body space must have {} components (translations) or "
                    "{} (translations and rotations), it has {}",
                    gdim, num_rigid, num_body));
  }
  return {gdim, num_body, num_body == num_rigid};
}

/// @brief Coefficient of body component `c` in foot component `j` of
/// @f$u = t + \theta \times r@f$, with @f$r = x - x_c@f$.
///
/// Components below `gdim` are the translation, the others the rotation
/// (3, 4, 5 in 3D, 2 in 2D).
template <std::floating_point U>
U rbe2_coefficient(int gdim, int j, int c, std::span<const U, 3> r)
{
  if (c < gdim)
    return c == j ? 1 : 0;
  // (theta x r)_j = sum_c sign[j][c] r[comp[j][c]] theta_c. In 3D
  // (theta x r)_x = theta_y r_z - theta_z r_y and cyclically, in 2D
  // (-theta r_y, theta r_x).
  constexpr std::array<std::array<int, 6>, 3> sign3
      = {{{0, 0, 0, 0, 1, -1}, {0, 0, 0, -1, 0, 1}, {0, 0, 0, 1, -1, 0}}};
  constexpr std::array<std::array<int, 6>, 3> comp3
      = {{{0, 0, 0, 0, 2, 1}, {0, 0, 0, 2, 0, 0}, {0, 0, 0, 1, 0, 0}}};
  constexpr std::array<std::array<int, 3>, 2> sign2 = {{{0, 0, -1}, {0, 0, 1}}};
  constexpr std::array<std::array<int, 3>, 2> comp2 = {{{0, 0, 1}, {0, 0, 0}}};
  if (gdim == 3)
    return sign3[j][c] * r[comp3[j][c]];
  return sign2[j][c] * r[comp2[j][c]];
}
} // namespace impl

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

/// @brief Tie every component of the blocked `dofs` to the rigid-body
/// motion of a spider (RBE2).
///
/// Foot `i`, at @f$x@f$, follows the spider with input index `spiders[i]`
/// at @f$x_c@f$: @f$u_j = t_j + (\theta \times (x - x_c))_j@f$. The
/// translation @f$t@f$ is the first `gdim` components of the spider's block
/// of dofs in `W`, the rotation @f$\theta@f$ the rest (components 3, 4, 5 in
/// 3D, 2 in 2D), and @f$x_c@f$ the coordinate of that block. The block,
/// owner and coordinate of each spider are found with `locate_spiders`.
///
/// Every rotation term is kept, also with a zero coefficient, so the masters
/// do not depend on the configuration and `update_rbe2` can recompute the
/// coefficients after the meshes move.
///
/// @param[in] V Space of the feet, with one component per dimension.
/// @param[in] dofs The feet, blocked dofs of `V` local to the process,
/// ghosts included.
/// @param[in] spiders Input index of the spider of each foot.
/// @param[in] W Space on the spider mesh.
/// @param[in] x Coordinates of the dofs of `V` local to the process
/// (row-major, shape `(num_dofs, 3)`), from
/// `V.tabulate_dof_coordinates(false)`.
/// @return The slaves, masters (global dofs of `W`), coefficients, owners
/// and offsets. The masters are in the block of `W`.
/// @note Collective.
template <typename T, std::floating_point U>
mpc_data<T> create_rbe2(const dolfinx::fem::FunctionSpace<U>& V,
                        std::span<const std::int32_t> dofs,
                        std::span<const std::int64_t> spiders,
                        const dolfinx::fem::FunctionSpace<U>& W,
                        std::span<const U> x)
{
  const auto [gdim, num_body, rotations] = impl::check_rbe2_spaces(V, W);
  if (dofs.size() != spiders.size())
  {
    throw std::invalid_argument(std::format("{} feet but {} spider indices",
                                            dofs.size(), spiders.size()));
  }

  std::vector<std::int64_t> needed(spiders.begin(), spiders.end());
  std::ranges::sort(needed);
  auto [first, last] = std::ranges::unique(needed);
  needed.erase(first, last);
  const auto [blocks, owners, x_c] = locate_spiders(W, needed);

  // The body components in foot component j: its translation, then the
  // rotations with a term in it
  std::vector<std::vector<int>> components(gdim);
  for (int j = 0; j < gdim; ++j)
  {
    components[j].push_back(j);
    for (int c = gdim; rotations and c < num_body; ++c)
    {
      const std::array<U, 3> unit = {1, 1, 1};
      if (impl::rbe2_coefficient<U>(gdim, j, c, unit) != 0)
        components[j].push_back(c);
    }
  }

  mpc_data<T> data;
  const std::size_t num_terms = components.front().size();
  data.slaves.reserve(gdim * dofs.size());
  data.masters.reserve(gdim * num_terms * dofs.size());
  data.coeffs.reserve(data.masters.capacity());
  data.owners.reserve(data.masters.capacity());
  data.offsets.reserve(data.slaves.capacity() + 1);
  data.offsets.push_back(0);
  for (std::size_t i = 0; i < dofs.size(); ++i)
  {
    const std::size_t k = std::distance(
        needed.begin(), std::ranges::lower_bound(needed, spiders[i]));
    std::array<U, 3> r = {0, 0, 0};
    for (int d = 0; d < gdim; ++d)
      r[d] = x[3 * dofs[i] + d] - x_c[3 * k + d];
    for (int j = 0; j < gdim; ++j)
    {
      data.slaves.push_back(gdim * dofs[i] + j);
      for (int c : components[j])
      {
        data.masters.push_back(blocks[k] * num_body + c);
        data.coeffs.push_back(impl::rbe2_coefficient<U>(gdim, j, c, r));
        data.owners.push_back(owners[k]);
      }
      data.offsets.push_back(static_cast<std::int32_t>(data.masters.size()));
    }
  }
  return data;
}

/// @brief Tie every component of the blocked `dofs` to the rigid-body
/// motion of a spider (RBE2), tabulating the dof coordinates of `V`.
///
/// See the overload taking the coordinates.
/// @note Collective.
template <typename T, std::floating_point U>
mpc_data<T> create_rbe2(const dolfinx::fem::FunctionSpace<U>& V,
                        std::span<const std::int32_t> dofs,
                        std::span<const std::int64_t> spiders,
                        const dolfinx::fem::FunctionSpace<U>& W)
{
  const std::vector<U> x = V.tabulate_dof_coordinates(false);
  return create_rbe2<T, U>(V, dofs, spiders, W, x);
}

/// @brief Recompute the coefficients of the masters in `W` from the current
/// dof coordinates of `V` and `W`.
///
/// For an RBE2 constraint from `create_rbe2`, after the meshes of `V` and
/// `W` have moved. The coordinate of a spider owned by another process
/// arrives by a forward scatter over the extended index map of `W`, which
/// holds it as a ghost. Every master in `W` is recomputed.
///
/// @param[in,out] mpc The finalized constraint on `V`.
/// @param[in] V Space of the feet, as given to `create_rbe2`.
/// @param[in] W Space on the spider mesh, as given to `create_rbe2`.
/// @param[in] block The block of `W` among `mpc.function_spaces()`.
/// @note Collective.
template <typename T, std::floating_point U>
void update_rbe2(MultiPointConstraint<T, U>& mpc,
                 const dolfinx::fem::FunctionSpace<U>& V,
                 const dolfinx::fem::FunctionSpace<U>& W, int block)
{
  const auto [gdim, num_body, rotations] = impl::check_rbe2_spaces(V, W);
  const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>&
      spaces = mpc.function_spaces();
  if (block < 0 or static_cast<std::size_t>(block) >= spaces.size())
  {
    throw std::out_of_range(
        std::format("Block {} is not one of the {} blocks of the constraint",
                    block, spaces.size()));
  }

  // The spider coordinates, ghosts included
  dolfinx::la::Vector<U> x_c(spaces[block]->dofmap()->index_map, 3);
  const std::vector<U> x_W = W.tabulate_dof_coordinates(false);
  const std::int32_t num_owned = W.dofmap()->index_map->size_local();
  std::copy_n(x_W.begin(), 3 * num_owned, x_c.array().begin());
  x_c.scatter_fwd();

  const std::vector<U> x = V.tabulate_dof_coordinates(false);
  auto [coeffs, offsets] = mpc.all_coefficients();
  const std::vector<std::int32_t> masters = mpc.all_masters();
  const std::vector<std::int32_t> blocks = mpc.all_master_blocks();
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i)
  {
    const std::size_t foot = i / gdim;
    const int j = static_cast<int>(i % gdim);
    for (std::int32_t k = offsets[i]; k < offsets[i + 1]; ++k)
    {
      if ((blocks.empty() ? mpc.block() : blocks[k]) != block)
        continue;
      const std::int32_t spider = masters[k] / num_body;
      std::array<U, 3> r = {0, 0, 0};
      for (int d = 0; d < gdim; ++d)
        r[d] = x[3 * foot + d] - x_c.array()[3 * spider + d];
      coeffs[k] = impl::rbe2_coefficient<U>(gdim, j, masters[k] % num_body, r);
    }
  }
  mpc.update_coefficients(coeffs);
}

} // namespace dolfinx_mpc

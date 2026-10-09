// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "mpi_utils.h"
#include "point_basis.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
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
#include <limits>
#include <map>
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

/// @brief Invert the row-major `n x n` matrix `A` in place, by Gauss-Jordan
/// elimination with partial pivoting.
/// @return false if a pivot is negligible relative to the diagonal entry of
/// its column, i.e. `A` is singular to working precision.
template <std::floating_point U>
bool invert(std::vector<U>& A, int n)
{
  std::vector<U> diagonal(n);
  for (int c = 0; c < n; ++c)
    diagonal[c] = std::abs(A[c * n + c]);
  std::vector<U> inverse(n * n, 0);
  for (int c = 0; c < n; ++c)
    inverse[c * n + c] = 1;
  const U tol = std::sqrt(std::numeric_limits<U>::epsilon());
  for (int c = 0; c < n; ++c)
  {
    int pivot = c;
    for (int r = c + 1; r < n; ++r)
      if (std::abs(A[r * n + c]) > std::abs(A[pivot * n + c]))
        pivot = r;
    if (std::abs(A[pivot * n + c]) <= tol * diagonal[c])
      return false;
    for (int d = 0; d < n; ++d)
    {
      std::swap(A[c * n + d], A[pivot * n + d]);
      std::swap(inverse[c * n + d], inverse[pivot * n + d]);
    }
    const U scale = 1 / A[c * n + c];
    for (int d = 0; d < n; ++d)
    {
      A[c * n + d] *= scale;
      inverse[c * n + d] *= scale;
    }
    for (int r = 0; r < n; ++r)
    {
      if (r == c)
        continue;
      const U factor = A[r * n + c];
      for (int d = 0; d < n; ++d)
      {
        A[r * n + d] -= factor * A[c * n + d];
        inverse[r * n + d] -= factor * inverse[c * n + d];
      }
    }
  }
  A = std::move(inverse);
  return true;
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
  // to its post office, this process included. The source of a row is the
  // owner of the point.
  std::vector<int> office;
  std::vector<std::int64_t> published;
  std::vector<U> published_x;
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::int32_t dof = dofmap->cell_dofs(c).front();
    if (dof < imap->size_local())
    {
      office.push_back(dolfinx::MPI::index_owner(size, original[c], K));
      published.insert(published.end(),
                       {original[c], imap->local_range()[0] + dof});
      published_x.insert(published_x.end(), std::next(x.begin(), 3 * dof),
                         std::next(x.begin(), 3 * (dof + 1)));
    }
  }
  const auto [recv_rows, recv_x, source] = impl::send_rows<std::int64_t, U>(
      comm, office, published, 2, published_x, 3);

  // The directory of this post office: (block, owner) and coordinate per row
  const std::array<std::int64_t, 2> range
      = dolfinx::common::local_range(rank, K, size);
  const std::int64_t num_rows = range[1] - range[0];
  std::vector<std::int64_t> directory(2 * num_rows, -1);
  std::vector<U> directory_x(3 * num_rows, 0);
  for (std::size_t i = 0; i < source.size(); ++i)
  {
    const std::int64_t row = recv_rows[2 * i] - range[0];
    directory[2 * row] = recv_rows[2 * i + 1];
    directory[2 * row + 1] = source[i];
    std::copy_n(std::next(recv_x.begin(), 3 * i), 3,
                std::next(directory_x.begin(), 3 * row));
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

/// @brief Tie the dofs of spiders to the motion of their feet (RBE3).
///
/// Each spider's motion is the rigid motion that best fits its feet, in the
/// weighted least-squares sense:
/// @f[
///   \min_{t, \theta} \sum_i w_i |u_i - t - \theta \times (x_i - x_c)|^2,
/// @f]
/// so @f$(t, \theta) = A^{-1} \sum_i w_i B_i^T u_i@f$, with @f$B_i@f$ the
/// map from @f$(t, \theta)@f$ to the rigid motion at foot @f$i@f$ and
/// @f$A = \sum_i w_i B_i^T B_i@f$. Without rotations, @f$t@f$ is the
/// weighted mean of the feet. Every dof of a spider is a slave, every
/// component of each of its feet a master. The feet are sent to the owner
/// of their spider's dofs, which builds its rows.
///
/// @param[in] W Space on the spider mesh, holding the slaves.
/// @param[in] V Spaces of the feet, with one component per dimension.
/// @param[in] dofs The feet in each space, blocked dofs local to the
/// process. Ghosts are ignored: each foot is sent by its owner.
/// @param[in] spiders Input index of the spider of each foot.
/// @param[in] weights Weight of each foot, non-negative.
/// @return (0) The slaves (dofs of `W`), masters (global dofs of their
/// space), coefficients, owners and offsets, and (1) the position in `V` of
/// the space of each master.
/// @note Collective.
template <typename T, std::floating_point U>
std::pair<mpc_data<T>, std::vector<std::int32_t>> create_rbe3(
    const dolfinx::fem::FunctionSpace<U>& W,
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    const std::vector<std::span<const std::int32_t>>& dofs,
    const std::vector<std::span<const std::int64_t>>& spiders,
    const std::vector<std::span<const U>>& weights)
{
  if (V.empty() or dofs.size() != V.size() or spiders.size() != V.size()
      or weights.size() != V.size())
  {
    throw std::invalid_argument(
        "One array of feet, spiders and weights is needed per space of feet");
  }
  const auto [gdim, num_body, rotations] = impl::check_rbe2_spaces(*V[0], W);
  for (const auto& V_s : V)
  {
    if (std::get<0>(impl::check_rbe2_spaces(*V_s, W)) != gdim)
      throw std::invalid_argument(
          "The spaces of the feet must have the same geometric dimension");
  }
  MPI_Comm comm = W.mesh()->comm();
  int bad_input = 0;
  for (std::size_t s = 0; s < V.size(); ++s)
  {
    bad_input |= dofs[s].size() != spiders[s].size()
                 or dofs[s].size() != weights[s].size()
                 or std::ranges::any_of(weights[s], [](U w) { return w < 0; });
  }
  MPI_Allreduce(MPI_IN_PLACE, &bad_input, 1, MPI_INT, MPI_MAX, comm);
  if (bad_input)
  {
    throw std::invalid_argument("Each foot needs one spider and one "
                                "non-negative weight");
  }

  std::vector<std::int64_t> needed;
  for (std::span<const std::int64_t> k : spiders)
    needed.insert(needed.end(), k.begin(), k.end());
  std::ranges::sort(needed);
  auto [first, last] = std::ranges::unique(needed);
  needed.erase(first, last);
  const std::vector<std::int32_t> spider_owners
      = std::get<1>(locate_spiders(W, needed));

  // Each owned foot goes to the owner of its spider, as (spider, global
  // block, space) and (coordinate, weight). The source of a row is the owner
  // of the foot.
  std::vector<int> dest;
  std::vector<std::int64_t> rows;
  std::vector<U> values;
  for (std::size_t s = 0; s < V.size(); ++s)
  {
    std::shared_ptr<const dolfinx::common::IndexMap> imap
        = V[s]->dofmap()->index_map;
    const std::vector<U> x = V[s]->tabulate_dof_coordinates(false);
    for (std::size_t i = 0; i < dofs[s].size(); ++i)
    {
      const std::int32_t dof = dofs[s][i];
      if (dof >= imap->size_local())
        continue;
      const std::size_t k = std::distance(
          needed.begin(), std::ranges::lower_bound(needed, spiders[s][i]));
      dest.push_back(spider_owners[k]);
      rows.insert(rows.end(), {spiders[s][i], imap->local_range()[0] + dof,
                               static_cast<std::int64_t>(s)});
      values.insert(values.end(), std::next(x.begin(), 3 * dof),
                    std::next(x.begin(), 3 * (dof + 1)));
      values.push_back(weights[s][i]);
    }
  }
  const auto [feet, feet_values, source]
      = impl::send_rows<std::int64_t, U>(comm, dest, rows, 3, values, 4);

  // The spiders owned here: input index -> local block of W
  std::shared_ptr<const dolfinx::mesh::Topology> topology
      = W.mesh()->topology();
  const std::int32_t num_cells
      = topology->index_map(topology->dim())->size_local();
  std::span<const std::int64_t> original
      = topology->original_cell_index.front();
  std::map<std::int64_t, std::int32_t> owned;
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::int32_t dof = W.dofmap()->cell_dofs(c).front();
    if (dof < W.dofmap()->index_map->size_local())
      owned.emplace(original[c], dof);
  }
  const std::vector<U> x_W = W.tabulate_dof_coordinates(false);

  // One spider at a time: its feet, A and the rows of its dofs
  std::vector<std::size_t> order(source.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, {},
                           [&feet](std::size_t i) { return feet[3 * i]; });
  const int n = num_body;
  mpc_data<T> data;
  std::vector<std::int32_t> spaces;
  data.offsets.push_back(0);
  std::int64_t singular = -1;
  for (std::size_t f0 = 0; f0 < order.size();)
  {
    const std::int64_t k = feet[3 * order[f0]];
    std::size_t f1 = f0;
    while (f1 < order.size() and feet[3 * order[f1]] == k)
      ++f1;
    const std::size_t num_feet = f1 - f0;
    const std::int32_t dof_W = owned.at(k);

    // B[f][j][c]: component j of the rigid motion at foot f per body dof c
    std::vector<U> B(num_feet * gdim * n);
    std::vector<U> A(n * n, 0);
    for (std::size_t f = 0; f < num_feet; ++f)
    {
      const std::size_t i = order[f0 + f];
      std::array<U, 3> r = {0, 0, 0};
      for (int d = 0; d < gdim; ++d)
        r[d] = feet_values[4 * i + d] - x_W[3 * dof_W + d];
      const U w = feet_values[4 * i + 3];
      for (int j = 0; j < gdim; ++j)
        for (int c = 0; c < n; ++c)
          B[(f * gdim + j) * n + c] = impl::rbe2_coefficient<U>(gdim, j, c, r);
      for (int j = 0; j < gdim; ++j)
        for (int c = 0; c < n; ++c)
          for (int d = 0; d < n; ++d)
            A[c * n + d]
                += w * B[(f * gdim + j) * n + c] * B[(f * gdim + j) * n + d];
    }
    if (!impl::invert(A, n))
    {
      singular = std::max(singular, k);
      f0 = f1;
      continue;
    }

    for (int c = 0; c < n; ++c)
    {
      data.slaves.push_back(dof_W * n + c);
      for (std::size_t f = 0; f < num_feet; ++f)
      {
        const std::size_t i = order[f0 + f];
        const U w = feet_values[4 * i + 3];
        for (int j = 0; j < gdim; ++j)
        {
          U coeff = 0;
          for (int d = 0; d < n; ++d)
            coeff += A[c * n + d] * w * B[(f * gdim + j) * n + d];
          data.masters.push_back(feet[3 * i + 1] * gdim + j);
          data.coeffs.push_back(coeff);
          data.owners.push_back(source[i]);
          spaces.push_back(static_cast<std::int32_t>(feet[3 * i + 2]));
        }
      }
      data.offsets.push_back(static_cast<std::int32_t>(data.masters.size()));
    }
    f0 = f1;
  }

  MPI_Allreduce(MPI_IN_PLACE, &singular, 1, MPI_INT64_T, MPI_MAX, comm);
  if (singular >= 0)
  {
    throw std::runtime_error(std::format(
        "The feet of spider {} do not determine its {}: too few, or {}",
        singular, rotations ? "translation and rotation" : "translation",
        rotations ? "on one line" : "all of weight zero"));
  }
  return {std::move(data), std::move(spaces)};
}

/// @brief Recompute the coefficients of an RBE3 constraint from the current
/// dof coordinates of the feet and the spiders.
///
/// The arguments are those given to `create_rbe3`, whose rows replace the
/// coefficients of the same slaves in `mpc`. Masters are matched by block
/// and global index, as finalization may reorder those of a slave. For use
/// after the meshes move.
///
/// @param[in,out] mpc The finalized constraint on `W`.
/// @param[in] blocks The block of each space in `V` among
/// `mpc.function_spaces()`.
/// @note Collective.
template <typename T, std::floating_point U>
void update_rbe3(
    MultiPointConstraint<T, U>& mpc, const dolfinx::fem::FunctionSpace<U>& W,
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
    std::span<const int> blocks,
    const std::vector<std::span<const std::int32_t>>& dofs,
    const std::vector<std::span<const std::int64_t>>& spiders,
    const std::vector<std::span<const U>>& weights)
{
  const auto [data, spaces] = create_rbe3<T, U>(W, V, dofs, spiders, weights);
  auto [coeffs, offsets] = mpc.all_coefficients();
  const std::vector<std::int32_t> masters = mpc.all_masters();
  const std::vector<std::int32_t> master_blocks = mpc.all_master_blocks();
  const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>&
      all_spaces = mpc.function_spaces();
  int mismatch = blocks.size() != V.size();
  for (std::size_t i = 0; i < data.slaves.size() and !mismatch; ++i)
  {
    // The new row, keyed by (block, global master)
    std::map<std::pair<int, std::int64_t>, T> row;
    for (std::int32_t k = data.offsets[i]; k < data.offsets[i + 1]; ++k)
      row.emplace(std::pair{blocks[spaces[k]], data.masters[k]},
                  data.coeffs[k]);

    const std::size_t slave = data.slaves[i];
    mismatch = slave + 1 >= offsets.size()
               or static_cast<std::size_t>(offsets[slave + 1] - offsets[slave])
                      != row.size();
    for (std::int32_t k = offsets[slave]; k < offsets[slave + 1] and !mismatch;
         ++k)
    {
      const int b = master_blocks.empty() ? mpc.block() : master_blocks[k];
      std::shared_ptr<const dolfinx::fem::DofMap> dofmap
          = all_spaces[b]->dofmap();
      const int bs = dofmap->index_map_bs();
      const std::int32_t local = masters[k] / bs;
      std::int64_t global;
      dofmap->index_map->local_to_global(std::span(&local, 1),
                                         std::span(&global, 1));
      auto it = row.find({b, global * bs + masters[k] % bs});
      mismatch = it == row.end();
      if (!mismatch)
        coeffs[k] = it->second;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_MAX, W.mesh()->comm());
  if (mismatch)
  {
    throw std::invalid_argument(
        "The constraint does not hold the RBE3 constraint given");
  }
  mpc.update_coefficients(coeffs);
}

} // namespace dolfinx_mpc

// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "point_basis.h"
#include "utils.h"
#include <algorithm>
#include <cstdint>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <format>
#include <functional>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace impl
{

/// @brief Tie each slave block to the dofs at its mapped point, found in a
/// cell among `master_cells`.
///
/// Component `b` of a slave is tied to component `b` of the masters, with the
/// basis values at the point, scaled by `scale`, as coefficients. Periodic
/// conditions map the slave coordinates by their relation; an inelastic
/// contact condition is the identity relation, restricted to the cells at the
/// master facets.
///
/// @param[in] V The input function space (possibly a collapsed sub space)
/// @param[in] slave_blocks The slave blocks of `V`
/// @param[in] relation Function mapping the coordinates of the slaves to the
/// points of their masters
/// @param[in] scale Scaling of the condition
/// @param[in] to_parent The dof in `parent_space` of each (unrolled) dof of
/// `V`. Empty if `V` is not collapsed.
/// @param[in] parent_space The parent space (The same space as V if not
/// collapsed)
/// @param[in] master_cells The cells (local to the process) to search for the
/// masters
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] allow_missing If true, a slave whose point is in no master cell
/// is not constrained. Otherwise it raises on every process.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
/// @note Collective.
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> _create_periodic_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    std::span<const std::int32_t> slave_blocks,
    const std::function<std::vector<U>(std::span<const U>)>& relation, T scale,
    std::span<const std::int32_t> to_parent,
    const dolfinx::fem::FunctionSpace<U>& parent_space,
    std::span<const std::int32_t> master_cells, U distance_tol,
    U coefficient_tol, bool allow_missing, std::size_t num_threads)
{
  const dolfinx::mesh::Mesh<U>& mesh = *V.mesh();
  const int bs = V.dofmap()->index_map_bs();
  const std::int32_t size_local = V.dofmap()->index_map->size_local();
  auto parent = [&to_parent](std::int32_t dof)
  { return to_parent.empty() ? dof : to_parent[dof]; };

  // Only work with local blocks
  std::vector<std::int32_t> local_blocks;
  local_blocks.reserve(slave_blocks.size());
  std::ranges::copy_if(slave_blocks, std::back_inserter(local_blocks),
                       [size_local](std::int32_t block)
                       { return block < size_local; });

  // Map the slave coordinates with the relation, to points of shape (n, 3)
  std::vector<U> points(local_blocks.size() * 3);
  {
    std::vector<std::int32_t> slave_cells
        = dolfinx_mpc::create_block_to_cell_map(*mesh.topology(), *V.dofmap(),
                                                local_blocks);
    auto [x, x_shape] = dolfinx_mpc::tabulate_dof_coordinates(
        V, local_blocks, slave_cells, true, num_threads);
    const std::vector<U> mapped_x = relation(x);
    for (std::size_t i = 0; i < local_blocks.size(); ++i)
      for (std::size_t j = 0; j < 3; ++j)
        points[3 * i + j] = mapped_x[j * x_shape[1] + i];
  }

  const dolfinx_mpc::point_basis<U> basis
      = dolfinx_mpc::evaluate_basis_at_points<U>(
          V, master_cells, points, distance_tol, distance_tol * distance_tol,
          to_parent, parent_space, num_threads);

  // Component b of a slave is tied to component b of the masters
  // A slave component has at most one master per dof of its master cell
  const int width = basis.num_dofs * bs;
  const std::size_t max_masters = local_blocks.size() * width;
  std::vector<std::int32_t> slaves;
  slaves.reserve(local_blocks.size() * bs);
  std::vector<std::int64_t> masters;
  masters.reserve(max_masters);
  std::vector<T> coeffs;
  coeffs.reserve(max_masters);
  std::vector<std::int32_t> owners;
  owners.reserve(max_masters);
  std::vector<std::int32_t> num_masters;
  num_masters.reserve(local_blocks.size() * bs);
  std::vector<std::int64_t> row_masters(basis.num_dofs);
  std::vector<T> row_coeffs(basis.num_dofs);
  std::vector<std::int32_t> row_owners(basis.num_dofs);
  std::int32_t num_missing = 0;
  for (std::size_t i = 0; i < local_blocks.size(); ++i)
  {
    if (!basis.found[i])
    {
      ++num_missing;
      continue;
    }
    for (int b = 0; b < bs; ++b)
    {
      slaves.push_back(parent(local_blocks[i] * bs + b));
      for (int j = 0; j < basis.num_dofs; ++j)
      {
        row_masters[j] = basis.dofs[i * width + j * bs + b];
        row_coeffs[j] = scale * basis.values[i * basis.num_dofs + j];
        row_owners[j] = basis.owners[i * width + j * bs + b];
      }
      num_masters.push_back(dolfinx_mpc::append_significant_masters<T, U>(
          row_masters, row_coeffs, row_owners, coefficient_tol, masters, coeffs,
          owners));
    }
  }
  if (!allow_missing)
  {
    MPI_Allreduce(MPI_IN_PLACE, &num_missing, 1, MPI_INT32_T, MPI_SUM,
                  mesh.comm());
    if (num_missing > 0)
    {
      throw std::runtime_error(std::format(
          "No masters found for {} slave block(s): their points are in none "
          "of the master cells. Make sure that the surfaces are in contact, or "
          "increase distance_tol.",
          num_missing));
    }
  }
  return dolfinx_mpc::add_ghost_rows<T>(
      std::move(slaves), std::move(masters), std::move(coeffs),
      std::move(owners), std::move(num_masters),
      parent_space.dofmap()->index_map, parent_space.dofmap()->index_map_bs());
}

/// Create a periodic MPC condition given a set of slave degrees of freedom
/// (blocked wrt. the input function space). The masters are searched among
/// all owned cells, and a slave whose mapped point is in no cell is not
/// constrained.
/// @param[in] V The input function space (possibly a collapsed sub space)
/// @param[in] slave_blocks The slave blocks of `V`
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] scale Scaling of the periodic condition
/// @param[in] to_parent The dof in `parent_space` of each (unrolled) dof of
/// `V`. Empty if `V` is not collapsed.
/// @param[in] parent_space The parent space (The same space as V if not
/// collapsed)
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> _create_periodic_condition(
    const dolfinx::fem::FunctionSpace<U>& V,
    std::span<const std::int32_t> slave_blocks,
    const std::function<std::vector<U>(std::span<const U>)>& relation, T scale,
    std::span<const std::int32_t> to_parent,
    const dolfinx::fem::FunctionSpace<U>& parent_space, U distance_tol,
    U coefficient_tol, std::size_t num_threads)
{
  const int tdim = V.mesh()->topology()->dim();
  std::vector<std::int32_t> cells(
      V.mesh()->topology()->index_map(tdim)->size_local());
  std::iota(cells.begin(), cells.end(), 0);
  return _create_periodic_condition<T, U>(
      V, slave_blocks, relation, scale, to_parent, parent_space, cells,
      distance_tol, coefficient_tol, true, num_threads);
}

/// Create a periodic MPC condition given a geometrical relation between the
/// slave and master surface
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] indicator Function marking tabulated degrees of freedom
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> geometrical_condition(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
    const std::function<std::vector<std::int8_t>(
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                         std::size_t, 3,
                         MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<U>(std::span<const U>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    T scale, bool collapse, U distance_tol, U coefficient_tol,
    std::size_t num_threads)
{
  std::vector<std::int32_t> reduced_blocks;
  if (collapse)
  {
    // Locate dofs in sub and parent space
    std::pair<dolfinx::fem::FunctionSpace<U>, std::vector<std::vector<int32_t>>>
        sub_space = V->collapse();
    const dolfinx::fem::FunctionSpace<U>& V_sub = sub_space.first;
    const std::vector<std::vector<std::int32_t>>& parent_map = sub_space.second;
    if (parent_map.size() != 1)
      throw std::runtime_error("Mixed topology not supported");
    // If sub-space is collapsed and has a block size (vector space) these are
    // not blocks, but the dofs
    std::array<std::vector<std::int32_t>, 2> slave_dofs
        = dolfinx::fem::locate_dofs_geometrical<U>({*V.get(), V_sub},
                                                   indicator);
    reduced_blocks.reserve(slave_dofs[0].size());
    // Remove dofs in Dirichlet bcs
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_dofs[0], bcs);
    const int sub_bs = V_sub.dofmap()->bs();
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_dofs[1][i] / sub_bs);
    // As block spaces will have duplicates, sort and remove
    std::ranges::sort(reduced_blocks);
    auto [unique_end, range_end] = std::ranges::unique(reduced_blocks);
    reduced_blocks.erase(unique_end, range_end);
    reduced_blocks.shrink_to_fit();

    return _create_periodic_condition<T>(V_sub, reduced_blocks, relation, scale,
                                         parent_map.front(), *V, distance_tol,
                                         coefficient_tol, num_threads);
  }
  else
  {
    std::vector<std::int32_t> slave_blocks
        = dolfinx::fem::locate_dofs_geometrical(*V, indicator);

    reduced_blocks.reserve(slave_blocks.size());
    // Remove blocks in Dirichlet bcs
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_blocks, bcs);
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[i]);
    return _create_periodic_condition<T>(*V, reduced_blocks, relation, scale,
                                         {}, *V, distance_tol, coefficient_tol,
                                         num_threads);
  }
}

/// Create a periodic MPC on a given set of mesh entities, mapped to the
/// master surface by a relation function.
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] meshtag Meshtag with set of entities
/// @param[in] tag The value of the mesh tag entities that should bec
/// considered as entities
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
template <typename T, std::floating_point U>
dolfinx_mpc::mpc_data<T> topological_condition(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<U>(std::span<const U>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    T scale, bool collapse, U distance_tol, U coefficient_tol,
    std::size_t num_threads)
{
  std::vector<std::int32_t> entities = meshtag->find(tag);
  V->mesh()->topology_mutable()->create_connectivity(
      meshtag->dim(), V->mesh()->topology()->dim());
  if (collapse)
  {
    // Locate dofs in sub and parent space
    std::pair<dolfinx::fem::FunctionSpace<U>, std::vector<std::vector<int32_t>>>
        sub_space = V->collapse();
    const dolfinx::fem::FunctionSpace<U>& V_sub = sub_space.first;
    const std::vector<std::vector<std::int32_t>>& parent_map = sub_space.second;
    if (parent_map.size() != 1)
      throw std::runtime_error("Mixed topology mesh not supported");
    // If sub-space is collapsed and has a block size (vector space) these are
    // not blocks, but the dofs
    std::array<std::vector<std::int32_t>, 2> slave_dofs
        = dolfinx::fem::locate_dofs_topological(*V->mesh()->topology_mutable(),
                                                {*V->dofmap(), *V_sub.dofmap()},
                                                meshtag->dim(), entities);
    // Remove DirichletBC dofs from sub space
    std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_dofs[0], bcs);
    std::vector<std::int32_t> reduced_blocks;
    const int sub_bs = V_sub.dofmap()->bs();
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_dofs[1][i] / sub_bs);

    std::ranges::sort(reduced_blocks);
    auto [unique_end, range_end] = std::ranges::unique(reduced_blocks);
    reduced_blocks.erase(unique_end, range_end);
    reduced_blocks.shrink_to_fit();

    return _create_periodic_condition<T>(V_sub, reduced_blocks, relation, scale,
                                         parent_map.front(), *V, distance_tol,
                                         coefficient_tol, num_threads);
  }
  else
  {
    std::vector<std::int32_t> slave_blocks
        = dolfinx::fem::locate_dofs_topological(*V->mesh()->topology_mutable(),
                                                *V->dofmap(), meshtag->dim(),
                                                entities);
    const std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<T>(*V, slave_blocks, bcs);
    std::vector<std::int32_t> reduced_blocks;
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        reduced_blocks.push_back(slave_blocks[i]);
    return _create_periodic_condition<T, U>(*V, reduced_blocks, relation, scale,
                                            {}, *V, distance_tol,
                                            coefficient_tol, num_threads);
  }
};

} // namespace impl

namespace dolfinx_mpc
{
/// Create a periodic condition on the dofs whose coordinates are marked by
/// `indicator`, mapped to the points of their masters by `relation`.
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] indicator Function marking tabulated degrees of freedom
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
/// @note Collective.
template <typename T, std::floating_point U>
mpc_data<T> create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
    const std::function<std::vector<std::int8_t>(
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                         std::size_t, 3,
                         MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>)>&
        indicator,
    const std::function<std::vector<U>(std::span<const U>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    T scale, bool collapse, U distance_tol = default_tolerance<U>(),
    U coefficient_tol = default_tolerance<U>(), std::size_t num_threads = 1)
{
  return impl::geometrical_condition<T, U>(V, indicator, relation, bcs, scale,
                                           collapse, distance_tol,
                                           coefficient_tol, num_threads);
}

/// Create a periodic condition on the dofs of the entities tagged with `tag`,
/// mapped to the points of their masters by `relation`.
/// @param[in] V The input function space (possibly a sub space)
/// @param[in] meshtag Meshtag with set of entities
/// @param[in] tag The value of the tagged entities of the slaves
/// @param[in] relation Function relating coordinates of the slave surface to
/// the master surface
/// @param[in] bcs List of Dirichlet BCs on the input space
/// @param[in] scale Scaling of the periodic condition
/// @param[in] collapse If true, the list of marked dofs is in the collapsed
/// input space
/// @param[in] distance_tol The largest distance from a mapped slave point to a
/// master cell for the point to be in the cell, and the padding of the bounding
/// boxes of the cells
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// master, so that the coefficients can later be rescaled.
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The multi point constraint
/// @note Collective.
template <typename T, std::floating_point U>
mpc_data<T> create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<std::vector<U>(std::span<const U>)>& relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    T scale, bool collapse, U distance_tol = default_tolerance<U>(),
    U coefficient_tol = default_tolerance<U>(), std::size_t num_threads = 1)
{
  return impl::topological_condition<T, U>(V, meshtag, tag, relation, bcs,
                                           scale, collapse, distance_tol,
                                           coefficient_tol, num_threads);
}
} // namespace dolfinx_mpc
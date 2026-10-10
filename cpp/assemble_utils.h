// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx_mpc
{

/// @brief Copy the coordinates of the geometry dofs of `cell` into
/// `coordinate_dofs`, three per dof, as the kernels expect them.
/// @param[in] x_dofmap Geometry dofmap
/// @param[in] x_g Geometry coordinates, three per node
/// @param[in] cell The cell
/// @param[out] coordinate_dofs Destination, of at least `3 *
/// x_dofmap.extent(1)` entries
template <std::floating_point U>
void gather_cell_coordinates(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x_dofmap,
    std::span<const U> x_g, std::int32_t cell, std::span<U> coordinate_dofs)
{
  auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
  assert(coordinate_dofs.size() >= 3 * x_dofs.size());
  for (std::size_t i = 0; i < x_dofs.size(); ++i)
  {
    std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                        std::next(coordinate_dofs.begin(), 3 * i));
  }
}

/// The integral types over one cell-local entity, given as (cell, local
/// entity) pairs, and assembled alike
inline constexpr std::array<dolfinx::fem::IntegralType, 3> entity_integral_types
    = {dolfinx::fem::IntegralType::exterior_facet,
       dolfinx::fem::IntegralType::ridge, dolfinx::fem::IntegralType::vertex};

/// @brief Permutations of the cell-local entities that an integral of type
/// `type` is over.
/// @param[in] mesh The integration domain
/// @param[in] type The integral type, over facets, ridges or vertices
/// @param[in] needed Whether the kernels of the form take permutations
/// @param[in] num_threads The number of threads to compute them with
/// @return The permutation of entity `e` of cell `c` at `c * n + e`, and `n`,
/// the number of such entities of a cell. Empty, and `n = 0`, if not needed or
/// for vertices, which have no orientation.
template <std::floating_point U>
std::pair<std::span<const std::uint8_t>, int>
entity_permutations(const dolfinx::mesh::Mesh<U>& mesh,
                    dolfinx::fem::IntegralType type, bool needed,
                    int num_threads)
{
  const int tdim = mesh.topology()->dim();
  const int dim = dolfinx::fem::integral_entity_dim(type, tdim);
  if (!needed or dim == 0 or dim == tdim)
    return {{}, 0};
  mesh.topology_mutable()->create_entity_permutations(dim, num_threads);
  const dolfinx::mesh::CellType cell_type
      = mesh.topology()->cell_types().front();
  return {mesh.topology()->get_entity_permutations(dim),
          dolfinx::mesh::cell_num_entities(cell_type, dim)};
}

/// For a set of unrolled dofs (slaves) compute the index (local to the cell
/// dofs)
/// @param[in] slaves List of unrolled dofs
/// @param[in] num_dofs Number of dofs (blocked)
/// @param[in] bs The block size
/// @param[in] cell_dofs The cell dofs (blocked)
/// @param[in] is_slave Array indicating if any dof (unrolled, local to process)
/// is a slave
/// @returns Map from position in slaves array to dof local to the cell
std::vector<std::int32_t>
compute_local_slave_index(std::span<const std::int32_t> slaves,
                          const std::uint32_t num_dofs, const int bs,
                          std::span<const std::int32_t> cell_dofs,
                          std::span<const std::int8_t> is_slave);

} // namespace dolfinx_mpc

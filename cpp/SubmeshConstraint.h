// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "point_basis.h"
#include "utils.h"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <format>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dolfinx_mpc
{
/// @brief Tie the dofs of a space to a space on a related mesh: a submesh and
/// its parent, related by an entity map.
///
/// Every dof of `V` in the closure of a cell related to a cell of `W` becomes
/// a slave, @f$u_V(x_i) = \mathrm{scale}\, u_W(x_i)@f$, with @f$u_W@f$
/// evaluated in the related cell and component `b` of `V` tied to component
/// `b` of `W`. With `V` on the submesh this is every dof of `V`. A related
/// cell of the parent is the entity of the map, or for a submesh of facets a
/// cell attached to it, so for a discontinuous `W` the side is arbitrary.
///
/// @param[in] V The space of the slaves, or a subspace of it
/// @param[in] W The space of the masters, or a subspace of it
/// @param[in] entity_map Relates the cells of the submesh to entities of the
/// parent, of codimension 0 or 1. The meshes of `V` and `W` are its two
/// topologies, either way round.
/// @param[in] bcs Dirichlet conditions on the root space of `V`. Their dofs
/// are not made slaves.
/// @param[in] scale Scaling of the masters
/// @param[in] coefficient_tol A master whose coefficient is below
/// `coefficient_tol` times the largest of its slave is dropped. 0 keeps every
/// basis function of the cell.
/// @param[in] num_threads The number of threads to use
/// @return The constraint, the masters numbered in the root space of `W`
/// @note Collective.
template <typename T, std::floating_point U>
mpc_data<T> create_submesh_constraint(
    const dolfinx::fem::FunctionSpace<U>& V,
    const dolfinx::fem::FunctionSpace<U>& W,
    const dolfinx::mesh::EntityMap& entity_map,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    T scale, U coefficient_tol = default_tolerance<U>(),
    std::size_t num_threads = 1)
{
  // A subspace is collapsed; its dofs are numbered in its root space
  std::optional<dolfinx::fem::FunctionSpace<U>> V_collapsed;
  std::vector<std::int32_t> V_to_root;
  if (!V.component().empty())
  {
    auto [space, map] = V.collapse();
    V_collapsed.emplace(std::move(space));
    V_to_root = std::move(map.front());
  }
  std::optional<dolfinx::fem::FunctionSpace<U>> W_collapsed;
  std::vector<std::int32_t> W_to_root;
  if (!W.component().empty())
  {
    auto [space, map] = W.collapse();
    W_collapsed.emplace(std::move(space));
    W_to_root = std::move(map.front());
  }
  const dolfinx::fem::FunctionSpace<U>& Vs = V_collapsed ? *V_collapsed : V;
  const dolfinx::fem::FunctionSpace<U>& Ws = W_collapsed ? *W_collapsed : W;

  // Checks, all local and with the same verdict on every process
  const bool slave_on_sub = entity_map.sub_topology() == Vs.mesh()->topology()
                            and entity_map.topology() == Ws.mesh()->topology();
  const bool master_on_sub = entity_map.sub_topology() == Ws.mesh()->topology()
                             and entity_map.topology() == Vs.mesh()->topology();
  if (!slave_on_sub and !master_on_sub)
  {
    throw std::invalid_argument(
        "The entity map must relate the meshes of the two spaces");
  }
  const int dim = entity_map.dim();
  if (dim != entity_map.sub_topology()->dim())
  {
    throw std::invalid_argument(
        "The entities of the map must be the cells of the submesh");
  }
  const int tdim = entity_map.topology()->dim();
  if (tdim - dim > 1)
  {
    throw std::invalid_argument(
        std::format("A submesh of codimension {} is not supported, only 0 or 1",
                    tdim - dim));
  }
  const int bs = Vs.dofmap()->index_map_bs();
  if (Ws.dofmap()->index_map_bs() != bs)
  {
    throw std::invalid_argument(std::format(
        "The two spaces must have the same number of components, not {} and "
        "{}",
        bs, Ws.dofmap()->index_map_bs()));
  }

  // The parent cell of an entity of the map, and the entity's local index in
  // it (-1 for a cell)
  const dolfinx::mesh::Mesh<U>& parent = slave_on_sub ? *Ws.mesh() : *Vs.mesh();
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> e_to_c;
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> c_to_e;
  if (dim < tdim)
  {
    parent.topology_mutable()->create_connectivity(dim, tdim);
    parent.topology_mutable()->create_connectivity(tdim, dim);
    e_to_c = parent.topology()->connectivity(dim, tdim);
    c_to_e = parent.topology()->connectivity(tdim, dim);
  }
  auto parent_cell
      = [&e_to_c, &c_to_e](std::int32_t e) -> std::pair<std::int32_t, int>
  {
    if (!e_to_c)
      return {e, -1};
    const std::int32_t cell = e_to_c->links(e).front();
    std::span<const std::int32_t> entities = c_to_e->links(cell);
    return {cell, static_cast<int>(std::ranges::distance(
                      entities.begin(), std::ranges::find(entities, e)))};
  };

  // Slave blocks of `Vs`, each with a cell of its own mesh and the related
  // cell of `W`'s mesh
  std::vector<std::int32_t> blocks;
  std::vector<std::int32_t> V_cells;
  std::vector<std::int32_t> W_cells;
  const std::shared_ptr<const dolfinx::common::IndexMap> V_map
      = Vs.dofmap()->index_map;
  if (slave_on_sub)
  {
    // Every owned dof is in a local cell of the submesh
    blocks.resize(V_map->size_local());
    std::iota(blocks.begin(), blocks.end(), 0);
    V_cells = create_block_to_cell_map(*Vs.mesh()->topology(), *Vs.dofmap(),
                                       blocks);
    const std::vector<std::int32_t> entities
        = entity_map.sub_topology_to_topology(std::span(V_cells), false);
    W_cells.reserve(blocks.size());
    for (std::int32_t e : entities)
      W_cells.push_back(parent_cell(e).first);
  }
  else
  {
    // The dofs in the closure of the related entities, owned or not: an owned
    // dof may be in no related entity on this process
    const std::int32_t num_cells
        = Ws.mesh()->topology()->index_map(dim)->size_local()
          + Ws.mesh()->topology()->index_map(dim)->num_ghosts();
    std::vector<std::int32_t> sub_cells(num_cells);
    std::iota(sub_cells.begin(), sub_cells.end(), 0);
    const std::vector<std::int32_t> entities
        = entity_map.sub_topology_to_topology(std::span(sub_cells), false);
    const std::int32_t num_blocks = V_map->size_local() + V_map->num_ghosts();
    std::vector<std::int32_t> block_V_cell(num_blocks, -1);
    std::vector<std::int32_t> block_W_cell(num_blocks, -1);
    const dolfinx::fem::ElementDofLayout& layout
        = Vs.dofmap()->element_dof_layout();
    std::vector<std::int32_t> closure;
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      const auto [cell, local] = parent_cell(entities[c]);
      std::span<const std::int32_t> cell_blocks = Vs.dofmap()->cell_dofs(cell);
      closure.clear();
      if (local < 0)
        closure.assign(cell_blocks.begin(), cell_blocks.end());
      else
      {
        for (int k : layout.entity_closure_dofs(dim, local))
          closure.push_back(cell_blocks[k]);
      }
      for (std::int32_t block : closure)
      {
        if (block_V_cell[block] == -1)
        {
          block_V_cell[block] = cell;
          block_W_cell[block] = c;
        }
      }
    }
    for (std::int32_t b = 0; b < num_blocks; ++b)
    {
      if (block_V_cell[b] != -1)
      {
        blocks.push_back(b);
        V_cells.push_back(block_V_cell[b]);
        W_cells.push_back(block_W_cell[b]);
      }
    }
  }

  // The basis of `W` at the coordinate of each slave block
  const std::vector<U> points
      = tabulate_dof_coordinates<U>(Vs, blocks, V_cells).first;
  const point_basis<U> basis = evaluate_basis_in_cells<U>(
      Ws, points, W_cells, W_to_root, W, default_tolerance<U>(), num_threads);

  // Send the basis of each slave block to its owner, itself included:
  // [global block, dofs, owners], values
  const int rank = dolfinx::MPI::rank(V_map->comm());
  const std::int32_t size_local = V_map->size_local();
  std::span<const int> ghost_owners = V_map->owners();
  const int width = basis.num_dofs * bs;
  std::vector<std::int64_t> global(blocks.size());
  V_map->local_to_global(blocks, global);
  std::vector<int> dest;
  std::vector<std::int64_t> rows;
  dest.reserve(blocks.size());
  rows.reserve(blocks.size() * (1 + 2 * width));
  for (std::size_t i = 0; i < blocks.size(); ++i)
  {
    dest.push_back(
        blocks[i] < size_local ? rank : ghost_owners[blocks[i] - size_local]);
    rows.push_back(global[i]);
    rows.insert(rows.end(), std::next(basis.dofs.begin(), i * width),
                std::next(basis.dofs.begin(), (i + 1) * width));
    rows.insert(rows.end(), std::next(basis.owners.begin(), i * width),
                std::next(basis.owners.begin(), (i + 1) * width));
  }
  const auto [recv_rows, recv_values, source]
      = impl::send_rows<std::int64_t, U>(V_map->comm(), dest, rows,
                                         1 + 2 * width, basis.values,
                                         basis.num_dofs);

  // The rows come by ascending source: the first for each block is used
  std::vector<std::int64_t> recv_global(source.size());
  for (std::size_t r = 0; r < source.size(); ++r)
    recv_global[r] = recv_rows[r * (1 + 2 * width)];
  std::vector<std::int32_t> recv_local(source.size());
  V_map->global_to_local(recv_global, recv_local);
  std::vector<std::int32_t> chosen(size_local, -1);
  for (std::size_t r = 0; r < source.size(); ++r)
  {
    assert(recv_local[r] >= 0 and recv_local[r] < size_local);
    if (chosen[recv_local[r]] == -1)
      chosen[recv_local[r]] = static_cast<std::int32_t>(r);
  }

  // Dirichlet dofs of the root space of `V` are not slaves
  const std::shared_ptr<const dolfinx::common::IndexMap> root_map
      = V.dofmap()->index_map;
  const int root_bs = V.dofmap()->index_map_bs();
  std::vector<std::int8_t> bc_marker(
      root_bs * (root_map->size_local() + root_map->num_ghosts()), 0);
  for (const auto& bc : bcs)
  {
    assert(bc);
    if (bc->function_space()->contains(V) or V.contains(*bc->function_space()))
      bc->mark_dofs(bc_marker);
  }

  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<T> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> num_masters;
  std::vector<std::int64_t> row_masters(basis.num_dofs);
  std::vector<T> row_coeffs(basis.num_dofs);
  std::vector<std::int32_t> row_owners(basis.num_dofs);
  for (std::int32_t block = 0; block < size_local; ++block)
  {
    const std::int32_t r = chosen[block];
    if (r == -1)
      continue;
    std::span<const std::int64_t> row(
        std::next(recv_rows.begin(), r * (1 + 2 * width) + 1), 2 * width);
    for (int b = 0; b < bs; ++b)
    {
      const std::int32_t dof = block * bs + b;
      const std::int32_t slave = V_to_root.empty() ? dof : V_to_root[dof];
      if (bc_marker[slave])
        continue;
      slaves.push_back(slave);
      for (int j = 0; j < basis.num_dofs; ++j)
      {
        row_masters[j] = row[j * bs + b];
        row_coeffs[j] = scale * recv_values[r * basis.num_dofs + j];
        row_owners[j] = static_cast<std::int32_t>(row[width + j * bs + b]);
      }
      num_masters.push_back(append_significant_masters<T, U>(
          row_masters, row_coeffs, row_owners, coefficient_tol, masters, coeffs,
          owners));
    }
  }
  return add_ghost_rows<T>(std::move(slaves), std::move(masters),
                           std::move(coeffs), std::move(owners),
                           std::move(num_masters), root_map, root_bs);
}
} // namespace dolfinx_mpc

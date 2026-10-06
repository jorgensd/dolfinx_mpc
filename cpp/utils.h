// Copyright (C) 2020-2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "mpi_utils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace impl
{
/// Create a map from each dof (block) found on the set of facets
/// topologically,
/// to the connecting facets
/// @param[in] V The function space
/// @param[in] dim The dimension of the entities
/// @param[in] entities The list of entities
/// @returns The map from each block (local + ghost) to the set of facets
dolfinx::graph::AdjacencyList<std::int32_t>
create_block_to_facet_map(dolfinx::mesh::Topology& topology,
                          const dolfinx::fem::DofMap& dofmap, std::int32_t dim,
                          std::span<const std::int32_t> entities)
{
  std::shared_ptr<const dolfinx::common::IndexMap> imap = dofmap.index_map;
  const std::int32_t tdim = topology.dim();
  // Locate all dofs for each facet
  topology.create_connectivity(dim, tdim);
  topology.create_connectivity(tdim, dim);
  auto e_to_c = topology.connectivity(dim, tdim);
  auto c_to_e = topology.connectivity(tdim, dim);

  const std::int32_t num_dofs = imap->size_local() + imap->num_ghosts();
  std::vector<std::int32_t> num_facets_per_dof(num_dofs);

  // Count how many facets each dof on process relates to
  std::vector<std::int32_t> local_indices(entities.size());
  std::vector<std::int32_t> cells(entities.size());
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    auto cell = e_to_c->links(entities[i]);
    assert(cell.size() == 1);
    cells[i] = cell[0];

    // Get local index of facet with respect to the cell
    auto cell_entities = c_to_e->links(cell[0]);
    const auto it
        = std::find(cell_entities.begin(), cell_entities.end(), entities[i]);
    assert(it != cell_entities.end());
    const auto local_entity = std::ranges::distance(cell_entities.begin(), it);
    local_indices[i] = local_entity;
    auto cell_blocks = dofmap.cell_dofs(cell[0]);
    auto closure_blocks
        = dofmap.element_dof_layout().entity_closure_dofs(dim, local_entity);
    std::ranges::for_each(closure_blocks,
                          [&num_facets_per_dof, &cell_blocks](auto block)
                          {
                            const int dof = cell_blocks[block];
                            num_facets_per_dof[dof]++;
                          });
  }

  // Compute offsets
  std::vector<std::int32_t> offsets(num_dofs + 1);
  offsets[0] = 0;
  std::partial_sum(num_facets_per_dof.begin(), num_facets_per_dof.end(),
                   offsets.begin() + 1);
  // Reuse data structure for insertion
  std::ranges::fill(num_facets_per_dof, 0);

  // Create dof->entities map
  std::vector<std::int32_t> data(offsets.back());
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    auto cell_blocks = dofmap.cell_dofs(cells[i]);
    auto closure_blocks = dofmap.element_dof_layout().entity_closure_dofs(
        dim, local_indices[i]);
    std::for_each(closure_blocks.begin(), closure_blocks.end(),
                  [&num_facets_per_dof, &data, &cell_blocks, &offsets,
                   entity = entities[i]](auto block)
                  {
                    const int dof = cell_blocks[block];
                    data[offsets[dof] + num_facets_per_dof[dof]++] = entity;
                  });
  }
  return dolfinx::graph::AdjacencyList<std::int32_t>(data, offsets);
}

} // namespace impl

namespace dolfinx_mpc
{

template <typename T>
struct mpc_data
{
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<T> coeffs;
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> owners;
};

template <typename T, std::floating_point U>
class MultiPointConstraint;

/// Given a function space, compute its shared entities
template <std::floating_point U>
dolfinx::graph::AdjacencyList<int>
compute_shared_indices(std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V)
{
  std::pair<std::vector<int>, std::vector<std::int32_t>> shared_indices
      = V->dofmap()->index_map->index_to_dest_ranks();
  return dolfinx::graph::AdjacencyList<int>(std::move(shared_indices.first),
                                            std::move(shared_indices.second));
}

template <std::floating_point U>
dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc1,
    const std::string& type = std::string())
{
  dolfinx::common::Timer timer("~MPC: Create Matrix");

  // Build sparsitypattern
  dolfinx::la::SparsityPattern pattern = create_sparsity_pattern(a, mpc0, mpc1);

  // Finalise communication
  dolfinx::common::Timer timer_s("~MPC: Assemble sparsity pattern");
  pattern.finalize();
  timer_s.stop();

  // Initialize matrix
  dolfinx::la::petsc::Matrix A(a.mesh()->comm(), pattern, type);

  return A;
}

template <std::floating_point U>
dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar, U>>
        mpc,
    const std::string& type = std::string())
{
  return dolfinx_mpc::create_matrix(a, mpc, mpc, type);
}

/// Creates a normal approximation for the dofs in the closure of the attached
/// facets, where the normal is an average if a dof belongs to multiple facets
/// FIXME: Remove petsc dependency here
template <std::floating_point U>
dolfinx::fem::Function<PetscScalar>
create_normal_approximation(std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V,
                            std::int32_t dim,
                            std::span<const std::int32_t> entities)
{
  dolfinx::graph::AdjacencyList<std::int32_t> block_to_entities
      = impl::create_block_to_facet_map(*V->mesh()->topology_mutable(),
                                        *V->dofmap(), dim, entities);

  // Create normal vector function and get local span
  dolfinx::fem::Function<PetscScalar> nh(V);
  Vec n_local;
  dolfinx::la::petsc::Vector n_vec(
      dolfinx::la::petsc::create_vector_wrap(*nh.x()), false);
  VecGhostGetLocalForm(n_vec.vec(), &n_local);
  PetscInt n = 0;
  VecGetSize(n_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(n_local, &array);
  std::span<PetscScalar> _n(array, n);

  const std::int32_t bs = V->dofmap()->index_map_bs();
  std::array<U, 3> normal;
  std::array<U, 3> n_0;
  for (std::int32_t i = 0; i < block_to_entities.num_nodes(); i++)
  {
    auto ents = block_to_entities.links(i);
    if (ents.empty())
      continue;
    // Sum all normal for entities
    std::vector<U> normals = dolfinx::mesh::cell_normals(*V->mesh(), dim, ents);
    std::ranges::copy_n(normals.begin(), 3, n_0.begin());
    std::ranges::copy_n(n_0.begin(), 3, normal.begin());
    for (std::size_t j = 1; j < normals.size() / 3; ++j)
    {
      // Align direction of normal vectors n_0 and n_j
      U n_nj = std::transform_reduce(
          n_0.begin(), n_0.end(), std::next(normals.begin(), 3 * j), 0.,
          std::plus{}, [](auto x, auto y) { return x * y; });
      auto sign = n_nj / std::abs(n_nj);
      // NOTE: Could probably use std::transform for this operation
      for (std::size_t k = 0; k < 3; ++k)
        normal[k] += sign * normals[3 * j + k];
    }
    std::ranges::copy_n(normal.begin(), bs, std::next(_n.begin(), i * bs));
  }
  // Receive normals from other processes with dofs on the facets
  VecGhostUpdateBegin(n_vec.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(n_vec.vec(), ADD_VALUES, SCATTER_REVERSE);
  // Normalize nh
  auto imap = V->dofmap()->index_map;
  std::int32_t num_blocks = imap->size_local();
  for (std::int32_t i = 0; i < num_blocks; i++)
  {
    PetscScalar acc = 0;
    for (std::int32_t j = 0; j < bs; j++)
      acc += _n[i * bs + j] * _n[i * bs + j];
    if (U abs = std::abs(acc); abs > 1e-10)
    {
      for (std::int32_t j = 0; j < bs; j++)
        _n[i * bs + j] /= abs;
    }
  }

  VecGhostUpdateBegin(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(n_vec.vec(), INSERT_VALUES, SCATTER_FORWARD);
  return nh;
}

/// @brief Reserve the diagonal entry of every owned slave in a pattern.
///
/// `slaves()` holds unrolled dof indices while the pattern is indexed by
/// blocks, so divide through by the block size before inserting. Reserving the
/// whole diagonal block is a superset of the single scalar entry
/// `insert_slave_diagonal` writes, which is harmless.
/// @param[in,out] pattern Pattern of a block whose rows and columns are both
/// the space of `mpc`
/// @param[in] mpc The constraint whose slaves are reserved
template <typename T, std::floating_point U>
void insert_slave_diagonal_pattern(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc)
{
  const int bs = mpc.function_space()->dofmap()->index_map_bs();
  std::span<const std::int32_t> slaves(mpc.slaves().data(),
                                       mpc.num_local_slaves());
  std::vector<std::int32_t> slave_blocks;
  slave_blocks.reserve(slaves.size());
  std::ranges::transform(slaves, std::back_inserter(slave_blocks),
                         [bs](std::int32_t dof) { return dof / bs; });
  std::ranges::sort(slave_blocks);
  slave_blocks.erase(std::unique(slave_blocks.begin(), slave_blocks.end()),
                     slave_blocks.end());
  pattern.insert_diagonal(slave_blocks);
}

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
///
/// @note This function is almost a copy of
/// dolfinx::fem::utils.h::create_sparsity_pattern
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
template <typename T>
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<T>& a)
{

  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (Classic)");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }
  // Get dof maps and mesh
  std::array<std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmap(),
      *a.function_spaces().at(1)->dofmap()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<dolfinx::fem::IntegralType> types = a.integral_types();
  if (types.find(dolfinx::fem::IntegralType::interior_facet) != types.end()
      or types.find(dolfinx::fem::IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  auto extract_cells = [](std::span<const std::int32_t> facets)
  {
    assert(facets.size() % 2 == 0);
    std::vector<std::int32_t> cells;
    cells.reserve(facets.size() / 2);
    for (std::size_t i = 0; i < facets.size(); i += 2)
      cells.push_back(facets[i]);
    return cells;
  };

  for (auto type : types)
  {
    switch (type)
    {
    case dolfinx::fem::IntegralType::cell:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        dolfinx::fem::sparsitybuild::cells(
            pattern,
            std::pair{a.domain_arg(type, 0, i, 0), a.domain_arg(type, 1, i, 0)},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case dolfinx::fem::IntegralType::interior_facet:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        std::vector<std::int32_t> cells0
            = extract_cells(a.domain_arg(type, 0, i, 0));
        std::vector<std::int32_t> cells1
            = extract_cells(a.domain_arg(type, 1, i, 0));
        dolfinx::fem::sparsitybuild::interior_facets(
            pattern,
            {std::span<const std::int32_t>(cells0),
             std::span<const std::int32_t>(cells1)},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case dolfinx::fem::IntegralType::exterior_facet:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        std::vector<std::int32_t> cells0
            = extract_cells(a.domain_arg(type, 0, i, 0));
        std::vector<std::int32_t> cells1
            = extract_cells(a.domain_arg(type, 1, i, 0));
        dolfinx::fem::sparsitybuild::cells(
            pattern,
            std::pair{std::span<const std::int32_t>(cells0),
                      std::span<const std::int32_t>(cells1)},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  timer.stop();
}

/// Create a map from dof blocks to one of the cells that contains the degree of
/// freedom
/// @param[in] topology The mesh topology
/// @param[in] dofmap The dofmap
/// @param[in] blocks The blocks (local to process) we want to map
std::vector<std::int32_t>
create_block_to_cell_map(const dolfinx::mesh::Topology& topology,
                         const dolfinx::fem::DofMap& dofmap,
                         std::span<const std::int32_t> blocks);

/// @brief Add the entries a multi point constraint adds to the pattern of a
/// form: the masters replacing the slaves in its rows and columns.
///
/// The test and trial spaces may live on different meshes (e.g. a submesh
/// coupled to its parent), in which case their cell numberings are unrelated.
/// Each axis is therefore walked with its own integration entities, taken from
/// `Form::domain_arg`, exactly as the assembler does.
///
/// @param[in] a The form, whose test space is constrained by `mpc_row` and
/// trial space by `mpc_col`
/// @param[in] mpc_row Constraint of the rows
/// @param[in] mpc_col Constraint of the columns
/// @param[in] insert Callable `insert(row_block, rows, col_block, cols)`
/// adding the dense block `rows x cols` (blocked indices) to the pattern of the
/// given blocks. A block is that of the constraint for the form's own rows or
/// columns, or the block of a master.
template <typename T, std::floating_point U, typename Insert>
void populate_mpc_pattern(
    const dolfinx::fem::Form<T>& a,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc_row,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc_col, Insert&& insert)
{
  const auto& V_row = mpc_row.function_space();
  const auto& V_col = mpc_col.function_space();
  const std::array<const dolfinx_mpc::MultiPointConstraint<T, U>*, 2> mpcs
      = {&mpc_row, &mpc_col};
  const std::array<int, 2> blocks = {mpc_row.block(), mpc_col.block()};

  // Map from cell index (local to the respective space's mesh) to the slave
  // dofs in that cell
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_row_slaves = mpc_row.cell_to_slaves();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_col_slaves = mpc_col.cell_to_slaves();

  // Scratch reused across entities to avoid reallocating
  std::vector<std::int32_t> row_dofs(2 * V_row->dofmap()->map().extent(1));
  std::vector<std::int32_t> col_dofs(2 * V_col->dofmap()->map().extent(1));
  // Masters of the slaves of an entity, per block, as blocked indices in the
  // extended space of that block
  const std::size_t nb = mpc_row.function_spaces().size();
  std::array<std::vector<std::vector<std::int32_t>>, 2> masters_by_block;
  masters_by_block[0].resize(nb);
  masters_by_block[1].resize(mpc_col.function_spaces().size());

  // Overwrite `dofs` with the dofs of `cells`, skipping a negative cell (no
  // cell on that side of an interface). Capacity is kept.
  auto gather_dofs
      = [](std::vector<std::int32_t>& dofs, const dolfinx::fem::DofMap& dofmap,
           std::span<const std::int32_t> cells)
  {
    const std::size_t ndofs = dofmap.map().extent(1);
    dofs.resize(cells.size() * ndofs);
    std::size_t n = 0;
    for (std::int32_t cell : cells)
    {
      if (cell < 0)
        continue;
      std::ranges::copy(dofmap.cell_dofs(cell), std::next(dofs.begin(), n));
      n += ndofs;
    }
    dofs.resize(n);
  };

  // Collect the masters of the slaves of `cells` on `axis`, by block
  auto gather_masters
      = [&mpcs, &masters_by_block](
            int axis, const dolfinx::graph::AdjacencyList<std::int32_t>& c,
            std::span<const std::int32_t> cells)
  {
    for (auto& m : masters_by_block[axis])
      m.clear();
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc = *mpcs[axis];
    const dolfinx::graph::AdjacencyList<std::int32_t>& master_map
        = *mpc.masters();
    std::span<const std::int32_t> blocks = mpc.master_blocks();
    const std::vector<std::int32_t>& offsets = master_map.offsets();
    for (std::int32_t cell : cells)
    {
      if (cell < 0)
        continue;
      for (std::int32_t slave : c.links(cell))
      {
        std::span<const std::int32_t> masters = master_map.links(slave);
        for (std::size_t k = 0; k < masters.size(); ++k)
        {
          const std::int32_t block = blocks[offsets[slave] + k];
          const int bs = mpc.function_spaces()[block]->dofmap()->index_map_bs();
          masters_by_block[axis][block].push_back(masters[k] / bs);
        }
      }
    }
  };

  // Insert the master rows/columns generated by one integration entity, a
  // cell or facet whose element tensor spans `cells_row` x `cells_col`: one
  // cell each for cell and exterior facet integrals, two for interior facet
  // integrals. The plain (non-master) entries are covered by the standard
  // pattern, so only entities carrying a slave do any work.
  auto insert_entity = [&](std::span<const std::int32_t> cells_row,
                           std::span<const std::int32_t> cells_col)
  {
    // `cell_to_slaves` has a node for every cell, ghosts included, so any
    // entity a form names can be looked up directly.
    auto has_slaves = [](const dolfinx::graph::AdjacencyList<std::int32_t>& c,
                         std::span<const std::int32_t> cells)
    {
      return std::ranges::any_of(
          cells, [&c](std::int32_t cell)
          { return cell >= 0 and c.num_links(cell) > 0; });
    };
    const bool row_has_slaves = has_slaves(*cell_to_row_slaves, cells_row);
    const bool col_has_slaves = has_slaves(*cell_to_col_slaves, cells_col);
    if (!row_has_slaves and !col_has_slaves)
      return;

    gather_dofs(col_dofs, *V_col->dofmap(), cells_col);
    if (col_has_slaves)
    {
      // NOTE: The slave columns could be dropped here -- `modify_mpc_cell`
      // zeroes the slave-slave block, so every (master, slave) entry inserted
      // below is an exact zero. Worth 2-9% of nnz for a periodic 3D
      // Laplacian, decaying as O(h).
      gather_masters(1, *cell_to_col_slaves, cells_col);
      // The columns gained masters, so the ordinary rows need them too
      gather_dofs(row_dofs, *V_row->dofmap(), cells_row);
      for (std::size_t cb = 0; cb < masters_by_block[1].size(); ++cb)
        if (!masters_by_block[1][cb].empty())
          insert(blocks[0], row_dofs, static_cast<int>(cb),
                 masters_by_block[1][cb]);
    }

    if (row_has_slaves)
    {
      gather_masters(0, *cell_to_row_slaves, cells_row);
      for (std::size_t rb = 0; rb < masters_by_block[0].size(); ++rb)
      {
        const std::vector<std::int32_t>& rows = masters_by_block[0][rb];
        if (rows.empty())
          continue;
        insert(static_cast<int>(rb), rows, blocks[1], col_dofs);
        if (col_has_slaves)
        {
          for (std::size_t cb = 0; cb < masters_by_block[1].size(); ++cb)
            if (!masters_by_block[1][cb].empty())
              insert(static_cast<int>(rb), rows, static_cast<int>(cb),
                     masters_by_block[1][cb]);
        }
      }
    }
  };

  for (auto type : a.integral_types())
  {
    switch (type)
    {
    case dolfinx::fem::IntegralType::cell:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        std::span<const std::int32_t> cells_row = a.domain_arg(type, 0, i, 0);
        std::span<const std::int32_t> cells_col = a.domain_arg(type, 1, i, 0);
        assert(cells_row.size() == cells_col.size());
        for (std::size_t e = 0; e < cells_row.size(); ++e)
          insert_entity(cells_row.subspan(e, 1), cells_col.subspan(e, 1));
      }
      break;
    case dolfinx::fem::IntegralType::exterior_facet:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        // Entities are (cell, local_facet) pairs
        std::span<const std::int32_t> facets_row = a.domain_arg(type, 0, i, 0);
        std::span<const std::int32_t> facets_col = a.domain_arg(type, 1, i, 0);
        assert(facets_row.size() == facets_col.size());
        assert(facets_row.size() % 2 == 0);
        for (std::size_t e = 0; e < facets_row.size(); e += 2)
          insert_entity(facets_row.subspan(e, 1), facets_col.subspan(e, 1));
      }
      break;
    case dolfinx::fem::IntegralType::interior_facet:
      for (int i = 0; i < a.num_integrals(type, 0); ++i)
      {
        // Entities are (cell, local_facet) for each of the two sides
        std::span<const std::int32_t> facets_row = a.domain_arg(type, 0, i, 0);
        std::span<const std::int32_t> facets_col = a.domain_arg(type, 1, i, 0);
        assert(facets_row.size() == facets_col.size());
        assert(facets_row.size() % 4 == 0);
        for (std::size_t e = 0; e < facets_row.size(); e += 4)
        {
          const std::array<std::int32_t, 2> cells_row
              = {facets_row[e], facets_row[e + 2]};
          const std::array<std::int32_t, 2> cells_col
              = {facets_col[e], facets_col[e + 2]};
          insert_entity(cells_row, cells_col);
        }
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }
}

/// Create sparsity pattern with multi point constraint additions to the rows
/// and the columns
/// @param[in] a bi-linear form for the current variational problem
/// (The one used to generate the standard sparsity-pattern)
/// @param[in] mpc0 The multi point constraint to apply to the rows of the
/// matrix.
/// @param[in] mpc1 The multi point constraint to apply to the columns of the
/// matrix.
/// @note The constraints must have their masters in their own blocks; see
/// `create_sparsity_patterns` otherwise.
template <typename T, std::floating_point U>
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> mpc0,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> mpc1)
{
  spdlog::info("Generating MPC sparsity pattern");
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  if (mpc0->has_cross_block_masters() or mpc1->has_cross_block_masters())
  {
    throw std::invalid_argument(
        "A constraint has masters in another block, so the matrix of a single "
        "form cannot hold its entries. Create the matrix of the whole blocked "
        "system.");
  }

  // Extract function space and index map from mpcs
  auto V0 = mpc0->function_space();
  auto V1 = mpc1->function_space();
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps
      = {V0->dofmap()->index_map, V1->dofmap()->index_map};
  std::array<int, 2> bs
      = {V0->dofmap()->index_map_bs(), V1->dofmap()->index_map_bs()};
  dolfinx::la::SparsityPattern pattern(a.mesh()->comm(), new_maps, bs);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  build_standard_pattern<T>(pattern, a);

  // Every master is in the block of its slave, so all entries go here
  populate_mpc_pattern<T, U>(a, *mpc0, *mpc1,
                             [&pattern](int, std::span<const std::int32_t> rows,
                                        int, std::span<const std::int32_t> cols)
                             { pattern.insert(rows, cols); });

  // `insert_slave_diagonal` writes a diagonal entry for every owned slave of a
  // diagonal block, so the pattern has to reserve those entries. They usually
  // fall inside the standard pattern already, but not when the block's form has
  // no integral covering them -- a `ufl.ZeroBaseForm` diagonal block, say --
  // and a missing entry is a PETSc allocation error at assembly rather than a
  // silently wrong matrix.
  if (mpc0 == mpc1)
    insert_slave_diagonal_pattern(pattern, *mpc0);

  return pattern;
}

/// @brief The sparsity patterns of every block of a blocked system under multi
/// point constraints, masters in other blocks included.
///
/// Block `(k, l)` has the rows of `mpcs0[k]` and the columns of `mpcs1[l]`.
/// Every block gets a pattern, even one without a form: a master in another
/// block puts entries there, and which blocks receive any is only known per
/// process, while the patterns are merged or finalized collectively.
/// @param[in] a Forms, with `nullptr` for a block that has none
/// @param[in] mpcs0 Constraint of each block row
/// @param[in] mpcs1 Constraint of each block column
/// @return The unfinalized patterns, `patterns[k][l]`
/// @pre If any constraint has masters in another block, `mpcs0` and `mpcs1`
/// are both the constraints in the order they were created in, so that the
/// block of a master is its position in either list.
template <typename T, std::floating_point U>
std::vector<std::vector<dolfinx::la::SparsityPattern>> create_sparsity_patterns(
    const std::vector<std::vector<const dolfinx::fem::Form<T>*>>& a,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs0,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs1)
{
  dolfinx::common::Timer timer("~MPC: Create block sparsity patterns");
  const bool cross
      = std::ranges::any_of(mpcs0, [](const auto& m)
                            { return m->has_cross_block_masters(); })
        or std::ranges::any_of(mpcs1, [](const auto& m)
                               { return m->has_cross_block_masters(); });
  if (cross)
  {
    // The block of a master is its position among the constraints created
    // together, which must then be the position in the system
    auto in_order = [](const auto& mpcs)
    {
      for (std::size_t k = 0; k < mpcs.size(); ++k)
      {
        if (mpcs[k]->block() != static_cast<int>(k)
            or mpcs[k]->function_spaces().size() != mpcs.size())
        {
          return false;
        }
      }
      return true;
    };
    if (!in_order(mpcs0) or !in_order(mpcs1))
    {
      throw std::invalid_argument(
          "A constraint has masters in another block, so the blocks of the "
          "system must be the constraints in the order they were finalized "
          "in, for the rows and for the columns.");
    }
  }

  MPI_Comm comm = mpcs0.front()->function_space()->mesh()->comm();
  std::vector<std::vector<dolfinx::la::SparsityPattern>> patterns(mpcs0.size());
  for (std::size_t k = 0; k < mpcs0.size(); ++k)
  {
    patterns[k].reserve(mpcs1.size());
    const auto& V0 = *mpcs0[k]->function_space();
    for (std::size_t l = 0; l < mpcs1.size(); ++l)
    {
      const auto& V1 = *mpcs1[l]->function_space();
      std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> maps
          = {V0.dofmap()->index_map, V1.dofmap()->index_map};
      patterns[k].emplace_back(comm, maps,
                               std::array<int, 2>{V0.dofmap()->index_map_bs(),
                                                  V1.dofmap()->index_map_bs()});
    }
  }

  for (std::size_t i = 0; i < mpcs0.size(); ++i)
  {
    for (std::size_t j = 0; j < mpcs1.size(); ++j)
    {
      const dolfinx::fem::Form<T>* form = a[i][j];
      if (!form)
        continue;
      build_standard_pattern<T>(patterns[i][j], *form);
      // A master in the form's own block goes to the form's block (i, j),
      // which is its position here also when the lists are not in creation
      // order; any other block is the position of the master's block.
      const int block0 = mpcs0[i]->block(), block1 = mpcs1[j]->block();
      populate_mpc_pattern<T, U>(*form, *mpcs0[i], *mpcs1[j],
                                 [&patterns, i, j, block0, block1](
                                     int rb, std::span<const std::int32_t> rows,
                                     int cb, std::span<const std::int32_t> cols)
                                 {
                                   const std::size_t r = rb == block0 ? i : rb;
                                   const std::size_t c = cb == block1 ? j : cb;
                                   patterns[r][c].insert(rows, cols);
                                 });
    }
  }

  // The slave diagonal of each diagonal block. A diagonal block without a
  // form reserves its whole owned diagonal, for the Dirichlet rows of its
  // space, which only the assembly knows.
  for (std::size_t k = 0; k < std::min(mpcs0.size(), mpcs1.size()); ++k)
  {
    if (mpcs0[k] != mpcs1[k])
      continue;
    if (a[k][k])
      insert_slave_diagonal_pattern(patterns[k][k], *mpcs0[k]);
    else
    {
      std::vector<std::int32_t> owned(
          mpcs0[k]->function_space()->dofmap()->index_map->size_local());
      std::iota(owned.begin(), owned.end(), 0);
      patterns[k][k].insert_diagonal(owned);
    }
  }

  return patterns;
}

/// @brief Create a PETSc matrix for a merged block sparsity pattern, with the
/// dofs of every block ordered `[owned, ghosts]`.
///
/// The local-to-global maps list the blocks one after another,
/// `[owned_0, ghosts_0, owned_1, ghosts_1, ...]`, as
/// `MatGetLocalSubMatrix` with the index sets of
/// `dolfinx::la::petsc::create_index_sets` expects.
///
/// @note This is a copy of the second half of
/// `dolfinx::fem::petsc::create_matrix_block` (`dolfinx/fem/petsc.h`), which
/// builds the pattern from forms. It is separate here so that the two can be
/// compared, and has to be kept in sync with it.
/// @param[in] comm The communicator of the matrix
/// @param[in] pattern The finalized merged pattern of all blocks
/// @param[in] maps Index map and block size of every block, for the rows and
/// the columns
/// @param[in] type The PETSc matrix type, or the default if empty
/// @return The matrix. The caller is responsible for destroying it.
inline Mat create_block_matrix(
    MPI_Comm comm, const dolfinx::la::SparsityPattern& pattern,
    const std::array<
        std::vector<std::pair<
            std::reference_wrapper<const dolfinx::common::IndexMap>, int>>,
        2>& maps,
    const std::string& type)
{
  // The maps of the rows and the columns coincide for a square system, so the
  // second is not computed again
  const bool square
      = maps[0].size() == maps[1].size()
        and std::ranges::equal(maps[0], maps[1],
                               [](const auto& x, const auto& y)
                               {
                                 return &x.first.get() == &y.first.get()
                                        and x.second == y.second;
                               });

  std::array<std::vector<PetscInt>, 2> l2g;
  for (int d = 0; d < 2; ++d)
  {
    if (d == 1 and square)
    {
      l2g[1] = l2g[0];
      continue;
    }
    const auto [rank_offset, local_offset, ghosts, _]
        = dolfinx::common::stack_index_maps(maps[d]);
    std::vector<PetscInt>& map = l2g[d];
    for (std::size_t f = 0; f < maps[d].size(); ++f)
    {
      const std::int32_t offset = local_offset[f];
      const dolfinx::common::IndexMap& imap = maps[d][f].first.get();
      const int bs = maps[d][f].second;
      for (std::int32_t i = 0; i < bs * imap.size_local(); ++i)
        map.push_back(i + rank_offset + offset);
      map.insert(map.end(), ghosts[f].begin(), ghosts[f].end());
    }
  }

  ISLocalToGlobalMapping l2g0 = nullptr, l2g1 = nullptr;
  dolfinx::common::petsc::check(
      ISLocalToGlobalMappingCreate(comm, 1, l2g[0].size(), l2g[0].data(),
                                   PETSC_COPY_VALUES, &l2g0),
      "ISLocalToGlobalMappingCreate");
  if (!square)
  {
    dolfinx::common::petsc::check(
        ISLocalToGlobalMappingCreate(comm, 1, l2g[1].size(), l2g[1].data(),
                                     PETSC_COPY_VALUES, &l2g1),
        "ISLocalToGlobalMappingCreate");
  }

  Mat A = dolfinx::la::petsc::create_matrix(comm, pattern, type, l2g0,
                                            l2g1 ? l2g1 : l2g0);
  dolfinx::common::petsc::check(ISLocalToGlobalMappingDestroy(&l2g0),
                                "ISLocalToGlobalMappingDestroy");
  if (l2g1)
  {
    dolfinx::common::petsc::check(ISLocalToGlobalMappingDestroy(&l2g1),
                                  "ISLocalToGlobalMappingDestroy");
  }
  return A;
}

/// @brief Create a monolithic matrix for a rectangular array of bilinear forms
/// under multi point constraints.
///
/// Block `(i, j)` is the form `a[i][j]` with constraint `mpcs0[i]` on its rows
/// and `mpcs1[j]` on its columns, so the matrix layout is that of
/// `dolfinx::fem::petsc::create_matrix_block`, with the extended index maps of
/// the constraints in place of the original ones. The local-to-global map
/// orders the dofs `[owned_0, ghosts_0, owned_1, ghosts_1, ...]`, which is what
/// `MatGetLocalSubMatrix` with the sets of
/// `dolfinx::la::petsc::create_index_sets` expects.
///
/// A diagonal block without a form still reserves the diagonal entries of its
/// slaves, which `insert_slave_diagonal` writes.
/// @param[in] a Forms, with `nullptr` for a block that has none
/// @param[in] mpcs0 Constraint of each block row
/// @param[in] mpcs1 Constraint of each block column
/// @param[in] type The PETSc matrix type, or the default if empty
/// @return The matrix. The caller is responsible for destroying it.
template <typename T, std::floating_point U>
Mat create_matrix_block(
    const std::vector<std::vector<const dolfinx::fem::Form<T>*>>& a,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs0,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs1,
    const std::optional<std::string>& type = std::nullopt)
{
  dolfinx::common::Timer timer("~MPC: Create block matrix");
  if (mpcs0.empty() or mpcs1.empty() or a.size() != mpcs0.size())
  {
    throw std::invalid_argument(
        "Expected one row of forms per row constraint, and at least one row "
        "and column.");
  }
  for (const std::vector<const dolfinx::fem::Form<T>*>& row : a)
  {
    if (row.size() != mpcs1.size())
    {
      throw std::invalid_argument(
          "Expected one form per column constraint in every row.");
    }
  }

  // Block patterns of the constraints, with the extended index maps
  std::vector<std::vector<dolfinx::la::SparsityPattern>> patterns
      = create_sparsity_patterns<T, U>(a, mpcs0, mpcs1);

  std::array<std::vector<std::pair<
                 std::reference_wrapper<const dolfinx::common::IndexMap>, int>>,
             2>
      maps;
  std::array<std::vector<int>, 2> bs_dofs;
  for (const auto& mpc : mpcs0)
  {
    const auto& V = *mpc->function_space();
    maps[0].emplace_back(*V.dofmap()->index_map, V.dofmap()->index_map_bs());
    bs_dofs[0].push_back(V.dofmap()->bs());
  }
  for (const auto& mpc : mpcs1)
  {
    const auto& V = *mpc->function_space();
    maps[1].emplace_back(*V.dofmap()->index_map, V.dofmap()->index_map_bs());
    bs_dofs[1].push_back(V.dofmap()->bs());
  }

  std::vector<std::vector<const dolfinx::la::SparsityPattern*>> p(
      patterns.size());
  for (std::size_t row = 0; row < patterns.size(); ++row)
    for (const auto& pattern : patterns[row])
      p[row].push_back(&pattern);

  MPI_Comm comm = mpcs0.front()->function_space()->mesh()->comm();
  dolfinx::la::SparsityPattern pattern(comm, p, maps, bs_dofs);
  pattern.finalize();

  return create_block_matrix(comm, pattern, maps, type.value_or(std::string()));
}

/// @brief Create a nest matrix for a rectangular array of bilinear forms under
/// multi point constraints.
///
/// A block gets a matrix if it has a form, if it is a diagonal block (which
/// holds the diagonal of its slaves), or, when a constraint has masters in
/// another block, always: those masters put entries in blocks that have no
/// form. The decision depends only on the forms and on globally reduced data,
/// so every process creates the same blocks.
/// @param[in] a Forms, with `nullptr` for a block that has none
/// @param[in] mpcs0 Constraint of each block row
/// @param[in] mpcs1 Constraint of each block column
/// @param[in] types PETSc matrix type of each block, or the default if unset
/// @return The matrix. The caller is responsible for destroying it.
template <typename T, std::floating_point U>
Mat create_matrix_nest(
    const std::vector<std::vector<const dolfinx::fem::Form<T>*>>& a,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs0,
    const std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>&
        mpcs1,
    const std::optional<std::vector<std::vector<std::optional<std::string>>>>&
        types = std::nullopt)
{
  dolfinx::common::Timer timer("~MPC: Create nest matrix");
  std::vector<std::vector<dolfinx::la::SparsityPattern>> patterns
      = create_sparsity_patterns<T, U>(a, mpcs0, mpcs1);
  const bool cross
      = std::ranges::any_of(mpcs0, [](const auto& m)
                            { return m->has_cross_block_masters(); })
        or std::ranges::any_of(mpcs1, [](const auto& m)
                               { return m->has_cross_block_masters(); });

  MPI_Comm comm = mpcs0.front()->function_space()->mesh()->comm();
  const std::size_t nr = mpcs0.size(), nc = mpcs1.size();
  std::vector<Mat> mats(nr * nc, nullptr);
  for (std::size_t k = 0; k < nr; ++k)
  {
    for (std::size_t l = 0; l < nc; ++l)
    {
      const bool needed = a[k][l] or (k == l and mpcs0[k] == mpcs1[l]) or cross;
      if (!needed)
        continue;
      patterns[k][l].finalize();
      std::string type;
      if (types and (*types)[k][l])
        type = *(*types)[k][l];
      mats[k * nc + l]
          = dolfinx::la::petsc::create_matrix(comm, patterns[k][l], type);
    }
  }

  Mat A;
  dolfinx::common::petsc::check(
      MatCreateNest(comm, nr, nullptr, nc, nullptr, mats.data(), &A),
      "MatCreateNest");
  // The nest holds its own reference to each block
  for (Mat& m : mats)
    if (m)
      MatDestroy(&m);
  return A;
}

/// Compute the dot product u . vs
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The dot product `u . v`. The type will be the same as value size
/// of u.
template <typename U, typename V>
typename U::value_type dot(const U& u, const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/// Distribute local slave->master data from owning process to ghost processes
/// @param[in] slaves List of local slaves indices (local to process, unrolled)
/// @param[in] masters The corresponding master dofs (global indices, unrolled)
/// @param[in] coeffs The master coefficients
/// @param[in] owners The owners of the corresponding master dof
/// @param[in] num_masters_per_slave The number of masters owned by each slave
/// @param[in] imap The index map
/// @param[in] bs The index map block size
/// @returns Data structure holding the received slave->master data
template <typename T>
dolfinx_mpc::mpc_data<T> distribute_ghost_data(
    std::span<const std::int32_t> slaves, std::span<const std::int64_t> masters,
    std::span<const T> coeffs, std::span<const std::int32_t> owners,
    std::span<const std::int32_t> num_masters_per_slave,
    std::shared_ptr<const dolfinx::common::IndexMap> imap, const int bs)
{
  std::shared_ptr<const dolfinx::common::IndexMap> slave_to_ghost;
  std::vector<int> parent_to_sub;
  parent_to_sub.reserve(slaves.size());

  // Create new index map for each slave block
  {
    // Fill in owned blocks
    std::vector<std::int32_t> blocks;
    blocks.reserve(slaves.size());
    std::ranges::transform(slaves, std::back_inserter(blocks),
                           [bs](auto& dof) { return dof / bs; });

    // Propagate local slave information to ghost processes
    dolfinx::la::Vector<std::int8_t> indicator(imap, 1);
    std::ranges::fill(indicator.array(), 0);
    std::vector<std::int8_t>& indicator_array = indicator.array();
    std::ranges::for_each(blocks, [&indicator_array](auto& block)
                          { indicator_array[block] = 1; });

    indicator.scatter_fwd();

    // Insert ghosts blocks with constraints into blocks
    const std::int32_t local_size = imap->size_local();
    for (std::size_t i = 0; i < imap->num_ghosts(); ++i)
      if (indicator_array[local_size + i] == 1)
        blocks.push_back(local_size + i);

    // Sort and delete duplicates
    std::ranges::sort(blocks);
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());

    // Create submap
    std::tuple<dolfinx::common::IndexMap, std::vector<int32_t>, bool>
        compressed_map = dolfinx::common::create_sub_index_map(
            *imap, blocks, dolfinx::common::IndexMapOrder::any);

    // Copy of function from https://github.com/FEniCS/dolfinx/pull/4479/
    // by Garth Wells, subject to LGPL-3.0 License.
    // Throw if any rank saw a dof acquire a new owner when building a
    // sub-index map. `create_sub_index_map` reports this per rank, so it is
    // reduced first: throwing on only some ranks would leave the others in
    // a later collective. Developer builds only, since the check requires
    // MPI communication.
    auto reject_owner_change
        = []([[maybe_unused]] const dolfinx::common::IndexMap& map,
             [[maybe_unused]] bool owners_changed)
    {
      int changed = owners_changed;
      int changed_any;
      const int ierr = MPI_Allreduce(&changed, &changed_any, 1, MPI_INT,
                                     MPI_LOR, map.comm());
      dolfinx::MPI::check_error(map.comm(), ierr);
      if (changed_any)
        throw std::runtime_error("Index owner change detected.");
    };
    reject_owner_change(*imap, std::get<2>(compressed_map));

    slave_to_ghost = std::make_shared<const dolfinx::common::IndexMap>(
        std::move(std::get<0>(compressed_map)));
    // Build map from new index map to slave indices (unrolled)
    //
    // `blocks` was sorted and deduplicated above, before being handed to
    // `create_sub_index_map`, so a binary search is valid here
    for (std::size_t i = 0; i < slaves.size(); i++)
    {
      const std::int32_t block = slaves[i] / bs;
      auto it = std::ranges::lower_bound(blocks, block);
      assert(it != blocks.end() and *it == block);
      std::size_t index = std::ranges::distance(blocks.begin(), it);
      parent_to_sub.push_back((int)index);
    }
  }

  // Get communicator for owner->ghost
  MPI_Comm local_to_ghost = create_owner_to_ghost_comm(*slave_to_ghost);
  std::span src_ranks_ghosts = slave_to_ghost->src();
  std::span dest_ranks_ghosts = slave_to_ghost->dest();

  // Compute number of outgoing slaves and masters for each process
  auto [im_data, im_offsets] = slave_to_ghost->index_to_dest_ranks();
  dolfinx::graph::AdjacencyList<int> shared_indices(std::move(im_data),
                                                    std::move(im_offsets));

  const std::size_t num_inc_proc = src_ranks_ghosts.size();
  const std::size_t num_out_proc = dest_ranks_ghosts.size();
  std::vector<std::int32_t> out_num_slaves(num_out_proc + 1, 0);
  std::vector<std::int32_t> out_num_masters(num_out_proc + 1, 0);
  for (std::size_t i = 0; i < slaves.size(); ++i)
  {
    for (auto proc : shared_indices.links(parent_to_sub[i]))
    {
      // Find index of process in local MPI communicator
      auto it = std::ranges::find(dest_ranks_ghosts, proc);
      const auto index = std::ranges::distance(dest_ranks_ghosts.begin(), it);
      out_num_masters[index] += num_masters_per_slave[i];
      out_num_slaves[index]++;
    }
  }

  // Communicate number of incoming slaves and masters
  std::vector<int> in_num_slaves(num_inc_proc + 1);
  std::vector<int> in_num_masters(num_inc_proc + 1);
  std::array<MPI_Request, 2> requests;
  std::array<MPI_Status, 2> states;
  MPI_Ineighbor_alltoall(out_num_slaves.data(), 1, MPI_INT,
                         in_num_slaves.data(), 1, MPI_INT, local_to_ghost,
                         &requests[0]);
  out_num_slaves.pop_back();
  in_num_slaves.pop_back();
  MPI_Ineighbor_alltoall(out_num_masters.data(), 1, MPI_INT,
                         in_num_masters.data(), 1, MPI_INT, local_to_ghost,
                         &requests[1]);
  out_num_masters.pop_back();
  in_num_masters.pop_back();
  // Compute out displacements for slaves and masters
  std::vector<std::int32_t> disp_out_masters(num_out_proc + 1, 0);
  std::partial_sum(out_num_masters.begin(), out_num_masters.end(),
                   disp_out_masters.begin() + 1);
  std::vector<std::int32_t> disp_out_slaves(num_out_proc + 1, 0);
  std::partial_sum(out_num_slaves.begin(), out_num_slaves.end(),
                   disp_out_slaves.begin() + 1);

  // Compute displacement of masters to able to insert them correctly
  std::vector<std::int32_t> local_offsets(slaves.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   local_offsets.begin() + 1);

  // Insertion counter
  std::vector<std::int32_t> insert_slaves(num_out_proc, 0);
  std::vector<std::int32_t> insert_masters(num_out_proc, 0);

  // Prepare arrays for sending ghost information
  std::vector<std::int64_t> masters_out(disp_out_masters.back());
  std::vector<T> coeffs_out(disp_out_masters.back());
  std::vector<std::int32_t> owners_out(disp_out_masters.back());
  std::vector<std::int32_t> slaves_out_loc(disp_out_slaves.back());
  std::vector<std::int64_t> slaves_out(disp_out_slaves.back());
  std::vector<std::int32_t> masters_per_slave(disp_out_slaves.back());
  for (std::size_t i = 0; i < slaves.size(); ++i)
  {
    // Find ghost processes for the ith local slave
    const std::int32_t master_start = local_offsets[i];
    const std::int32_t master_end = local_offsets[i + 1];
    for (auto proc : shared_indices.links(parent_to_sub[i]))
    {
      // Find index of process in local MPI communicator
      auto it = std::ranges::find(dest_ranks_ghosts, proc);
      const auto index = std::ranges::distance(dest_ranks_ghosts.begin(), it);

      // Insert slave and num masters per slave
      slaves_out_loc[disp_out_slaves[index] + insert_slaves[index]] = slaves[i];
      masters_per_slave[disp_out_slaves[index] + insert_slaves[index]]
          = num_masters_per_slave[i];
      insert_slaves[index]++;

      // Insert global master dofs to send
      std::ranges::copy(masters.begin() + master_start,
                        masters.begin() + master_end,
                        masters_out.begin() + disp_out_masters[index]
                            + insert_masters[index]);
      // Insert owners to send
      std::ranges::copy(
          owners.begin() + master_start, owners.begin() + master_end,
          owners_out.begin() + disp_out_masters[index] + insert_masters[index]);
      // Insert coeffs to send
      std::ranges::copy(
          coeffs.begin() + master_start, coeffs.begin() + master_end,
          coeffs_out.begin() + disp_out_masters[index] + insert_masters[index]);
      insert_masters[index] += num_masters_per_slave[i];
    }
  }
  // Map slaves to global index
  {
    std::vector<std::int32_t> blocks(slaves_out_loc.size());
    std::vector<std::int32_t> rems(slaves_out_loc.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
      std::div_t pos = std::div(slaves_out_loc[i], bs);
      blocks[i] = pos.quot;
      rems[i] = pos.rem;
    }
    imap->local_to_global(blocks, slaves_out);
    std::ranges::transform(slaves_out, rems, slaves_out.begin(),
                           [bs](auto dof, auto rem) { return dof * bs + rem; });
  }

  // Create in displacements for slaves
  MPI_Wait(&requests[0], &states[0]);
  std::vector<std::int32_t> disp_in_slaves(num_inc_proc + 1, 0);
  std::partial_sum(in_num_slaves.begin(), in_num_slaves.end(),
                   disp_in_slaves.begin() + 1);

  // Create in displacements for masters
  MPI_Wait(&requests[1], &states[1]);
  std::vector<std::int32_t> disp_in_masters(num_inc_proc + 1, 0);
  std::partial_sum(in_num_masters.begin(), in_num_masters.end(),
                   disp_in_masters.begin() + 1);

  // Send data to ghost processes
  std::vector<MPI_Request> ghost_requests(5);
  std::vector<MPI_Status> ghost_status(5);

  // Receive slaves from owner
  std::vector<std::int64_t> recv_slaves(disp_in_slaves.back());
  MPI_Ineighbor_alltoallv(
      slaves_out.data(), out_num_slaves.data(), disp_out_slaves.data(),
      dolfinx::MPI::mpi_t<std::int64_t>, recv_slaves.data(),
      in_num_slaves.data(), disp_in_slaves.data(),
      dolfinx::MPI::mpi_t<std::int64_t>, local_to_ghost, &ghost_requests[0]);

  // Receive number of masters from owner
  std::vector<std::int32_t> recv_num(disp_in_slaves.back());
  MPI_Ineighbor_alltoallv(
      masters_per_slave.data(), out_num_slaves.data(), disp_out_slaves.data(),
      dolfinx::MPI::mpi_t<std::int32_t>, recv_num.data(), in_num_slaves.data(),
      disp_in_slaves.data(), dolfinx::MPI::mpi_t<std::int32_t>, local_to_ghost,
      &ghost_requests[1]);

  // Convert slaves to local index
  MPI_Wait(&ghost_requests[0], &ghost_status[0]);
  std::vector<std::int64_t> recv_block;
  recv_block.reserve(recv_slaves.size());
  std::vector<std::int32_t> recv_rem;
  recv_rem.reserve(recv_slaves.size());
  std::ranges::for_each(recv_slaves,
                        [bs, &recv_rem, &recv_block](const auto dof)
                        {
                          recv_rem.push_back(dof % bs);
                          recv_block.push_back(dof / bs);
                        });
  std::vector<std::int32_t> recv_local(recv_slaves.size());
  imap->global_to_local(recv_block, recv_local);
  for (std::size_t i = 0; i < recv_local.size(); i++)
    recv_local[i] = recv_local[i] * bs + recv_rem[i];

  MPI_Wait(&ghost_requests[1], &ghost_status[1]);

  // Receive masters, coeffs and owners from owning processes
  std::vector<std::int64_t> recv_masters(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(
      masters_out.data(), out_num_masters.data(), disp_out_masters.data(),
      dolfinx::MPI::mpi_t<std::int64_t>, recv_masters.data(),
      in_num_masters.data(), disp_in_masters.data(),
      dolfinx::MPI::mpi_t<std::int64_t>, local_to_ghost, &ghost_requests[2]);
  std::vector<std::int32_t> recv_owners(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(
      owners_out.data(), out_num_masters.data(), disp_out_masters.data(),
      dolfinx::MPI::mpi_t<std::int32_t>, recv_owners.data(),
      in_num_masters.data(), disp_in_masters.data(),
      dolfinx::MPI::mpi_t<std::int32_t>, local_to_ghost, &ghost_requests[3]);
  std::vector<T> recv_coeffs(disp_in_masters.back());
  MPI_Ineighbor_alltoallv(coeffs_out.data(), out_num_masters.data(),
                          disp_out_masters.data(), dolfinx::MPI::mpi_t<T>,
                          recv_coeffs.data(), in_num_masters.data(),
                          disp_in_masters.data(), dolfinx::MPI::mpi_t<T>,
                          local_to_ghost, &ghost_requests[4]);

  int err = MPI_Comm_free(&local_to_ghost);
  dolfinx::MPI::check_error(imap->comm(), err);

  mpc_data<T> ghost_data;
  ghost_data.slaves = recv_local;
  ghost_data.offsets = recv_num;

  MPI_Wait(&ghost_requests[2], &ghost_status[2]);
  ghost_data.masters = recv_masters;
  MPI_Wait(&ghost_requests[3], &ghost_status[3]);
  ghost_data.owners = recv_owners;
  MPI_Wait(&ghost_requests[4], &ghost_status[4]);
  ghost_data.coeffs = recv_coeffs;
  return ghost_data;
}

/// @brief The default distance and coefficient tolerance of the constraints:
/// 500 machine epsilon of `U`.
template <std::floating_point U>
constexpr U default_tolerance()
{
  return 500 * std::numeric_limits<U>::epsilon();
}

/// @brief Append the masters of one slave, without those whose coefficient is
/// below `coefficient_tol` times the largest in magnitude.
/// @param[in] row_masters The candidate masters of the slave (global)
/// @param[in] row_coeffs The coefficient of each candidate
/// @param[in] row_owners The process owning each candidate
/// @param[in] coefficient_tol The relative tolerance. 0 keeps every master.
/// @param[in,out] masters The masters, appended to
/// @param[in,out] coeffs The coefficients, appended to
/// @param[in,out] owners The owners, appended to
/// @return The number of masters appended
template <typename T, std::floating_point U>
std::int32_t append_significant_masters(
    std::span<const std::int64_t> row_masters, std::span<const T> row_coeffs,
    std::span<const std::int32_t> row_owners, U coefficient_tol,
    std::vector<std::int64_t>& masters, std::vector<T>& coeffs,
    std::vector<std::int32_t>& owners)
{
  U largest = 0;
  for (const T& c : row_coeffs)
    largest = std::max<U>(largest, std::abs(c));
  const U cut = coefficient_tol * largest;
  std::int32_t num = 0;
  for (std::size_t j = 0; j < row_coeffs.size(); ++j)
  {
    if (std::abs(row_coeffs[j]) >= cut)
    {
      masters.push_back(row_masters[j]);
      coeffs.push_back(row_coeffs[j]);
      owners.push_back(row_owners[j]);
      ++num;
    }
  }
  return num;
}

/// @brief Complete the rows of owned slaves with the rows of the slaves that
/// are ghosts on this process.
/// @param[in] slaves The owned slaves (local, unrolled)
/// @param[in] masters The masters of each slave (global, unrolled)
/// @param[in] coeffs The coefficient of each master
/// @param[in] owners The process owning each master
/// @param[in] num_masters The number of masters of each slave
/// @param[in] imap The index map of the slaves' space
/// @param[in] bs The block size of `imap`
/// @return The constraint, owned slaves first
/// @note Collective.
template <typename T>
mpc_data<T>
add_ghost_rows(std::vector<std::int32_t>&& slaves,
               std::vector<std::int64_t>&& masters, std::vector<T>&& coeffs,
               std::vector<std::int32_t>&& owners,
               std::vector<std::int32_t>&& num_masters,
               std::shared_ptr<const dolfinx::common::IndexMap> imap, int bs)
{
  mpc_data<T> ghosts = distribute_ghost_data<T>(slaves, masters, coeffs, owners,
                                                num_masters, imap, bs);
  slaves.insert(slaves.end(), ghosts.slaves.begin(), ghosts.slaves.end());
  masters.insert(masters.end(), ghosts.masters.begin(), ghosts.masters.end());
  coeffs.insert(coeffs.end(), ghosts.coeffs.begin(), ghosts.coeffs.end());
  owners.insert(owners.end(), ghosts.owners.begin(), ghosts.owners.end());
  num_masters.insert(num_masters.end(), ghosts.offsets.begin(),
                     ghosts.offsets.end());

  mpc_data<T> out;
  out.offsets.assign(num_masters.size() + 1, 0);
  std::partial_sum(num_masters.begin(), num_masters.end(),
                   std::next(out.offsets.begin()));
  out.slaves = std::move(slaves);
  out.masters = std::move(masters);
  out.coeffs = std::move(coeffs);
  out.owners = std::move(owners);
  return out;
}

//-----------------------------------------------------------------------------
/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3), flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] u The values at the points. Values are not computed
/// for points with a negative cell index. This argument must be
/// passed with the correct size.
/// @param[in] tol The stopping tolerance for the l2 norm of the Newton update
/// when pulling back a point on a non-affine cell. For affine cells this value
/// is not used.
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size). Flattened row major
/// @param[in] num_threads The number of threads to use for certain operations.
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 3>>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace<U>& V,
                         std::span<const U> x,
                         std::span<const std::int32_t> cells, const U tol,
                         const std::size_t num_threads)
{
  assert(x.size() % 3 == 0);
  const std::size_t num_points = x.size() / 3;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get mesh
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get geometry data
  if (mesh->geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmaps().front();

  const dolfinx::fem::CoordinateElement<U>& cmap
      = mesh->geometry().cmaps().front();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();

  // Get element
  auto element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size = element->reference_value_size();

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error(
        "Evaluation of basis functions is not supported for mixed "
        "elements. Extract subspaces.");
  }

  // Return early if we have no points
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);

  assert(basis_shape[2]
         == std::size_t(element->space_dimension() / bs_element));
  assert(basis_shape[3] == std::size_t(element->reference_value_size()));
  std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<U> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), 1, std::multiplies{}));

  if (num_points == 0)
    return {output_basis, reference_shape};

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_cell_permutations(num_threads);
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  const std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
  const std::size_t phi_size
      = std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{});
  std::vector<U> phi0_b(phi_size);
  cmdspan4_t phi0(phi0_b.data(), phi_shape);
  cmap.tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Reference coordinates and geometry data at each point
  std::vector<U> Xb(num_points * tdim);
  mdspan2_t X(Xb.data(), num_points, tdim);
  std::vector<U> J_b(num_points * gdim * tdim);
  mdspan3_t J(J_b.data(), num_points, gdim, tdim);
  std::vector<U> K_b(num_points * tdim * gdim);
  mdspan3_t K(K_b.data(), num_points, tdim, gdim);
  std::vector<U> detJ(num_points);

  // Basis on the reference element at each point
  const std::size_t num_reference_dofs = basis_shape[2];
  const std::size_t num_basis_values = num_reference_dofs * basis_shape[3];
  std::vector<U> reference_basisb(num_points * num_basis_values);

  using xu_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xU_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xJ_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xK_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation = element->template dof_transformation_fn<U>(
      dolfinx::fem::doftransform::standard);
  const bool transform_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation);
  mdspan3_t full_basis(output_basis.data(), reference_shape);

  // Evaluate the basis at the points [p0, p1). Each call has its own scratch
  // and writes only to the data of its points, so calls can run concurrently.
  auto evaluate_points
      = [&cells, &x, &x_dofmap, &x_g, &cmap, &dphi0, &X, &Xb, &J, &K, &detJ,
         &reference_basisb, &element, &cell_info, &apply_dof_transformation,
         &push_forward_fn, &full_basis, &phi_shape, num_dofs_g, gdim, tdim,
         phi_size, num_reference_dofs, num_basis_values, transform_set,
         reference_value_size, tol](std::size_t p0, std::size_t p1)
  {
    std::vector<U> coord_dofs_b(num_dofs_g * gdim);
    mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
    std::vector<U> xp_b(gdim);
    mdspan2_t xp(xp_b.data(), 1, gdim);

    // Geometry basis at a point, used in non-affine case
    std::vector<U> phi_b(phi_size);
    cmdspan4_t phi(phi_b.data(), phi_shape);
    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi, std::pair(1, tdim + 1), 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    std::vector<U> pull_back_scratch(
        cmap.is_affine() ? 0 : cmap.pull_back_working_size(gdim));
    std::vector<U> det_scratch(2 * gdim * tdim);

    for (std::size_t p = p0; p < p1; ++p)
    {
      const std::int32_t cell_index = cells[p];

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Get cell geometry (coordinate dofs)
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cell_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const std::int32_t pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coord_dofs(i, j) = x_g[pos + j];
      }

      for (std::size_t j = 0; j < gdim; ++j)
        xp(0, j) = x[3 * p + j];

      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

      std::array<U, 3> Xpb = {0, 0, 0};
      MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
          U,
          MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
              std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
          Xp(Xpb.data(), 1, tdim);

      // Compute reference coordinates X, and J, detJ and K
      if (cmap.is_affine())
      {
        dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs,
                                                             _J);
        dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
        std::array<U, 3> x0 = {0, 0, 0};
        for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
          x0[i] += coord_dofs(0, i);
        dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K, x0, xp);
      }
      else
      {
        // Pull-back physical point xp to reference coordinate Xp
        cmap.pull_back_nonaffine(Xp, xp, coord_dofs, pull_back_scratch, tol,
                                 15);
        cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
        dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi, coord_dofs,
                                                             _J);
        dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      }
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);

      for (std::size_t j = 0; j < X.extent(1); ++j)
        X(p, j) = Xpb[j];
    }

    // Compute basis on reference element
    std::span<U> reference_basis
        = std::span(reference_basisb)
              .subspan(p0 * num_basis_values, (p1 - p0) * num_basis_values);
    element->tabulate(
        reference_basis,
        std::span<const U>(Xb).subspan(p0 * tdim, (p1 - p0) * tdim),
        {p1 - p0, tdim}, 0);

    // Data structure to hold basis for transformation
    std::vector<U> basis_valuesb(num_basis_values);
    mdspan2_t basis_values(basis_valuesb.data(), num_reference_dofs,
                           reference_value_size);
    for (std::size_t p = p0; p < p1; ++p)
    {
      const std::int32_t cell_index = cells[p];
      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Permute the reference values to account for the cell's orientation
      std::ranges::copy_n(
          std::next(reference_basis.begin(), num_basis_values * (p - p0)),
          num_basis_values, basis_valuesb.begin());
      if (transform_set)
      {
        apply_dof_transformation(basis_valuesb, cell_info, cell_index,
                                 (int)reference_value_size);
      }

      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          full_basis, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      push_forward_fn(_U, basis_values, _J, detJ[p], _K);
    }
  };

  const int num_chunks = std::max<std::size_t>(
      1, std::min<std::size_t>(num_threads, num_points));
  if (num_chunks < 2)
    evaluate_points(0, num_points);
  else
  {
    // An exception must not escape a spawned thread: each stores its own,
    // which is rethrown once all have joined
    auto try_evaluate_points
        = [&evaluate_points](std::size_t p0, std::size_t p1,
                             std::exception_ptr& error)
    {
      try
      {
        evaluate_points(p0, p1);
      }
      catch (...)
      {
        error = std::current_exception();
      }
    };

    std::vector<std::exception_ptr> errors(num_chunks - 1);
    {
      std::vector<std::jthread> threads;
      for (int i = 1; i < num_chunks; ++i)
      {
        auto [p0, p1] = dolfinx::common::local_range(i, num_points, num_chunks);
        threads.emplace_back(try_evaluate_points, p0, p1,
                             std::ref(errors[i - 1]));
      }
      auto [p0, p1] = dolfinx::common::local_range(0, num_points, num_chunks);
      evaluate_points(p0, p1);
    }
    for (const std::exception_ptr& error : errors)
      if (error)
        std::rethrow_exception(error);
  }
  return {output_basis, reference_shape};
}

//-----------------------------------------------------------------------------
/// Tabuilate dof coordinates (not unrolled for block size) for a set of points
/// and corresponding cells.
/// @param[in] V The function space
/// @param[in] dofs Array of dofs (not unrolled with block size)
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of a cell that contains dofs[i]
/// @param[in] transposed If true return coordiantes in xxyyzz format. Else
/// xyzxzyxzy
/// @param[in] num_threads The number of threads to use for certain operations.
/// @returns The dof coordinates flattened in the appropriate format
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 2>> tabulate_dof_coordinates(
    const dolfinx::fem::FunctionSpace<U>& V, std::span<const std::int32_t> dofs,
    std::span<const std::int32_t> cells, bool transposed = false,
    const std::size_t num_threads = 1)
{
  if (!V.component().empty())
  {
    throw std::runtime_error("Cannot tabulate coordinates for a "
                             "FunctionSpace that is a subspace.");
  }
  auto element = V.element();
  assert(element);
  if (V.element()->is_mixed())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a mixed FunctionSpace.");
  }

  auto mesh = V.mesh();
  assert(mesh);

  const std::size_t gdim = mesh->geometry().dim();

  // Get dofmap local size
  auto dofmap = V.dofmap();
  assert(dofmap);
  std::shared_ptr<const dolfinx::common::IndexMap> index_map
      = V.dofmap()->index_map;
  assert(index_map);

  const int element_block_size = element->block_size();
  const std::size_t space_dimension
      = element->space_dimension() / element_block_size;

  // Get the dof coordinates on the reference element
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  auto [X_b, X_shape] = element->interpolation_points();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement<U>& cmap
      = mesh->geometry().cmaps().front();

  // Prepare cell geometry
  if (mesh->geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmaps().front();
  std::span<const U> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = x_dofmap.extent(1);

  // Array to hold coordinates to return

  std::array<std::size_t, 2> coord_shape = {dofs.size(), 3};
  if (transposed)
    coord_shape = {3, dofs.size()};
  std::vector<U> coordsb(std::reduce(coord_shape.cbegin(), coord_shape.cend(),
                                     1, std::multiplies{}));

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

  assert(space_dimension == X_shape[0]);
  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_cell_permutations(num_threads);
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  const auto apply_dof_transformation
      = element->template dof_transformation_fn<U>(
          dolfinx::fem::doftransform::standard);
  const bool transform_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation);
  const std::array<std::size_t, 4> bsize = cmap.tabulate_shape(0, X_shape[0]);
  std::vector<U> phi_b(
      std::reduce(bsize.begin(), bsize.end(), 1, std::multiplies{}));
  cmap.tabulate(0, X_b, X_shape, phi_b);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>
      phi_full(phi_b.data(), bsize);
  auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Tabulate the coordinates of the dofs [c0, c1). Each call has its own
  // scratch and writes only the coordinates of its dofs.
  auto tabulate
      = [&dofs, &cells, &x_dofmap, &x_g, &phi, &dofmap, &cell_info,
         &apply_dof_transformation, &coordsb, num_dofs_g, gdim, space_dimension,
         transform_set, transposed](std::size_t c0, std::size_t c1)
  {
    std::vector<U> xb(space_dimension * gdim);
    mdspan2_t x(xb.data(), space_dimension, gdim);
    std::vector<U> coordinate_dofs_b(num_dofs_g * gdim);
    mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);
    for (std::size_t c = c0; c < c1; ++c)
    {
      // Fetch the coordinates of the cell
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cells[c], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const std::int32_t pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[pos + j];
      }
      // Tabulate dof coordinates on cell
      dolfinx::fem::CoordinateElement<U>::push_forward(x, coordinate_dofs, phi);
      if (transform_set)
      {
        apply_dof_transformation(xb, cell_info, cells[c],
                                 static_cast<int>(gdim));
      }

      // Copy the coordinates of the dof
      std::span<const std::int32_t> cell_dofs = dofmap->cell_dofs(cells[c]);
      const std::size_t loc = std::ranges::distance(
          cell_dofs.begin(), std::ranges::find(cell_dofs, dofs[c]));
      for (std::size_t j = 0; j < gdim; ++j)
      {
        if (transposed)
          coordsb[j * dofs.size() + c] = x(loc, j);
        else
          coordsb[c * 3 + j] = x(loc, j);
      }
    }
  };

  const int num_chunks = std::max<std::size_t>(
      1, std::min<std::size_t>(num_threads, cells.size()));
  if (num_chunks < 2)
    tabulate(0, cells.size());
  else
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_chunks; ++i)
    {
      auto [c0, c1] = dolfinx::common::local_range(i, cells.size(), num_chunks);
      threads.emplace_back(tabulate, c0, c1);
    }
    auto [c0, c1] = dolfinx::common::local_range(0, cells.size(), num_chunks);
    tabulate(c0, c1);
  }

  return {coordsb, coord_shape};
}

/// From a Mesh, find which cells collide with a set of points.
/// @note Uses the GJK algorithm, see dolfinx::geometry::compute_distance_gjk
/// for details
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points The points to check for collision, shape=(num_points, 3).
/// Flattened row major.
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
/// @return Adjacency list where the ith node is the closest entity whose
/// squared distance is within eps2
/// @note There may be nodes with no entries in the adjacency list
template <std::floating_point U>
dolfinx::graph::AdjacencyList<int> compute_colliding_cells(
    const dolfinx::mesh::Mesh<U>& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    std::span<const U> points, const U eps2)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  const int tdim = mesh.topology()->dim();
  std::vector<std::int32_t> result;
  for (std::int32_t i = 0; i < candidate_cells.num_nodes(); i++)
  {
    auto cells = candidate_cells.links(i);
    if (cells.empty())
    {
      offsets.push_back((std::int32_t)colliding_cells.size());
      continue;
    }
    // Create span of
    std::vector<U> distances_sq(cells.size());
    for (std::size_t j = 0; j < cells.size(); j++)
    {
      distances_sq[j]
          = dolfinx::geometry::squared_distance(mesh, tdim, cells.subspan(j, 1),
                                                points.subspan(3 * i, 3))
                .front();
    }
    // Only push back closest cell
    if (auto cell_idx = std::ranges::min_element(distances_sq);
        *cell_idx < eps2)
    {
      auto pos = std::ranges::distance(distances_sq.begin(), cell_idx);
      colliding_cells.push_back(cells[pos]);
    }
    offsets.push_back((std::int32_t)colliding_cells.size());
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(colliding_cells),
                                                     std::move(offsets));
}

/// Given a mesh and corresponding bounding box tree and a set of points,check
/// which cells (local to process) collide with each point.
/// Return an array of the same size as the number of points, where the ith
/// entry corresponds to the first cell colliding with the ith point.
/// @note If no colliding point is found, the index -1 is returned.
/// @param[in] mesh The mesh
/// @param[in] tree The boundingbox tree of all cells (local to process) in
/// the mesh
/// @param[in] points The points to check collision with, shape (num_points,
/// 3). Flattened row major.
/// @param[in] eps2 The tolerance for the squared distance to be considered a
/// collision
template <std::floating_point U>
std::vector<std::int32_t>
find_local_collisions(const dolfinx::mesh::Mesh<U>& mesh,
                      const dolfinx::geometry::BoundingBoxTree<U>& tree,
                      std::span<const U> points, const U eps2)
{
  assert(points.size() % 3 == 0);

  // Compute collisions for each point with BoundingBoxTree
  dolfinx::graph::AdjacencyList<std::int32_t> bbox_collisions
      = dolfinx::geometry::compute_collisions(tree, points);

  // Compute exact collision
  auto cell_collisions = dolfinx_mpc::compute_colliding_cells(
      mesh, bbox_collisions, points, eps2);

  // Extract first collision
  std::vector<std::int32_t> collisions(points.size() / 3, -1);
  for (int i = 0; i < cell_collisions.num_nodes(); i++)
  {
    auto local_cells = cell_collisions.links(i);
    if (!local_cells.empty())
      collisions[i] = local_cells[0];
  }
  return collisions;
}

/// @brief The dof blocks of `V` on the closure of the entities tagged with
/// `marker`.
///
/// A wrapper of `dolfinx::fem::locate_dofs_topological` on the dofmap of `V`.
/// For a blocked space, such as a vector space, it returns blocks, not
/// unrolled dofs: block `b` holds the dofs `b * bs + c` of every component
/// `c`, with `bs` the block size of `V`'s index map.
///
/// @param[in] V The function space
/// @param[in] meshtags Tags on entities, of any dimension, of the mesh of `V`
/// @param[in] marker The value of the tagged entities
/// @return The blocks, local to the process, ghosts included
/// @pre The connectivities between the tagged entities and the cells of the
/// mesh have been computed.
template <std::floating_point U>
std::vector<std::int32_t>
locate_tagged_blocks(const dolfinx::fem::FunctionSpace<U>& V,
                     const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                     std::int32_t marker)
{
  assert(V.mesh()->topology() == meshtags.topology());
  return dolfinx::fem::locate_dofs_topological(
      *meshtags.topology(), *V.dofmap(), meshtags.dim(), meshtags.find(marker));
}

/// Given an input array of dofs from a function space, return an array with
/// true/false if the degree of freedom is in a DirichletBC
/// @param[in] V The function space
/// @param[in] blocks The degrees of freedom (not unrolled for dofmap block
/// size)
/// @param[in] bcs List of Dirichlet BCs on V
template <typename T, std::floating_point U>
std::vector<std::int8_t> is_bc(
    const dolfinx::fem::FunctionSpace<U>& V,
    std::span<const std::int32_t> blocks,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{
  auto dofmap = V.dofmap();
  assert(dofmap);
  auto imap = dofmap->index_map;
  assert(imap);
  const int bs = dofmap->index_map_bs();
  std::int32_t dim = bs * (imap->size_local() + imap->num_ghosts());
  std::vector<std::int8_t> dof_marker(dim, false);
  std::ranges::for_each(bcs,
                        [&dof_marker, &V](auto bc)
                        {
                          assert(bc);
                          assert(bc->function_space());
                          if (bc->function_space()->contains(V))
                            bc->mark_dofs(dof_marker);
                        });
  // Remove slave blocks contained in DirichletBC
  std::vector<std::int8_t> bc_marker(blocks.size(), 0);
  const int dofmap_bs = dofmap->bs();
  for (std::size_t i = 0; i < blocks.size(); i++)
  {
    auto& block = blocks[i];
    for (int j = 0; j < dofmap_bs; j++)
    {
      if (dof_marker[block * dofmap_bs + j])
      {
        bc_marker[i] = 1;
        break;
      }
    }
  }
  return bc_marker;
}

} // namespace dolfinx_mpc

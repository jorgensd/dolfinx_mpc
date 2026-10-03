// Copyright (C) 2020 Jorgen S. Dokken and Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx_mpc;
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
dolfinx_mpc::create_block_to_cell_map(const dolfinx::mesh::Topology& topology,
                                      const dolfinx::fem::DofMap& dofmap,
                                      std::span<const std::int32_t> blocks)
{
  std::vector<std::int32_t> cells;
  cells.reserve(blocks.size());
  // Create block -> cells map

  // Compute number of cells each dof is in
  auto imap = dofmap.index_map;
  const int size_local = imap->size_local();
  std::span<const int> ghost_owners = imap->owners();
  std::vector<std::int32_t> num_cells_per_dof(size_local + ghost_owners.size());

  const int tdim = topology.dim();
  auto cell_imap = topology.index_map(tdim);
  const int num_cells_local = cell_imap->size_local();
  const int num_ghost_cells = cell_imap->num_ghosts();
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap.cell_dofs(i);
    for (auto dof : dofs)
      num_cells_per_dof[dof]++;
  }
  std::vector<std::int32_t> cell_dofs_disp(num_cells_per_dof.size() + 1, 0);
  std::partial_sum(num_cells_per_dof.begin(), num_cells_per_dof.end(),
                   cell_dofs_disp.begin() + 1);
  std::vector<std::int32_t> cell_map(cell_dofs_disp.back());
  // Reuse num_cells_per_dof for insertion
  std::ranges::fill(num_cells_per_dof, 0);

  // Create the block -> cells map
  for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
  {
    auto dofs = dofmap.cell_dofs(i);
    for (auto dof : dofs)
      cell_map[cell_dofs_disp[dof] + num_cells_per_dof[dof]++] = i;
  }

  // Populate map from slaves to corresponding cell (choose first cell in map)
  std::ranges::for_each(blocks,
                        [&cell_dofs_disp, &cell_map, &cells](const auto dof)
                        { cells.push_back(cell_map[cell_dofs_disp[dof]]); });
  assert(cells.size() == blocks.size());
  return cells;
}

//-----------------------------------------------------------------------------

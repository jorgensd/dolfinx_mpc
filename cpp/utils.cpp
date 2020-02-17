// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN-X MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/MeshIterator.h>

using namespace dolfinx_mpc;

std::pair<std::vector<std::int64_t>,
          std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>
dolfinx_mpc::locate_cells_with_dofs(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    std::vector<std::int64_t> dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const std::vector<int64_t>& global_indices
      = mesh.topology().global_indices(mesh.topology().dim());
  std::array<std::int64_t, 2> local_range = dofmap.index_map->local_range();

  /// Data structures
  std::vector<std::int64_t> cell_to_dofs;
  std::vector<std::int64_t> cell_to_dofs_offsets;
  std::vector<std::int64_t> cells_with_dofs;

  /// Loop over all cells on this process
  std::int64_t j = 0;
  cell_to_dofs_offsets.push_back(j);
  for (auto& cell : dolfinx::mesh::MeshRange(mesh, mesh.topology().dim()))
  {
    const int cell_index = cell.index();
    auto cell_dofs = dofmap.cell_dofs(cell_index);
    bool in_cell = false;
    for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
    {
      for (auto gdof : dofs)
      {
        /// Check if dof is owned by the process and if is on the cell
        if ((local_range[0] <= gdof) && (gdof < local_range[1])
            && (unsigned(cell_dofs[i] + local_range[0]) == gdof))
        {
          cell_to_dofs.push_back(gdof);
          j++;
          in_cell = true;
        }
      }
    }
    if (in_cell)
    {
      cells_with_dofs.push_back(cell_index);
      cell_to_dofs_offsets.push_back(j);
    }
  }
  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
      cells_to_dofs_map(cell_to_dofs, cell_to_dofs_offsets);

  return std::make_pair(cells_with_dofs, cells_to_dofs_map);
}

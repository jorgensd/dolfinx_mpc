// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshIterator.h>

using namespace dolfinx_mpc;

std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
          std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
                    Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>>
dolfinx_mpc::locate_cells_with_dofs(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dofs)
{
  const dolfinx::mesh::Mesh& mesh = *(V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(V->dofmap());
  const std::vector<std::int64_t> global_indices
      = dofmap.index_map->global_indices(false);

  /// Data structures
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_to_dofs;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_to_dofs_offsets;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cells_with_dofs;

  /// Loop over all cells on this process
  std::int64_t offset_index = 0;
  cell_to_dofs_offsets.conservativeResize(1);
  cell_to_dofs_offsets.tail(1) = offset_index;
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  for (int cell_index = 0; cell_index < num_cells; cell_index++)
  {

    auto cell_dofs = dofmap.cell_dofs(cell_index);

    /// Check if dof is in cell
    bool in_cell = false;
    for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
    {
      for (Eigen::Index j = 0; j < dofs.size(); ++j)
      {
        if (global_indices[cell_dofs[i]] == dofs[j])
        {
          cell_to_dofs.conservativeResize(cell_to_dofs.size() + 1);
          cell_to_dofs.tail(1) = dofs[j];
          offset_index++;
          in_cell = true;
        }
      }
    }
    if (in_cell)
    {
      cells_with_dofs.conservativeResize(cells_with_dofs.size() + 1);
      cells_with_dofs.tail(1) = cell_index;
      cell_to_dofs_offsets.conservativeResize(cell_to_dofs_offsets.size() + 1);
      cell_to_dofs_offsets.tail(1) = offset_index;
    }
  }
  std::pair<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>,
            Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
      cells_to_dofs_map(cell_to_dofs, cell_to_dofs_offsets);

  return std::make_pair(cells_with_dofs, cells_to_dofs_map);
}

void dolfinx_mpc::build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                                         const dolfinx::fem::Form& a)
{
  // Get dof maps
  std::array<const dolfinx::fem::DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::cell) > 0)
  {
    dolfinx::fem::SparsityPatternBuilder::cells(pattern, mesh.topology(),
                                                {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::interior_facet)
      > 0)
  {

    mesh.create_entities(mesh.topology().dim() - 1);
    mesh.create_connectivity(mesh.topology().dim() - 1, mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::interior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(
          dolfinx::fem::FormIntegrals::Type::exterior_facet)
      > 0)
  {
    mesh.create_entities(mesh.topology().dim() - 1);
    mesh.create_connectivity(mesh.topology().dim() - 1, mesh.topology().dim());
    dolfinx::fem::SparsityPatternBuilder::exterior_facets(
        pattern, mesh.topology(), {{dofmaps[0], dofmaps[1]}});
  }
}

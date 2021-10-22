// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/assembler.h>
#include <iostream>

namespace
{
template <typename T>
void modify_mpc_cell_vec(
    const int num_dofs, xtl::span<T>& b, xtl::span<T>& b_local,
    const xtl::span<const int32_t>& dofs, int bs,
    const xtl::span<const int32_t>& slave_indices,
    const std::shared_ptr<
        const dolfinx::graph::AdjacencyList<std::int32_t>>& masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>&
        coeffs,
    const std::vector<std::int32_t>& local_indices,
    const std::vector<bool>& is_slave)
{
  std::vector<std::int32_t> flattened_masters;
  std::vector<std::int32_t> flattened_slaves;
  std::vector<std::int32_t> slaves_loc;
  std::vector<T> flattened_coeffs;

  for (std::int32_t i = 0; i < slave_indices.size(); ++i)
  {
    xtl::span<const std::int32_t> local_masters
        = masters->links(slave_indices[i]);
    xtl::span<const T> local_coeffs = coeffs->links(slave_indices[i]);

    for (std::int32_t j = 0; j < local_masters.size(); ++j)
    {
      slaves_loc.push_back(i);
      flattened_slaves.push_back(local_indices[i]);
      flattened_masters.push_back(local_masters[j]);
      flattened_coeffs.push_back(local_coeffs[j]);
    }
  }

  const auto b_local_copy = b_local.copy();

  for (std::int32_t slv_idx = 0; slv_idx < flattened_slaves.size(); ++slv_idx)
  {
    const std::int32_t& local_index = flattened_slaves[slv_idx];
    const std::int32_t& master = flattened_masters[slv_idx];
    const T& coeff = flattened_coeffs[slv_idx];

    for (std::int32_t i = 0; i < num_dofs; ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        b[master] += coeff * b_local_copy[i*bs + j];
        b_local[i*bs + j] = 0;
      }
    }
  }

}
//-----------------------------------------------------------------------------
template <typename T>
void _assemble_cells(
    xtl::span<T> b,
    const dolfinx::mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const xtl::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc,
    const std::function<void(T*, const T*, const T*, const double*, 
        const int*, const std::uint8_t*)>& kernel)
{
  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters_local();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>
      coefficients = mpc->coeffs();
  const xtl::span<const std::int32_t> slaves = mpc->slaves();

  xtl::span<const std::int32_t> slave_cells = mpc->slave_cells();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc->cell_to_slaves();

  // Compute local indices for slave cells
  std::vector<bool> is_slave_entity = std::vector<bool>(active_entities.size(), false);
  std::vector<std::int32_t> slave_cell_index
      = std::vector<std::int32_t>(active_entities.size(), -1);

  for (std::int32_t i = 0; i < active_entities.size(); ++i)
  {
    const std::int32_t cell = active_entities[i];
    for (std::int32_t j = 0; j < slave_cells.size(); ++j)
    if (slave_cells[j] == cell)
    {
        is_slave_entity[i] = true;
        slave_cell_index[i] = j;
        break;
    }
  }

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  const int num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  const int num_dofs = dofmap.links(0).size();
//   std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<T> be(bs * num_dofs);
  const xtl::span<T> _be(be);

  for (std::int32_t l = 0; l < active_entities.size(); ++l)
  {
    const std::int32_t slave_cell = active_entities[l];

    const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
    const xt::xarray<double> coordinate_dofs(
    xt::view(x_g, xt::keep(x_dofs), xt::all()));

    // Tabulate tensor
    std::fill(_be.data(), _be.data() + _be.size(), 0);
    kernel(be.data(), coeffs.data() + slave_cell * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);

    modify_mpc_cell_vec(num_dofs, b, be, dofs, bs, slave_indices,
        masters, coefficients, local_indices, is_slave);
  }
}
}
//-----------------------------------------------------------------------------
template <typename T>
void dolfinx_mpc::assemble_vector(
    xtl::span<T> b,
    const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
  assert(dofmap);
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();

  const int num_dofs = dofs.links(0).size();
  const std::uint32_t ndim = bs * num_dofs;

  // Prepare constants & coefficients
  const std::vector<T> constants = pack_constants(L);
  const auto coeffs = dolfinx::fem::pack_coefficients(L);

  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = L.function_spaces().at(0)->element();
  std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation = element->get_dof_transformation_function<T>();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Prepare tranformation data
  const bool needs_transformation_data
      = element->needs_dof_transformations()
        or L.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  if (L.num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    // const auto fetch_cells = [&](const std::int32_t& entity){
    //   return entity; };

    for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
    {
    //   const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);

    //   // Assembly lambda employing the form's kernel
    //   const auto assemble_local_cell_matrix = [&](
    //     xt::xtensor<T, 2>& Ae, const std::int32_t cell)
    //     {
    //       const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
    //       const xt::xarray<double> coordinate_dofs(
    //         xt::view(x_g, xt::keep(x_dofs), xt::all()));

    //       // Tabulate tensor
    //       std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    //       fn(Ae.data(), coeffs.first.data() + cell*coeffs.second, constants.data(),
    //             coordinate_dofs.data(), nullptr, nullptr);

    //       // Apply any required transformations
    //       const xtl::span<T> _Ae(Ae);
    //       apply_dof_transformation(_Ae, cell_info, cell, ndim1);
    //       apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);
    //     };

      const std::vector<std::int32_t>& active_cells = L.cell_domains(i);
      _assemble_cells<T>(
          *mesh, active_cells,
          dofs, bs,
          coeffs.first, coeffs.second, constants, cell_info, mpc);
    }
  }

}
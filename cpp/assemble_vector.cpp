// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <iostream>

namespace
{

/// Assemble an integration kernel over a set of active entities, described
/// through into vector of type T, and apply the multipoint constraint
/// @param[in, out] b The vector to assemble into
/// @param[in] active_entities The set of active entities.
/// @param[in] dofmap The dofmap
/// @param[in] bs The block size of the dofmap
/// @param[in] mpc The multipoint constraint
/// @param[in] fetch_cells Function that fetches the cell index for an entity
/// in active_entities
/// @param[in] assemble_local_element_matrix Function f(be, index) that
/// assembles into a local element matrix for a given entity
template <typename T, typename E_DESC>
void _assemble_entities_impl(
    xtl::span<T> b, const std::vector<E_DESC>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc,
    const std::function<const std::int32_t(const E_DESC&)> fetch_cells,
    const std::function<void(xtl::span<T>, E_DESC)>
        assemble_local_element_vector)
{

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc->coefficients();
  const std::vector<std::int8_t>& is_slave = mpc->is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc->cell_to_slaves();

  // NOTE: Assertion that all links have the same size (no P refinement)
  const int num_dofs = dofmap.links(0).size();
  std::vector<T> be(bs * num_dofs);
  const xtl::span<T> _be(be);
  std::vector<T> be_copy(bs * num_dofs);
  const xtl::span<T> _be_copy(be_copy);

  // Assemble over all entities
  for (std::size_t e = 0; e < active_entities.size(); ++e)
  {
    // Assemble into element vector
    assemble_local_element_vector(_be, active_entities[e]);

    const std::int32_t cell = fetch_cells(active_entities[e]);
    auto dofs = dofmap.links(cell);
    // Modify local element matrix if entity is connected to a slave cell
    xtl::span<const int32_t> slaves = cell_to_slaves->links(cell);
    if (slaves.size() > 0)
    {
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      std::copy(be.begin(), be.end(), be_copy.begin());
      dolfinx_mpc::modify_mpc_vec<T>(b, _be, _be_copy, dofs, num_dofs, bs,
                                     is_slave, slaves, masters, coefficients);
    }

    // Add local contribution to b
    for (int i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}

template <typename T>
void _assemble_vector(
    xtl::span<T> b, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
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

  // Prepare constants & coefficients
  const std::vector<T> constants = pack_constants(L);
  const auto coeffs = dolfinx::fem::pack_coefficients(L);

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Prepare dof tranformation data
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = L.function_spaces().at(0)->element();
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element->get_dof_transformation_function<T>();
  const bool needs_transformation_data
      = element->needs_dof_transformations() or L.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(cell);
  // FIXME: Reconsider when using mixed topology (mixed celltypes)
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  if (L.num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    const auto fetch_cells = [&](const std::int32_t& entity) { return entity; };
    for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
    {
      const auto& fn = L.kernel(dolfinx::fem::IntegralType::cell, i);
      /// Assemble local cell kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] cell The cell index
      const auto assemble_local_cell_vector
          = [&](xtl::span<T> be, std::int32_t cell)
      {
        // Fetch the coordinates of the cell
        const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }
        // Tabulate tensor
        std::fill(be.data(), be.data() + be.size(), 0);
        fn(be.data(), coeffs.first.data() + cell * coeffs.second,
           constants.data(), coordinate_dofs.data(), nullptr, nullptr);

        // Apply any required transformations
        dof_transform(be, cell_info, cell, 1);
      };

      // Assemble over all active cells
      const std::vector<std::int32_t>& active_cells = L.cell_domains(i);
      _assemble_entities_impl<T, std::int32_t>(b, active_cells, dofs, bs, mpc,
                                               fetch_cells,
                                               assemble_local_cell_vector);
    }
  }
  // Prepare permutations for exterior and interior facet integrals
  if (L.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0
      or L.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (L.needs_facet_permutations())
    {
      mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
      mesh->topology_mutable().create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology().get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    // Create lambda function fetching cell index from exterior facet entity
    const auto fetch_cells_ext = [&](const std::pair<std::int32_t, int>& entity)
    { return entity.first; };

    // Get number of cells per facet to be able to get the facet permutation
    const int tdim = mesh->topology().dim();
    const int num_cell_facets = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);

    // Assemble exterior facet integral kernels
    for (int i : L.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& fn = L.kernel(dolfinx::fem::IntegralType::exterior_facet, i);

      /// Assemble local exterior facet kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] entity The entity, given as a cell index and the local
      /// index relative to the cell
      const auto assemble_local_exterior_facet_vector
          = [&](xtl::span<T> be, std::pair<std::int32_t, int> entity)
      {
        // Fetch the coordiantes of the cell
        const std::int32_t cell = entity.first;
        const int local_facet = entity.second;
        const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }

        std::uint8_t perm = get_perm(cell * num_cell_facets + local_facet);

        // Tabulate tensor
        std::fill(be.data(), be.data() + be.size(), 0);
        fn(be.data(), coeffs.first.data() + cell * coeffs.second,
           constants.data(), coordinate_dofs.data(), &local_facet, &perm);

        // Apply any required transformations
        dof_transform(be, cell_info, cell, 1);
      };

      // Assemble over all active cells
      const std::vector<std::pair<std::int32_t, int>>& active_facets
          = L.exterior_facet_domains(i);
      _assemble_entities_impl<T, std::pair<std::int32_t, int>>(
          b, active_facets, dofs, bs, mpc, fetch_cells_ext,
          assemble_local_exterior_facet_vector);
    }
    if (L.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
    {
      throw std::runtime_error(
          "Interior facet integrals currently not supported");
    }
  }
}
} // namespace
//-----------------------------------------------------------------------------

void dolfinx_mpc::assemble_vector(
    xtl::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc)
{
  _assemble_vector<double>(b, L, mpc);
}

void dolfinx_mpc::assemble_vector(
    xtl::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc)
{
  _assemble_vector<std::complex<double>>(b, L, mpc);
}
//-----------------------------------------------------------------------------

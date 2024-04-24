// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_vector.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <iostream>

using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

namespace
{

/// Assemble an integration kernel over a set of active entities, described
/// through into vector of type T, and apply the multipoint constraint
/// @param[in, out] b The vector to assemble into
/// @param[in] active_entities The set of active entities.
/// @param[in] dofmap The dofmap
/// @param[in] mpc The multipoint constraint
/// @param[in] fetch_cells Function that fetches the cell index for an entity
/// in active_entities
/// @param[in] assemble_local_element_matrix Function f(be, index) that
/// assembles into a local element matrix for a given entity
/// @tparam T Scalar type for vector
/// @tparam e stride Stride for each entity in active_entities
template <typename T, std::floating_point U, std::size_t estride>
void _assemble_entities_impl(
    std::span<T> b, std::span<const std::int32_t> active_entities,
    const dolfinx::fem::DofMap& dofmap,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    const std::function<const std::int32_t(std::span<const std::int32_t>)>
        fetch_cells,
    const std::function<void(std::span<T>, std::span<const std::int32_t>,
                             std::size_t)>
        assemble_local_element_vector)
{

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc->coefficients();
  std::span<const std::int8_t> is_slave = mpc->is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc->cell_to_slaves();

  // NOTE: Assertion that all links have the same size (no P refinement)
  const std::size_t num_dofs = dofmap.map().extent(1);
  int bs = dofmap.bs();
  std::vector<T> be(bs * num_dofs);
  const std::span<T> _be(be);
  std::vector<T> be_copy(bs * num_dofs);
  const std::span<T> _be_copy(be_copy);

  // Assemble over all entities
  for (std::size_t e = 0; e < active_entities.size(); e += estride)
  {
    std::span<const std::int32_t> entity = active_entities.subspan(e, estride);
    // Assemble into element vector
    assemble_local_element_vector(_be, entity, e / estride);

    const std::int32_t cell = fetch_cells(entity);
    auto dofs = dofmap.cell_dofs(cell);

    // Modify local element matrix if entity is connected to a slave cell
    std::span<const std::int32_t> slaves = cell_to_slaves->links(cell);
    if (!slaves.empty())
    {
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      std::copy(be.begin(), be.end(), be_copy.begin());
      dolfinx_mpc::modify_mpc_vec<T>(b, _be, _be_copy, dofs, num_dofs, bs,
                                     is_slave, slaves, masters, coefficients);
    }

    // Add local contribution to b
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}

template <typename T, std::floating_point U>
void _assemble_vector(
    std::span<T> b, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc)
{

  const auto mesh = L.mesh();
  assert(mesh);

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
  assert(dofmap);

  // Prepare constants & coefficients
  const std::vector<T> constants = pack_constants(L);
  auto coeff_vec = dolfinx::fem::allocate_coefficient_storage(L);
  dolfinx::fem::pack_coefficients(L, coeff_vec);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);

  // Prepare cell geometry
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();
  std::span<const U> x_g = mesh->geometry().x();

  // Prepare dof tranformation data
  auto element = L.function_spaces().at(0)->element();
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element->template dof_transformation_fn<T>(
          dolfinx::fem::doftransform::standard);
  const bool needs_transformation_data
      = element->needs_dof_transformations() or L.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }
  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::vector<U> coordinate_dofs(3 * num_dofs_g);

  if (L.num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    const auto fetch_cell
        = [&](std::span<const std::int32_t> entity) { return entity.front(); };
    for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
    {
      const auto& coeffs
          = coefficients.at({dolfinx::fem::IntegralType::cell, i});
      const auto& fn = L.kernel(dolfinx::fem::IntegralType::cell, i);
      /// Assemble local cell kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] cell The cell index
      /// @param[in] index The index of the cell in the active_cells (To fetch
      /// the appropriate coefficients)
      const auto assemble_local_cell_vector
          = [&](std::span<T> be, std::span<const std::int32_t> entity,
                std::int32_t index)
      {
        auto cell = entity.front();

        // Fetch the coordinates of the cell
        auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }

        // Tabulate tensor
        std::fill(be.data(), be.data() + be.size(), 0);
        fn(be.data(), coeffs.first.data() + index * coeffs.second,
           constants.data(), coordinate_dofs.data(), nullptr, nullptr);

        // Apply any required transformations
        dof_transform(be, cell_info, cell, 1);
      };

      // Assemble over all active cells
      std::span<const std::int32_t> active_cells
          = L.domain(dolfinx::fem::IntegralType::cell, i);
      _assemble_entities_impl<T, U, 1>(b, active_cells, *dofmap, mpc,
                                       fetch_cell, assemble_local_cell_vector);
    }
  }
  // Prepare permutations for exterior and interior facet integrals
  if (L.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0)
  {

    // Create lambda function fetching cell index from exterior facet entity
    auto fetch_cell = [](auto entity) { return entity.front(); };

    // Assemble exterior facet integral kernels
    for (int i : L.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& fn = L.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
      const auto& coeffs
          = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
      /// Assemble local exterior facet kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] entity The entity, given as a cell index and the local
      /// index relative to the cell
      /// @param[in] index The index of entity in active_facets
      const auto assemble_local_exterior_facet_vector
          = [&](std::span<T> be, std::span<const std::int32_t> entity,
                std::size_t index)
      {
        // Fetch the coordinates of the cell
        const std::int32_t cell = entity[0];
        const int local_facet = entity[1];
        auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }

        // Tabulate tensor
        std::fill(be.data(), be.data() + be.size(), 0);
        fn(be.data(), coeffs.first.data() + index * coeffs.second,
           constants.data(), coordinate_dofs.data(), &local_facet, nullptr);

        // Apply any required transformations
        dof_transform(be, cell_info, cell, 1);
      };

      // Assemble over all active cells
      std::span<const std::int32_t> active_facets
          = L.domain(dolfinx::fem::IntegralType::exterior_facet, i);
      _assemble_entities_impl<T, U, 2>(b, active_facets, *dofmap, mpc,
                                       fetch_cell,
                                       assemble_local_exterior_facet_vector);
    }
  }

  if (L.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    throw std::runtime_error(
        "Interior facet integrals currently not supported");
    // std::function<std::uint8_t(std::size_t)> get_perm;
    // if (L.needs_facet_permutations())
    // {
    //   const int tdim = mesh->topology()->dim();
    //   mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    //   mesh->topology_mutable().create_entity_permutations();
    //   const std::vector<std::uint8_t>& perms
    //       = mesh->topology()->get_facet_permutations();
    //   get_perm = [&perms](std::size_t i) { return perms[i]; };
    // }
    // else
    //   get_perm = [](std::size_t) { return 0; };
  }
}
} // namespace
//-----------------------------------------------------------------------------

void dolfinx_mpc::assemble_vector(
    std::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc)
{
  _assemble_vector<double>(b, L, mpc);
}

void dolfinx_mpc::assemble_vector(
    std::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc)
{
  _assemble_vector<std::complex<double>>(b, L, mpc);
}

void dolfinx_mpc::assemble_vector(
    std::span<float> b, const dolfinx::fem::Form<float>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc)
{
  _assemble_vector<float>(b, L, mpc);
}

void dolfinx_mpc::assemble_vector(
    std::span<std::complex<float>> b,
    const dolfinx::fem::Form<std::complex<float>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc)
{
  _assemble_vector<std::complex<float>>(b, L, mpc);
}
//-----------------------------------------------------------------------------

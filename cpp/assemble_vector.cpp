// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_vector.h"
#include "assemble_utils.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/cell_types.h>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

namespace
{

/// Assemble an integration kernel over a set of active entities, described
/// through into vector of type T, and apply the multipoint constraint
/// @param[in, out] b The vector to assemble into
/// @param[in] active_entities The set of active entities.
/// @param[in] active_cells0 The corresponding cells for the test function space
/// @param[in] dofmap The dofmap
/// @param[in] mpc The multipoint constraint
/// @param[in] assemble_local_element_vector Callable `f(be, entity, cell0,
/// index)` tabulating the (transformed) element vector of an entity. This is
/// the standard DOLFINx tabulation; everything specific to the constraint
/// happens here.
/// @tparam T Scalar type for vector
/// @tparam e stride Stride for each entity in active_entities
template <typename T, std::floating_point U, std::size_t estride,
          typename Tabulate>
  requires std::invocable<Tabulate&, std::span<T>,
                          std::span<const std::int32_t>, std::int32_t,
                          std::size_t>
void _assemble_entities_impl(
    const dolfinx_mpc::VectorTarget<T>& b,
    std::span<const std::int32_t> active_entities,
    std::span<const std::int32_t> active_cells0,
    const dolfinx::fem::DofMap& dofmap,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    Tabulate&& assemble_local_element_vector)
{

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters();
  std::span<const std::int32_t> master_blocks = mpc->master_blocks();
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
    // The entity indexes the integration mesh; `cell0` indexes the test
    // function's own mesh. They coincide only when the two are the same mesh,
    // so the dofmap, the constraint and the dof transformation must all be
    // keyed on `active_cells0`, never on `entity`.
    const std::int32_t cell0 = active_cells0.subspan(e, estride).front();
    // Assemble into element vector
    assemble_local_element_vector(_be, entity, cell0, e / estride);

    auto dofs = dofmap.cell_dofs(cell0);

    // Modify local element matrix if entity is connected to a slave cell
    std::span<const std::int32_t> slaves = cell_to_slaves->links(cell0);
    if (!slaves.empty())
    {
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      std::ranges::copy(be, be_copy.begin());
      dolfinx_mpc::modify_mpc_vec<T>(b, _be, _be_copy, dofs, num_dofs, bs,
                                     is_slave, slaves, masters, coefficients,
                                     master_blocks);
    }

    // Add local contribution to b
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b.at(b.block, bs * dofs[i] + k) += be[bs * i + k];
  }
}

/// Assemble an interior facet kernel into a vector and apply the multipoint
/// constraint.
///
/// The element vector spans the two cells of the facet. Each half is a cell
/// vector of the test space, so the constraint is applied to each side on its
/// own; a side on which the test function has no cell (negative entry in
/// `facets0`, e.g. on an interface between two subdomains) is skipped.
/// @param[in, out] b The vector to assemble into
/// @param[in] facets The facets of the integration mesh, as (cell, local facet)
/// for each side
/// @param[in] facets0 The corresponding entities for the test function space
/// @param[in] dofmap The dofmap of the test function space
/// @param[in] mpc The multipoint constraint
/// @param[in] tabulate Callable `tabulate(be, f)` tabulating the (transformed)
/// element vector of the `f`-th facet. A template parameter rather than a
/// `std::function`, so the call is inlined.
template <typename T, std::floating_point U, typename Tabulate>
  requires std::invocable<Tabulate&, std::span<T>, std::size_t>
void _assemble_interior_facets(
    const dolfinx_mpc::VectorTarget<T>& b, std::span<const std::int32_t> facets,
    std::span<const std::int32_t> facets0, const dolfinx::fem::DofMap& dofmap,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    Tabulate&& tabulate)
{
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters();
  std::span<const std::int32_t> master_blocks = mpc->master_blocks();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc->coefficients();
  std::span<const std::int8_t> is_slave = mpc->is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc->cell_to_slaves();

  const std::size_t num_dofs = dofmap.map().extent(1);
  const int bs = dofmap.bs();
  const std::size_t ndim = bs * num_dofs;
  std::vector<T> be(2 * ndim);
  std::vector<T> be_copy(ndim);
  for (std::size_t f = 0; f < facets.size() / 4; ++f)
  {
    tabulate(std::span<T>(be), f);
    for (int s = 0; s < 2; ++s)
    {
      const std::int32_t cell0 = facets0[4 * f + 2 * s];
      if (cell0 < 0)
        continue;
      std::span<T> be_s(be.data() + s * ndim, ndim);
      std::span<const std::int32_t> dofs = dofmap.cell_dofs(cell0);
      std::span<const std::int32_t> slaves = cell_to_slaves->links(cell0);
      if (!slaves.empty())
      {
        std::ranges::copy(be_s, be_copy.begin());
        dolfinx_mpc::modify_mpc_vec<T>(b, be_s, be_copy, dofs, num_dofs, bs,
                                       is_slave, slaves, masters, coefficients,
                                       master_blocks);
      }
      for (std::size_t i = 0; i < num_dofs; ++i)
        for (int k = 0; k < bs; ++k)
          b.at(b.block, bs * dofs[i] + k) += be_s[bs * i + k];
    }
  }
}

template <typename T, std::floating_point U>
void _assemble_vector(
    const dolfinx_mpc::VectorTarget<T>& b, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    std::size_t num_threads)
{

  const auto mesh = L.mesh();
  assert(mesh);

  // Test function mesh
  auto mesh0 = L.function_spaces().at(0)->mesh();
  assert(mesh0);

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
  if (mesh->geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmaps().front();
  std::span<const U> x_g = mesh->geometry().x();

  // Prepare dof tranformation data
  auto element = L.function_spaces().at(0)->element();
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element->template dof_transformation_fn<T>(
          dolfinx::fem::doftransform::standard);
  const bool transform_set = dolfinx::fem::is_transform_set(dof_transform);
  const bool needs_transformation_data
      = element->needs_dof_transformations() or L.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info0;
  if (needs_transformation_data)
  {
    mesh0->topology_mutable()->create_cell_permutations(num_threads);
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
  }

  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::vector<U> coordinate_dofs(3 * num_dofs_g);
  const int num_cell_types = mesh->topology()->cell_types().size();
  if (num_cell_types > 1)
    throw std::runtime_error("Not implemented for mixed cell types");

  for (int i = 0; i < L.num_integrals(dolfinx::fem::IntegralType::cell, 0); ++i)
  {
    const auto& coeffs = coefficients.at({dolfinx::fem::IntegralType::cell, i});

    const auto& fn = L.kernel(dolfinx::fem::IntegralType::cell, i, 0);
    /// Assemble local cell kernels into a vector
    /// @param[in] be The local element vector
    /// @param[in] entity The cell index (for the geometry)
    /// @param[in] cell0 The cell index (for the test function space)
    /// @param[in] index The index of the cell in the active_cells (To fetch
    /// the appropriate coefficients and the correct cell information)
    const auto assemble_local_cell_vector
        = [&](std::span<T> be, std::span<const std::int32_t> entity,
              std::int32_t cell0, std::int32_t index)
    {
      auto cell = entity.front();

      // Fetch the coordinates of the cell
      dolfinx_mpc::gather_cell_coordinates(x_dofmap, x_g, cell,
                                           std::span(coordinate_dofs));

      // Tabulate tensor
      std::ranges::fill(be, 0);
      fn(be.data(), coeffs.first.data() + index * coeffs.second,
         constants.data(), coordinate_dofs.data(), nullptr, nullptr, nullptr);

      // Apply any required transformations
      if (transform_set)
        dof_transform(be, cell_info0, cell0, 1);
    };

    // Assemble over all active cells
    std::span cells = L.domain(dolfinx::fem::IntegralType::cell, i, 0);
    std::span cells0 = L.domain_arg(dolfinx::fem::IntegralType::cell, 0, i, 0);
    _assemble_entities_impl<T, U, 1>(b, cells, cells0, *dofmap, mpc,
                                     assemble_local_cell_vector);
  }
  // Integrals over one entity of a cell: exterior facets, ridges and vertices.
  // The kernels take the permutation of the entity whenever they ask for one,
  // for instance when an argument space lives on another mesh than the
  // integration domain.
  for (dolfinx::fem::IntegralType type : dolfinx_mpc::entity_integral_types)
  {
    const auto [perms, num_entities_per_cell]
        = dolfinx_mpc::entity_permutations(*mesh, type,
                                           L.needs_facet_permutations(),
                                           static_cast<int>(num_threads));
    for (int i = 0; i < L.num_integrals(type, 0); ++i)
    {
      const auto& fn = L.kernel(type, i, 0);
      const auto& coeffs = coefficients.at({type, i});
      /// Assemble the kernel of an entity into a vector
      /// @param[in] be The local element vector
      /// @param[in] entity The entity, given as a cell index and the local
      /// index relative to the cell
      /// @param[in] cell0 The cell of the test function space
      /// @param[in] index The index of the entity in the active entities
      const auto assemble_local_entity_vector
          = [&](std::span<T> be, std::span<const std::int32_t> entity,
                std::int32_t cell0, std::size_t index)
      {
        const std::int32_t cell = entity[0];
        const int local_entity = entity[1];
        dolfinx_mpc::gather_cell_coordinates(x_dofmap, x_g, cell,
                                             std::span(coordinate_dofs));

        // A kernel that asks for the permutation would dereference a null
        // pointer if it were not supplied
        const std::uint8_t perm
            = perms.empty()
                  ? 0
                  : perms[cell * num_entities_per_cell + local_entity];
        std::ranges::fill(be, 0);
        fn(be.data(), coeffs.first.data() + index * coeffs.second,
           constants.data(), coordinate_dofs.data(), &local_entity, &perm,
           nullptr);

        if (transform_set)
          dof_transform(be, cell_info0, cell0, 1);
      };

      std::span<const std::int32_t> entities = L.domain(type, i, 0);
      std::span<const std::int32_t> cells0 = L.domain_arg(type, 0, i, 0);
      _assemble_entities_impl<T, U, 2>(b, entities, cells0, *dofmap, mpc,
                                       assemble_local_entity_vector);
    }
  }

  const auto [perms, num_facets_per_cell] = dolfinx_mpc::entity_permutations(
      *mesh, dolfinx::fem::IntegralType::interior_facet,
      L.needs_facet_permutations(), static_cast<int>(num_threads));
  for (int i = 0;
       i < L.num_integrals(dolfinx::fem::IntegralType::interior_facet, 0); ++i)
  {
    const auto& fn = L.kernel(dolfinx::fem::IntegralType::interior_facet, i, 0);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::interior_facet, i});
    std::span<const std::int32_t> facets
        = L.domain(dolfinx::fem::IntegralType::interior_facet, i, 0);
    std::span<const std::int32_t> facets0
        = L.domain_arg(dolfinx::fem::IntegralType::interior_facet, 0, i, 0);
    std::vector<U> facet_coordinate_dofs(2 * 3 * num_dofs_g);
    const std::size_t ndim = dofmap->bs() * dofmap->map().extent(1);

    const auto tabulate = [&](std::span<T> be, std::size_t f)
    {
      const std::array<std::int32_t, 2> cells
          = {facets[4 * f], facets[4 * f + 2]};
      const std::array<int, 2> local_facet
          = {facets[4 * f + 1], facets[4 * f + 3]};
      std::span<U> cdofs(facet_coordinate_dofs);
      dolfinx_mpc::gather_cell_coordinates(x_dofmap, x_g, cells[0], cdofs);
      dolfinx_mpc::gather_cell_coordinates(x_dofmap, x_g, cells[1],
                                           cdofs.subspan(3 * num_dofs_g));
      const std::array<std::uint8_t, 2> perm
          = perms.empty()
                ? std::array<std::uint8_t, 2>{0, 0}
                : std::array{
                      perms[cells[0] * num_facets_per_cell + local_facet[0]],
                      perms[cells[1] * num_facets_per_cell + local_facet[1]]};
      std::ranges::fill(be, 0);
      fn(be.data(), coeffs.data() + f * 2 * cstride, constants.data(),
         facet_coordinate_dofs.data(), local_facet.data(), perm.data(),
         nullptr);

      // Each half of the element vector belongs to one cell of the test space
      if (transform_set)
      {
        for (int s = 0; s < 2; ++s)
        {
          const std::int32_t cell0 = facets0[4 * f + 2 * s];
          if (cell0 >= 0)
            dof_transform(be.subspan(s * ndim, ndim), cell_info0, cell0, 1);
        }
      }
    };
    _assemble_interior_facets<T, U>(b, facets, facets0, *dofmap, mpc, tabulate);
  }
}
/// Assemble into the vector of a single block, whose constraint has all its
/// masters in that block
template <typename T, std::floating_point U>
void _assemble_vector_single(
    std::span<T> b, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    std::size_t num_threads)
{
  if (mpc->has_cross_block_masters())
  {
    throw std::invalid_argument(
        "The constraint has masters in another block. Assemble into the "
        "vectors of every block.");
  }
  const std::array<std::span<T>, 1> blocks = {b};
  _assemble_vector<T, U>(dolfinx_mpc::VectorTarget<T>{blocks, 0, mpc->block()},
                         L, mpc, num_threads);
}
} // namespace

//-----------------------------------------------------------------------------

void dolfinx_mpc::assemble_vector(
    std::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc,
    std::size_t num_threads)
{
  _assemble_vector_single<double>(b, L, mpc, num_threads);
}

void dolfinx_mpc::assemble_vector(
    std::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc,
    std::size_t num_threads)
{
  _assemble_vector_single<std::complex<double>>(b, L, mpc, num_threads);
}

void dolfinx_mpc::assemble_vector(
    std::span<float> b, const dolfinx::fem::Form<float>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc,
    std::size_t num_threads)
{
  _assemble_vector_single<float>(b, L, mpc, num_threads);
}

void dolfinx_mpc::assemble_vector(
    std::span<std::complex<float>> b,
    const dolfinx::fem::Form<std::complex<float>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc,
    std::size_t num_threads)
{
  _assemble_vector_single<std::complex<float>>(b, L, mpc, num_threads);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
template <typename T, std::floating_point U>
void dolfinx_mpc::assemble_vector_blocks(
    std::span<const std::span<T>> b, int i, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    std::size_t num_threads)
{
  if (i < 0 or static_cast<std::size_t>(i) >= b.size())
  {
    throw std::invalid_argument(
        std::format("Block {} is not one of the {} vectors.", i, b.size()));
  }
  if (mpc->has_cross_block_masters()
      and b.size() != mpc->function_spaces().size())
  {
    throw std::invalid_argument(
        std::format("The constraint has masters in other blocks; expected the "
                    "vectors of all {} blocks, got {}.",
                    mpc->function_spaces().size(), b.size()));
  }
  _assemble_vector<T, U>(
      dolfinx_mpc::VectorTarget<T>{b, static_cast<std::size_t>(i),
                                   mpc->block()},
      L, mpc, num_threads);
}

template void dolfinx_mpc::assemble_vector_blocks<double, double>(
    std::span<const std::span<double>>, int, const dolfinx::fem::Form<double>&,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>&,
    std::size_t);
template void dolfinx_mpc::assemble_vector_blocks<float, float>(
    std::span<const std::span<float>>, int, const dolfinx::fem::Form<float>&,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>&,
    std::size_t);
template void dolfinx_mpc::assemble_vector_blocks<std::complex<double>, double>(
    std::span<const std::span<std::complex<double>>>, int,
    const dolfinx::fem::Form<std::complex<double>>&,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&,
    std::size_t);
template void dolfinx_mpc::assemble_vector_blocks<std::complex<float>, float>(
    std::span<const std::span<std::complex<float>>>, int,
    const dolfinx::fem::Form<std::complex<float>>&,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&,
    std::size_t);

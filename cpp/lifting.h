// Copyright (C) 2021-2025 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "assemble_vector.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/cell_types.h>
#include <format>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace impl
{

/// Implementation of bc application (lifting) for an given a set of integration
/// entities
/// @tparam T The scalar type
/// @tparam E_DESC Description of the set of entities
/// @param[in, out] b The vector to apply lifting to
/// @param[in] active_entities Set of active entities (either cells, exterior
/// facets or interior facets in their specified format)
/// @param[in] active_entities0 The active entities for the rows of the matrix
/// @param[in] active_entities1 The active entities for the columns of the
/// matrix
/// @param[in] dofmap0 The dofmap for the rows of the matrix
/// @param[in] dofmap1 The dofmap for the columns of the matrix
/// @param[in] bc_values1 Array of Dirichlet condition values for dofs local to
/// process
/// @param[in] bc_markers1 Array indicating what dofs local to process is in a
/// DirichletBC
/// @param[in] mpc1 Multipoint constraints to apply to the rows of the vector
/// @param[in] fetch_cells Function that fetches the cell index for each active
/// entity
/// @param[in] lift_local_vector Function that lift local matrix Ae into local
/// vector be, i.e. be <- be - scale * (A (g - x0))
/// @tparam T Scalartype of local vector
/// @tparam estride Stride in actiave entities
template <typename T, std::size_t estride, std::floating_point U>
void lift_bc_entities(
    std::span<T> b, std::span<const std::int32_t> active_entities,
    std::span<const std::int32_t> active_entities0,
    std::span<const std::int32_t> active_entities1,
    const dolfinx::fem::DofMap& dofmap0, const dolfinx::fem::DofMap& dofmap1,
    std::span<const T> bc_values1, std::span<const std::int8_t> bc_markers1,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc0,
    const std::function<const std::int32_t(std::span<const std::int32_t>)>
        fetch_cells,
    const std::function<void(std::span<T>, std::span<T>, const int, const int,
                             std::span<const std::int32_t>, std::int32_t,
                             std::int32_t, std::size_t)>
        lift_local_vector)
{
  const int bs0 = dofmap0.bs();
  const int bs1 = dofmap1.bs();
  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc0.masters();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc0.coefficients();
  std::span<const std::int8_t> is_slave = mpc0.is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc0.cell_to_slaves();

  const int num_dofs0 = dofmap0.map().extent(1);
  std::vector<T> be;
  std::vector<T> be_copy;
  std::vector<T> Ae;

  // Assemble over all entities
  for (std::size_t e = 0; e < active_entities.size(); e += estride)
  {
    auto entity = active_entities.subspan(e, estride);
    const std::int32_t cell = fetch_cells(entity);
    const std::int32_t cell0
        = fetch_cells(active_entities0.subspan(e, estride));
    const std::int32_t cell1
        = fetch_cells(active_entities1.subspan(e, estride));
    // Size data structure for assembly
    auto dmap0 = dofmap0.cell_dofs(cell0);
    auto dmap1 = dofmap1.cell_dofs(cell1);
    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();
    be.resize(num_rows);
    Ae.resize(num_rows * num_cols);

    // Check if bc is applied to entity
    bool has_bc = false;
    std::ranges::for_each(dmap1,
                          [&bc_markers1, bs1, &has_bc](const auto dof)
                          {
                            for (int k = 0; k < bs1; ++k)
                            {
                              assert(bs1 * dof + k < (int)bc_markers1.size());
                              if (bc_markers1[bs1 * dof + k])
                              {
                                has_bc = true;
                                break;
                              }
                            }
                          });
    if (!has_bc)
      continue;

    // Lift into local element vector
    const std::span<T> _be(be);
    const std::span<T> _Ae(Ae);
    lift_local_vector(_be, _Ae, num_rows, num_cols, entity, cell0, cell1,
                      e / estride);
    // Modify local element matrix if entity is connected to a slave cell.
    // `cell_to_slaves`, `is_slave`, `masters` and `dmap0` all belong to
    // `mpc0`, which constrains the rows, so the lookup is keyed on the test
    // space's cell. `cell0` and `cell1` coincide only when both spaces live
    // on the integration mesh.
    std::span<const std::int32_t> slaves = cell_to_slaves->links(cell0);

    if (slaves.size() > 0)
    {
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      be_copy.resize(num_rows);
      std::ranges::copy(be, be_copy.begin());
      const std::span<T> _be_copy(be_copy);
      dolfinx_mpc::modify_mpc_vec<T>(b, _be, _be_copy, dmap0, dmap0.size(), bs0,
                                     is_slave, slaves, masters, coefficients);
    }
    // Add local contribution to b
    for (int i = 0; i < num_dofs0; ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
};

/// Lift Dirichlet values through an interior facet kernel into b.
///
/// The element tensor spans the two cells of each facet. Columns are lifted
/// for every side on which the trial function has a cell, and the result is
/// constrained and added for every side on which the test function has one,
/// each side on its own (a negative cell in `facets0`/`facets1` means the
/// argument has no cell on that side, e.g. on an interface between two
/// subdomains).
template <typename T, std::floating_point U>
void lift_bc_interior_facets(
    std::span<T> b, std::span<const std::int32_t> facets,
    std::span<const std::int32_t> facets0,
    std::span<const std::int32_t> facets1, const dolfinx::mesh::Mesh<U>& mesh,
    const dolfinx::fem::DofMap& dofmap0, const dolfinx::fem::DofMap& dofmap1,
    std::span<const T> bc_values1, std::span<const std::int8_t> bc_markers1,
    std::span<const T> x0, T scale,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc0,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*, void*)>& kernel,
    std::span<const T> coeffs, int cstride, std::span<const T> constants,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    std::span<const std::uint8_t> perms, int num_facets_per_cell,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose)
{
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc0.masters();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc0.coefficients();
  std::span<const std::int8_t> is_slave = mpc0.is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc0.cell_to_slaves();

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh.geometry().dofmaps().front();
  std::span<const U> x_g = mesh.geometry().x();
  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::vector<U> coordinate_dofs(2 * 3 * num_dofs_g);

  const int bs0 = dofmap0.bs();
  const int bs1 = dofmap1.bs();
  const std::size_t num_dofs0 = dofmap0.map().extent(1);
  const std::size_t num_dofs1 = dofmap1.map().extent(1);
  const std::size_t ndim0 = bs0 * num_dofs0;
  const std::size_t ndim1 = bs1 * num_dofs1;
  const std::size_t num_rows = 2 * ndim0;
  const std::size_t num_cols = 2 * ndim1;
  std::vector<T> Ae(num_rows * num_cols);
  std::vector<T> be(num_rows);
  std::vector<T> be_copy(ndim0);

  const bool transform_set0 = dolfinx::fem::is_transform_set(dof_transform);
  const bool transform_set1
      = dolfinx::fem::is_transform_set(dof_transform_to_transpose);

  for (std::size_t f = 0; f < facets.size() / 4; ++f)
  {
    const std::array<std::int32_t, 2> cells
        = {facets[4 * f], facets[4 * f + 2]};
    const std::array<int, 2> local_facet
        = {facets[4 * f + 1], facets[4 * f + 3]};
    const std::array<std::int32_t, 2> cells0
        = {facets0[4 * f], facets0[4 * f + 2]};
    const std::array<std::int32_t, 2> cells1
        = {facets1[4 * f], facets1[4 * f + 2]};

    // Skip the facet unless a trial side carries a lifted value
    bool has_bc = false;
    for (int s = 0; s < 2 and !has_bc; ++s)
    {
      if (cells1[s] < 0)
        continue;
      for (std::int32_t dof : dofmap1.cell_dofs(cells1[s]))
        for (int k = 0; k < bs1; ++k)
          has_bc = has_bc or bc_markers1[bs1 * dof + k];
    }
    if (!has_bc)
      continue;

    for (int s = 0; s < 2; ++s)
    {
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cells[s], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::ranges::copy_n(
            std::next(x_g.begin(), 3 * x_dofs[i]), 3,
            std::next(coordinate_dofs.begin(), 3 * (s * num_dofs_g + i)));
      }
    }
    const std::array<std::uint8_t, 2> perm
        = perms.empty()
              ? std::array<std::uint8_t, 2>{0, 0}
              : std::array{
                    perms[cells[0] * num_facets_per_cell + local_facet[0]],
                    perms[cells[1] * num_facets_per_cell + local_facet[1]]};
    std::ranges::fill(Ae, T(0));
    kernel(Ae.data(), coeffs.data() + f * 2 * cstride, constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data(), nullptr);

    std::span<T> _Ae(Ae);
    if (transform_set0 and cells0[0] >= 0)
      dof_transform(_Ae, cell_info0, cells0[0], num_cols);
    if (transform_set0 and cells0[1] >= 0)
    {
      dof_transform(_Ae.subspan(ndim0 * num_cols, ndim0 * num_cols), cell_info0,
                    cells0[1], num_cols);
    }
    if (transform_set1 and cells1[0] >= 0)
      dof_transform_to_transpose(_Ae, cell_info1, cells1[0], num_rows);
    if (transform_set1 and cells1[1] >= 0)
    {
      for (std::size_t row = 0; row < num_rows; ++row)
      {
        dof_transform_to_transpose(_Ae.subspan(row * num_cols + ndim1, ndim1),
                                   cell_info1, cells1[1], 1);
      }
    }

    // be <- -scale * A (g - x0), over the lifted columns of each trial side
    std::ranges::fill(be, T(0));
    for (int s = 0; s < 2; ++s)
    {
      if (cells1[s] < 0)
        continue;
      std::span<const std::int32_t> dmap1 = dofmap1.cell_dofs(cells1[s]);
      for (std::size_t j = 0; j < num_dofs1; ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          const std::int32_t jj = bs1 * dmap1[j] + k;
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? T(0) : x0[jj];
            const std::size_t col = s * ndim1 + bs1 * j + k;
            for (std::size_t m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + col] * scale * (bc - _x0);
          }
        }
      }
    }

    // Constrain and add each test side
    for (int s = 0; s < 2; ++s)
    {
      if (cells0[s] < 0)
        continue;
      std::span<T> be_s(be.data() + s * ndim0, ndim0);
      std::span<const std::int32_t> dmap0 = dofmap0.cell_dofs(cells0[s]);
      std::span<const std::int32_t> slaves = cell_to_slaves->links(cells0[s]);
      if (!slaves.empty())
      {
        std::ranges::copy(be_s, be_copy.begin());
        dolfinx_mpc::modify_mpc_vec<T>(b, be_s, be_copy, dmap0, num_dofs0, bs0,
                                       is_slave, slaves, masters, coefficients);
      }
      for (std::size_t i = 0; i < num_dofs0; ++i)
        for (int k = 0; k < bs0; ++k)
          b[bs0 * dmap0[i] + k] += be_s[bs0 * i + k];
    }
  }
}

/// @brief Lift a set of column values into the vector b.
///
/// Computes `b <- b - scale * K^T (A (g - x0))`, where the columns to lift and
/// the values `g` are given by `bc_markers1`/`bc_values1`. These describe
/// either a set of Dirichlet conditions or the inhomogeneity of a multi point
/// constraint on the trial space.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form generating A
/// @param[in] bc_markers1 Marker for each dof of the trial space local to the
/// process, indicating whether its column should be lifted
/// @param[in] bc_values1 Value to lift for each dof of the trial space
/// @param[in] x0 Vector subtracted from the lifted values. Treated as zero if
/// empty
/// @param[in] scale Scaling to apply
/// @param[in] mpc0 Multi point constraint applied to the rows of the vector
/// @param[in] num_threads The number of threads to use for certain operations
template <typename T, std::floating_point U>
void lift_values(
    std::span<T> b, const std::shared_ptr<const dolfinx::fem::Form<T>> a,
    std::span<const std::int8_t> bc_markers1, std::span<const T> bc_values1,
    const std::span<const T>& x0, T scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    std::size_t num_threads = 1)
{
  const std::vector<T> constants = pack_constants(*a);
  auto coeff_vec = dolfinx::fem::allocate_coefficient_storage(*a);
  dolfinx::fem::pack_coefficients(*a, coeff_vec);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);

  assert(a->function_spaces().at(1));
  auto V1 = a->function_spaces().at(1);
  const int bs1 = V1->dofmap()->index_map_bs();

  // Extract dofmaps for columns and rows of a
  assert(a->function_spaces().at(0));
  auto dofmap1 = V1->dofmap();
  auto element1 = V1->element();
  auto dofmap0 = a->function_spaces()[0]->dofmap();
  const int bs0 = a->function_spaces()[0]->dofmap()->bs();
  auto element0 = a->function_spaces()[0]->element();

  auto mesh0 = a->function_spaces()[0]->mesh();
  assert(mesh0);
  auto mesh1 = a->function_spaces()[1]->mesh();
  assert(mesh1);

  // Prepare cell geometry

  auto mesh = a->mesh();
  assert(mesh);
  if (mesh->geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmaps().front();
  std::span<const U> x_g = mesh->geometry().x();
  const int tdim = mesh->topology()->dim();
  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::vector<U> coordinate_dofs(3 * num_dofs_g);

  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a->needs_facet_permutations();

  if (needs_transformation_data)
  {
    mesh0->topology_mutable()->create_cell_permutations(num_threads);
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    mesh1->topology_mutable()->create_cell_permutations(num_threads);
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  // Facet permutations of the integration domain. Needed whenever the kernel
  // asks for them, which happens for instance when the two argument spaces
  // live on different meshes.
  std::span<const std::uint8_t> perms;
  int num_facets_per_cell = 0;
  if (a->needs_facet_permutations())
  {
    const dolfinx::mesh::CellType cell_type
        = mesh->topology()->cell_types().front();
    std::size_t fdim = mesh->topology()->dim() - 1;
    num_facets_per_cell = dolfinx::mesh::cell_num_entities(cell_type, fdim);
    mesh->topology_mutable()->create_entity_permutations(fdim, num_threads);
    perms = std::span(mesh->topology()->get_entity_permutations(fdim));
  }

  // Get dof-transformations for the element matrix
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element0->template dof_transformation_fn<T>(
          dolfinx::fem::doftransform::standard);
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform_to_transpose
      = element1->template dof_transformation_right_fn<T>(
          dolfinx::fem::doftransform::transpose);
  const bool transform_set0 = dolfinx::fem::is_transform_set(dof_transform);
  const bool transform_set1
      = dolfinx::fem::is_transform_set(dof_transform_to_transpose);
  // Loop over cell integrals and lift bc

  const auto fetch_cells
      = [&](std::span<const std::int32_t> entity) { return entity.front(); };
  for (int i = 0; i < a->num_integrals(dolfinx::fem::IntegralType::cell, 0);
       ++i)
  {
    const auto& coeffs = coefficients.at({dolfinx::fem::IntegralType::cell, i});
    const auto& kernel = a->kernel(dolfinx::fem::IntegralType::cell, i, 0);

    // Function that lift bcs for cell kernels
    const auto lift_bcs_cell
        = [&](std::span<T> be, std::span<T> Ae, std::int32_t num_rows,
              std::int32_t num_cols, std::span<const std::int32_t> entity,
              std::int32_t cell0, std::int32_t cell1, std::size_t index)
    {
      auto cell = entity.front();

      // Fetch the coordinates of the cell
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
      }

      // Tabulate tensor
      std::ranges::fill(Ae, 0);
      kernel(Ae.data(), coeffs.first.data() + index * coeffs.second,
             constants.data(), coordinate_dofs.data(), nullptr, nullptr,
             nullptr);
      if (transform_set0)
        dof_transform(Ae, cell_info0, cell0, num_cols);
      if (transform_set1)
        dof_transform_to_transpose(Ae, cell_info1, cell1, num_rows);

      auto dmap1 = dofmap1->cell_dofs(cell1);
      std::ranges::fill(be, 0);
      for (std::size_t j = 0; j < dmap1.size(); ++j)
      {
        for (std::int32_t k = 0; k < bs1; k++)
        {
          const std::int32_t jj = bs1 * dmap1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0.0 : x0[jj];
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
          }
        }
      }
    };
    // Assemble over all active cells
    std::span<const std::int32_t> cells
        = a->domain(dolfinx::fem::IntegralType::cell, i, 0);
    std::span<const std::int32_t> active_cells0
        = a->domain_arg(dolfinx::fem::IntegralType::cell, 0, i, 0);
    std::span<const std::int32_t> active_cells1
        = a->domain_arg(dolfinx::fem::IntegralType::cell, 1, i, 0);
    lift_bc_entities<T, 1>(b, cells, active_cells0, active_cells1, *dofmap0,
                           *dofmap1, bc_values1, bc_markers1, *mpc0,
                           fetch_cells, lift_bcs_cell);
  }

  // Get number of cells per facet to be able to get the facet permutation
  for (int i = 0;
       i < a->num_integrals(dolfinx::fem::IntegralType::exterior_facet, 0); ++i)
  {
    const auto& coeffs
        = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
    const auto& kernel
        = a->kernel(dolfinx::fem::IntegralType::exterior_facet, i, 0);

    /// Assemble local exterior facet kernels into a vector
    /// @param[in] be The local element vector
    /// @param[in] entity The entity, given as a cell index and the local
    /// index relative to the cell
    /// @param[in] index The index of the facet in the active_facets (To fetch
    /// the appropriate coefficients)
    const auto lift_bc_exterior_facet
        = [&](std::span<T> be, std::span<T> Ae, int num_rows, int num_cols,
              std::span<const std::int32_t> entity, std::int32_t cell0,
              std::int32_t cell1, std::size_t index)
    {
      // Fetch the coordiantes of the cell
      const std::int32_t cell = entity[0];
      const int local_facet = entity[1];

      // Fetch the coordinates of the cell
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
      }

      // Tabulate tensor. A kernel that asks for the facet permutation would
      // dereference a null pointer if it were not supplied.
      const std::uint8_t perm
          = perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];
      std::ranges::fill(Ae, 0);
      kernel(Ae.data(), coeffs.first.data() + index * coeffs.second,
             constants.data(), coordinate_dofs.data(), &local_facet, &perm,
             nullptr);
      if (transform_set0)
        dof_transform(Ae, cell_info0, cell0, num_cols);
      if (transform_set1)
        dof_transform_to_transpose(Ae, cell_info1, cell1, num_rows);

      auto dmap1 = dofmap1->cell_dofs(cell1);
      std::ranges::fill(be, 0);
      for (std::size_t j = 0; j < dmap1.size(); ++j)
      {
        for (std::int32_t k = 0; k < bs1; k++)
        {
          const std::int32_t jj = bs1 * dmap1[j] + k;
          assert(jj < (int)bc_markers1.size());
          if (bc_markers1[jj])
          {
            const T bc = bc_values1[jj];
            const T _x0 = x0.empty() ? 0.0 : x0[jj];
            for (int m = 0; m < num_rows; ++m)
              be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
          }
        }
      }
    };

    // Assemble over all active cells
    std::span<const std::int32_t> active_facets
        = a->domain(dolfinx::fem::IntegralType::exterior_facet, i, 0);
    std::span<const std::int32_t> active_facets0
        = a->domain_arg(dolfinx::fem::IntegralType::exterior_facet, 0, i, 0);
    std::span<const std::int32_t> active_facets1
        = a->domain_arg(dolfinx::fem::IntegralType::exterior_facet, 1, i, 0);

    impl::lift_bc_entities<T, 2>(
        b, active_facets, active_facets0, active_facets1, *dofmap0, *dofmap1,
        bc_values1, bc_markers1, *mpc0, fetch_cells, lift_bc_exterior_facet);
  }
  for (int i = 0;
       i < a->num_integrals(dolfinx::fem::IntegralType::interior_facet, 0); ++i)
  {
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::interior_facet, i});
    const auto& kernel
        = a->kernel(dolfinx::fem::IntegralType::interior_facet, i, 0);
    std::span<const std::int32_t> facets
        = a->domain(dolfinx::fem::IntegralType::interior_facet, i, 0);
    std::span<const std::int32_t> facets0
        = a->domain_arg(dolfinx::fem::IntegralType::interior_facet, 0, i, 0);
    std::span<const std::int32_t> facets1
        = a->domain_arg(dolfinx::fem::IntegralType::interior_facet, 1, i, 0);
    lift_bc_interior_facets<T, U>(
        b, facets, facets0, facets1, *mesh, *dofmap0, *dofmap1, bc_values1,
        bc_markers1, x0, scale, *mpc0, kernel, coeffs, cstride, constants,
        cell_info0, cell_info1, perms, num_facets_per_cell, dof_transform,
        dof_transform_to_transpose);
  }
}
/// @brief Apply lifting for the inhomogeneity of a multi point constraint.
///
/// Computes `b <- b - scale * K^T (A g)` where `g` is the constraint offset of
/// `mpc1` on the trial space of `a`, i.e. the term that arises in
/// `K^T A K x_red = K^T (b - A g)` for the affine constraint `x = K x_red + g`.
///
/// @note No `x0` is subtracted. The offset is a property of the constraint
/// rather than of the current iterate, and the residual assembled at an
/// iterate that already satisfies the constraint contains `K^T A g` already.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form generating A
/// @param[in] scale Scaling to apply
/// @param[in] mpc0 Multi point constraint applied to the rows of the vector
/// @param[in] mpc1 Multi point constraint on the trial space of `a`, supplying
/// the offset `g`
/// @param[in] num_threads The number of threads to use for certain operations
template <typename T, std::floating_point U>
void apply_mpc_lifting(
    std::span<T> b, const std::shared_ptr<const dolfinx::fem::Form<T>> a,
    T scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1,
    std::size_t num_threads = 1)
{
  assert(a->function_spaces().at(1));
  auto V1 = a->function_spaces().at(1);
  auto map1 = V1->dofmap()->index_map;
  assert(map1);
  const int bs1 = V1->dofmap()->index_map_bs();
  const std::int32_t crange = bs1 * (map1->size_local() + map1->num_ghosts());

  // A homogeneous constraint costs nothing. The marker is globally reduced, so
  // every process takes the same branch and the collectives inside
  // `lift_values` are reached by all or by none.
  if (!mpc1->has_inhomogeneity())
    return;

  std::span<const std::int8_t> is_slave = mpc1->is_slave();
  const std::vector<T>& constants = mpc1->constant_values();
  if (is_slave.size() != static_cast<std::size_t>(crange))
  {
    throw std::invalid_argument(std::format(
        "Column constraint spans {} dofs but the trial space of the form has "
        "{}. The constraint must be the one built on that space.",
        is_slave.size(), crange));
  }

  std::vector<std::int8_t> markers(crange, false);
  std::vector<T> values(crange, 0.0);
  for (std::int32_t i = 0; i < crange; ++i)
  {
    if (is_slave[i] and constants[i] != T(0))
    {
      markers[i] = true;
      values[i] = constants[i];
    }
  }

  lift_values<T, U>(b, a, markers, values, std::span<const T>(), scale, mpc0,
                    num_threads);
}
} // namespace impl

namespace dolfinx_mpc
{

/// @brief Modify `b` such that
///
///   b <- b - scale * K^T (A_j (g_j - x0_j))
///
/// where `j` is a block column index, `K^T` the reduction matrix of the row
/// constraint `mpc`, and `g_j` the Dirichlet values on the trial space of
/// `a[j]`. This is the overload the library calls internally.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where `a[j]` generates `A_j`
/// @param[in] bc_markers1 Constrained dof markers on the trial space of `a[j]`,
/// owned and ghost (unrolled). An empty `bc_markers1[j]` skips block `j`.
/// @param[in] bc_values1 Dirichlet values on the trial space of `a[j]`, read
/// only where `bc_markers1[j]` is non-zero. Same length as `bc_markers1[j]`.
/// @param[in] x0 Vectors subtracted from the values. Treated as zero if empty
/// @param[in] scale Scaling to apply
/// @param[in] mpc The multi point constraint on the rows of `b`
/// @param[in] num_threads The number of threads to use for certain operations
template <typename T, std::floating_point U>
void apply_lifting(
    std::span<T> b,
    const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
    const std::vector<std::span<const std::int8_t>>& bc_markers1,
    const std::vector<std::span<const T>>& bc_values1,
    const std::vector<std::span<const T>>& x0, T scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    std::size_t num_threads = 1)
{
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::invalid_argument(std::format(
        "Mismatch between number of forms ({}) and x0 ({}) in lifting.",
        a.size(), x0.size()));
  }
  if (bc_markers1.size() != a.size() or bc_values1.size() != a.size())
  {
    throw std::invalid_argument(std::format(
        "Mismatch between number of forms ({}), markers ({}) and values ({}) "
        "in lifting.",
        a.size(), bc_markers1.size(), bc_values1.size()));
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (!a[j] or bc_markers1[j].empty())
      continue;

    std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V1
        = a[j]->function_spaces().at(1);
    assert(V1);
    const dolfinx::common::IndexMap& map1 = *V1->dofmap()->index_map;
    const std::size_t crange = V1->dofmap()->index_map_bs()
                               * (map1.size_local() + map1.num_ghosts());
    if (bc_markers1[j].size() != crange or bc_values1[j].size() != crange)
    {
      throw std::invalid_argument(std::format(
          "Block {}: markers ({}) and values ({}) must span the {} dofs of the "
          "trial space.",
          j, bc_markers1[j].size(), bc_values1[j].size(), crange));
    }

    impl::lift_values<T, U>(b, a[j], bc_markers1[j], bc_values1[j],
                            x0.empty() ? std::span<const T>() : x0[j], scale,
                            mpc, num_threads);
  }
}

/// @brief Modify `b` such that
///
///   b <- b - scale * K^T (A_j (g_j - x0_j))
///
/// for the Dirichlet conditions `bcs1[j]` on the trial space of `a[j]`.
/// @note Rebuilds the markers and values on every call. Callers that lift
/// repeatedly should build them once and call the marker-taking overload.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where `a[j]` generates `A_j`
/// @param[in] bcs1 Dirichlet conditions for each `a[j]`. Only those defined on
/// the trial space of `a[j]`, or a subspace of it, are applied.
/// @param[in] x0 Vectors subtracted from the values. Treated as zero if empty
/// @param[in] scale Scaling to apply
/// @param[in] mpc The multi point constraint on the rows of `b`
/// @param[in] num_threads The number of threads to use for certain operations
template <typename T, std::floating_point U>
void apply_lifting(
    std::span<T> b,
    const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>>&
        bcs1,
    const std::vector<std::span<const T>>& x0, T scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
    std::size_t num_threads = 1)
{
  if (bcs1.size() != a.size())
  {
    throw std::invalid_argument(std::format(
        "Mismatch between number of forms ({}) and bcs ({}) in lifting.",
        a.size(), bcs1.size()));
  }

  std::vector<std::vector<std::int8_t>> markers(a.size());
  std::vector<std::vector<T>> values(a.size());
  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (!a[j] or bcs1[j].empty())
      continue;
    std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V1
        = a[j]->function_spaces().at(1);
    const dolfinx::common::IndexMap& map1 = *V1->dofmap()->index_map;
    const std::size_t crange = V1->dofmap()->index_map_bs()
                               * (map1.size_local() + map1.num_ghosts());
    for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>& bc :
         bcs1[j])
    {
      // `mark_dofs` writes into an array sized for the condition's own space
      // and bounds-checks only in debug builds, so a condition on another space
      // must not reach it
      if (!V1->contains(*bc->function_space()))
        continue;
      if (markers[j].empty())
      {
        markers[j].resize(crange, 0);
        values[j].resize(crange, T(0));
      }
      bc->mark_dofs(markers[j]);
      bc->set(values[j], std::nullopt, 1);
    }
  }

  apply_lifting<T, U>(
      b, a,
      std::vector<std::span<const std::int8_t>>(markers.begin(), markers.end()),
      std::vector<std::span<const T>>(values.begin(), values.end()), x0, scale,
      mpc, num_threads);
}

/// @brief Modify b to account for the inhomogeneity of a multi point
/// constraint:
///
///   b <- b - scale * K^T (A_j g_j)
///
/// where j is a block (nest) column index, K^T is the reduction matrix of the
/// row constraint and g_j is the constraint offset of `mpc1[j]`. This is the
/// term arising in `K^T A K x_red = K^T (b - A g)` for the affine constraint
/// `x = K x_red + g`, and is a no-op for a homogeneous constraint.
///
/// @note Only required when solving directly for `x_red`. A residual assembled
/// at an iterate that already satisfies the constraint contains `K^T A g`
/// already, so the Newton/SNES path must not call this.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] generates A[j]
/// @param[in] scale Scaling to apply
/// @param[in] mpc0 The multi point constraint on the rows of b
/// @param[in] mpc1 The multi point constraints on the columns, one per block
/// @param[in] num_threads The number of threads to use for certain operations
template <typename T, std::floating_point U>
void apply_mpc_lifting(
    std::span<T> b,
    const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>> a, T scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::vector<
        std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>>& mpc1,
    std::size_t num_threads = 1)
{
  if (a.size() != mpc1.size())
  {
    throw std::runtime_error(std::format(
        "Mismatch between number of forms ({}) and number of column "
        "constraints ({}) in apply_mpc_lifting.",
        a.size(), mpc1.size()));
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (a[j] and mpc1[j])
      impl::apply_mpc_lifting<T, U>(b, a[j], scale, mpc0, mpc1[j], num_threads);
  }
}
} // namespace dolfinx_mpc

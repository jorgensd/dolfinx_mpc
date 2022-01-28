// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "assemble_vector.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <xtl/xspan.hpp>

namespace
{

/// Implementation of bc application (lifting) for an given a set of integartion
/// entities
/// @tparam T The scalar type
/// @tparam E_DESC Description of the set of entities
/// @param[in, out] b The vector to apply lifting to
template <typename T, typename E_DESC>
void _lift_bc_entities(
    xtl::span<T> b, const std::vector<E_DESC>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs0,
    int bs1, const xtl::span<const T>& bc_values1,
    const std::vector<std::int8_t>& bc_markers1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1,
    const std::function<const std::int32_t(const E_DESC&)> fetch_cells,
    const std::function<void(xtl::span<T>, xtl::span<T>, const int, const int,
                             E_DESC, std::size_t)>
        lift_local_vector)
{

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc1->masters();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc1->coefficients();
  const std::vector<std::int8_t>& is_slave = mpc1->is_slave();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc1->cell_to_slaves();

  // NOTE: Assertion that all links have the same size (no P refinement)
  const int num_dofs0 = dofmap0.links(0).size();
  std::vector<T> be;
  std::vector<T> be_copy;
  std::vector<T> Ae;

  // Assemble over all entities
  for (std::size_t e = 0; e < active_entities.size(); ++e)
  {
    const std::int32_t cell = fetch_cells(active_entities[e]);
    // Size data structure for assembly
    auto dmap0 = dofmap0.links(cell);
    auto dmap1 = dofmap1.links(cell);
    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();
    be.resize(num_rows);
    Ae.resize(num_rows * num_cols);

    // Check if bc is applied to entity
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        assert(bs1 * dmap1[j] + k < (int)bc_markers1.size());
        if (bc_markers1[bs1 * dmap1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }
    if (!has_bc)
      continue;
    // // Lift into local element vector
    const xtl::span<T> _be(be);
    const xtl::span<T> _Ae(Ae);
    lift_local_vector(_be, _Ae, num_rows, num_cols, active_entities[e], e);

    // Modify local element matrix if entity is connected to a slave cell
    xtl::span<const int32_t> slaves = cell_to_slaves->links(cell);

    if (slaves.size() > 0)
    {
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      be_copy.resize(num_rows);
      std::copy(be.begin(), be.end(), be_copy.begin());
      const xtl::span<T> _be_copy(be_copy);
      dolfinx_mpc::modify_mpc_vec<T>(b, _be, _be_copy, dmap0, dmap0.size(), bs0,
                                     is_slave, slaves, masters, coefficients);
    }
    // Add local contribution to b
    for (int i = 0; i < num_dofs0; ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
};

/// Modify b such that:
///
///   b <- b - scale * K^T (A (g - x0))
///
/// The boundary conditions bcs are on the trial spaces V_j.
/// The forms in [a] must have the same test space as L (from
/// which b was built), but the trial space may differ
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a is the form that
/// generates A
/// @param[in] bcs List of boundary conditions
/// @param[in] x0 The function to subtract
/// @param[in] scale Scale of lifting
/// @param[in] mpc1 The multi point constraints
template <typename T>
void _apply_lifting(
    xtl::span<T> b, const std::shared_ptr<const dolfinx::fem::Form<T>> a,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    const xtl::span<const T>& x0, double scale,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  const std::vector<T> constants = pack_constants(*a);
  auto coeff_vec = dolfinx::fem::allocate_coefficient_storage(*a);
  dolfinx::fem::pack_coefficients(*a, coeff_vec);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);
  const std::vector<std::int8_t>& is_slave = mpc1->is_slave();

  // Create 1D arrays of bc values and bc indicator
  std::vector<std::int8_t> bc_markers1;
  std::vector<T> bc_values1;
  assert(a->function_spaces().at(1));
  auto V1 = a->function_spaces().at(1);
  auto map1 = V1->dofmap()->index_map;
  const int bs1 = V1->dofmap()->index_map_bs();
  assert(map1);
  const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
  bc_markers1.assign(crange, false);
  bc_values1.assign(crange, 0.0);
  for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : bcs)
  {
    bc->mark_dofs(bc_markers1);
    bc->dof_values(bc_values1);
  }

  // Extract dofmaps for columns and rows of a
  assert(a->function_spaces().at(0));
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1
      = V1->dofmap()->list();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1 = V1->element();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0
      = a->function_spaces()[0]->dofmap()->list();
  const int bs0 = a->function_spaces()[0]->dofmap()->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a->function_spaces()[0]->element();

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a->needs_facet_permutations();

  auto mesh = a->function_spaces()[0]->mesh();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  xtl::span<const double> x_g = mesh->geometry().x();
  const int tdim = mesh->topology().dim();

  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dof-transformations for the element matrix
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element0->get_dof_transformation_function<T>();
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform_to_transpose
      = element1->get_dof_transformation_to_transpose_function<T>();

  // Loop over cell integrals and lift bc
  if (a->num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    const auto fetch_cells = [&](const std::int32_t& entity) { return entity; };
    for (int i : a->integral_ids(dolfinx::fem::IntegralType::cell))
    {
      const auto& coeffs
          = coefficients.at({dolfinx::fem::IntegralType::cell, i});
      const auto& kernel = a->kernel(dolfinx::fem::IntegralType::cell, i);

      // Function that lift bcs for cell kernels
      const auto lift_bcs_cell
          = [&](xtl::span<T> be, xtl::span<T> Ae, std::int32_t num_rows,
                std::int32_t num_cols, std::int32_t cell, std::size_t index)
      {
        // FIXME: Add proper interface for num coordinate dofs
        const std::size_t num_dofs_g = x_dofmap.num_links(cell);
        // FIXME: Reconsider when using mixed topology (mixed celltypes)
        std::vector<double> coordinate_dofs(3 * num_dofs_g);

        // Fetch the coordinates of the cell
        const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          dolfinx::common::impl::copy_N<3>(
              std::next(x_g.begin(), 3 * x_dofs[i]),
              std::next(coordinate_dofs.begin(), 3 * i));
        }
        // Tabulate tensor
        std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
        kernel(Ae.data(), coeffs.first.data() + index * coeffs.second,
               constants.data(), coordinate_dofs.data(), nullptr, nullptr);
        dof_transform(Ae, cell_info, cell, num_cols);
        dof_transform_to_transpose(Ae, cell_info, cell, num_rows);

        auto dmap1 = dofmap1.links(cell);
        std::fill(be.begin(), be.end(), 0);
        for (std::size_t j = 0; j < dmap1.size(); ++j)
        {
          for (std::int32_t k = 0; k < bs1; k++)
          {
            const std::int32_t jj = bs1 * dmap1[j] + k;
            assert(jj < (int)bc_markers1.size());
            // Add this in once we have interface for rhs of MPCs
            // MPCs overwrite Dirichlet conditions
            // if (is_slave[jj])
            // {
            //   // Lift MPC values
            //   const T val = mpc_consts[jj];
            //   for (int m = 0; m < num_rows; ++m)
            //     be[m] -= Ae[m * num_cols + bs1 * j + k] * val;
            // }
            // else
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
      const std::vector<std::int32_t>& cells = a->cell_domains(i);
      _lift_bc_entities<T, std::int32_t>(b, cells, dofmap0, dofmap1, bs0, bs1,
                                         bc_values1, bc_markers1, mpc1,
                                         fetch_cells, lift_bcs_cell);
    }
  }

  // Prepare permutations for exterior and interior facet integrals
  if (a->num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0)
  {
    // Create lambda function fetching cell index from exterior facet entity
    const auto fetch_cells_ext = [&](const std::pair<std::int32_t, int>& entity)
    { return entity.first; };

    // Get number of cells per facet to be able to get the facet permutation
    const int tdim = mesh->topology().dim();
    const int num_cell_facets = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);
    for (int i : a->integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& coeffs
          = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
      const auto& kernel
          = a->kernel(dolfinx::fem::IntegralType::exterior_facet, i);

      /// Assemble local exterior facet kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] entity The entity, given as a cell index and the local
      /// index relative to the cell
      /// @param[in] index The index of the facet in the active_facets (To fetch
      /// the appropriate coefficients)
      const auto lift_bc_exterior_facet
          = [&](xtl::span<T> be, xtl::span<T> Ae, int num_rows, int num_cols,
                std::pair<std::int32_t, int> entity, std::size_t index)
      {
        // Fetch the coordiantes of the cell
        const std::int32_t cell = entity.first;
        const int local_facet = entity.second;
        const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
        // FIXME: Add proper interface for num coordinate dofs
        const std::size_t num_dofs_g = x_dofmap.num_links(cell);
        // FIXME: Reconsider when using mixed topology (mixed celltypes)

        std::vector<double> coordinate_dofs(3 * num_dofs_g);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          dolfinx::common::impl::copy_N<3>(
              std::next(x_g.begin(), 3 * x_dofs[i]),
              std::next(coordinate_dofs.begin(), 3 * i));
        }

        // Tabulate tensor
        std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
        kernel(Ae.data(), coeffs.first.data() + index * coeffs.second,
               constants.data(), coordinate_dofs.data(), &local_facet, nullptr);
        dof_transform(Ae, cell_info, cell, num_cols);
        dof_transform_to_transpose(Ae, cell_info, cell, num_rows);

        auto dmap1 = dofmap1.links(cell);
        std::fill(be.begin(), be.end(), 0);
        for (std::size_t j = 0; j < dmap1.size(); ++j)
        {
          for (std::int32_t k = 0; k < bs1; k++)
          {
            const std::int32_t jj = bs1 * dmap1[j] + k;
            assert(jj < (int)bc_markers1.size());
            // Add this in once we have interface for rhs of MPCs
            // MPCs overwrite Dirichlet conditions
            // if (is_slave[jj])
            // {
            //   // Lift MPC values
            //   const T val = mpc_consts[jj];
            //   for (int m = 0; m < num_rows; ++m)
            //     be[m] -= Ae[m * num_cols + bs1 * j + k] * val;
            // }
            // else

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
      const std::vector<std::pair<std::int32_t, int>>& active_facets
          = a->exterior_facet_domains(i);
      _lift_bc_entities<T, std::pair<std::int32_t, int>>(
          b, active_facets, dofmap0, dofmap1, bs0, bs1, bc_values1, bc_markers1,
          mpc1, fetch_cells_ext, lift_bc_exterior_facet);
    }
  }
  if (a->num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    throw std::runtime_error(
        "Interior facet integrals currently not supported");

    //   std::function<std::uint8_t(std::size_t)> get_perm;
    //   if (a->needs_facet_permutations())
    //   {
    //     mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    //     mesh->topology_mutable().create_entity_permutations();
    //     const std::vector<std::uint8_t>& perms
    //         = mesh->topology().get_facet_permutations();
    //     get_perm = [&perms](std::size_t i) { return perms[i]; };
    //   }
    //   else
    //     get_perm = [](std::size_t) { return 0; };
  }
}
} // namespace

namespace dolfinx_mpc
{

/// Modify b such that:
///
///   b <- b - scale * K^T (A_j (g_j 0 x0_j))
///
/// where j is a block (nest) row index and K^T is the reduction matrix stemming
/// from the multi point constraint. For non - blocked problems j = 0.
/// The boundary conditions bcs1 are on the trial spaces V_j.
/// The forms in [a] must have the same test space as L (from
/// which b was built), but the trial space may differ. If x0 is not
/// supplied, then it is treated as 0.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear formss, where a[j] is the form that
/// generates A[j]
/// @param[in] bcs List of boundary conditions for each block, i.e. bcs1[2]
/// are the boundary conditions applied to the columns of a[2] / x0[2]
/// block.
/// @param[in] x0 The vectors used in the lifitng.
/// @param[in] scale Scaling to apply
/// @param[in] mpc The multi point constraints
void apply_lifting(
    xtl::span<double> b,
    const std::vector<std::shared_ptr<const dolfinx::fem::Form<double>>> a,
    const std::vector<
        std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>>&
        bcs1,
    const std::vector<xtl::span<const double>>& x0, double scale,
    std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc)
{
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::runtime_error(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }
  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (x0.empty())
    {
      _apply_lifting<double>(b, a[j], bcs1[j], xtl::span<const double>(), scale,
                             mpc);
    }
    else
    {
      _apply_lifting<double>(b, a[j], bcs1[j], x0[j], scale, mpc);
    }
  }
}
/// Modify b such that:
///
///   b <- b - scale * K^T (A_j (g_j 0 x0_j))
///
/// where j is a block (nest) row index and K^T is the reduction matrix stemming
/// from the multi point constraint. For non - blocked problems j = 0.
/// The boundary conditions bcs1 are on the trial spaces V_j.
/// The forms in [a] must have the same test space as L (from
/// which b was built), but the trial space may differ. If x0 is not
/// supplied, then it is treated as 0.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear formss, where a[j] is the form that
/// generates A[j]
/// @param[in] bcs List of boundary conditions for each block, i.e. bcs1[2]
/// are the boundary conditions applied to the columns of a[2] / x0[2]
/// block.
/// @param[in] x0 The vectors used in the lifitng.
/// @param[in] scale Scaling to apply
/// @param[in] mpc The multi point constraints
void apply_lifting(
    xtl::span<std::complex<double>> b,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::Form<std::complex<double>>>>
        a,
    const std::vector<std::vector<std::shared_ptr<
        const dolfinx::fem::DirichletBC<std::complex<double>>>>>& bcs1,
    const std::vector<xtl::span<const std::complex<double>>>& x0, double scale,

    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc)
{
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::runtime_error(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }
  for (std::size_t j = 0; j < a.size(); ++j)
  {
    if (x0.empty())
    {
      _apply_lifting<std::complex<double>>(
          b, a[j], bcs1[j], xtl::span<const std::complex<double>>(), scale,
          mpc);
    }
    else
    {
      _apply_lifting<std::complex<double>>(b, a[j], bcs1[j], x0[j], scale, mpc);
    }
  }
}
} // namespace dolfinx_mpc
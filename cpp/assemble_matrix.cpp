// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_matrix.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <xtensor/xtensor.hpp>
#include <iostream>
#include <xtensor/xio.hpp>

namespace
{
template <typename T>
void modify_mpc_cell(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const std::array<const int, 2>& num_dofs, xt::xtensor<T, 2>& Ae,
    const std::array<const xtl::span<const int32_t>, 2>& dofs, const std::array<const int, 2>& bs,
    const std::array<const xtl::span<const int32_t>, 2>& slaves,
    const std::array<const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>&
        masters,
    const std::array<const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>& coeffs,
    const std::array<const std::vector<std::int8_t>, 2>& is_slave)
{

  // Locate which local dofs are slave dofs and compute the local index of the
  // slave
  std::array<std::vector<std::int32_t>, 2> local_index;
  std::array<std::size_t, 2> num_flattened_masters = {0, 0};
  for (int axis = 0; axis < 2; ++axis)
  {
    local_index[axis] = std::vector<std::int32_t>(slaves[axis].size());
    for (std::int32_t i = 0; i < num_dofs[axis]; i++)
      for (std::int32_t j = 0; j < bs[axis]; j++)
      {
        const std::int32_t slave = dofs[axis][i] * bs[axis] + j;
        if (is_slave[axis][slave])
        {
          auto it = std::find(slaves[axis].begin(), slaves[axis].end(), slave);
          const std::int32_t slave_index = std::distance(slaves[axis].begin(), it);
          local_index[axis][slave_index] = i * bs[axis] + j;
          num_flattened_masters[axis] += masters[axis]->links(slave).size();
        }
      }
  }
  // Create copy to use for distribution to master dofs
  xt::xtensor<T, 2> Ae_original = Ae;
  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  xt::xtensor<T, 2> Ae_stripped = 
    xt::empty<T>({bs[0] * num_dofs[0], bs[1] * num_dofs[1]});

  // Strip Ae of all entries where both i and j are slaves
  for (std::int32_t i = 0; i < num_dofs[0]; i++)
    for (std::int32_t b = 0; b < bs[0]; b++)
    {
      const bool is_slave0 = is_slave[0][dofs[0][i] * bs[0] + b];
      for (std::int32_t j = 0; j < num_dofs[1]; j++)
        for (std::int32_t c = 0; c < bs[1]; c++)
        {
          const bool is_slave1 = is_slave[1][dofs[1][j] * bs[1] + c];
          Ae_stripped(i * bs[0] + b, j * bs[1] + c)
              = (is_slave0 && is_slave1) ? T(0.0) : Ae(i * bs[0] + b, j * bs[1] + c);
        }
    }

  // Zero out slave entries in element matrix
  const int ndim0 = bs[0] * num_dofs[0];
  const int ndim1 = bs[1] * num_dofs[1];
  for (std::size_t i = 0; i < local_index[0].size(); i++)
  {
    const int dof = local_index[0][i];
    // Zero slave row
    std::fill_n(std::next(Ae.begin(), ndim1 * dof), ndim1, 0.0);
  }
  for (std::size_t i = 0; i < local_index[1].size(); i++)
  {
    const int dof = local_index[1][i];
    // Zero slave column
    for (int row = 0; row < ndim0; ++row)
      Ae[row * ndim1 + dof] = 0.0;
  }

  // Flatten slaves, masters and coeffs for efficient modification of the
  // matrices
  std::array<std::vector<std::int32_t>, 2> flattened_masters;
  std::array<std::vector<std::int32_t>, 2> flattened_slaves;
  std::array<std::vector<T>, 2> flattened_coeffs;
  for (std::int8_t axis = 0; axis < 2; axis++)
  {
    flattened_masters[axis].reserve(num_flattened_masters[axis]);
    flattened_slaves[axis].reserve(num_flattened_masters[axis]);
    flattened_coeffs[axis].reserve(num_flattened_masters[axis]);
    for (std::size_t i = 0; i < slaves[axis].size(); i++)
    {
      auto _masters = masters[axis]->links(slaves[axis][i]);
      auto _coeffs = coeffs[axis]->links(slaves[axis][i]);
      for (std::size_t j = 0; j < _masters.size(); j++)
      {
        flattened_slaves[axis].push_back(local_index[axis][i]);
        flattened_masters[axis].push_back(_masters[j]);
        flattened_coeffs[axis].push_back(_coeffs[j]);
      }
    }
  }

  for (std::int8_t axis = 0; axis < 2; ++axis)
    assert(num_flattened_masters[axis] == flattened_masters[axis].size());
  // Data structures used for insertion of master contributions
  std::array<std::int32_t, 1> row;
  std::array<std::int32_t, 1> col;
  std::array<T, 1> A0;
  xt::xarray<T> Arow(ndim0);
  xt::xarray<T> Acol(ndim1);

  // Loop over each master
  std::vector<std::int32_t> unrolled_dofs(ndim1);
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {
    Acol = flattened_coeffs[0][i] * xt::row(Ae_stripped, flattened_slaves[0][i]);

    // Unroll dof blocks
    for (std::int32_t j = 0; j < num_dofs[1]; ++j)
      for (std::int32_t k = 0; k < bs[1]; ++k)
        unrolled_dofs[j * bs[1] + k] = dofs[1][j] * bs[1] + k;

    // Insert modified entries
    row[0] = flattened_masters[0][i];
    mat_set(1, row.data(), int(ndim1), unrolled_dofs.data(), Acol.data());
  }

  unrolled_dofs.resize(ndim0);
  for (std::size_t i = 0; i < num_flattened_masters[1]; ++i)
  {
    Arow = flattened_coeffs[1][i] * xt::col(Ae_stripped, flattened_slaves[1][i]);

    // Unroll dof blocks
    for (std::int32_t j = 0; j < num_dofs[0]; ++j)
      for (std::int32_t k = 0; k < bs[0]; ++k)
        unrolled_dofs[j * bs[0] + k] = dofs[0][j] * bs[0] + k;

    // Insert modified entries
    col[0] = flattened_masters[1][i];
    mat_set(int(ndim0), unrolled_dofs.data(), 1, col.data(), Arow.data());
  }

  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {
    // Loop through other masters on the same cell and add in contribution
    for (std::size_t j = 0; j < num_flattened_masters[1]; ++j)
    {
      // if (i == j)
      //   continue;

      row[0] = flattened_masters[0][i];
      col[0] = flattened_masters[1][j];
      A0[0] = flattened_coeffs[0][i] * flattened_coeffs[1][j]
              * Ae_original(flattened_slaves[0][i], flattened_slaves[1][j]);
      mat_set(1, row.data(), 1, col.data(), A0.data());
    }
  }
} // namespace

//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Mesh& mesh,
    const std::vector<std::pair<std::int32_t, int>>& facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, size_t bs0,
    const std::function<
        void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
             std::int32_t, int)>& apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, size_t bs1,
    const std::vector<std::int8_t>& bc0, const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
{

  // // Get MPC data
  // const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  //     masters = mpc->masters();
  // const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coefficients
  //     = mpc->coefficients();
  // const std::vector<std::int8_t>& is_slave = mpc->is_slave();
  // const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
  //     cell_to_slaves
  //     = mpc->cell_to_slaves();

  // // Get mesh data
  // const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
  //     = mesh.geometry().dofmap();
  // // FIXME: Add proper interface for num coordinate dofs
  // const int num_dofs_g = x_dofmap.num_links(0);
  // xtl::span<const double> x_g = mesh.geometry().x();

  // // Iterate over all facets
  // const size_t num_dofs0 = dofmap0.links(0).size();
  // const size_t num_dofs1 = dofmap1.links(0).size();
  // const std::uint32_t ndim0 = bs0 * num_dofs0;
  // const std::uint32_t ndim1 = bs1 * num_dofs1;
  // xt::xtensor<T, 2> Ae({ndim0, ndim1});
  // const xtl::span<T> _Ae(Ae);
  // std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // for (std::size_t l = 0; l < facets.size(); ++l)
  // {

  //   const std::int32_t cell = facets[l].first;
  //   const int local_facet = facets[l].second;

  //   // Get cell vertex coordinates
  //   xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
  //   for (std::size_t i = 0; i < x_dofs.size(); ++i)
  //   {
  //     dolfinx::common::impl::copy_N<3>(
  //         std::next(x_g.begin(), 3 * x_dofs[i]),
  //         std::next(coordinate_dofs.begin(), 3 * i));
  //   }
  //   // Tabulate tensor
  //   std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
  //   kernel(Ae.data(), coeffs.data() + l * cstride, constants.data(),
  //          coordinate_dofs.data(), &local_facet, nullptr);
  //   apply_dof_transformation(_Ae, cell_info, cell, ndim1);
  //   apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);

  //   // Zero rows/columns for essential bcs
  //   xtl::span<const std::int32_t> dmap0 = dofmap0.links(cell);
  //   xtl::span<const std::int32_t> dmap1 = dofmap1.links(cell);
  //   if (!bc0.empty())
  //   {
  //     for (std::size_t i = 0; i < num_dofs0; ++i)
  //     {
  //       for (std::size_t k = 0; k < bs0; ++k)
  //       {
  //         if (bc0[bs0 * dmap0[i] + k])
  //         {
  //           // Zero row bs0 * i + k
  //           const int row = bs0 * i + k;
  //           std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0.0);
  //         }
  //       }
  //     }
  //   }
  //   if (!bc1.empty())
  //   {
  //     for (std::size_t j = 0; j < num_dofs1; ++j)
  //     {
  //       for (std::size_t k = 0; k < bs1; ++k)
  //       {
  //         if (bc1[bs1 * dmap1[j] + k])
  //         {
  //           // Zero column bs1 * j + k
  //           const int col = bs1 * j + k;
  //           for (std::uint32_t row = 0; row < ndim0; ++row)
  //             Ae[row * ndim1 + col] = 0.0;
  //         }
  //       }
  //     }
  //   }

  //   // Modify local element matrix Ae and insert contributions into master
  //   // locations
  //   if (cell_to_slaves->num_links(cell) > 0)
  //   {
  //     xtl::span<const std::int32_t> slave_indices = cell_to_slaves->links(cell);
  //     // Assuming test and trial space has same number of dofs and dofs per
  //     // cell
  //     modify_mpc_cell<T>(mat_add_values, num_dofs0, Ae, dmap0, bs0,
  //                        slave_indices, masters, coefficients, is_slave);
  //   }
  //   mat_add_block_values(dmap0.size(), dmap0.data(), dmap1.size(), dmap1.data(),
  //                        Ae.data());
  // }
} // namespace
//-----------------------------------------------------------------------------
template <typename T>
void assemble_cells_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<std::int8_t>& bc0, const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T>& coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  // Get MPC data
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters(), mpc1->masters()};
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2> coefficients
      = {mpc0->coefficients(), mpc1->coefficients()};
  const std::array<const std::vector<std::int8_t>, 2>
    is_slave = {mpc0->is_slave(), mpc1->is_slave()};

  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  xtl::span<const double> x_g = geometry.x();

  // Iterate over active cells
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const size_t ndim0 = num_dofs0 * bs0;
  const size_t ndim1 = num_dofs1 * bs1;
  const std::array<const int, 2> num_dofs = {num_dofs0, num_dofs1};
  const std::array<const int, 2> bs = {bs0, bs1};
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  for (std::size_t c = 0; c < active_cells.size(); c++)
  {
    const std::int32_t cell = active_cells[c];
    // Get cell coordinates/geometry
    xtl::span<const int32_t> x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      dolfinx::common::impl::copy_N<3>(
          std::next(x_g.begin(), 3 * x_dofs[i]),
          std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    kernel(Ae.data(), coeffs.data() + c * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    apply_dof_transformation(_Ae, cell_info, cell, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);

    // Zero rows/columns for essential bcs
    xtl::span<const int32_t> dofs0 = dofmap0.links(cell);
    xtl::span<const int32_t> dofs1 = dofmap1.links(cell);
    if (!bc0.empty())
    {
      for (std::int32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
            xt::row(Ae, bs0 * i + k).fill(0);
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::int32_t j = 0; j < num_dofs1; ++j)
      {
        for (std::int32_t k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dofs1[j] + k])
            xt::col(Ae, bs1 * j + k) = xt::zeros<T>({num_dofs0 * bs0});
        }
      }
    }
    // Modify local element matrix Ae and insert contributions into master
    // locations
    if ((cell_to_slaves[0]->num_links(cell) > 0)
      or (cell_to_slaves[1]->num_links(cell) > 0))
    {
      const std::array<const xtl::span<const int32_t>, 2> slaves = 
        {cell_to_slaves[0]->links(cell), cell_to_slaves[1]->links(cell)};
      const std::array<const xtl::span<const int32_t>, 2> dofs = {dofs0, dofs1};
      modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, dofs, bs, slaves,
                         masters, coefficients, is_slave);
    }
    mat_add_block_values(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(),
                         Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_matrix_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::fem::Form<T>& a, const std::vector<std::int8_t>& bc0,
    const std::vector<std::int8_t>& bc1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
  assert(mesh);

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();
  // Prepare constants
  const std::vector<T> constants = pack_constants(a);

  // Prepare coefficients
  const auto coeff_vec = dolfinx::fem::pack_coefficients(a);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);
  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a.function_spaces().at(0)->element();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1
      = a.function_spaces().at(1)->element();
  std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation = element0->get_dof_transformation_function<T>();
  std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation_to_transpose
      = element1->get_dof_transformation_to_transpose_function<T>();

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::cell, i});
    const std::vector<std::int32_t>& active_cells = a.cell_domains(i);
    assemble_cells_impl<T>(
        mat_add_block_values, mat_add_values, mesh->geometry(), active_cells,
        apply_dof_transformation, dofs0, bs0,
        apply_dof_transformation_to_transpose, dofs1, bs1, bc0, bc1, fn, coeffs,
        cstride, constants, cell_info, mpc0, mpc1);
  }

  // for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
  // {
  //   const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
  //   const auto& [coeffs, cstride]
  //       = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
  //   const std::vector<std::pair<std::int32_t, int>>& facets
  //       = a.exterior_facet_domains(i);
  //   assemble_exterior_facets<T>(mat_add_block_values, mat_add_values, *mesh,
  //                               facets, apply_dof_transformation, dofs0, bs0,
  //                               apply_dof_transformation_to_transpose, dofs1,
  //                               bs1, bc0, bc1, fn, coeffs, cstride, constants,
  //                               cell_info, mpc);
  // }

  // if (a.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  // {
  //   throw std::runtime_error("Not implemented yet");
  //   // const int tdim = mesh->topology().dim();
  //   // mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  //   // mesh->topology_mutable().create_entity_permutations();

  //   // std::function<std::uint8_t(std::size_t)> get_perm;
  //   // if (a.needs_facet_permutations())
  //   // {
  //   //   mesh->topology_mutable().create_entity_permutations();
  //   //   const std::vector<std::uint8_t>& perms
  //   //       = mesh->topology().get_facet_permutations();
  //   //   get_perm = [&perms](std::size_t i) { return perms[i]; };
  //   // }
  //   // else
  //   //   get_perm = [](std::size_t) { return 0; };
  // }
}
//-----------------------------------------------------------------------------
template <typename T>
void _assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add,
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    const T diagval)
{
  dolfinx::common::Timer timer("~MPC: Assemble matrix (C++)");

  // Index maps for dof ranges
  std::shared_ptr<const dolfinx::common::IndexMap> map0
      = a.function_spaces().at(0)->dofmap()->index_map;
  std::shared_ptr<const dolfinx::common::IndexMap> map1
      = a.function_spaces().at(1)->dofmap()->index_map;
  int bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  int bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

  // Build dof markers
  std::vector<std::int8_t> dof_marker0, dof_marker1;
  std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
  std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a.function_spaces().at(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }
    if (a.function_spaces().at(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  assemble_matrix_impl<T>(mat_add_block, mat_add, a, dof_marker0, dof_marker1,
                          mpc0, mpc1);

  // Add diagval on diagonal for slave dofs
  if (mpc0->function_space() == mpc1->function_space())
  {
    const std::vector<std::int32_t>& slaves = mpc0->slaves();
    const std::int32_t num_local_slaves = mpc0->num_local_slaves();
    std::vector<std::int32_t> diag_dof(1);
    std::vector<T> diag_value(1);
    diag_value[0] = diagval;
    for (std::int32_t i = 0; i < num_local_slaves; ++i)
    {
      diag_dof[0] = slaves[i];
      mat_add(1, diag_dof.data(), 1, diag_dof.data(), diag_value.data());
    }
  }
  timer.stop();
}
} // namespace
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const double*)>& mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const double*)>& mat_add,
    const dolfinx::fem::Form<double>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    const double diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const std::complex<double>*)>&
        mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const std::complex<double>*)>&
        mat_add,
    const dolfinx::fem::Form<std::complex<double>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc1,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    const std::complex<double> diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}

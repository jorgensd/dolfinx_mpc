// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <iostream>

namespace
{
template <typename T>
void modify_mpc_cell(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const int num_dofs, xt::xtensor<T, 2>& Ae,
    const xtl::span<const int32_t>& dofs, int bs,
    const xtl::span<const int32_t>& slave_indices,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coeffs,
    const std::vector<std::int32_t>& local_indices,
    const std::vector<bool>& is_slave)
{
  // Arrays for flattened master slave data
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

  // Create copy to use for distribution to master dofs
  xt::xtensor<T, 2> Ae_original = Ae;
  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  xt::xtensor<T, 2> Ae_stripped = xt::empty<T>({bs * num_dofs, bs * num_dofs});
  for (std::int32_t i = 0; i < bs * num_dofs; ++i)
    for (std::int32_t j = 0; j < bs * num_dofs; ++j)
      Ae_stripped(i, j) = (is_slave[i] && is_slave[j]) ? 0 : Ae(i, j);

  // Unroll dof blocks
  std::vector<std::int32_t> mpc_dofs(bs * dofs.size());
  for (std::int32_t j = 0; j < dofs.size(); ++j)
    for (std::int32_t k = 0; k < bs; ++k)
      mpc_dofs[j * bs + k] = dofs[j] * bs + k;

  for (std::int32_t i = 0; i < flattened_slaves.size(); ++i)
  {
    const std::int32_t& local_index = flattened_slaves[i];
    const std::int32_t& master = flattened_masters[i];
    const T& coeff = flattened_coeffs[i];

    // Remove contributions form Ae
    xt::col(Ae, local_index).fill(0);
    xt::row(Ae, local_index).fill(0);

    const xt::xarray<T> Arow = coeff * xt::col(Ae_stripped, local_index);
    const xt::xarray<T> Acol = coeff * xt::row(Ae_stripped, local_index);
    const T Amaster = coeff * coeff * Ae_original(local_index, local_index);

    // Insert modified entries
    const auto mpc_dof_old = mpc_dofs[local_index]; // store old mpc value
    mpc_dofs[local_index] = master;
    mat_set(bs * num_dofs, mpc_dofs.data(), 1, &master, Arow.data());
    mat_set(1, &master, bs * num_dofs, mpc_dofs.data(), Acol.data());
    mat_set(1, &master, 1, &master, &Amaster);
    mpc_dofs[local_index] = mpc_dof_old;

    // Loop through other masters on the same cell
    std::vector<std::int32_t> other_slaves(flattened_slaves.size() - 1);
    std::iota(std::begin(other_slaves), std::end(other_slaves), 0);
    if (i < flattened_slaves.size() - 1)
      other_slaves[i] = flattened_slaves.size() - 1;
    for (const auto j : other_slaves)
    {
      const std::int32_t& other_local_index = flattened_slaves[j];
      const std::int32_t& other_master = flattened_masters[j];
      const T& other_coeff = flattened_coeffs[j];
      const T Am0m1 = coeff * other_coeff * Ae_original(local_index, other_local_index);
      mat_set(1, &master, 1, &other_master, &Am0m1);
    }
  }
}
}

namespace
{
//-----------------------------------------------------------------------------
template <typename T>
void modify_mpc_cell_rect(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const std::array<int, 2> num_dofs, xt::xtensor<T, 2>& Ae,
    const std::array<xtl::span<const int32_t>, 2>& dofs, std::array<int, 2> bs,
    const std::array<xtl::span<const int32_t>, 2>& slave_indices,
    const std::array<const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>&
        masters,
    const std::array<const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>& coeffs,
    const std::array<std::vector<std::int32_t>, 2>& local_indices,
    const std::array<std::vector<bool>, 2>& is_slave)
{

  // Arrays for flattened master slave data
  std::array<std::vector<std::int32_t>, 2> flattened_masters;
  std::array<std::vector<std::int32_t>, 2> flattened_slaves;
  std::array<std::vector<std::int32_t>, 2> slaves_loc;
  std::array<std::vector<T>, 2> flattened_coeffs;
  for (int axis = 0; axis < 2; ++axis)
  {
    for (std::int32_t i = 0; i < slave_indices[axis].size(); ++i)
    {
      xtl::span<const std::int32_t> local_masters
          = masters[axis]->links(slave_indices[axis][i]);
      xtl::span<const T> local_coeffs = coeffs[axis]->links(slave_indices[axis][i]);

      for (std::int32_t j = 0; j < local_masters.size(); ++j)
      {
        slaves_loc[axis].push_back(i);
        flattened_slaves[axis].push_back(local_indices[axis][i]);
        flattened_masters[axis].push_back(local_masters[j]);
        flattened_coeffs[axis].push_back(local_coeffs[j]);
      }
    }
  }

  // Create copy to use for distribution to master dofs
  const xt::xtensor<T, 2> Ae_original = Ae;
  // Build matrix where all (slave_i, slave_j) entries are 0 for usage in row and
  // column addition
  xt::xtensor<T, 2> Ae_stripped = xt::empty<T>({bs[0] * num_dofs[0], bs[1] * num_dofs[1]});
  for (std::int32_t i = 0; i < bs[0] * num_dofs[0]; ++i)
    for (std::int32_t j = 0; j < bs[1] * num_dofs[1]; ++j)
      Ae_stripped(i, j) = (is_slave[0][i] && is_slave[1][j]) ? 0 : Ae(i, j);

  // Unroll dof blocks
  std::array<std::vector<std::int32_t>, 2> mpc_dofs =
    {std::vector<std::int32_t>(bs[0] * dofs[0].size()),
     std::vector<std::int32_t>(bs[1] * dofs[1].size())};
  for (int axis = 0; axis < 2; ++axis)
    for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
      for (std::int32_t k = 0; k < bs[axis]; ++k)
        mpc_dofs[axis][j * bs[axis] + k] = dofs[axis][j] * bs[axis] + k;

  const std::array<const std::function<void(const std::int32_t&, const std::int32_t&, const T&)>, 2>
    inserters = {
      [&](const std::int32_t& local_index, const std::int32_t& master, const T& coeff)->void
      {
        xt::row(Ae, local_index).fill(0);
        const xt::xarray<T> Acols = coeff * xt::row(Ae_stripped, local_index);
        mat_set(1, &master, bs[1] * num_dofs[1], mpc_dofs[1].data(), Acols.data());
      [&](const std::int32_t& local_index, const std::int32_t& master, const T& coeff)->void
      {
        xt::col(Ae, local_index).fill(0);
        const xt::xarray<T> Arows = coeff * xt::col(Ae_stripped, local_index);
        mat_set(bs[0] * num_dofs[0], mpc_dofs[0].data(), 1, &master, Arows.data());
      }
    };

  // Insert: (slave_i, :) rows to (master_i, :)
  //         (:, slave_j) cols to (:, master_j)
  for (int axis = 0; axis < 2; ++axis)
  {
    for (std::int32_t i = 0; i < flattened_slaves[axis].size(); ++i)
    {
      const std::int32_t& local_index = flattened_slaves[axis][i];
      const std::int32_t& master = flattened_masters[axis][i];
      const T& coeff = flattened_coeffs[axis][i];
      inserters[axis](local_index, master, coeff);
    }
  }

  // Insert the (slave_i, slave_j) contributions to (master_i, master_j)
  for (std::int32_t i = 0; i < flattened_slaves[0].size(); ++i)
  {
    for (std::int32_t j = 0; j < flattened_slaves[1].size(); ++j)
    {
      const std::int32_t& local_index0 = flattened_slaves[0][i];
      const std::int32_t& local_index1 = flattened_slaves[1][j];
      const std::int32_t& master0 = flattened_masters[0][i];
      const std::int32_t& master1 = flattened_masters[1][j];
      const T& coeff0 = flattened_coeffs[0][i];
      const T& coeff1 = flattened_coeffs[1][j];

      const T Amaster = coeff0 * coeff1 * Ae_original(local_index0, local_index1);
      mat_set(1, &master0, 1, &master1, &Amaster);
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
    const std::vector<std::int32_t>& active_facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<
        void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
             std::int32_t, int)>& apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const dolfinx::array2d<T>& coeffs, const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble exterior facets");

  // Get MPC data
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters_local(), mpc1->masters_local()};
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coeffs(), mpc1->coeffs()};
  const std::array<const xtl::span<const std::int32_t>, 2> 
    slaves = {mpc0->slaves(), mpc1->slaves()};

  std::array<xtl::span<const std::int32_t>, 2>
    slave_cells = {mpc0->slave_cells(), mpc1->slave_cells()};
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  // Compute local indices for slave cells
  std::array<std::vector<bool>, 2> is_slave_facet =
    {std::vector<bool>(active_facets.size(), false),
     std::vector<bool>(active_facets.size(), false)};
  std::array<std::vector<std::int32_t>, 2> slave_cell_index
    = {std::vector<std::int32_t>(active_facets.size(), -1),
       std::vector<std::int32_t>(active_facets.size(), -1)};
  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (std::int32_t i = 0; i < active_facets.size(); ++i)
  {
    const std::int32_t f = active_facets[i];
    xtl::span<const std::int32_t> cells = f_to_c->links(f);
    assert(cells.size() == 1);
    for (std::int32_t axis = 0; axis < 2; ++axis)
    {
      for (std::int32_t j = 0; j < slave_cells[axis].size(); ++j)
        if (slave_cells[axis][j] == cells[0])
        {
          is_slave_facet[axis][i] = true;
          slave_cell_index[axis][i] = j;
          break;
        }
    }
  }

  // Iterate over all facets
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  for (std::int32_t l = 0; l < active_facets.size(); ++l)
  {
    const std::int32_t f = active_facets[l];
    xtl::span<const std::int32_t> cells = f_to_c->links(f);

    // Get local index of facet with respect to the cell
    xtl::span<const std::int32_t> facets = c_to_f->links(cells[0]);
    const auto* it = std::find(facets.data(), facets.data() + facets.size(), f);
    assert(it != (facets.data() + facets.size()));
    const int local_facet = std::distance(facets.data(), it);

    // Get cell vertex coordinates
    xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cells[0]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    kernel(Ae.data(), coeffs.row(cells[0]).data(), constants.data(),
           coordinate_dofs.data(), &local_facet,
           &perms[cells[0] * facets.size() + local_facet]);

    apply_dof_transformation(_Ae, cell_info, cells[0], ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, cells[0], ndim0);

    // Zero rows/columns for essential bcs
    xtl::span<const std::int32_t> dmap0 = dofmap0.links(cells[0]);
    xtl::span<const std::int32_t> dmap1 = dofmap1.links(cells[0]);
    if (!bc0.empty())
    {
      for (std::int32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dmap0[i] + k])
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
          if (bc1[bs1 * dmap1[j] + k])
            xt::col(Ae, bs1 * j + k).fill(0);
        }
      }
    }

    const std::array<int, 2> bs = {bs0, bs1};
    const std::array<xtl::span<const int32_t>, 2> dofs = {dmap0, dmap1};
    const std::array<int, 2> num_dofs = {num_dofs0, num_dofs1};
    std::array<xtl::span<const int32_t>, 2> slave_indices;
    std::array<std::vector<bool>, 2> is_slave =
      {std::vector<bool>(bs[0] * num_dofs[0], false),
       std::vector<bool>(bs[1] * num_dofs[1], false)};
    std::array<std::vector<std::int32_t>, 2> local_indices;

    if (is_slave_facet[0][l] or is_slave_facet[1][l])
    {
      for (int axis = 0; axis < 2; ++axis)
      {
        // Assuming test and trial space has same number of dofs and dofs per
        // cell
        slave_indices[axis] = cell_to_slaves[axis]->links(slave_cell_index[axis][l]);
        // Find local position of every slave
        local_indices[axis] = std::vector<std::int32_t>(slave_indices[axis].size());

        for (std::int32_t i = 0; i < slave_indices[axis].size(); ++i)
        {
          bool found = false;
          for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
          {
            for (std::int32_t k = 0; k < bs[axis]; ++k)
            {
              if (bs[axis] * dofs[axis][j] + k == slaves[axis][slave_indices[axis][i]])
              {
                local_indices[axis][i] = bs[axis] * j + k;
                is_slave[axis][bs[axis] * j + k] = true;
                found = true;
                break;
              }
            }
            if (found)
              break;
          }
        }
      }
      modify_mpc_cell_rect<T>(mat_add_values, num_dofs, Ae, dofs, bs,
                              slave_indices, masters, coefficients, local_indices,
                              is_slave);
    }
    mat_add_block_values(dmap0.size(), dmap0.data(), dmap1.size(), dmap1.data(),
                         Ae.data());
  }
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
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const dolfinx::array2d<T>& coeffs, const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble cells");
  
  // Get MPC data
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters_local(), mpc1->masters_local()};
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coeffs(), mpc1->coeffs()};
  const std::array<const xtl::span<const std::int32_t>, 2> 
    slaves = {mpc0->slaves(), mpc1->slaves()};

  std::array<xtl::span<const std::int32_t>, 2>
    slave_cells = {mpc0->slave_cells(), mpc1->slave_cells()};
  const std::array<
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = geometry.x();

  // Compute local indices for slave cells
  std::array<std::vector<bool>, 2> is_slave_cell = 
    {std::vector<bool>(active_cells.size(), false),
     std::vector<bool>(active_cells.size(), false)};
  std::array<std::vector<std::int32_t>, 2> slave_cell_index
    = {std::vector<std::int32_t>(active_cells.size(), -1),
       std::vector<std::int32_t>(active_cells.size(), -1)};
  for (std::int32_t j = 0; j < active_cells.size(); ++j)
  {
    for (std::int32_t axis = 0; axis < 2; ++axis)
    {
      for (std::int32_t i = 0; i < slave_cells[axis].size(); ++i)
      {
        if (slave_cells[axis][i] == active_cells[j])
        {
          is_slave_cell[axis][j] = true;
          slave_cell_index[axis][j] = i;
          break;
        }
      }
    }
  }

  // Iterate over active cells
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const size_t ndim0 = num_dofs0 * bs0;
  const size_t ndim1 = num_dofs1 * bs1;
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  for (std::int32_t l = 0; l < active_cells.size(); ++l)
  {
    // Get cell coordinates/geometry
    const std::int32_t c = active_cells[l];
    xtl::span<const int32_t> x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    kernel(Ae.data(), coeffs.row(c).data(), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    apply_dof_transformation(_Ae, cell_info, c, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, c, ndim0);

    // Zero rows/columns for essential bcs
    xtl::span<const int32_t> dofs0 = dofmap0.links(c);
    xtl::span<const int32_t> dofs1 = dofmap1.links(c);
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
            xt::col(Ae, bs1 * j + k).fill(0);
        }
      }
    }

    const std::array<int, 2> bs = {bs0, bs1};
    const std::array<xtl::span<const int32_t>, 2> dofs = {dofs0, dofs1};
    const std::array<int, 2> num_dofs = {num_dofs0, num_dofs1};
    std::array<xtl::span<const int32_t>, 2> slave_indices;
    std::array<std::vector<bool>, 2> is_slave = 
      {std::vector<bool>(bs[0] * num_dofs[0], false),
       std::vector<bool>(bs[1] * num_dofs[1], false)};
    std::array<std::vector<std::int32_t>, 2> local_indices;

    if (is_slave_cell[0][l] or is_slave_cell[1][l])
    {
      for (int axis = 0; axis < 2; ++axis)
      {
        slave_indices[axis] = cell_to_slaves[axis]->links(slave_cell_index[axis][l]);
        // Find local position of every slave
        local_indices[axis] = std::vector<std::int32_t>(slave_indices[axis].size());

        for (std::int32_t i = 0; i < slave_indices[axis].size(); ++i)
        {
          bool found = false;
          for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
          {
            for (std::int32_t k = 0; k < bs[axis]; ++k)
            {
              if (bs[axis] * dofs[axis][j] + k == slaves[axis][slave_indices[axis][i]])
              {
                local_indices[axis][i] = bs[axis] * j + k;
                is_slave[axis][bs[axis] * j + k] = true;
                break;
              }
            }
            if (found)
              break;
          }
        }
      }
      modify_mpc_cell_rect<T>(mat_add_values, num_dofs, Ae, dofs, bs,
                              slave_indices, masters, coefficients, local_indices,
                              is_slave);
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
    const dolfinx::fem::Form<T>& a, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

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
  const dolfinx::array2d<T> coeffs = dolfinx::fem::pack_coefficients(a);

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
    const std::vector<std::int32_t>& active_cells
        = a.domains(dolfinx::fem::IntegralType::cell, i);
    assemble_cells_impl<T>(mat_add_block_values, mat_add_values,
                           mesh->geometry(), active_cells,
                           apply_dof_transformation, dofs0, bs0,
                           apply_dof_transformation_to_transpose, dofs1, bs1,
                           bc0, bc1, fn, coeffs, constants, cell_info,
                           mpc0, mpc1);
  }
  if (a.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0
      or a.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const int facets_per_cell = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);
    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();

    for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(dolfinx::fem::IntegralType::exterior_facet, i);
      assemble_exterior_facets<T>(
          mat_add_block_values, mat_add_values, *mesh, active_facets,
          apply_dof_transformation, dofs0, bs0,
          apply_dof_transformation_to_transpose, dofs1, bs1, bc0, bc1, fn,
          coeffs, constants, cell_info, perms, mpc0, mpc1);
    }

    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(dolfinx::fem::IntegralType::interior_facet))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(dolfinx::fem::IntegralType::interior_facet, i);
      throw std::runtime_error("Not implemented yet");

      //   impl::assemble_interior_facets(
      //       mat_set_values, *mesh, active_facets, *dofmap0, *dofmap1, bc0,
      //       bc1, fn, coeffs, c_offsets, constants, cell_info, perms);
    }
  }
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
  assert(a.function_spaces().at(0) == mpc0->function_space());
  assert(a.function_spaces().at(1) == mpc1->function_space());

  dolfinx::common::Timer timer_s("~MPC: Assembly (C++)");

  // Index maps for dof ranges
  std::shared_ptr<const dolfinx::common::IndexMap> map0
      = a.function_spaces().at(0)->dofmap()->index_map;
  std::shared_ptr<const dolfinx::common::IndexMap> map1
      = a.function_spaces().at(1)->dofmap()->index_map;
  int bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  int bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

  // Build dof markers
  std::vector<bool> dof_marker0, dof_marker1;
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
    const xtl::span<const std::int32_t> slaves = mpc0->slaves();
    const std::int32_t num_local_slaves = mpc0->num_local_slaves();
    for (std::int32_t i = 0; i < num_local_slaves; ++i)
      mat_add(1, &slaves[i], 1, &slaves[i], &diagval);
  }
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

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
//-----------------------------------------------------------------------------
template <typename T>
void modify_mpc_cell_rect(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const std::array<int, 2> num_dofs, xt::xtensor<T, 2>& Ae,
    const std::array<xtl::span<const int32_t>, 2>& dofs, std::array<int, 2> bs,
    const std::array<xtl::span<const int32_t>, 2>& slave_indices,
    const std::array<const std::shared_ptr<
                         const dolfinx::graph::AdjacencyList<std::int32_t>>,
                     2>& masters,
    const std::array<
        const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>&
        coeffs,
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
      xtl::span<const T> local_coeffs
          = coeffs[axis]->links(slave_indices[axis][i]);

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

  assert(num_dofs[0] == dofs[0].size());
  assert(num_dofs[1] == dofs[1].size());
  const std::array<std::int32_t, 2> num_dofs_unrolled
      = {bs[0] * num_dofs[0], bs[1] * num_dofs[1]};
  // Build matrix where all (slave_i, slave_j) entries are 0 for usage in row
  // and column addition
  xt::xtensor<T, 2> Ae_stripped
      = xt::empty<T>({num_dofs_unrolled[0], num_dofs_unrolled[1]});
  for (std::int32_t i = 0; i < num_dofs_unrolled[0]; ++i)
    for (std::int32_t j = 0; j < num_dofs_unrolled[1]; ++j)
      Ae_stripped(i, j) = (is_slave[0][i] && is_slave[1][j]) ? 0 : Ae(i, j);

  // Unroll dof blocks
  std::array<std::vector<std::int32_t>, 2> mpc_dofs
      = {std::vector<std::int32_t>(num_dofs_unrolled[0]),
         std::vector<std::int32_t>(num_dofs_unrolled[1])};
  for (int axis = 0; axis < 2; ++axis)
    for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
      for (std::int32_t k = 0; k < bs[axis]; ++k)
        mpc_dofs[axis][j * bs[axis] + k] = dofs[axis][j] * bs[axis] + k;

  const std::array<const std::function<void(const std::int32_t&,
                                            const std::int32_t&, const T&)>,
                   2>
      inserters = {[&](const std::int32_t& local_index,
                       const std::int32_t& master, const T& coeff) -> void
                   {
                     xt::row(Ae, local_index).fill(0);
                     const xt::xarray<T> Acols
                         = coeff * xt::row(Ae_stripped, local_index);
                     mat_set(1, &master, num_dofs_unrolled[0],
                             mpc_dofs[1].data(), Acols.data());
                   },
                   [&](const std::int32_t& local_index,
                       const std::int32_t& master, const T& coeff) -> void
                   {
                     xt::col(Ae, local_index).fill(0);
                     const xt::xarray<T> Arows
                         = coeff * xt::col(Ae_stripped, local_index);
                     mat_set(num_dofs_unrolled[1], mpc_dofs[0].data(), 1,
                             &master, Arows.data());
                   }};

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

      const T Amaster
          = coeff0 * coeff1 * Ae_original(local_index0, local_index1);
      mat_set(1, &master0, 1, &master1, &Amaster);
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T, typename E_DESC>  // E_DESC, the entity description
void assemble_entity_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Mesh& mesh,
    const std::vector<E_DESC>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const xtl::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1,
    const std::function<const std::int32_t(const E_DESC&)> fetch_cells,
    const std::function<void(xt::xtensor<T, 2>&, const E_DESC&)>
      assemble_local_element_matrix)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble entity");

  // Get MPC data
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>,
      2>
      masters = {mpc0->masters_local(), mpc1->masters_local()};
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coeffs(), mpc1->coeffs()};
  const std::array<const xtl::span<const std::int32_t>, 2> slaves
      = {mpc0->slaves(), mpc1->slaves()};

  std::array<xtl::span<const std::int32_t>, 2> slave_cells
      = {mpc0->slave_cells(), mpc1->slave_cells()};
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>,
      2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Compute local indices for slave cells
  std::array<std::vector<bool>, 2> is_slave_entity
      = {std::vector<bool>(active_entities.size(), false),
         std::vector<bool>(active_entities.size(), false)};
  std::array<std::vector<std::int32_t>, 2> slave_cell_index
      = {std::vector<std::int32_t>(active_entities.size(), -1),
         std::vector<std::int32_t>(active_entities.size(), -1)};

  for (std::int32_t i = 0; i < active_entities.size(); ++i)
  {
    const std::int32_t cell = fetch_cells(active_entities[i]);
    for (std::int32_t axis = 0; axis < 2; ++axis)
    {
      for (std::int32_t j = 0; j < slave_cells[axis].size(); ++j)
        if (slave_cells[axis][j] == cell)
        {
          is_slave_entity[axis][i] = true;
          slave_cell_index[axis][i] = j;
          break;
        }
    }
  }

  // Iterate over all entities
  // NOTE: Assertion that all links have the same size (no P refinement)
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  for (std::int32_t l = 0; l < active_entities.size(); ++l)
  {
    const std::int32_t cell = fetch_cells(active_entities[l]);
    assemble_local_element_matrix(Ae, active_entities[l]);

    // Zero rows/columns for essential bcs
    xtl::span<const std::int32_t> dmap0 = dofmap0.links(cell);
    xtl::span<const std::int32_t> dmap1 = dofmap1.links(cell);
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
    std::array<std::vector<bool>, 2> is_slave
        = {std::vector<bool>(ndim0, false), std::vector<bool>(ndim1, false)};
    std::array<std::vector<std::int32_t>, 2> local_indices;

    if (is_slave_entity[0][l] or is_slave_entity[1][l])
    {
      for (int axis = 0; axis < 2; ++axis)
      {
        // Assuming test and trial space has same number of dofs and dofs per
        // cell
        slave_indices[axis]
            = cell_to_slaves[axis]->links(slave_cell_index[axis][l]);
        // Find local position of every slave
        local_indices[axis]
            = std::vector<std::int32_t>(slave_indices[axis].size());

        for (std::int32_t i = 0; i < slave_indices[axis].size(); ++i)
        {
          bool found = false;
          for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
          {
            for (std::int32_t k = 0; k < bs[axis]; ++k)
            {
              if (bs[axis] * dofs[axis][j] + k
                  == slaves[axis][slave_indices[axis][i]])
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
                              slave_indices, masters, coefficients,
                              local_indices, is_slave);
    }
    mat_add_block_values(dmap0.size(), dmap0.data(), dmap1.size(), dmap1.data(),
                         Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename T>  // E_DESC, the entity description
void assemble_interior_facets_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Mesh& mesh,
    const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const xtl::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const dolfinx::fem::DofMap& fulldofmap0, const dolfinx::fem::DofMap& fulldofmap1,
    const std::vector<int>& offsets,
    const std::function<std::uint8_t(std::size_t)>& get_perm,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble interior facet");

  // // Get MPC data
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>,
      2>
      masters = {mpc0->masters_local(), mpc1->masters_local()};
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coeffs(), mpc1->coeffs()};
  const std::array<const xtl::span<const std::int32_t>, 2> slaves
      = {mpc0->slaves(), mpc1->slaves()};

  std::array<xtl::span<const std::int32_t>, 2> slave_cells
      = {mpc0->slave_cells(), mpc1->slave_cells()};
  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Compute local indices for slave cells
  std::array<std::vector<bool>, 2> is_slave_entity0
      = {std::vector<bool>(active_entities.size(), false),
         std::vector<bool>(active_entities.size(), false)};
  std::array<std::vector<bool>, 2> is_slave_entity1
      = {std::vector<bool>(active_entities.size(), false),
         std::vector<bool>(active_entities.size(), false)};
  std::array<std::vector<std::int32_t>, 2> slave_cell_index0
      = {std::vector<std::int32_t>(active_entities.size(), -1),
         std::vector<std::int32_t>(active_entities.size(), -1)};
  std::array<std::vector<std::int32_t>, 2> slave_cell_index1
      = {std::vector<std::int32_t>(active_entities.size(), -1),
         std::vector<std::int32_t>(active_entities.size(), -1)};

  for (std::int32_t i = 0; i < active_entities.size(); ++i)
  {
    const std::int32_t cell0 = std::get<0>(active_entities[i]);
    const std::int32_t cell1 = std::get<2>(active_entities[i]);
    for (std::int32_t axis = 0; axis < 2; ++axis)
    {
      bool break0 = false;
      bool break1 = false;
      for (std::int32_t j = 0; j < slave_cells[axis].size(); ++j)
      {
        if (slave_cells[axis][j] == cell0)
        {
          is_slave_entity0[axis][i] = true;
          slave_cell_index0[axis][i] = j;
          break0 = true;
        }
        if (slave_cells[axis][j] == cell1)
        {
          is_slave_entity1[axis][i] = true;
          slave_cell_index1[axis][i] = j;
          break1 = true;
        }
        if (break0 && break1)
          break;
      }
    }
  }
  
  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh.geometry().dofmap();
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  // Data structures used in assembly
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  const int num_cell_facets
      = dolfinx::mesh::cell_num_entities(
          mesh.topology().cell_type(), mesh.topology().dim() - 1);

  // Iterate over all entities
  // NOTE: Assertion that all links have the same size (no P refinement)
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae;
  
  for (std::int32_t l = 0; l < active_entities.size(); ++l)
  {
    const std::array<std::int32_t, 2> cell
        = {std::get<0>(active_entities[l]), std::get<2>(active_entities[l])};
    const std::array<int, 2> local_facet
        = {std::get<1>(active_entities[l]), std::get<3>(active_entities[l])};

    const std::array<xtl::span<const std::int32_t>, 2> x_dofs
        = {x_dofmap.links(cell[0]), x_dofmap.links(cell[1])};
    const xt::xtensor<double, 3> coordinate_dofs = 
      xt::concatenate(xt::xtuple(
        xt::xarray<double>(xt::view(x_g, xt::keep(x_dofs[0]), xt::all())),
        xt::xarray<double>(xt::view(x_g, xt::keep(x_dofs[1]), xt::all()))));


    // Get dof maps for cells and pack
    xtl::span<const std::int32_t> dmap0_cell0 = fulldofmap0.cell_dofs(cell[0]);
    xtl::span<const std::int32_t> dmap0_cell1 = fulldofmap0.cell_dofs(cell[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    xtl::span<const std::int32_t> dmap1_cell0 = fulldofmap1.cell_dofs(cell[0]);
    xtl::span<const std::int32_t> dmap1_cell1 = fulldofmap1.cell_dofs(cell[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::copy(dmap1_cell0.begin(), dmap1_cell0.end(), dmapjoint1.begin());
    std::copy(dmap1_cell1.begin(), dmap1_cell1.end(),
              std::next(dmapjoint1.begin(), dmap1_cell0.size()));

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    const T* coeff_cell0 = coeffs.data() + cell[0] * cstride;
    const T* coeff_cell1 = coeffs.data() + cell[1] * cstride;

    // Loop over coefficients
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    {
      // Loop over entries for coefficient i
      const int num_entries = offsets[i + 1] - offsets[i];
      std::copy_n(coeff_cell0 + offsets[i], num_entries,
                  std::next(coeff_array.begin(), 2 * offsets[i]));
      std::copy_n(coeff_cell1 + offsets[i], num_entries,
                  std::next(coeff_array.begin(), offsets[i + 1] + offsets[i]));
    }

    const int num_rows = bs0 * dmapjoint0.size();
    const int num_cols = bs1 * dmapjoint1.size();

    // Tabulate tensor
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);

    const std::array perm{
        get_perm(cell[0] * num_cell_facets + local_facet[0]),
        get_perm(cell[1] * num_cell_facets + local_facet[1])};
    fn(Ae.data(), coeff_array.data(), constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data());

    const xtl::span<T> _Ae(Ae);
    const xtl::span<T> sub_Ae0
        = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                      bs0 * dmap0_cell1.size() * num_cols);
    const xtl::span<T> sub_Ae1
        = _Ae.subspan(bs1 * dmap1_cell0.size(),
                      num_rows * num_cols - bs1 * dmap1_cell0.size());


    // Need to apply DOF transformations for parts of the matrix due to cell 0
    // and cell 1. For example, if the space has 3 DOFs, then Ae will be 6 by 6
    // (3 rows/columns for each cell). Subspans are used to offset to the right
    // blocks of the matrix

    dof_transform(_Ae, cell_info, cell[0], num_cols);
    dof_transform(sub_Ae0, cell_info, cell[1], num_cols);
    dof_transform_to_transpose(_Ae, cell_info, cell[0], num_rows);
    dof_transform_to_transpose(sub_Ae1, cell_info, cell[1], num_rows);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (std::size_t i = 0; i < dmapjoint0.size(); ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dmapjoint0[i] + k])
          {
            // Zero row bs0 * i + k
            std::fill_n(std::next(Ae.begin(), num_cols * (bs0 * i + k)),
                        num_cols, 0.0);
          }
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::size_t j = 0; j < dmapjoint1.size(); ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dmapjoint1[j] + k])
          {
            // Zero column bs1 * j + k
            for (int m = 0; m < num_rows; ++m)
              Ae[m * num_cols + bs1 * j + k] = 0.0;
          }
        }
      }
    }

    const std::array<int, 2> bs = {bs0, bs1};
    const std::array<xtl::span<const int32_t>, 2> dofs = {dmap0_cell0, dmap1_cell0};
    const std::array<int, 2> num_dofs = {num_dofs0, num_dofs1};
    std::array<xtl::span<const int32_t>, 2> slave_indices;
    std::array<std::vector<bool>, 2> is_slave
        = {std::vector<bool>(ndim0, false), std::vector<bool>(ndim1, false)};
    std::array<std::vector<std::int32_t>, 2> local_indices;

    xt::xtensor<T, 2> sub_Ae0_tensor()

    if (is_slave_entity0[0][l] or is_slave_entity0[1][l])
    {
      for (int axis = 0; axis < 2; ++axis)
      {
        slave_indices[axis]
            = cell_to_slaves[axis]->links(slave_cell_index0[axis][l]);
        // Find local position of every slave
        local_indices[axis]
            = std::vector<std::int32_t>(slave_indices[axis].size());

        for (std::int32_t i = 0; i < slave_indices[axis].size(); ++i)
        {
          bool found = false;
          for (std::int32_t j = 0; j < dofs[axis].size(); ++j)
          {
            for (std::int32_t k = 0; k < bs[axis]; ++k)
            {
              if (bs[axis] * dofs[axis][j] + k
                  == slaves[axis][slave_indices[axis][i]])
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
      modify_mpc_cell_rect<T>(mat_add_values, num_dofs, sub_Ae0, dofs, bs,
                              slave_indices, masters, coefficients,
                              local_indices, is_slave);
    }
    mat_add_block_values(dmapjoint0.size(), dmapjoint0.data(), 
      dmapjoint1.size(), dmapjoint1.data(), Ae.data());
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

  const int num_dofs0 = dofs0.links(0).size();
  const int num_dofs1 = dofs1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;

  // Prepare constants
  const std::vector<T> constants = pack_constants(a);

  // Prepare coefficients
  const auto coeffs = dolfinx::fem::pack_coefficients(a);

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

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Prepare tranformation data
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

  if (a.num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    const auto fetch_cells = [&](const std::int32_t& entity){
      return entity; };

    // Prepare assembly lambda
    for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);

      // Assembly lambda employing the form's kernel
      const auto assemble_local_cell_matrix = [&](
        xt::xtensor<T, 2>& Ae, const std::int32_t cell)
        {
          const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
          const xt::xarray<double> coordinate_dofs(
            xt::view(x_g, xt::keep(x_dofs), xt::all()));

          // Tabulate tensor
          std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
          fn(Ae.data(), coeffs.first.data() + cell*coeffs.second, constants.data(),
                coordinate_dofs.data(), nullptr, nullptr);

          // Apply any required transformations
          const xtl::span<T> _Ae(Ae);
          apply_dof_transformation(_Ae, cell_info, cell, ndim1);
          apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);
        };

      const std::vector<std::int32_t>& active_cells = a.cell_domains(i);
      assemble_entity_impl<T, std::int32_t>(
          mat_add_block_values, mat_add_values, *mesh, active_cells,
          dofs0, bs0, dofs1, bs1, bc0, bc1,
          coeffs.first, coeffs.second, constants, cell_info, mpc0, mpc1,
          fetch_cells, assemble_local_cell_matrix);
    }
  }

  if (a.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0)
  {
    const int num_cell_facets
        = dolfinx::mesh::cell_num_entities(
            mesh->topology().cell_type(), tdim - 1);
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable().create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology().get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    // Prepare assembly lambdas
    const auto fetch_cells = [](const std::pair<std::int32_t, int>& entity){
      return entity.first; };

    // Assemble all the exterior facet subdomains
    for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);

      // Assembly lambda employing the form's kernel
      const auto assemble_local_facet_matrix = [&](
        xt::xtensor<T, 2>& Ae, const std::pair<std::int32_t, int>& entity)
        {
          const std::int32_t cell = fetch_cells(entity);
          const int local_facet = entity.second;
          const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
          const xt::xarray<double> coordinate_dofs(
            xt::view(x_g, xt::keep(x_dofs), xt::all()));

          // Tabulate tensor
          std::uint8_t perm = get_perm(cell * num_cell_facets + local_facet);
          std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
          fn(Ae.data(), coeffs.first.data() + cell*coeffs.second, constants.data(),
                coordinate_dofs.data(), &local_facet, &perm);

          // Apply any required transformations
          const xtl::span<T> _Ae(Ae);
          apply_dof_transformation(_Ae, cell_info, cell, ndim1);
          apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);
        };

      const std::vector<std::pair<std::int32_t, int>>& active_facets
          = a.exterior_facet_domains(i);
      assemble_entity_impl<T, std::pair<std::int32_t, int>>(
          mat_add_block_values, mat_add_values, *mesh, active_facets,
          dofs0, bs0, dofs1, bs1, bc0, bc1,
          coeffs.first, coeffs.second, constants, cell_info, mpc0, mpc1,
          fetch_cells, assemble_local_facet_matrix);
    }
  }

  if (a.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    const std::vector<int> c_offsets = a.coefficient_offsets();

    // Perm data
    const int num_cell_facets
        = dolfinx::mesh::cell_num_entities(
            mesh->topology().cell_type(), mesh->topology().dim() - 1);
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable().create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology().get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    for (int i : a.integral_ids(dolfinx::fem::IntegralType::interior_facet))
    {
      const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>& active_facets
          = a.interior_facet_domains(i);
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::interior_facet, i);
      assemble_interior_facets_impl<T>(
            mat_add_block_values, mat_add_values, *mesh, active_facets,
            dofs0, bs0, dofs1, bs1, bc0, bc1,
            coeffs.first, coeffs.second, constants, cell_info, mpc0, mpc1,
            fn, *dofmap0, *dofmap1, c_offsets, get_perm,
            apply_dof_transformation, apply_dof_transformation_to_transpose);
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
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc1,
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

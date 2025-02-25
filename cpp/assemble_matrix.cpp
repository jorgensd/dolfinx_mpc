// Copyright (C) 2020-2022 Jorgen S. Dokken, Nathan Sime, and Connor D. Pierce
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_matrix.h"
#include <assemble_utils.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>

using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

namespace
{

/// Given an assembled element matrix Ae, remove all entries (i,j) where both i
/// and j corresponds to a slave degree of freedom
/// @param[in,out] Ae_stripped The matrix Ae stripped of all other entries
/// @param[in] Ae The element matrix
/// @param[in] num_dofs The number of degrees of freedom in each row and column
/// (blocked)
/// @param[in] bs The block size for the rows and columns
/// @param[in] is_slave Marker indicating if a dof (local to process) is a slave
/// degree of freedom
/// @param[in] dofs Map from index local to cell to index local to process for
/// rows rows and columns
/// @returns The matrix stripped of slave contributions
template <typename T>
void fill_stripped_matrix(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        Ae_stripped,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        Ae,
    const std::array<const std::uint32_t, 2>& num_dofs,
    const std::array<const int, 2>& bs,
    const std::array<std::span<const std::int8_t>, 2>& is_slave,
    const std::array<std::span<const std::int32_t>, 2>& dofs)
{
  const auto& [row_bs, col_bs] = bs;
  const auto& [num_row_dofs, num_col_dofs] = num_dofs;
  const auto& [row_dofs, col_dofs] = dofs;
  const auto& [slave_rows, slave_cols] = is_slave;

  assert(Ae_stripped.extent(0) == Ae.extent(0));
  assert(Ae_stripped.extent(1) == Ae.extent(1));

  // Strip Ae of all entries where both i and j are slaves
  bool slave_row;
  bool slave_col;
  for (std::uint32_t i = 0; i < num_row_dofs; i++)
  {
    const int row_block = row_dofs[i] * row_bs;
    for (int row = 0; row < row_bs; row++)
    {
      slave_row = slave_rows[row_block + row];
      const int l_row = i * row_bs + row;
      for (std::uint32_t j = 0; j < num_col_dofs; j++)
      {
        const int col_block = col_dofs[j] * col_bs;
        for (int col = 0; col < col_bs; col++)
        {
          slave_col = slave_cols[col_block + col];
          const int l_col = j * col_bs + col;
          Ae_stripped(l_row, l_col)
              = (slave_row && slave_col) ? T(0.0) : Ae(l_row, l_col);
        }
      }
    }
  }
};

/// Modify local element matrix Ae with MPC contributions, and insert non-local
/// contributions in the correct places
///
/// @param[in] mat_set Function that sets a local matrix into specified
/// positions of the global matrix A
/// @param[in] num_dofs The number of degrees of freedom in each row and column
/// (blocked)
/// @param[in, out] Ae The local element matrix
/// @param[in] dofs The local indices of the row and column dofs (blocked)
/// @param[in] bs The row and column block size
/// @param[in] slaves The row and column slave indices (local to process)
/// @param[in] masters Row and column map from the slave indices (local to
/// process) to the master dofs (local to process)
/// @param[in] coefficients row and column map from the slave indices (local to
/// process) to the corresponding coefficients
/// @param[in] is_slave Marker indicating if a dof (local to process) is a slave
/// dof
/// @param[in] scratch_memory Memory used in computations of additional element
/// matrices and rows. Should be at least 2 * num_rows(Ae) * num_cols(Ae) +
/// num_cols(Ae) + num_rows(Ae)
template <typename T>
void modify_mpc_cell(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>, std::span<const T>)>&
        mat_set,
    const std::array<const std::uint32_t, 2>& num_dofs,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        Ae,
    const std::array<std::span<const std::int32_t>, 2>& dofs,
    const std::array<const int, 2>& bs,
    const std::array<std::span<const std::int32_t>, 2>& slaves,
    const std::array<
        std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>&
        masters,
    const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>,
                     2>& coeffs,
    const std::array<std::span<const std::int8_t>, 2>& is_slave,
    std::span<T> scratch_memory)
{
  std::array<std::size_t, 2> num_flattened_masters = {0, 0};
  std::array<std::vector<std::int32_t>, 2> local_index;
  for (int axis = 0; axis < 2; ++axis)
  {
    // NOTE: Should this be moved into the MPC constructor?
    // Locate which local dofs are slave dofs and compute the local index of the
    // slave
    local_index[axis] = dolfinx_mpc::compute_local_slave_index(
        slaves[axis], num_dofs[axis], bs[axis], dofs[axis], is_slave[axis]);

    // Count number of masters in flattened structure for the rows and columns
    for (std::uint32_t i = 0; i < num_dofs[axis]; i++)
    {
      for (int j = 0; j < bs[axis]; j++)
      {
        const std::int32_t dof = dofs[axis][i] * bs[axis] + j;
        if (is_slave[axis][dof])
          num_flattened_masters[axis] += masters[axis]->links(dof).size();
      }
    }
  }

  const int ndim0 = bs[0] * num_dofs[0];
  const int ndim1 = bs[1] * num_dofs[1];
  assert(scratch_memory.size()
         >= std::size_t(2 * ndim0 * ndim1 + ndim0 + ndim1));
  std::ranges::fill(scratch_memory, T(0));

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Ae_original(scratch_memory.data(), ndim0, ndim1);

  // Copy Ae into new matrix for distirbution of master dofs
  std::ranges::copy_n(Ae.data_handle(), ndim0 * ndim1,
                      Ae_original.data_handle());

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Ae_stripped(std::next(scratch_memory.data(), ndim0 * ndim1), ndim0,
                  ndim1);
  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  fill_stripped_matrix(Ae_stripped, Ae, num_dofs, bs, is_slave, dofs);

  // Zero out slave entries in element matrix
  // Zero slave row
  std::ranges::for_each(local_index[0],
                        [&Ae, ndim1](const auto dof)
                        {
                          std::ranges::fill_n(
                              std::next(Ae.data_handle(), ndim1 * dof), ndim1,
                              0.0);
                        });
  // Zero slave column
  std::ranges::for_each(local_index[1],
                        [&Ae, ndim0](const auto dof)
                        {
                          for (int row = 0; row < ndim0; ++row)
                            Ae(row, dof) = 0.0;
                        });

  // Flatten slaves, masters and coeffs for efficient
  // modification of the matrices
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
  auto Arow = scratch_memory.subspan(2 * ndim0 * ndim1, ndim0);
  auto Acol = scratch_memory.subspan(2 * ndim0 * ndim1 + ndim0, ndim1);
  // Loop over all masters for the MPC applied to rows.
  // Insert contributions in columns
  std::vector<std::int32_t> unrolled_dofs(ndim1);
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {
    // Use the standard transpose for type double, Hermitian transpose
    // for type std::complex<double>. Do this outside the j-loop so
    // std::conj() is only computed once per entry in flattened_masters.
    T coeff_i;
    if constexpr (std::is_scalar_v<T>)
      coeff_i = flattened_coeffs[0][i];
    else
      coeff_i = std::conj(flattened_coeffs[0][i]);

    // Unroll dof blocks and add column contribution
    for (std::uint32_t j = 0; j < num_dofs[1]; ++j)
      for (int k = 0; k < bs[1]; ++k)
      {
        Acol[j * bs[1] + k]
            = coeff_i * Ae_stripped(flattened_slaves[0][i], j * bs[1] + k);
        unrolled_dofs[j * bs[1] + k] = dofs[1][j] * bs[1] + k;
      }

    // Insert modified entries
    row[0] = flattened_masters[0][i];
    mat_set(row, unrolled_dofs, Acol);

    // Loop through other masters on the same cell and add in contribution
    for (std::size_t j = 0; j < num_flattened_masters[1]; ++j)
    {
      col[0] = flattened_masters[1][j];
      A0[0] = coeff_i * flattened_coeffs[1][j]
              * Ae_original(flattened_slaves[0][i], flattened_slaves[1][j]);
      mat_set(row, col, A0);
    }
  }

  // Loop over all masters for the MPC applied to columns.
  // Insert contributions in rows
  unrolled_dofs.resize(ndim0);
  for (std::size_t i = 0; i < num_flattened_masters[1]; ++i)
  {

    // Unroll dof blocks and compute row contribution
    for (std::uint32_t j = 0; j < num_dofs[0]; ++j)
      for (int k = 0; k < bs[0]; ++k)
      {
        Arow[j * bs[0] + k]
            = flattened_coeffs[1][i]
              * Ae_stripped(j * bs[0] + k, flattened_slaves[1][i]);
        unrolled_dofs[j * bs[0] + k] = dofs[0][j] * bs[0] + k;
      }

    // Insert modified entries
    col[0] = flattened_masters[1][i];
    mat_set(unrolled_dofs, col, Arow);
  }
} // namespace

//-----------------------------------------------------------------------------
template <typename T, std::floating_point U>
void assemble_exterior_facets(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_values,
    const dolfinx::mesh::Mesh<U>& mesh, std::span<const std::int32_t> facets,
    std::span<const std::int32_t> facets0,
    std::span<const std::int32_t> facets1,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& apply_dof_transformation,
    const dolfinx::fem::DofMap& dofmap0,
    const std::function<
        void(const std::span<T>&, const std::span<const std::uint32_t>&,
             std::int32_t, int)>& apply_dof_transformation_to_transpose,
    const dolfinx::fem::DofMap& dofmap1, const std::vector<std::int8_t>& bc0,
    const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*)>& kernel,
    const std::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info0,
    const std::span<const std::uint32_t>& cell_info1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1)
{
  // Get MPC data
  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters(), mpc1->masters()};
  const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coefficients(), mpc1->coefficients()};
  const std::array<std::span<const std::int8_t>, 2> is_slave
      = {mpc0->is_slave(), mpc1->is_slave()};

  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Get mesh data
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh.geometry().dofmap();

  const int num_dofs_g = x_dofmap.extent(1);
  std::span<const U> x_g = mesh.geometry().x();

  // Iterate over all facets
  std::vector<U> coordinate_dofs(3 * num_dofs_g);
  const auto num_dofs0 = (std::uint32_t)dofmap0.map().extent(1);
  const auto num_dofs1 = (std::uint32_t)dofmap1.map().extent(1);
  int bs0 = dofmap0.bs();
  int bs1 = dofmap1.bs();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  const std::array<const std::uint32_t, 2> num_dofs = {num_dofs0, num_dofs1};
  const std::array<const int, 2> bs = {bs0, bs1};

  std::vector<T> Aeb(ndim0 * ndim1);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Ae(Aeb.data(), ndim0, ndim1);
  const std::span<T> _Ae(Aeb);

  std::vector<T> scratch_memory(2 * ndim0 * ndim1 + ndim0 + ndim1);
  for (std::size_t l = 0; l < facets.size(); l += 2)
  {
    const std::int32_t cell0 = facets0[l];
    const std::int32_t cell1 = facets1[l];
    const std::int32_t cell = facets[l];
    const int local_facet = facets[l + 1];

    // Get cell vertex coordinates

    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                          std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::ranges::fill(Aeb, 0);
    kernel(Aeb.data(), coeffs.data() + l / 2 * cstride, constants.data(),
           coordinate_dofs.data(), &local_facet, nullptr);
    apply_dof_transformation(_Ae, cell_info0, cell0, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info1, cell1, ndim0);

    // Zero rows/columns for essential bcs
    auto dmap0 = dofmap0.cell_dofs(cell0);
    auto dmap1 = dofmap1.cell_dofs(cell1);
    if (!bc0.empty())
    {
      for (std::uint32_t i = 0; i < num_dofs0; ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dmap0[i] + k])
          {
            // Zero row bs0 * i + k
            const int row = bs0 * i + k;
            std::ranges::fill_n(std::next(Ae.data_handle(), ndim1 * row), ndim1,
                                0.0);
          }
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::size_t j = 0; j < num_dofs1; ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dmap1[j] + k])
          {
            // Zero column bs1 * j + k
            const int col = bs1 * j + k;
            for (std::uint32_t row = 0; row < ndim0; ++row)
              Aeb[row * ndim1 + col] = 0.0;
          }
        }
      }
    }

    // Modify local element matrix Ae and insert contributions into master
    // locations
    if ((cell_to_slaves[0]->num_links(cell0) > 0)
        || (cell_to_slaves[1]->num_links(cell1) > 0))
    {
      const std::array<std::span<const std::int32_t>, 2> slaves
          = {cell_to_slaves[0]->links(cell0), cell_to_slaves[1]->links(cell1)};
      const std::array<std::span<const std::int32_t>, 2> dofs = {dmap0, dmap1};
      modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, dofs, bs, slaves,
                         masters, coefficients, is_slave, scratch_memory);
    }
    mat_add_block_values(dmap0, dmap1, Aeb);
  }
} // namespace
//-----------------------------------------------------------------------------
template <typename T, std::floating_point U>
void assemble_cells_impl(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_values,
    const dolfinx::mesh::Geometry<U>& geometry,
    std::span<const std::int32_t> active_cells,
    std::span<const std::int32_t> active_cells0,
    std::span<const std::int32_t> active_cells1,
    std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation,
    const dolfinx::fem::DofMap& dofmap0,
    std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation_to_transpose,
    const dolfinx::fem::DofMap& dofmap1, const std::vector<std::int8_t>& bc0,
    const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*)>& kernel,
    const std::span<const T>& coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info0,
    const std::span<const std::uint32_t>& cell_info1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1)
{
  // Get MPC data
  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters(), mpc1->masters()};
  const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coefficients(), mpc1->coefficients()};
  const std::array<std::span<const std::int8_t>, 2> is_slave
      = {mpc0->is_slave(), mpc1->is_slave()};

  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>,
      2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Prepare cell geometry
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::span<const U> x_g = geometry.x();

  // Iterate over active cells
  std::vector<U> coordinate_dofs(3 * num_dofs_g);
  const auto num_dofs0 = (std::uint32_t)dofmap0.map().extent(1);
  const auto num_dofs1 = (std::uint32_t)dofmap1.map().extent(1);
  const std::array<const int, 2> bs = {dofmap0.bs(), dofmap1.bs()};
  const std::uint32_t ndim0 = num_dofs0 * bs.front();
  const std::uint32_t ndim1 = num_dofs1 * bs.back();
  const std::array<const std::uint32_t, 2> num_dofs = {num_dofs0, num_dofs1};

  std::vector<T> Aeb(ndim0 * ndim1);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Ae(Aeb.data(), ndim0, ndim1);
  const std::span<T> _Ae(Aeb);
  std::vector<T> scratch_memory(2 * ndim0 * ndim1 + ndim0 + ndim1);

  for (std::size_t index = 0; index < active_cells0.size(); index++)
  {
    const std::int32_t cell = active_cells[index];
    const std::int32_t cell0 = active_cells0[index];
    const std::int32_t cell1 = active_cells1[index];

    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::ranges::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                          std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::ranges::fill(Aeb, 0);
    kernel(Aeb.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    apply_dof_transformation(_Ae, cell_info0, cell0, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info1, cell1, ndim0);

    // Zero rows/columns for essential bcs
    std::span<const std::int32_t> dofs0 = dofmap0.cell_dofs(cell0);
    std::span<const std::int32_t> dofs1 = dofmap1.cell_dofs(cell1);
    if (!bc0.empty())
    {
      for (std::uint32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs.front(); ++k)
        {
          if (bc0[bs.front() * dofs0[i] + k])
            std::ranges::fill_n(
                std::next(Aeb.begin(), ndim1 * (bs.front() * i + k)), ndim1,
                T(0));
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::uint32_t j = 0; j < num_dofs1; ++j)
        for (std::int32_t k = 0; k < bs.back(); ++k)
          if (bc1[bs.back() * dofs1[j] + k])
            for (std::size_t l = 0; l < ndim0; ++l)
              Aeb[l * ndim1 + bs.back() * j + k] = 0;
    }

    // Modify local element matrix Ae and insert contributions into master
    // locations
    if ((cell_to_slaves[0]->num_links(cell0) > 0)
        || (cell_to_slaves[1]->num_links(cell1) > 0))
    {
      const std::array<std::span<const std::int32_t>, 2> slaves
          = {cell_to_slaves[0]->links(cell0), cell_to_slaves[1]->links(cell1)};
      const std::array<std::span<const std::int32_t>, 2> dofs = {dofs0, dofs1};
      modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, dofs, bs, slaves,
                         masters, coefficients, is_slave, scratch_memory);
    }
    mat_add_block_values(dofs0, dofs1, _Ae);
  }
}
//-----------------------------------------------------------------------------
template <typename T, std::floating_point U>
void assemble_matrix_impl(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>)>& mat_add_values,
    const dolfinx::fem::Form<T>& a, const std::vector<std::int8_t>& bc0,
    const std::vector<std::int8_t>& bc1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1)
{
  // Integration domain mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);

  // Test function mesh
  auto mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);

  // Trial function mesh
  auto mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  // Prepare constants
  const std::vector<T> constants = pack_constants(a);

  // Prepare coefficients
  auto coeff_vec = dolfinx::fem::allocate_coefficient_storage(a);
  dolfinx::fem::pack_coefficients(a, coeff_vec);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);

  auto element0 = a.function_spaces().at(0)->element();
  auto element1 = a.function_spaces().at(1)->element();
  std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation = element0->template dof_transformation_fn<T>(
          dolfinx::fem::doftransform::standard);
  std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation_to_transpose
      = element1->template dof_transformation_right_fn<T>(
          dolfinx::fem::doftransform::transpose);

  const int num_cell_types = mesh->topology()->cell_types().size();
  if (num_cell_types > 1)
    throw std::runtime_error("Not implemented for mixed cell types");

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  if (needs_transformation_data)
  {
    mesh0->topology_mutable()->create_entity_permutations();
    mesh1->topology_mutable()->create_entity_permutations();
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }
  for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i, 0);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::cell, i});
    std::span<const std::int32_t> active_cells
        = a.domain(dolfinx::fem::IntegralType::cell, i, 0);
    std::span<const std::int32_t> active_cells0
        = a.domain_arg(dolfinx::fem::IntegralType::cell, 0, i, 0);
    std::span<const std::int32_t> active_cells1
        = a.domain_arg(dolfinx::fem::IntegralType::cell, 1, i, 0);
    assemble_cells_impl<T>(
        mat_add_block_values, mat_add_values, mesh->geometry(), active_cells,
        active_cells0, active_cells1, apply_dof_transformation, *dofmap0,
        apply_dof_transformation_to_transpose, *dofmap1, bc0, bc1, fn, coeffs,
        cstride, constants, cell_info0, cell_info1, mpc0, mpc1);
  }

  for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i, 0);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
    std::span<const std::int32_t> facets
        = a.domain(dolfinx::fem::IntegralType::exterior_facet, i, 0);
    std::span<const std::int32_t> active_facets0
        = a.domain_arg(dolfinx::fem::IntegralType::exterior_facet, 0, i, 0);
    std::span<const std::int32_t> active_facets1
        = a.domain_arg(dolfinx::fem::IntegralType::exterior_facet, 1, i, 0);
    assemble_exterior_facets<T>(
        mat_add_block_values, mat_add_values, *mesh, facets, active_facets0,
        active_facets1, apply_dof_transformation, *dofmap0,
        apply_dof_transformation_to_transpose, *dofmap1, bc0, bc1, fn, coeffs,
        cstride, constants, cell_info0, cell_info1, mpc0, mpc1);
  }

  if (a.integral_ids(dolfinx::fem::IntegralType::interior_facet).size() > 0)
    throw std::runtime_error("Not implemented yet");
}
//-----------------------------------------------------------------------------
template <typename T, std::floating_point U>
void _assemble_matrix(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>&)>& mat_add_block,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>&)>& mat_add,
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1,
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
      mat_add(diag_dof, diag_dof, diag_value);
    }
  }
  timer.stop();
}
} // namespace
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const double>&)>& mat_add_block,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const double>&)>& mat_add,
    const dolfinx::fem::Form<double>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<double, double>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    const double diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<double>>&)>& mat_add_block,
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<double>>&)>& mat_add,
    const dolfinx::fem::Form<std::complex<double>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>, double>>&
        mpc1,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    const std::complex<double> diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const float>&)>& mat_add_block,
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const float>&)>& mat_add,
    const dolfinx::fem::Form<float>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<float, float>>& mpc1,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<float>>>&
        bcs,
    const float diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<float>>&)>& mat_add_block,
    const std::function<
        int(std::span<const std::int32_t>, std::span<const std::int32_t>,
            const std::span<const std::complex<float>>&)>& mat_add,
    const dolfinx::fem::Form<std::complex<float>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc0,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<float>, float>>&
        mpc1,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<float>>>>&
        bcs,
    const std::complex<float> diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, bcs, diagval);
}

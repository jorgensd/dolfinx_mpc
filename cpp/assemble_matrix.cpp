// Copyright (C) 2020-2022 Jorgen S. Dokken and Nathan Sime
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
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>
        Ae_stripped,
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>
        Ae,
    const std::array<const std::uint32_t, 2>& num_dofs,
    const std::array<const int, 2>& bs,
    const std::array<const std::vector<std::int8_t>, 2>& is_slave,
    const std::array<const std::span<const int32_t>, 2>& dofs)
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

/// Loop over all masters for the MPC applied to rows. Insert contributions in columns
void modify_mpc_cell_rows(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const double>)>& mat_set,
    const int ndim0, const int ndim1,
    const std::array<const std::uint32_t, 2>& num_dofs,
    const std::array<const std::span<const int32_t>, 2>& dofs,
    std::array<std::size_t, 2> num_flattened_masters,
    const std::array<const int, 2>& bs,
    std::span<double> scratch_memory,
    std::array<std::vector<std::int32_t>, 2> flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_slaves,
    std::array<std::vector<double>, 2> flattened_coeffs,
    std::vector<std::int32_t> unrolled_dofs,
    std::experimental::mdspan<double, std::experimental::dextents<std::size_t, 2>> Ae_stripped)
{
  std::array<std::int32_t, 1> row;
  auto Acol = scratch_memory.subspan(2 * ndim0 * ndim1 + ndim0, ndim1);
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {

    // Unroll dof blocks and add column contribution
    for (std::uint32_t j = 0; j < num_dofs[1]; ++j)
      for (int k = 0; k < bs[1]; ++k)
      {
        Acol[j * bs[1] + k]
            = flattened_coeffs[0][i]
              * Ae_stripped(flattened_slaves[0][i], j * bs[1] + k);
        unrolled_dofs[j * bs[1] + k] = dofs[1][j] * bs[1] + k;
      }

    // Insert modified entries
    row[0] = flattened_masters[0][i];
    mat_set(row, unrolled_dofs, Acol);
  }
}

/// Loop over all masters for the MPC applied to rows. Insert contributions in columns
void modify_mpc_cell_rows(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const std::complex<double>>)>& mat_set,
    const int ndim0, const int ndim1,
    const std::array<const std::uint32_t, 2>& num_dofs,
    const std::array<const std::span<const int32_t>, 2>& dofs,
    std::array<std::size_t, 2> num_flattened_masters,
    const std::array<const int, 2>& bs,
    std::span<std::complex<double>> scratch_memory,
    std::array<std::vector<std::int32_t>, 2> flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_slaves,
    std::array<std::vector<std::complex<double>>, 2> flattened_coeffs,
    std::vector<std::int32_t> unrolled_dofs,
    std::experimental::mdspan<std::complex<double>, 
                              std::experimental::dextents<std::size_t, 2>> Ae_stripped)
{
  std::array<std::int32_t, 1> row;
  auto Acol = scratch_memory.subspan(2 * ndim0 * ndim1 + ndim0, ndim1);
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {

    // Unroll dof blocks and add column contribution
    for (std::uint32_t j = 0; j < num_dofs[1]; ++j)
      for (int k = 0; k < bs[1]; ++k)
      {
        Acol[j * bs[1] + k]
            = std::conj(flattened_coeffs[0][i])
              * Ae_stripped(flattened_slaves[0][i], j * bs[1] + k);
        unrolled_dofs[j * bs[1] + k] = dofs[1][j] * bs[1] + k;
      }

    // Insert modified entries
    row[0] = flattened_masters[0][i];
    mat_set(row, unrolled_dofs, Acol);
  }
}

// Loop through masters on a cell and add the quadratic contribution of the
// multi-point constraint (real version)
void modify_mpc_cell_masters(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const double>)>& mat_set,
    const std::array<const std::span<const int32_t>, 2>& dofs,
    std::array<std::size_t, 2> num_flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_slaves,
    std::array<std::vector<double>, 2> flattened_coeffs,
    std::experimental::mdspan<double, std::experimental::dextents<std::size_t, 2>> Ae_original)
{
  std::array<std::int32_t, 1> row;
  std::array<std::int32_t, 1> col;
  std::array<double, 1> A0;
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {
    // Loop through other masters on the same cell and add in contribution
    for (std::size_t j = 0; j < num_flattened_masters[1]; ++j)
    {

      row[0] = flattened_masters[0][i];
      col[0] = flattened_masters[1][j];
      A0[0] = flattened_coeffs[0][i] * flattened_coeffs[1][j]
              * Ae_original(flattened_slaves[0][i], flattened_slaves[1][j]);
      mat_set(row, col, A0);
    }
  }
}

// Loop through masters on a cell and add the quadratic contribution of the
// multi-point constraint (complex version)
void modify_mpc_cell_masters(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const std::complex<double>>)>& mat_set,
    const std::array<const std::span<const int32_t>, 2>& dofs,
    std::array<std::size_t, 2> num_flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_masters,
    std::array<std::vector<std::int32_t>, 2> flattened_slaves,
    std::array<std::vector<std::complex<double>>, 2> flattened_coeffs,
    std::experimental::mdspan<std::complex<double>, 
                              std::experimental::dextents<std::size_t, 2>> Ae_original)
{
  std::array<std::int32_t, 1> row;
  std::array<std::int32_t, 1> col;
  std::array<std::complex<double>, 1> A0;
  for (std::size_t i = 0; i < num_flattened_masters[0]; ++i)
  {
    // Loop through other masters on the same cell and add in contribution
    for (std::size_t j = 0; j < num_flattened_masters[1]; ++j)
    {

      row[0] = flattened_masters[0][i];
      col[0] = flattened_masters[1][j];
      A0[0] = std::conj(flattened_coeffs[0][i]) * flattened_coeffs[1][j]
              * Ae_original(flattened_slaves[0][i], flattened_slaves[1][j]);
      mat_set(row, col, A0);
    }
  }
}

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
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_set,
    const std::array<const std::uint32_t, 2>& num_dofs,
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>
        Ae,
    const std::array<const std::span<const int32_t>, 2>& dofs,
    const std::array<const int, 2>& bs,
    const std::array<const std::span<const int32_t>, 2>& slaves,
    const std::array<
        std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>&
        masters,
    const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>,
                     2>& coeffs,
    const std::array<const std::vector<std::int8_t>, 2>& is_slave,
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
  std::fill(scratch_memory.begin(), scratch_memory.end(), T(0));

  std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>
      Ae_original(scratch_memory.data(), ndim0, ndim1);

  // Copy Ae into new matrix for distirbution of master dofs
  std::copy_n(Ae.data_handle(), ndim0 * ndim1, Ae_original.data_handle());

  std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>
      Ae_stripped(std::next(scratch_memory.data(), ndim0 * ndim1), ndim0,
                  ndim1);
  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  fill_stripped_matrix(Ae_stripped, Ae, num_dofs, bs, is_slave, dofs);

  // Zero out slave entries in element matrix
  // Zero slave row
  std::for_each(
      local_index[0].cbegin(), local_index[0].cend(),
      [&Ae, ndim1](const auto dof)
      { std::fill_n(std::next(Ae.data_handle(), ndim1 * dof), ndim1, 0.0); });
  // Zero slave column
  std::for_each(local_index[1].cbegin(), local_index[1].cend(),
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
  std::vector<std::int32_t> unrolled_dofs(ndim0);
  auto Arow = scratch_memory.subspan(2 * ndim0 * ndim1, ndim0);
  std::array<std::int32_t, 1> col;

  // Loop over all masters for the MPC applied to columns.
  // Insert contributions in rows
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

  // Loop over all masters for the MPC applied to rows.
  // Insert contributions in columns
  unrolled_dofs.resize(ndim1);
  modify_mpc_cell_rows(mat_set, ndim0, ndim1, num_dofs, dofs, num_flattened_masters, bs,
                       scratch_memory, flattened_masters, flattened_slaves, flattened_coeffs,
                       unrolled_dofs, Ae_stripped);

  // Loop through other masters on the same cell and add in contribution
  modify_mpc_cell_masters(mat_set, dofs, num_flattened_masters, flattened_masters,
                          flattened_slaves, flattened_coeffs, Ae_original);
} // namespace

//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_values,
    const dolfinx::mesh::Mesh& mesh,
    const std::span<const std::int32_t>& facets,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<
        void(const std::span<T>&, const std::span<const std::uint32_t>&,
             std::int32_t, int)>& apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<std::int8_t>& bc0, const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const std::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  // Get MPC data
  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters(), mpc1->masters()};
  const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coefficients(), mpc1->coefficients()};
  const std::array<const std::vector<std::int8_t>, 2> is_slave
      = {mpc0->is_slave(), mpc1->is_slave()};

  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Get mesh data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  std::span<const double> x_g = mesh.geometry().x();

  // Iterate over all facets
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const auto num_dofs0 = (std::uint32_t)dofmap0.links(0).size();
  const auto num_dofs1 = (std::uint32_t)dofmap1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  const std::array<const std::uint32_t, 2> num_dofs = {num_dofs0, num_dofs1};
  const std::array<const int, 2> bs = {bs0, bs1};

  std::vector<T> Aeb(ndim0 * ndim1);
  std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>> Ae(
      Aeb.data(), ndim0, ndim1);
  const std::span<T> _Ae(Aeb);

  std::vector<T> scratch_memory(2 * ndim0 * ndim1 + ndim0 + ndim1);
  for (std::size_t l = 0; l < facets.size(); l += 2)
  {

    const std::int32_t cell = facets[l];
    const int local_facet = facets[l + 1];

    // Get cell vertex coordinates
    std::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::fill(Aeb.begin(), Aeb.end(), 0);
    kernel(Aeb.data(), coeffs.data() + l / 2 * cstride, constants.data(),
           coordinate_dofs.data(), &local_facet, nullptr);
    apply_dof_transformation(_Ae, cell_info, cell, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);

    // Zero rows/columns for essential bcs
    std::span<const std::int32_t> dmap0 = dofmap0.links(cell);
    std::span<const std::int32_t> dmap1 = dofmap1.links(cell);
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
            std::fill_n(std::next(Ae.data_handle(), ndim1 * row), ndim1, 0.0);
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
    if ((cell_to_slaves[0]->num_links(cell) > 0)
        || (cell_to_slaves[1]->num_links(cell) > 0))
    {
      const std::array<const std::span<const int32_t>, 2> slaves
          = {cell_to_slaves[0]->links(cell), cell_to_slaves[1]->links(cell)};
      const std::array<const std::span<const int32_t>, 2> dofs = {dmap0, dmap1};
      modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, dofs, bs, slaves,
                         masters, coefficients, is_slave, scratch_memory);
    }
    mat_add_block_values(dmap0, dmap1, Aeb);
  }
} // namespace
//-----------------------------------------------------------------------------
template <typename T>
void assemble_cells_impl(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_values,
    const dolfinx::mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<std::int8_t>& bc0, const std::vector<std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const std::span<const T>& coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc1)
{
  // Get MPC data
  const std::array<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      masters = {mpc0->masters(), mpc1->masters()};
  const std::array<std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>, 2>
      coefficients = {mpc0->coefficients(), mpc1->coefficients()};
  const std::array<const std::vector<std::int8_t>, 2> is_slave
      = {mpc0->is_slave(), mpc1->is_slave()};

  const std::array<
      const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>,
      2>
      cell_to_slaves = {mpc0->cell_to_slaves(), mpc1->cell_to_slaves()};

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  std::span<const double> x_g = geometry.x();

  // Iterate over active cells
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const auto num_dofs0 = (std::uint32_t)dofmap0.links(0).size();
  const auto num_dofs1 = (std::uint32_t)dofmap1.links(0).size();
  const std::uint32_t ndim0 = num_dofs0 * bs0;
  const std::uint32_t ndim1 = num_dofs1 * bs1;
  const std::array<const std::uint32_t, 2> num_dofs = {num_dofs0, num_dofs1};
  const std::array<const int, 2> bs = {bs0, bs1};

  std::vector<T> Aeb(ndim0 * ndim1);
  std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>> Ae(
      Aeb.data(), ndim0, ndim1);
  const std::span<T> _Ae(Aeb);
  std::vector<T> scratch_memory(2 * ndim0 * ndim1 + ndim0 + ndim1);

  for (std::size_t c = 0; c < active_cells.size(); c++)
  {
    const std::int32_t cell = active_cells[c];
    // Get cell coordinates/geometry
    std::span<const int32_t> x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::fill(Aeb.begin(), Aeb.end(), 0);
    kernel(Aeb.data(), coeffs.data() + c * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    apply_dof_transformation(_Ae, cell_info, cell, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);

    // Zero rows/columns for essential bcs
    std::span<const int32_t> dofs0 = dofmap0.links(cell);
    std::span<const int32_t> dofs1 = dofmap1.links(cell);
    if (!bc0.empty())
    {
      for (std::uint32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
            std::fill_n(std::next(Aeb.begin(), ndim1 * (bs0 * i + k)), ndim1,
                        T(0));
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::uint32_t j = 0; j < num_dofs1; ++j)
        for (std::int32_t k = 0; k < bs1; ++k)
          if (bc1[bs1 * dofs1[j] + k])
            for (std::size_t l = 0; l < ndim0; ++l)
              Aeb[l * ndim1 + bs1 * j + k] = 0;
    }

    // Modify local element matrix Ae and insert contributions into master
    // locations
    if ((cell_to_slaves[0]->num_links(cell) > 0)
        || (cell_to_slaves[1]->num_links(cell) > 0))
    {
      const std::array<const std::span<const int32_t>, 2> slaves
          = {cell_to_slaves[0]->links(cell), cell_to_slaves[1]->links(cell)};
      const std::array<const std::span<const int32_t>, 2> dofs = {dofs0, dofs1};

      modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, dofs, bs, slaves,
                         masters, coefficients, is_slave, scratch_memory);
    }
    mat_add_block_values(dofs0, dofs1, _Ae);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_matrix_impl(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_block_values,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>)>& mat_add_values,
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
  auto coeff_vec = dolfinx::fem::allocate_coefficient_storage(a);
  dolfinx::fem::pack_coefficients(a, coeff_vec);
  auto coefficients = dolfinx::fem::make_coefficients_span(coeff_vec);

  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a.function_spaces().at(0)->element();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1
      = a.function_spaces().at(1)->element();
  std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation = element0->get_dof_transformation_function<T>();
  std::function<void(std::span<T>, const std::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation_to_transpose
      = element1->get_dof_transformation_to_transpose_function<T>();

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
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

  for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::exterior_facet, i});
    const std::vector<std::int32_t>& facets = a.exterior_facet_domains(i);
    assemble_exterior_facets<T>(mat_add_block_values, mat_add_values, *mesh,
                                facets, apply_dof_transformation, dofs0, bs0,
                                apply_dof_transformation_to_transpose, dofs1,
                                bs1, bc0, bc1, fn, coeffs, cstride, constants,
                                cell_info, mpc0, mpc1);
  }

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
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>&)>& mat_add_block,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const T>&)>& mat_add,
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
      mat_add(diag_dof, diag_dof, diag_value);
    }
  }
  timer.stop();
}
} // namespace
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const double>&)>& mat_add_block,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const double>&)>& mat_add,
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
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const std::complex<double>>&)>&
        mat_add_block,
    const std::function<int(const std::span<const std::int32_t>&,
                            const std::span<const std::int32_t>&,
                            const std::span<const std::complex<double>>&)>&
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

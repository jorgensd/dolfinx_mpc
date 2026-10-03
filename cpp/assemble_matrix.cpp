// Copyright (C) 2020-2026 Jorgen S. Dokken, Nathan Sime, and Connor D. Pierce
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "assemble_matrix.h"
#include <algorithm>
#include <array>
#include <assemble_utils.h>
#include <cstdint>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <span>
#include <vector>

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

/// Zero the rows and columns of the row-major element matrix `Ae` whose dofs
/// carry a Dirichlet condition. Empty markers are skipped.
template <typename T>
void zero_dirichlet(std::span<T> Ae, std::span<const std::int32_t> dofs0,
                    int bs0, std::span<const std::int8_t> bc0,
                    std::span<const std::int32_t> dofs1, int bs1,
                    std::span<const std::int8_t> bc1)
{
  const std::size_t num_rows = bs0 * dofs0.size();
  const std::size_t num_cols = bs1 * dofs1.size();
  if (!bc0.empty())
  {
    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        if (bc0[bs0 * dofs0[i] + k])
          std::fill_n(std::next(Ae.begin(), num_cols * (bs0 * i + k)), num_cols,
                      T(0));
  }
  if (!bc1.empty())
  {
    for (std::size_t j = 0; j < dofs1.size(); ++j)
      for (int k = 0; k < bs1; ++k)
        if (bc1[bs1 * dofs1[j] + k])
          for (std::size_t row = 0; row < num_rows; ++row)
            Ae[row * num_cols + bs1 * j + k] = T(0);
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
    const dolfinx::fem::DofMap& dofmap1, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*, void*)>& kernel,
    const std::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info0,
    const std::span<const std::uint32_t>& cell_info1,
    std::span<const std::uint8_t> perms, int num_facets_per_cell,
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
  if (mesh.geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh.geometry().dofmaps().front();

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
  const bool is_transform0_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation);
  const bool is_transform1_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation_to_transpose);
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
    // Tabulate tensor.
    const std::uint8_t perm
        = perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];
    std::ranges::fill(Aeb, 0);
    kernel(Aeb.data(), coeffs.data() + l / 2 * cstride, constants.data(),
           coordinate_dofs.data(), &local_facet, &perm, nullptr);
    if (is_transform0_set)
      apply_dof_transformation(_Ae, cell_info0, cell0, ndim1);
    if (is_transform1_set)
      apply_dof_transformation_to_transpose(_Ae, cell_info1, cell1, ndim0);

    // Zero rows/columns for essential bcs
    auto dmap0 = dofmap0.cell_dofs(cell0);
    auto dmap1 = dofmap1.cell_dofs(cell1);
    zero_dirichlet<T>(_Ae, dmap0, bs0, bc0, dmap1, bs1, bc1);

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
/// Assemble interior facet integrals.
///
/// The element tensor of a facet is the 2x2 block matrix
/// [A++, A+-; A-+, A--], where block (s, t) couples the test function on the
/// cell of side s with the trial function on the cell of side t. Each block is
/// an ordinary cell-cell element matrix, so the constraint is applied to each
/// block on its own with `modify_mpc_cell`. Assembly is linear in the element
/// tensor, so this equals constraining the whole tensor, and it avoids the
/// joint dof list, in which a dof on the facet appears once per cell. A block
/// is skipped when an argument has no cell on its side (a negative cell, e.g.
/// on an interface between two subdomains).
template <typename T, std::floating_point U>
void assemble_interior_facets(
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
    const dolfinx::fem::DofMap& dofmap1, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*, void*)>& kernel,
    const std::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const std::span<const std::uint32_t>& cell_info0,
    const std::span<const std::uint32_t>& cell_info1,
    std::span<const std::uint8_t> perms, int num_facets_per_cell,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1)
{
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

  if (mesh.geometry().dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  mdspan2_t x_dofmap = mesh.geometry().dofmaps().front();
  const std::size_t num_dofs_g = x_dofmap.extent(1);
  std::span<const U> x_g = mesh.geometry().x();
  std::vector<U> coordinate_dofs(2 * 3 * num_dofs_g);

  const std::array<const std::uint32_t, 2> num_dofs
      = {static_cast<std::uint32_t>(dofmap0.map().extent(1)),
         static_cast<std::uint32_t>(dofmap1.map().extent(1))};
  const std::array<const int, 2> bs = {dofmap0.bs(), dofmap1.bs()};
  const std::size_t ndim0 = bs[0] * num_dofs[0];
  const std::size_t ndim1 = bs[1] * num_dofs[1];
  const std::size_t num_rows = 2 * ndim0;
  const std::size_t num_cols = 2 * ndim1;

  // The joint tensor, and one (ndim0, ndim1) block of it
  std::vector<T> Ab(num_rows * num_cols);
  std::vector<T> Ae_block(ndim0 * ndim1);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Ae(Ae_block.data(), ndim0, ndim1);
  std::vector<T> scratch_memory(2 * ndim0 * ndim1 + ndim0 + ndim1);
  std::vector<std::int32_t> joint_dofs0(2 * num_dofs[0]);
  std::vector<std::int32_t> joint_dofs1(2 * num_dofs[1]);

  const bool transform0_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation);
  const bool transform1_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation_to_transpose);
  for (std::size_t f = 0; f < facets.size() / 4; ++f)
  {
    // Entities are (cell, local facet) for each side
    const std::array<std::int32_t, 2> cells
        = {facets[4 * f], facets[4 * f + 2]};
    const std::array<int, 2> local_facet
        = {facets[4 * f + 1], facets[4 * f + 3]};
    const std::array<std::int32_t, 2> cells0
        = {facets0[4 * f], facets0[4 * f + 2]};
    const std::array<std::int32_t, 2> cells1
        = {facets1[4 * f], facets1[4 * f + 2]};

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
    std::ranges::fill(Ab, T(0));
    kernel(Ab.data(), coeffs.data() + f * 2 * cstride, constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data(), nullptr);

    // Transform each block row and block column whose cell exists
    if (transform0_set and cells0[0] >= 0)
      apply_dof_transformation(Ab, cell_info0, cells0[0], num_cols);
    if (transform0_set and cells0[1] >= 0)
    {
      std::span<T> sub(Ab.data() + ndim0 * num_cols, ndim0 * num_cols);
      apply_dof_transformation(sub, cell_info0, cells0[1], num_cols);
    }
    if (transform1_set and cells1[0] >= 0)
    {
      apply_dof_transformation_to_transpose(Ab, cell_info1, cells1[0],
                                            num_rows);
    }
    if (transform1_set and cells1[1] >= 0)
    {
      // The second cell's columns are not contiguous, so transform row by row
      for (std::size_t row = 0; row < num_rows; ++row)
      {
        std::span<T> sub(Ab.data() + row * num_cols + ndim1, ndim1);
        apply_dof_transformation_to_transpose(sub, cell_info1, cells1[1], 1);
      }
    }

    auto has_slaves
        = [](const dolfinx::graph::AdjacencyList<std::int32_t>& c,
             std::int32_t cell) { return cell >= 0 and c.num_links(cell) > 0; };
    const bool all_cells = cells0[0] >= 0 and cells0[1] >= 0 and cells1[0] >= 0
                           and cells1[1] >= 0;
    const bool any_slaves = has_slaves(*cell_to_slaves[0], cells0[0])
                            or has_slaves(*cell_to_slaves[0], cells0[1])
                            or has_slaves(*cell_to_slaves[1], cells1[0])
                            or has_slaves(*cell_to_slaves[1], cells1[1]);

    // Common case: both arguments on both sides and no slaves, so the joint
    // tensor is inserted in one go
    if (all_cells and !any_slaves)
    {
      for (int s = 0; s < 2; ++s)
      {
        std::ranges::copy(dofmap0.cell_dofs(cells0[s]),
                          std::next(joint_dofs0.begin(), s * num_dofs[0]));
        std::ranges::copy(dofmap1.cell_dofs(cells1[s]),
                          std::next(joint_dofs1.begin(), s * num_dofs[1]));
      }
      zero_dirichlet<T>(Ab, joint_dofs0, bs[0], bc0, joint_dofs1, bs[1], bc1);
      mat_add_block_values(joint_dofs0, joint_dofs1, Ab);
      continue;
    }

    for (int s = 0; s < 2; ++s)
    {
      if (cells0[s] < 0)
        continue;
      std::span<const std::int32_t> dofs0 = dofmap0.cell_dofs(cells0[s]);
      for (int t = 0; t < 2; ++t)
      {
        if (cells1[t] < 0)
          continue;
        std::span<const std::int32_t> dofs1 = dofmap1.cell_dofs(cells1[t]);

        // Copy block (s, t) out of the joint tensor
        for (std::size_t row = 0; row < ndim0; ++row)
        {
          std::copy_n(
              std::next(Ab.begin(), (s * ndim0 + row) * num_cols + t * ndim1),
              ndim1, std::next(Ae_block.begin(), row * ndim1));
        }
        zero_dirichlet<T>(Ae_block, dofs0, bs[0], bc0, dofs1, bs[1], bc1);

        if (has_slaves(*cell_to_slaves[0], cells0[s])
            or has_slaves(*cell_to_slaves[1], cells1[t]))
        {
          const std::array<std::span<const std::int32_t>, 2> slaves
              = {cell_to_slaves[0]->links(cells0[s]),
                 cell_to_slaves[1]->links(cells1[t])};
          modify_mpc_cell<T>(mat_add_values, num_dofs, Ae, {dofs0, dofs1}, bs,
                             slaves, masters, coefficients, is_slave,
                             scratch_memory);
        }
        mat_add_block_values(dofs0, dofs1, Ae_block);
      }
    }
  }
}
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
    const dolfinx::fem::DofMap& dofmap1, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1,
    const std::function<void(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*, void*)>& kernel,
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
  if (geometry.dofmaps().size() != 1)
    throw std::runtime_error(
        "Currently only supports meshes with one geometry dofmap.");
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmaps().front();
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

  const bool transform0_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation);
  const bool transform1_set
      = dolfinx::fem::is_transform_set(apply_dof_transformation_to_transpose);
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
           coordinate_dofs.data(), nullptr, nullptr, nullptr);
    if (transform0_set)
    {
      apply_dof_transformation(_Ae, cell_info0, cell0, ndim1);
    }
    if (transform1_set)
    {
      apply_dof_transformation_to_transpose(_Ae, cell_info1, cell1, ndim0);
    }

    // Zero rows/columns for essential bcs
    std::span<const std::int32_t> dofs0 = dofmap0.cell_dofs(cell0);
    std::span<const std::int32_t> dofs1 = dofmap1.cell_dofs(cell1);
    zero_dirichlet<T>(_Ae, dofs0, bs.front(), bc0, dofs1, bs.back(), bc1);

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
    const dolfinx::fem::Form<T>& a, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1,
    std::size_t num_threads)
{
  // Integration domain mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = a.mesh();
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
    mesh0->topology_mutable()->create_cell_permutations(num_threads);
    mesh1->topology_mutable()->create_cell_permutations(num_threads);
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  // Facet permutations of the integration domain. Needed whenever the kernel
  // asks for them, which happens for instance when the two argument spaces
  // live on different meshes.
  std::span<const std::uint8_t> perms;
  int num_facets_per_cell = 0;
  if (a.needs_facet_permutations())
  {
    const dolfinx::mesh::CellType cell_type
        = mesh->topology()->cell_types().front();
    const std::size_t fdim = mesh->topology()->dim() - 1;
    num_facets_per_cell = dolfinx::mesh::cell_num_entities(cell_type, fdim);
    mesh->topology_mutable()->create_entity_permutations(fdim, num_threads);
    perms = std::span(mesh->topology()->get_entity_permutations(fdim));
  }
  for (int i = 0; i < a.num_integrals(dolfinx::fem::IntegralType::cell, 0); ++i)
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

  for (int i = 0;
       i < a.num_integrals(dolfinx::fem::IntegralType::exterior_facet, 0); ++i)

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
        cstride, constants, cell_info0, cell_info1, perms, num_facets_per_cell,
        mpc0, mpc1);
  }

  for (int i = 0;
       i < a.num_integrals(dolfinx::fem::IntegralType::interior_facet, 0); ++i)
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::interior_facet, i, 0);
    const auto& [coeffs, cstride]
        = coefficients.at({dolfinx::fem::IntegralType::interior_facet, i});
    std::span<const std::int32_t> facets
        = a.domain(dolfinx::fem::IntegralType::interior_facet, i, 0);
    std::span<const std::int32_t> facets0
        = a.domain_arg(dolfinx::fem::IntegralType::interior_facet, 0, i, 0);
    std::span<const std::int32_t> facets1
        = a.domain_arg(dolfinx::fem::IntegralType::interior_facet, 1, i, 0);
    assemble_interior_facets<T>(
        mat_add_block_values, mat_add_values, *mesh, facets, facets0, facets1,
        apply_dof_transformation, *dofmap0,
        apply_dof_transformation_to_transpose, *dofmap1, bc0, bc1, fn, coeffs,
        cstride, constants, cell_info0, cell_info1, perms, num_facets_per_cell,
        mpc0, mpc1);
  }
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
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads)
{
  dolfinx::common::Timer timer("~MPC: Assemble matrix (C++)");
  assemble_matrix_impl<T>(mat_add_block, mat_add, a, dof_marker0, dof_marker1,
                          mpc0, mpc1, num_threads);
  timer.stop();
}
//-----------------------------------------------------------------------------
/// @brief Mark the dofs of `V` constrained by the conditions in `bcs` that are
/// defined on `V` or a subspace of it.
///
/// Mirrors `dolfinx::fem::impl::bc_dof_markers`. Only the convenience
/// overloads use it; the library itself is handed markers.
/// @return Markers, or an empty array if no condition applies.
template <typename T, std::floating_point U>
std::vector<std::int8_t> _bc_dof_markers(
    const dolfinx::fem::FunctionSpace<U>& V,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{
  std::vector<std::int8_t> markers;
  for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>>& bc : bcs)
  {
    assert(bc);
    assert(bc->function_space());
    if (V.contains(*bc->function_space()))
    {
      if (markers.empty())
      {
        const dolfinx::fem::DofMap& dofmap = *V.dofmap();
        const std::shared_ptr<const dolfinx::common::IndexMap> map
            = dofmap.index_map;
        markers.resize(
            dofmap.index_map_bs() * (map->size_local() + map->num_ghosts()), 0);
      }
      bc->mark_dofs(markers);
    }
  }
  return markers;
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
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, dof_marker0,
                   dof_marker1, num_threads);
}
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
    std::size_t num_threads)
{
  // Convenience overload: it rebuilds the markers on every call, so the
  // library calls the overload above instead.
  const std::vector<std::int8_t> marker0
      = _bc_dof_markers(*a.function_spaces().at(0), bcs);
  const std::vector<std::int8_t> marker1
      = _bc_dof_markers(*a.function_spaces().at(1), bcs);
  assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1,
                  std::span<const std::int8_t>(marker0),
                  std::span<const std::int8_t>(marker1), num_threads);
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
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, dof_marker0,
                   dof_marker1, num_threads);
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
    std::size_t num_threads)
{
  // Convenience overload: it rebuilds the markers on every call, so the
  // library calls the overload above instead.
  const std::vector<std::int8_t> marker0
      = _bc_dof_markers(*a.function_spaces().at(0), bcs);
  const std::vector<std::int8_t> marker1
      = _bc_dof_markers(*a.function_spaces().at(1), bcs);
  assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1,
                  std::span<const std::int8_t>(marker0),
                  std::span<const std::int8_t>(marker1), num_threads);
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
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, dof_marker0,
                   dof_marker1, num_threads);
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
    std::size_t num_threads)
{
  // Convenience overload: it rebuilds the markers on every call, so the
  // library calls the overload above instead.
  const std::vector<std::int8_t> marker0
      = _bc_dof_markers(*a.function_spaces().at(0), bcs);
  const std::vector<std::int8_t> marker1
      = _bc_dof_markers(*a.function_spaces().at(1), bcs);
  assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1,
                  std::span<const std::int8_t>(marker0),
                  std::span<const std::int8_t>(marker1), num_threads);
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
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1, dof_marker0,
                   dof_marker1, num_threads);
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
    std::size_t num_threads)
{
  // Convenience overload: it rebuilds the markers on every call, so the
  // library calls the overload above instead.
  const std::vector<std::int8_t> marker0
      = _bc_dof_markers(*a.function_spaces().at(0), bcs);
  const std::vector<std::int8_t> marker1
      = _bc_dof_markers(*a.function_spaces().at(1), bcs);
  assemble_matrix(mat_add_block, mat_add, a, mpc0, mpc1,
                  std::span<const std::int8_t>(marker0),
                  std::span<const std::int8_t>(marker1), num_threads);
}

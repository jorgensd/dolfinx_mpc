// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "MultiPointConstraint.h"
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <format>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace dolfinx_mpc
{
template <typename T, std::floating_point U>
class MultiPointConstraint;

/// @brief Add a value on the diagonal of every slave row owned by the process.
///
/// Kept out of `assemble_matrix` because the diagonal belongs to a block, not
/// to a form: a block may carry slaves without having a diagonal bilinear form
/// to assemble, and a block appearing in several forms must still receive
/// exactly one diagonal entry. Callers run this once per diagonal block, after
/// every form has been assembled.
///
/// @param[in] mat_add Function for adding values into the matrix
/// @param[in] mpc Constraint whose slave rows are given a diagonal entry
/// @param[in] diagval Value to add on the diagonal
template <typename T, std::floating_point U>
void insert_slave_diagonal(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>&)>& mat_add,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc, T diagval)
{
  const std::vector<std::int32_t>& slaves = mpc.slaves();
  const std::int32_t num_local_slaves = mpc.num_local_slaves();
  std::array<std::int32_t, 1> dof;
  const std::array<T, 1> value = {diagval};
  for (std::int32_t i = 0; i < num_local_slaves; ++i)
  {
    dof[0] = slaves[i];
    mat_add(dof, dof, value);
  }
}

/// @brief Assemble a bilinear form into a matrix, given constrained dofs.
///
/// This is the overload the library calls internally. Rows marked in
/// `dof_marker0` and columns marked in `dof_marker1` are zeroed; the diagonal
/// entry is not set. An empty marker array marks nothing.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] dof_marker0 Constrained dof markers on the test space of `a`
/// (owned and ghost, unrolled), or empty if none are constrained
/// @param[in] dof_marker1 Constrained dof markers on the trial space of `a`
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads = 1);
//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix.
///
/// @note Convenience overload for callers holding boundary conditions. It
/// rebuilds the dof markers on every call; the library uses the overload
/// above.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] bcs Boundary conditions to apply. For boundary condition
/// dofs the row and column are zeroed. The diagonal entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::size_t num_threads = 1);

//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix, given constrained dofs.
///
/// This is the overload the library calls internally. Rows marked in
/// `dof_marker0` and columns marked in `dof_marker1` are zeroed; the diagonal
/// entry is not set. An empty marker array marks nothing.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] dof_marker0 Constrained dof markers on the test space of `a`
/// (owned and ghost, unrolled), or empty if none are constrained
/// @param[in] dof_marker1 Constrained dof markers on the trial space of `a`
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads = 1);
//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix.
///
/// @note Convenience overload for callers holding boundary conditions. It
/// rebuilds the dof markers on every call; the library uses the overload
/// above.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] bcs Boundary conditions to apply. For boundary condition
/// dofs the row and column are zeroed. The diagonal entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::size_t num_threads = 1);

/// @brief Assemble a bilinear form into a matrix, given constrained dofs.
///
/// This is the overload the library calls internally. Rows marked in
/// `dof_marker0` and columns marked in `dof_marker1` are zeroed; the diagonal
/// entry is not set. An empty marker array marks nothing.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] dof_marker0 Constrained dof markers on the test space of `a`
/// (owned and ghost, unrolled), or empty if none are constrained
/// @param[in] dof_marker1 Constrained dof markers on the trial space of `a`
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads = 1);
//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix.
///
/// @note Convenience overload for callers holding boundary conditions. It
/// rebuilds the dof markers on every call; the library uses the overload
/// above.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] bcs Boundary conditions to apply. For boundary condition
/// dofs the row and column are zeroed. The diagonal entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::size_t num_threads = 1);

//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix, given constrained dofs.
///
/// This is the overload the library calls internally. Rows marked in
/// `dof_marker0` and columns marked in `dof_marker1` are zeroed; the diagonal
/// entry is not set. An empty marker array marks nothing.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] dof_marker0 Constrained dof markers on the test space of `a`
/// (owned and ghost, unrolled), or empty if none are constrained
/// @param[in] dof_marker1 Constrained dof markers on the trial space of `a`
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads = 1);
//-----------------------------------------------------------------------------
/// @brief Assemble a bilinear form into a matrix.
///
/// @note Convenience overload for callers holding boundary conditions. It
/// rebuilds the dof markers on every call; the library uses the overload
/// above.
/// @param[in] mat_add_block The function for adding block values into the
/// matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] bcs Boundary conditions to apply. For boundary condition
/// dofs the row and column are zeroed. The diagonal entry is not set.
/// @param[in] num_threads The number of threads to use for certain operations.
void assemble_matrix(
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
    std::size_t num_threads = 1);

/// @brief Route the insertions of a constrained form to the matrices of the
/// blocks of a system.
///
/// The form `a[i][j]` of a blocked system has its rows constrained by `mpc0`
/// and its columns by `mpc1`. Its own entries belong to block `(i, j)`, while a
/// master of either constraint may be in another block, given by its
/// `MultiPointConstraint::master_blocks()`. The returned function sends each
/// insertion to the matrix of the block it belongs to, and is the
/// `mat_add_blocks` argument of `assemble_matrix_blocks`.
///
/// A block is identified by its position in the system, which for a master in
/// another block is its block index: the constraints must then be the blocks
/// of the system in the order they were created in. A master in the
/// constraint's own block goes to `(i, j)` whatever the order.
///
/// @param[in] mat_add Function adding values to the matrix of block `(k, l)`,
/// `mat_add[k][l](rows, cols, values)`, with unrolled indices local to those
/// blocks. A block without a matrix has an empty function.
/// @param[in] i Position of the form's row block in the system
/// @param[in] j Position of the form's column block in the system
/// @param[in] mpc0 Constraint on the rows of the form
/// @param[in] mpc1 Constraint on the columns of the form
/// @return `mat_add_blocks(row_block, rows, col_block, cols, values)`. It
/// throws `std::runtime_error` if an insertion falls in a block without a
/// matrix, which means the sparsity pattern did not include it.
template <typename T, std::floating_point U>
std::function<int(int, std::span<const std::int32_t>, int,
                  std::span<const std::int32_t>, std::span<const T>)>
make_mat_add_blocks(
    std::vector<std::vector<
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>, std::span<const T>)>>>
        mat_add,
    std::size_t i, std::size_t j,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc0,
    const dolfinx_mpc::MultiPointConstraint<T, U>& mpc1)
{
  if (i >= mat_add.size() or j >= mat_add[i].size() or !mat_add[i][j])
  {
    throw std::invalid_argument(
        std::format("Block ({}, {}) of the form has no matrix.", i, j));
  }
  return [mat_add = std::move(mat_add), i, j, block0 = mpc0.block(),
          block1 = mpc1.block()](int rb, std::span<const std::int32_t> rows,
                                 int cb, std::span<const std::int32_t> cols,
                                 std::span<const T> vals) -> int
  {
    const std::size_t r = rb == block0 ? i : static_cast<std::size_t>(rb);
    const std::size_t c = cb == block1 ? j : static_cast<std::size_t>(cb);
    if (r >= mat_add.size() or c >= mat_add[r].size() or !mat_add[r][c])
    {
      throw std::runtime_error(std::format(
          "A master puts entries in block ({}, {}), which has no matrix.", r,
          c));
    }
    return mat_add[r][c](rows, cols, vals);
  };
}

/// @brief Assemble a bilinear form into the matrices of a blocked system,
/// given constrained dofs.
///
/// The entries of the form's own block go through `mat_add_block`, with
/// blocked indices. A constraint with masters in other blocks adds the rows and
/// columns of those masters to other blocks of the system, so every entry the
/// constraint adds goes through `mat_add_blocks` with the blocks it belongs to.
/// @param[in] mat_add_block Adds an element matrix to the form's own block,
/// with blocked indices
/// @param[in] mat_add_blocks `mat_add_blocks(row_block, rows, col_block, cols,
/// values)` adds values to the given block, with unrolled local indices of
/// those blocks. A block is the index of a constraint among those created
/// together; the form's own blocks are `mpc0->block()` and `mpc1->block()`.
/// @param[in] a The bilinear form to assemble
/// @param[in] mpc0 Constraint applied to the rows
/// @param[in] mpc1 Constraint applied to the columns
/// @param[in] dof_marker0 Constrained dof markers on the test space of `a`, or
/// empty if none are constrained
/// @param[in] dof_marker1 Constrained dof markers on the trial space of `a`
/// @param[in] num_threads The number of threads to use for certain operations.
template <typename T, std::floating_point U>
void assemble_matrix_blocks(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>,
                            const std::span<const T>&)>& mat_add_block,
    const std::function<int(int, std::span<const std::int32_t>, int,
                            std::span<const std::int32_t>, std::span<const T>)>&
        mat_add_blocks,
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1,
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1, std::size_t num_threads = 1);

} // namespace dolfinx_mpc
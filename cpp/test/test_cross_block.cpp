// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
//
// Assembly of a blocked system whose constraint has masters in another
// block, through `make_mat_add_blocks` and `assemble_matrix_blocks`, into
// `la::MatrixCSR` matrices. Two P1 spaces on one mesh, a block-diagonal form,
// and the slaves of the first space on x = 1 tied to the second space at the
// same points.

#include "forms.h"
#include <array>
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx_mpc/MultiPointConstraint.h>
#include <dolfinx_mpc/assemble_matrix.h>
#include <dolfinx_mpc/utils.h>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

namespace
{
using T = double;
using mpc_t = dolfinx_mpc::MultiPointConstraint<T, double>;
using mat_add_t
    = std::function<int(std::span<const std::int32_t>,
                        std::span<const std::int32_t>, std::span<const T>)>;
using mat_add_ref_t = std::function<int(std::span<const std::int32_t>,
                                        std::span<const std::int32_t>,
                                        const std::span<const T>&)>;

constexpr T coeff = 2.0;

struct System
{
  std::shared_ptr<const fem::FunctionSpace<double>> V, Q;
  std::shared_ptr<const fem::Form<T>> a0, a1;
  std::vector<std::shared_ptr<mpc_t>> mpcs;
  std::vector<std::int32_t> slaves;
};

/// Block 0 is V, block 1 is Q. The slaves of V on x = 1 are `coeff` times Q
/// at the same dofs, which have the same local index as V and Q are the same
/// element on the same mesh.
System create_system()
{
  MPI_Comm comm = MPI_COMM_WORLD;
  auto mesh
      = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle<double>(
          comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 3}, mesh::CellType::triangle));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto create_space = [&mesh, &element]()
  {
    return std::make_shared<const fem::FunctionSpace<double>>(
        fem::create_functionspace<double>(
            mesh, std::make_shared<fem::FiniteElement<double>>(
                      element, mesh->geometry().dim())));
  };
  System s;
  s.V = create_space();
  s.Q = create_space();
  s.a0 = std::make_shared<const fem::Form<T>>(
      fem::create_form<T>(*form_forms_a, {s.V, s.V}, {}, {}, {}, {}));
  s.a1 = std::make_shared<const fem::Form<T>>(
      fem::create_form<T>(*form_forms_a, {s.Q, s.Q}, {}, {}, {}, {}));

  s.slaves = fem::locate_dofs_geometrical(
      *s.V,
      [](auto x)
      {
        std::vector<std::int8_t> marker(x.extent(1));
        for (std::size_t i = 0; i < x.extent(1); ++i)
          marker[i] = std::abs(x(0, i) - 1.0) < 1e-10;
        return marker;
      });

  const common::IndexMap& imap = *s.Q->dofmap()->index_map;
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int64_t> masters(s.slaves.size());
  imap.local_to_global(s.slaves, masters);
  std::vector<std::int32_t> owners(s.slaves.size(), rank);
  std::span<const int> ghost_owners = imap.owners();
  for (std::size_t i = 0; i < s.slaves.size(); ++i)
    if (s.slaves[i] >= imap.size_local())
      owners[i] = ghost_owners[s.slaves[i] - imap.size_local()];
  std::vector<T> coeffs(s.slaves.size(), coeff);
  std::vector<std::int32_t> offsets(s.slaves.size() + 1);
  for (std::size_t i = 0; i < offsets.size(); ++i)
    offsets[i] = i;
  std::vector<std::int32_t> master_blocks(s.slaves.size(), 1);

  static constexpr std::int32_t no_offsets[1] = {0};
  s.mpcs = dolfinx_mpc::create_multipointconstraints<T, double>(
      {s.V, s.Q},
      {dolfinx_mpc::mpc_block_view<T>{
           s.slaves, masters, coeffs, owners, offsets, {}, {}, master_blocks},
       dolfinx_mpc::mpc_block_view<T>{
           {},
           {},
           {},
           {},
           std::span<const std::int32_t>(no_offsets, 1),
           {},
           {},
           {}}});
  return s;
}

/// The matrix of every block, with the pattern of the constrained system
std::vector<std::vector<la::MatrixCSR<T>>> create_matrices(const System& s)
{
  std::vector<std::vector<la::SparsityPattern>> patterns
      = dolfinx_mpc::create_sparsity_patterns<T, double>(
          {{s.a0.get(), nullptr}, {nullptr, s.a1.get()}}, s.mpcs, s.mpcs);
  std::vector<std::vector<la::MatrixCSR<T>>> A(2);
  for (std::size_t k = 0; k < 2; ++k)
  {
    for (std::size_t l = 0; l < 2; ++l)
    {
      patterns[k][l].finalize();
      A[k].emplace_back(patterns[k][l]);
    }
  }
  return A;
}

/// Dense, unconstrained matrix of a form
std::vector<T> dense(const fem::Form<T>& a)
{
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);
  pattern.finalize();
  la::MatrixCSR<T> A(pattern);
  fem::assemble_matrix(A.mat_add_values(), a, std::span<const std::int8_t>(),
                       std::span<const std::int8_t>());
  A.scatter_rev();
  return A.to_dense();
}
} // namespace

TEST_CASE("Masters in another block assemble to K^T A K", "[cross-block]")
{
  if (dolfinx::MPI::size(MPI_COMM_WORLD) > 1)
    SKIP("The dense reference is built on one process");

  System s = create_system();
  REQUIRE(s.mpcs[0]->has_cross_block_masters());
  REQUIRE(!s.slaves.empty());

  std::vector<std::vector<la::MatrixCSR<T>>> A = create_matrices(s);
  std::vector<std::vector<mat_add_t>> mat_add(2);
  for (std::size_t k = 0; k < 2; ++k)
    for (std::size_t l = 0; l < 2; ++l)
      mat_add[k].push_back(A[k][l].mat_add_values());

  // Each form in its own block, the masters' entries in theirs
  for (std::size_t i = 0; i < 2; ++i)
  {
    const fem::Form<T>& a = i == 0 ? *s.a0 : *s.a1;
    dolfinx_mpc::assemble_matrix_blocks<T, double>(
        mat_add_ref_t(A[i][i].mat_add_values()),
        dolfinx_mpc::make_mat_add_blocks<T, double>(mat_add, i, i, *s.mpcs[i],
                                                    *s.mpcs[i]),
        a, s.mpcs[i], s.mpcs[i], {}, {});
  }
  for (std::size_t k = 0; k < 2; ++k)
  {
    dolfinx_mpc::insert_slave_diagonal<T, double>(
        mat_add_ref_t(A[k][k].mat_add_values()), *s.mpcs[k], 1.0);
  }

  // Reference: x = K x_red, with the slave columns of K zero, applied to the
  // unconstrained block-diagonal matrix
  const std::size_t n = s.V->dofmap()->index_map->size_local();
  const std::size_t N = 2 * n;
  std::vector<T> A0 = dense(*s.a0), A1 = dense(*s.a1);
  std::vector<T> A_full(N * N, 0);
  for (std::size_t r = 0; r < n; ++r)
  {
    for (std::size_t c = 0; c < n; ++c)
    {
      A_full[r * N + c] = A0[r * n + c];
      A_full[(n + r) * N + n + c] = A1[r * n + c];
    }
  }
  std::vector<T> K(N * N, 0);
  std::vector<bool> is_slave(N, false);
  for (std::int32_t s0 : s.slaves)
    is_slave[s0] = true;
  for (std::size_t r = 0; r < N; ++r)
    if (!is_slave[r])
      K[r * N + r] = 1;
  for (std::int32_t s0 : s.slaves)
    K[s0 * N + n + s0] = coeff;
  std::vector<T> KtAK(N * N, 0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      for (std::size_t p = 0; p < N; ++p)
        for (std::size_t q = 0; q < N; ++q)
          KtAK[i * N + j] += K[p * N + i] * A_full[p * N + q] * K[q * N + j];

  // The assembled blocks, side by side
  std::vector<T> A_mpc(N * N, 0);
  for (std::size_t k = 0; k < 2; ++k)
  {
    for (std::size_t l = 0; l < 2; ++l)
    {
      A[k][l].scatter_rev();
      std::vector<T> A_kl = A[k][l].to_dense();
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
          A_mpc[(k * n + r) * N + l * n + c] = A_kl[r * n + c];
    }
  }

  // Off-diagonal blocks from a block-diagonal form
  T coupling = 0;
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t c = 0; c < n; ++c)
      coupling = std::max(coupling, std::abs(A_mpc[r * N + n + c]));
  CHECK(coupling > 0);

  // Free rows and columns are K^T A K, slave rows and columns the identity
  for (std::size_t i = 0; i < N; ++i)
  {
    for (std::size_t j = 0; j < N; ++j)
    {
      const T expected = (is_slave[i] or is_slave[j]) ? (i == j ? T(1) : T(0))
                                                      : KtAK[i * N + j];
      CHECK_THAT(A_mpc[i * N + j], Catch::Matchers::WithinAbs(expected, 1e-12));
    }
  }
}

TEST_CASE("make_mat_add_blocks needs a matrix for every block it routes to",
          "[cross-block]")
{
  System s = create_system();
  std::vector<std::vector<la::MatrixCSR<T>>> A = create_matrices(s);

  // The off-diagonal blocks are missing
  std::vector<std::vector<mat_add_t>> mat_add(2, std::vector<mat_add_t>(2));
  mat_add[0][0] = A[0][0].mat_add_values();
  mat_add[1][1] = A[1][1].mat_add_values();

  // The form's own block has no matrix
  auto route_to_missing_block = [&]()
  {
    return dolfinx_mpc::make_mat_add_blocks<T, double>(mat_add, 0, 1,
                                                       *s.mpcs[0], *s.mpcs[1]);
  };
  CHECK_THROWS_AS(route_to_missing_block(), std::invalid_argument);

  // Only processes that assemble a cell with a slave of block 0 route an entry
  // to block (1, 0)
  const std::int32_t num_cells
      = s.V->mesh()->topology()->index_map(2)->size_local();
  bool has_slaves = false;
  for (std::int32_t c = 0; c < num_cells; ++c)
    has_slaves = has_slaves or s.mpcs[0]->cell_to_slaves()->num_links(c) > 0;
  auto assemble = [&]()
  {
    dolfinx_mpc::assemble_matrix_blocks<T, double>(
        mat_add_ref_t(A[0][0].mat_add_values()),
        dolfinx_mpc::make_mat_add_blocks<T, double>(mat_add, 0, 0, *s.mpcs[0],
                                                    *s.mpcs[0]),
        *s.a0, s.mpcs[0], s.mpcs[0], {}, {});
  };
  if (has_slaves)
    CHECK_THROWS_AS(assemble(), std::runtime_error);
  else
    CHECK_NOTHROW(assemble());
}

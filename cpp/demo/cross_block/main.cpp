// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
//
// # Masters in another block (C++)
//
// Two fields `u` and `p` live in their own P1 spaces `V` and `Q` on one
// mesh, and solve different reaction-diffusion problems:
//
// $$
// -\nabla\cdot(\kappa\nabla u) + u = \sin(\pi y), \qquad
// -\Delta p + p = xy,
// $$
//
// with a constant diffusion coefficient $\kappa$ for `u` only. The forms do
// not couple the fields. The coupling is a multi point constraint whose masters
// are in the other block:
//
// $$
// u(x) = c\,p(x) \quad\text{for } x \text{ on the right edge},
// $$
//
// so every slave of `V` on $x = 1$ has its master in `Q`. The reduced operator
// $K^T A K$ then has off-diagonal blocks although the form has none. The demo
// assembles the system into a PETSc nest matrix with
// `dolfinx_mpc::assemble_matrix_blocks`, routing the entries of each master to
// the matrix of its block with `dolfinx_mpc::make_mat_add_blocks`, and solves
// it with conjugate gradients.

#include "cross_block.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx_mpc/MultiPointConstraint.h>
#include <dolfinx_mpc/assemble_matrix.h>
#include <dolfinx_mpc/assemble_vector.h>
#include <dolfinx_mpc/utils.h>
#include <format>
#include <functional>
#include <memory>
#include <petscksp.h>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  common::petsc::check(PetscInitialize(&argc, &argv, nullptr, nullptr),
                       "PetscInitialize");
  {
    MPI_Comm comm = MPI_COMM_WORLD;

    // ## Spaces and forms
    // The same element on the same mesh, created twice, gives two spaces
    // with the same dof numbering but independent blocks of the system.
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32}, mesh::CellType::triangle));
    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto create_space = [&mesh, &element]()
    {
      return std::make_shared<const fem::FunctionSpace<U>>(
          fem::create_functionspace<U>(mesh,
                                       std::make_shared<fem::FiniteElement<U>>(
                                           element, mesh->geometry().dim())));
    };
    auto V = create_space();
    auto Q = create_space();

    // The forms of the blocked system, `a[i][j]` coupling the test functions
    // of block `i` to the trial functions of block `j`. The fields are not
    // coupled by the PDEs, so only the diagonal blocks have a form.
    auto kappa = std::make_shared<const fem::Constant<T>>(T(0.1));
    std::vector<std::vector<std::shared_ptr<const fem::Form<T>>>> a
        = {{std::make_shared<const fem::Form<T>>(fem::create_form<T>(
                *form_cross_block_a_u, {V, V}, {}, {{"kappa", kappa}}, {}, {})),
            nullptr},
           {nullptr, std::make_shared<const fem::Form<T>>(fem::create_form<T>(
                         *form_cross_block_a_p, {Q, Q}, {}, {}, {}, {}))}};
    std::array<fem::Form<T>, 2> L
        = {fem::create_form<T>(*form_cross_block_L_u, {V}, {}, {}, {}, {}),
           fem::create_form<T>(*form_cross_block_L_p, {Q}, {}, {}, {}, {})};

    // ## The constraint
    // The slaves are the dofs of `V` on the right edge, ghosts included. As
    // `V` and `Q` number their dofs alike, the master of each is the dof with
    // the same local index in `Q`, given in the global numbering of `Q` and
    // with its owner. `master_blocks` places every master in block 1.
    const U c = 2.0;
    std::vector<std::int32_t> slaves = fem::locate_dofs_geometrical(
        *V,
        [](auto x)
        {
          std::vector<std::int8_t> marker(x.extent(1));
          for (std::size_t i = 0; i < x.extent(1); ++i)
            marker[i] = std::abs(x(0, i) - 1.0) < 1e-10;
          return marker;
        });
    const common::IndexMap& imap = *Q->dofmap()->index_map;
    std::vector<std::int64_t> masters(slaves.size());
    imap.local_to_global(slaves, masters);
    std::vector<std::int32_t> owners(slaves.size(), dolfinx::MPI::rank(comm));
    for (std::size_t i = 0; i < slaves.size(); ++i)
      if (slaves[i] >= imap.size_local())
        owners[i] = imap.owners()[slaves[i] - imap.size_local()];
    std::vector<T> coeffs(slaves.size(), c);
    std::vector<std::int32_t> offsets(slaves.size() + 1);
    for (std::size_t i = 0; i < offsets.size(); ++i)
      offsets[i] = i;
    std::vector<std::int32_t> master_blocks(slaves.size(), 1);

    // The constraints of both blocks are created together, so that the
    // extended index map of `Q` gains the masters that `V` puts there. `Q`
    // has no slaves of its own.
    static constexpr std::int32_t no_offsets[1] = {0};
    std::vector<std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>> mpcs
        = dolfinx_mpc::create_multipointconstraints<T, U>(
            {V, Q}, {dolfinx_mpc::mpc_block_view<T>{slaves,
                                                    masters,
                                                    coeffs,
                                                    owners,
                                                    offsets,
                                                    {},
                                                    {},
                                                    master_blocks},
                     dolfinx_mpc::mpc_block_view<T>{
                         {},
                         {},
                         {},
                         {},
                         std::span<const std::int32_t>(no_offsets, 1),
                         {},
                         {},
                         {}}});

    // ## The matrix
    // Every block of the nest exists, as the masters put entries in the
    // off-diagonal blocks. Each form goes to its own block, and
    // `make_mat_add_blocks` sends the entries of a master to the block of
    // that master.
    std::vector<std::vector<const fem::Form<T>*>> a_ptr(2);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        a_ptr[i].push_back(a[i][j].get());
    Mat A = dolfinx_mpc::create_matrix_nest<T, U>(a_ptr, mpcs, mpcs);
    std::array<std::array<Mat, 2>, 2> blocks;
    std::vector<std::vector<
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>, std::span<const T>)>>>
        mat_add(2);
    for (int k = 0; k < 2; ++k)
    {
      for (int l = 0; l < 2; ++l)
      {
        MatNestGetSubMat(A, k, l, &blocks[k][l]);
        mat_add[k].push_back(
            la::petsc::Matrix::set_fn(blocks[k][l], ADD_VALUES));
      }
    }
    // Assemble every block that has a form. Here that is only the diagonal,
    // as the PDEs do not couple `u` and `p`. The off-diagonal blocks are
    // nevertheless filled: a slave of `u` moves its row and column to its
    // master in `p`, which `make_mat_add_blocks` sends to block (1, 0) and
    // (0, 1).
    for (std::size_t i = 0; i < 2; ++i)
    {
      for (std::size_t j = 0; j < 2; ++j)
      {
        if (!a[i][j])
          continue;
        dolfinx_mpc::assemble_matrix_blocks<T, U>(
            la::petsc::Matrix::set_block_fn(blocks[i][j], ADD_VALUES),
            dolfinx_mpc::make_mat_add_blocks<T, U>(mat_add, i, j, *mpcs[i],
                                                   *mpcs[j]),
            *a[i][j], mpcs[i], mpcs[j], {}, {});
      }
    }
    // The slave rows hold the identity, once per diagonal block
    for (std::size_t k = 0; k < 2; ++k)
    {
      dolfinx_mpc::insert_slave_diagonal<T, U>(
          la::petsc::Matrix::set_fn(blocks[k][k], ADD_VALUES), *mpcs[k], T(1));
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    PetscReal coupling = 0;
    MatNorm(blocks[1][0], NORM_FROBENIUS, &coupling);
    if (dolfinx::MPI::rank(comm) == 0)
    {
      std::cout << std::format(
          "Norm of block (1, 0), which no form fills: {:.6e}\n", coupling);
    }

    // ## The vector
    // One vector per block, over the extended index map of its constraint.
    // A slave's entry moves to its master, in the vector of the master's
    // block, and the ghost contributions are sent to their owners.
    std::array<la::Vector<T>, 2> b
        = {la::Vector<T>(mpcs[0]->function_space()->dofmap()->index_map, 1),
           la::Vector<T>(mpcs[1]->function_space()->dofmap()->index_map, 1)};
    std::vector<std::span<T>> b_spans = {b[0].array(), b[1].array()};
    for (std::size_t i = 0; i < 2; ++i)
    {
      dolfinx_mpc::assemble_vector_blocks<T, U>(b_spans, static_cast<int>(i),
                                                L[i], mpcs[i]);
    }
    for (la::Vector<T>& b_k : b)
      b_k.scatter_rev(std::plus<T>());

    // ## Solve
    // The unknowns live in the extended spaces of the constraints.
    std::array<fem::Function<T, U>, 2> u
        = {fem::Function<T, U>(mpcs[0]->function_space()),
           fem::Function<T, U>(mpcs[1]->function_space())};
    std::array<Vec, 2> b_vec = {la::petsc::create_vector_wrap(b[0]),
                                la::petsc::create_vector_wrap(b[1])};
    std::array<Vec, 2> u_vec = {la::petsc::create_vector_wrap(*u[0].x()),
                                la::petsc::create_vector_wrap(*u[1].x())};
    Vec b_nest, u_nest;
    VecCreateNest(comm, 2, nullptr, b_vec.data(), &b_nest);
    VecCreateNest(comm, 2, nullptr, u_vec.data(), &u_nest);

    KSP ksp;
    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    // The system is symmetric positive definite: K^T A K on the free dofs, the
    // identity on the slaves. PETSc's Jacobi preconditioner cannot work on a
    // nest matrix, so this small problem is solved unpreconditioned.
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    const U rtol
        = std::max(U(1e-10), U(100) * std::numeric_limits<U>::epsilon());
    KSPSetTolerances(ksp, rtol, PETSC_CURRENT, PETSC_CURRENT, 5000);
    KSPSolve(ksp, b_nest, u_nest);
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    PetscInt iterations;
    KSPGetIterationNumber(ksp, &iterations);
    if (reason <= 0)
    {
      throw std::runtime_error(std::format(
          "The linear solver did not converge: reason {} after {} iterations",
          static_cast<int>(reason), iterations));
    }

    // ## Back-substitution
    // A slave of `V` reads its master from the function of `Q`, so the
    // functions of both blocks are passed, with their ghosts up to date.
    for (fem::Function<T, U>& u_k : u)
      u_k.x()->scatter_fwd();
    std::vector<std::span<T>> u_spans = {u[0].x()->array(), u[1].x()->array()};
    for (const auto& mpc : mpcs)
      mpc->backsubstitution(u_spans);

    // The relation holds on every slave
    U error = 0;
    for (std::int32_t s : slaves)
      error = std::max(error, std::abs(u_spans[0][s] - c * u_spans[1][s]));
    U max_error = 0;
    MPI_Allreduce(&error, &max_error, 1, dolfinx::MPI::mpi_t<U>, MPI_MAX, comm);
    if (dolfinx::MPI::rank(comm) == 0)
    {
      std::cout << std::format("CG iterations: {}\n", iterations);
      std::cout << std::format("max |u - c p| on x = 1: {:.3e}\n", max_error);
    }
    if (max_error > 1e3 * std::numeric_limits<U>::epsilon())
      throw std::runtime_error("The constraint does not hold");

    // The solution does not depend on how the mesh is partitioned
    const U norm_u = la::norm(*u[0].x()), norm_p = la::norm(*u[1].x());
    if (dolfinx::MPI::rank(comm) == 0)
      std::cout << std::format("|u| = {:.10e}, |p| = {:.10e}\n", norm_u,
                               norm_p);

    KSPDestroy(&ksp);
    VecDestroy(&b_nest);
    VecDestroy(&u_nest);
    for (Vec& v : b_vec)
      VecDestroy(&v);
    for (Vec& v : u_vec)
      VecDestroy(&v);
    MatDestroy(&A);
  }
  PetscFinalize();
  return 0;
}

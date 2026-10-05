# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""0-D springs between spiders of two spider meshes, related by :func:`dolfinx_mpc.create_spider_pair`.

Spring `k` joins spider `k` of mesh A to spider `k` of mesh B, which start at the same point. The
spring is a form, :math:`\\int K (w_A - w_B) \\cdot (v_A - v_B)` over mesh A, with a symmetric 6x6
stiffness coupling translations and rotations. The tests assemble through PETSc, so they use its
scalar type.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import numpy as np
import pytest
import scipy.sparse
import ufl
from dolfinx import default_real_type, default_scalar_type, fem

import dolfinx_mpc
import dolfinx_mpc.utils

_tol = 1e3 * np.finfo(default_real_type).eps


def _springs():
    """Two spider meshes of the same points, the spaces on them, the pairing and the spring form."""
    comm = MPI.COMM_WORLD
    # More springs than processes, so that the spiders are spread over the processes
    num = 2 * comm.size + 1
    points = np.column_stack([np.linspace(1.0, 2.0, num), np.full(num, 0.5), np.full(num, 0.5)])
    points = points.astype(default_real_type)
    spiders_A = dolfinx_mpc.create_spider_mesh(comm, points)
    spiders_B = dolfinx_mpc.create_spider_mesh(comm, points)
    emap = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)

    element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type)
    W_A = fem.functionspace(spiders_A, element)
    W_B = fem.functionspace(spiders_B, element)

    # Symmetric positive definite, with translation-rotation coupling
    M = np.random.default_rng(3).standard_normal((6, 6))
    K = (M @ M.T + 6 * np.eye(6)).astype(default_scalar_type)

    W = ufl.MixedFunctionSpace(W_A, W_B)
    u_A, u_B = ufl.TrialFunctions(W)
    v_A, v_B = ufl.TestFunctions(W)
    a = ufl.inner(fem.Constant(spiders_A, K) * (u_A - u_B), v_A - v_B) * ufl.dx(spiders_A)
    return num, W_A, W_B, emap, K, a


def _constraints(W_A, W_B):
    mpcs = [
        dolfinx_mpc.MultiPointConstraint(W_A, dtype=default_scalar_type),
        dolfinx_mpc.MultiPointConstraint(W_B, dtype=default_scalar_type),
    ]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)
    return mpcs


def _dense(A):
    """The assembled matrix on the first process, `None` elsewhere."""
    if A.getType() == "nest":
        gather = dolfinx_mpc.utils.gather_PETScMatrix
        blocks = [[gather(A.getNestSubMatrix(i, j), root=0) for j in range(2)] for i in range(2)]
        return scipy.sparse.bmat(blocks).toarray() if MPI.COMM_WORLD.rank == 0 else None
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=0)
    return A_csr.toarray() if MPI.COMM_WORLD.rank == 0 else None


def test_spring_operator():
    """Both layouts assemble the operator DOLFINx does, which applies K (u_A - u_B) and its negative
    to the two ends of every spring, and leaves the joint motion of each pair free."""
    comm = MPI.COMM_WORLD
    num, W_A, W_B, emap, K, a = _springs()
    a_blocks = fem.form(ufl.extract_blocks(a), entity_maps=[emap], dtype=default_scalar_type)
    A_ref = dolfinx.fem.petsc.assemble_matrix(a_blocks, kind="nest")
    A_ref.assemble()
    mpcs = _constraints(W_A, W_B)
    A = {}
    for kind in ("nest", "mpi"):
        A[kind] = dolfinx_mpc.assemble_matrix(a_blocks, mpcs, kind=kind)
        A[kind].assemble()

    dense = [_dense(M) for M in (A_ref, A["nest"], A["mpi"])]
    if comm.rank == 0:
        values = [np.linalg.svd(D, compute_uv=False) for D in dense]
        scale = values[0].max()
        for v in values[1:]:
            np.testing.assert_allclose(v, values[0], atol=_tol * scale)
        assert np.linalg.matrix_rank(dense[0], tol=_tol * scale) == 6 * num

    # The action on random motions of the spiders, spring by spring
    rng = np.random.default_rng(5)
    u = [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]
    y = [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]
    for f in u:
        f.x.array[:] = rng.standard_normal(f.x.array.shape).astype(default_scalar_type)
        f.x.scatter_forward()
    x_nest = PETSc.Vec().createNest([f.x.petsc_vec for f in u])
    y_nest = PETSc.Vec().createNest([f.x.petsc_vec for f in y])
    A["nest"].mult(x_nest, y_nest)
    for f in y:
        f.x.scatter_forward()
    for k in range(num):
        u_A, u_B = (dolfinx_mpc.spider_values(f, k) for f in u)
        y_A, y_B = (dolfinx_mpc.spider_values(f, k) for f in y)
        expected = K @ (u_A - u_B)
        np.testing.assert_allclose(y_A, expected, atol=_tol * np.abs(K).max())
        np.testing.assert_allclose(y_B, -expected, atol=_tol * np.abs(K).max())


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_spring_solve(kind):
    """With every spider of A held fixed and a force and moment on each spider of B, spring k
    stretches by u_B = K^{-1} f_k."""
    num, W_A, W_B, emap, K, a = _springs()
    held = fem.locate_dofs_geometrical(W_A, lambda x: np.full(x.shape[1], True))
    bc = fem.dirichletbc(np.zeros(6, dtype=default_scalar_type), held, W_A)

    # A different load on each spring
    load = fem.Function(W_B, dtype=default_scalar_type)
    x_B = W_B.tabulate_dof_coordinates()
    values = np.zeros((len(x_B), 6), dtype=default_scalar_type)
    values[:, 1] = 2.0 * (x_B[:, 0] - 1.5)
    values[:, 2] = -1.0
    values[:, 3] = 0.5 * x_B[:, 0]
    load.x.array[:] = values.reshape(-1)
    v_A, v_B = ufl.TestFunction(W_A), ufl.TestFunction(W_B)
    L = [ufl.ZeroBaseForm((v_A,)), ufl.inner(load, v_B) * ufl.dx(W_B.mesh)]

    problem = dolfinx_mpc.LinearProblem(
        ufl.extract_blocks(a),
        L,
        _constraints(W_A, W_B),
        bcs=[bc],
        kind=kind,
        entity_maps=[emap],
        petsc_options_prefix=f"test_spider_spring_{kind}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    w_A, w_B = problem.solve()
    for k in range(num):
        expected = np.linalg.solve(K, dolfinx_mpc.spider_values(load, k))
        np.testing.assert_allclose(dolfinx_mpc.spider_values(w_A, k), 0.0, atol=_tol)
        np.testing.assert_allclose(dolfinx_mpc.spider_values(w_B, k), expected, atol=_tol * np.abs(expected).max())

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import numpy.testing as nt
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc
import dolfinx_mpc.utils


@pytest.mark.parametrize("u_from_mpc", [True, False])
def test_pipeline(u_from_mpc):
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    d = fem.Constant(mesh, default_scalar_type(0.08))
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - d * ufl.inner(u, v) * ufl.dx
    rhs = ufl.inner(f, v) * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(bilinear_form)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Create multipoint constraint
    def periodic_relation(x):
        out_x = np.copy(x)
        out_x[0] = 1 - x[0]
        return out_x

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
    arg_sort = np.argsort(facets)
    mt = meshtags(mesh, mesh.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, mt, 2, periodic_relation, [], 1.0)
    mpc.finalize()

    if u_from_mpc:
        uh = fem.Function(mpc.function_space)
        problem = dolfinx_mpc.LinearProblem(
            bilinear_form,
            linear_form,
            mpc,
            bcs=[],
            u=uh,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem.solve()

        root = 0
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, problem.A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, problem.b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
        scipy_dtype = np.complex128 if is_complex else np.float64
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
            reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K.astype(scipy_dtype) @ d
            nt.assert_allclose(
                uh_numpy.astype(u_mpc.dtype),
                u_mpc,
                rtol=500 * np.finfo(default_scalar_type).resolution,
                atol=500 * np.finfo(default_scalar_type).resolution,
            )
        L_org.destroy()
        A_org.destroy()

    else:
        uh = fem.Function(V)
        with pytest.raises(ValueError):
            problem = dolfinx_mpc.LinearProblem(
                bilinear_form,
                linear_form,
                mpc,
                bcs=[],
                u=uh,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )
            problem.solve()


def test_bc_markers_are_built_once(monkeypatch):
    """Repeated solves must not rebuild the dof markers.

    Markers and diagonal rows depend only on the function spaces and the
    conditions, both fixed for the lifetime of a `LinearProblem`, so they are
    computed on first use and reused thereafter. Rebuilding them per assembly
    is what the marker interface exists to avoid, and a Newton iteration would
    otherwise pay it on every Jacobian evaluation.
    """
    import dolfinx.fem.assemble as _dfa

    calls = {"n": 0}
    original = _dfa._bc_dof_markers

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_dfa, "_bc_dof_markers", counted)

    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    L = ufl.inner(fem.Constant(mesh, default_scalar_type(1.0)), v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0)))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.finalize()
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpc,
        bcs=[bc],
        petsc_options_prefix="test_marker_cache_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    # The cache fills on first use, then never again: one array per function
    # space, shared by the operator and the preconditioner.
    problem.solve()
    after_first_solve = calls["n"]
    assert after_first_solve > 0

    for _ in range(3):
        problem.solve()
    assert calls["n"] == after_first_solve


def test_dirichlet_values_are_reread_with_cached_markers():
    """The lifting markers are cached but the Dirichlet values must not be.

    With no source the solution is linear in the boundary data, so doubling the
    data must double the solution even though the second solve reuses the
    markers built by the first.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(mesh, default_scalar_type(0.0)), v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: 1 + x[1] * (1 - x[1]))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0)))

    # Periodic top to bottom, so the constraint is non-trivial
    def periodic_relation(x):
        out_x = x.copy()
        out_x[1] = 1 - x[1]
        return out_x

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(
        V, lambda x: np.isclose(x[1], 1.0) & (x[0] > 1e-12), periodic_relation, [bc]
    )
    mpc.finalize()
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpc,
        bcs=[bc],
        petsc_options_prefix="test_bc_values_reread_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    u1 = problem.solve().x.array.copy()
    assert np.linalg.norm(u1) > 0.5

    u_bc.x.array[:] *= 2
    u2 = problem.solve().x.array
    tol = 500 * np.finfo(default_scalar_type).eps
    nt.assert_allclose(u2, 2 * u1, rtol=tol, atol=tol)

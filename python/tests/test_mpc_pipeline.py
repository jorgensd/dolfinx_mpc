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
from dolfinx.common import Timer, list_timings
from dolfinx.mesh import create_unit_square

import dolfinx_mpc
import dolfinx_mpc.utils


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
def test_pipeline(master_point, get_assemblers):  # noqa: F811
    assemble_matrix, assemble_vector = get_assemblers

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    d = fem.Constant(mesh, default_scalar_type(1.5))
    c = fem.Constant(mesh, default_scalar_type(2))
    x = ufl.SpatialCoordinate(mesh)
    f = c * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    g = fem.Function(V)
    g.interpolate(lambda x: np.sin(x[0]) * x[1])
    h = fem.Function(V)
    h.interpolate(lambda x: 2 + x[1] * x[0])

    a = d * g * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = h * ufl.inner(f, v) * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(bilinear_form)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

    s_m_c = {
        l2b([1, 0]): {l2b([0, 1]): 0.43, l2b([1, 1]): 0.11},
        l2b([0, 0]): {l2b(master_point): 0.69},
    }

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    A = assemble_matrix(bilinear_form, mpc)
    b = assemble_vector(linear_form, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)

    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)

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
            )

    list_timings(comm)


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
def test_linearproblem(master_point):
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    d = fem.Constant(mesh, default_scalar_type(1.5))
    c = fem.Constant(mesh, default_scalar_type(2))
    x = ufl.SpatialCoordinate(mesh)
    f = c * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    g = fem.Function(V)
    g.interpolate(lambda x: np.sin(x[0]) * x[1])
    h = fem.Function(V)
    h.interpolate(lambda x: 2 + x[1] * x[0])

    a = d * g * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = h * ufl.inner(f, v) * ufl.dx
    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(fem.form(a))
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(fem.form(rhs))
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

    s_m_c = {
        l2b([1, 0]): {l2b([0, 1]): 0.43, l2b([1, 1]): 0.11},
        l2b([0, 0]): {l2b(master_point): 0.69},
    }

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    problem = dolfinx_mpc.LinearProblem(a, rhs, mpc, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
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
            )

    list_timings(comm)

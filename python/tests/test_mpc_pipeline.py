# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
def test_pipeline(master_point):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 5)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    d = dolfinx.Constant(mesh, 1.5)
    c = dolfinx.Constant(mesh, 2)
    x = ufl.SpatialCoordinate(mesh)
    f = c * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    g = dolfinx.Function(V)
    g.interpolate(lambda x: np.sin(x[0]) * x[1])
    h = dolfinx.Function(V)
    h.interpolate(lambda x: 2 + x[1] * x[0])

    a = d * g * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = h * ufl.inner(f, v) * ufl.dx
    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]):
             {l2b([0, 1]): 0.43,
              l2b([1, 1]): 0.11},
             l2b([0, 0]):
                 {l2b(master_point): 0.69}}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    A = dolfinx_mpc.assemble_matrix(a, mpc)
    b = dolfinx_mpc.assemble_vector(rhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)

    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
def test_linearproblem(master_point):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 5)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    d = dolfinx.Constant(mesh, 1.5)
    c = dolfinx.Constant(mesh, 2)
    x = ufl.SpatialCoordinate(mesh)
    f = c * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    g = dolfinx.Function(V)
    g.interpolate(lambda x: np.sin(x[0]) * x[1])
    h = dolfinx.Function(V)
    h.interpolate(lambda x: 2 + x[1] * x[0])

    a = d * g * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = h * ufl.inner(f, v) * ufl.dx
    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]):
             {l2b([0, 1]): 0.43,
              l2b([1, 1]): 0.11},
             l2b([0, 0]):
                 {l2b(master_point): 0.69}}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    problem = dolfinx_mpc.LinearProblem(a, rhs, mpc, bcs=[],
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    for i in range(2):
        uh = problem.solve()

    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, problem._A, mpc, root=root)
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, problem._b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.vector, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)

    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])

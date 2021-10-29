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


@pytest.mark.parametrize("Nx", [4])
@pytest.mark.parametrize("Ny", [2, 3])
@pytest.mark.parametrize("slave_space", [0, 1])
@pytest.mark.parametrize("master_space", [0, 1])
def test_vector_possion(Nx, Ny, slave_space, master_space):
    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, Nx, Ny)

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    def boundary(x):
        return np.isclose(x.T, [0, 0, 0]).all(axis=1)

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bdofsV = dolfinx.fem.locate_dofs_geometrical(V, boundary)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsV)
    bcs = [bc]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((-5 * x[1], 7 * x[0]))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(f, v) * ufl.dx

    # Setup LU solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]): {l2b([1, 1]): 0.1, l2b([0.5, 1]): 0.3}}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, slave_space, master_space)
    mpc.finalize()

    with dolfinx.common.Timer("~TEST: Assemble matrix"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    with dolfinx.common.Timer("~TEST: Assemble matrix (C++)"):
        A_cpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc, bcs=bcs)

    with dolfinx.common.Timer("~TEST: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

    with dolfinx.common.Timer("~TEST: Assemble vector (cached)"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    with dolfinx.common.Timer("~TEST: Assemble vector (C++)"):
        b_cpp = dolfinx_mpc.assemble_vector_cpp(rhs, mpc)
    dolfinx.fem.apply_lifting(b_cpp, [a], [bcs])
    b_cpp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_cpp, bcs)
    assert np.allclose(b.array, b_cpp.array)

    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)

    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    # Generate reference matrices for unconstrained problem
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()

    L_org = dolfinx.fem.assemble_vector(rhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(A_cpp, root=root)
        A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
        if MPI.COMM_WORLD.rank == root:
            dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A_cpp, mpc, root=root)
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

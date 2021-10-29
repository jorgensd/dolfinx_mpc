# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import scipy.sparse.linalg
import dolfinx
import dolfinx.io


def test_cell_domains():
    """
    Periodic MPC conditions over integral with different cell subdomains
    """
    N = 5
    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 15, N)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    def left_side(x):
        return x[0] < 0.5

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    values = []
    for cell_index in range(num_cells):
        if left_side(cell_midpoints[cell_index]):
            values.append(1)
        else:
            values.append(2)
    ct = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim,
                               np.arange(num_cells, dtype=np.int32),
                               np.array(values, dtype=np.intc))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    c1 = dolfinx.Constant(mesh, PETSc.ScalarType(2))
    c2 = dolfinx.Constant(mesh, PETSc.ScalarType(10))

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    a = c1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(1) +\
        c2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(2)\
        + 0.01 * ufl.inner(u, v) * dx(1)

    rhs = ufl.inner(x[1], v) * dx(1) + \
        ufl.inner(dolfinx.Constant(mesh, PETSc.ScalarType(1)), v) * dx(2)

    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()

    s_m_c = {}
    for i in range(0, N + 1):
        s_m_c[l2b([1, i / N])] = {l2b([0, i / N]): 1}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    # Setup MPC system
    with dolfinx.common.Timer("~TEST: Assemble matrix old"):
        A = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (cached)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (C++)"):
        Acpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc)

    with dolfinx.common.Timer("~TEST: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    with dolfinx.common.Timer("~TEST: Assemble vector (C++)"):
        b_cpp = dolfinx_mpc.assemble_vector_cpp(rhs, mpc)
    b_cpp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    assert np.allclose(b.array, b_cpp.array)

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

        A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(Acpp, root=root)
        A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
        if MPI.COMM_WORLD.rank == root:
            dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)
    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])

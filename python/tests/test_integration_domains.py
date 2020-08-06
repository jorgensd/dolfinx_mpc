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
                               range(num_cells),
                               np.array(values, dtype=np.intc))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    c1 = dolfinx.Constant(mesh, 2)
    c2 = dolfinx.Constant(mesh, 10)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    a = c1*ufl.inner(ufl.grad(u), ufl.grad(v))*dx(1) +\
        c2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(2)\
        + 0.01*ufl.inner(u, v)*dx(1)

    rhs = ufl.inner(x[1], v)*dx(1) + \
        ufl.inner(dolfinx.Constant(mesh, 1), v)*dx(2)

    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()

    s_m_c = {}
    for i in range(0, N+1):
        s_m_c[l2b([1, i/N])] = {l2b([0, i/N]): 1}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    # Setup MPC system
    with dolfinx.common.Timer("~Test: Assemble matrix"):
        A = dolfinx_mpc.assemble_matrix(a, mpc)

    with dolfinx.common.Timer("~Test: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    dolfinx_mpc.backsubstitution(mpc, uh)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)
    # Create reduced A
    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    # Created reduced L
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    # Solve linear system
    d = np.linalg.solve(reduced_A, reduced_L)
    # # Backsubstitution to full solution vector
    uh_numpy = np.dot(K, d)
    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, mpc)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, mpc)

    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

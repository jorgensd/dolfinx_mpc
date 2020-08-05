# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.io


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
    f = c*ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    g = dolfinx.Function(V)
    g.interpolate(lambda x: np.sin(x[0])*x[1])
    h = dolfinx.Function(V)
    h.interpolate(lambda x: 2+x[1]*x[0])

    a = d*g*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    rhs = h*ufl.inner(f, v)*ufl.dx
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

    mpc = dolfinx_mpc.create_dictionary_constraint(V, s_m_c)
    A = dolfinx_mpc.assemble_matrix(a, mpc)
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
    A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

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
    # Backsubstitution to full solution vector
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
    dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)

    # Move solution to global array
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                          uh.owner_range[1]])

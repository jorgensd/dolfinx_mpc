# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import time

import numpy as np
import pytest

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


dolfinx_mpc.utils.cache_numba(matrix=True, vector=True, backsubstitution=True)


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
    lhs = h*ufl.inner(f, v)*ufl.dx
    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    # Create MPC
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0]):
             {lambda x: dof_at(x, [0, 1]): 0.43,
              lambda x: dof_at(x, [1, 1]): 0.11},
             lambda x: dof_at(x, [0, 0]):
             {lambda x: dof_at(x, master_point): 0.69}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c)
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Setup MPC system
    start = time.time()
    A = dolfinx_mpc.assemble_matrix(a, mpc)
    end = time.time()
    print("Runtime: {0:.2e}".format(end-start))

    b = dolfinx_mpc.assemble_vector(lhs, mpc)
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

    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
                                                       masters, coeffs,
                                                       offsets)
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
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)

    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

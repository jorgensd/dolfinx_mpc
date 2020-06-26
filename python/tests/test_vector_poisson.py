# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import time
import pytest
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


dolfinx_mpc.utils.cache_numba(matrix=True, vector=True, backsubstitution=True)


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
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bdofsV = dolfinx.fem.locate_dofs_geometrical(V, boundary)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsV)
    bcs = [bc]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((-5*x[1], 7*x[0]))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(f, v)*ufl.dx

    # Create MPC
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0]): {lambda x: dof_at(x, [1, 1]): 0.1,
                                           lambda x: dof_at(x, [0.5, 1]): 0.3}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c,
                                                           slave_space,
                                                           master_space)

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Setup MPC system
    start = time.time()
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    end = time.time()
    print("Runtime: {0:.2e}".format(end-start))

    b = dolfinx_mpc.assemble_vector(lhs, mpc)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

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

    # Generate reference matrices for unconstrained problem
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
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

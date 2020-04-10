# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import time

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


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

    lhs = ufl.inner(x[1], v)*dx(1) + \
        ufl.inner(dolfinx.Constant(mesh, 1), v)*dx(2)

    # Generate reference matrices
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    # Create MPC (map all left dofs to right)
    dof_at = dolfinx_mpc.dof_close_to

    def slave_locater(i, N):
        return lambda x: dof_at(x, [1, float(i/N)])

    def master_locater(i, N):
        return lambda x: dof_at(x, [0, float(i/N)])

    s_m_c = {}
    for i in range(0, N+1):
        s_m_c[slave_locater(i, N)] = {master_locater(i, N): 1}

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
    # Create functionspace and function for mpc vector
    # Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
    #                                               mpc.mpc_dofmap())
    # Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # # Write solution to file
    # u_h = dolfinx.Function(Vmpc)
    # u_h.vector.setArray(uh.array)
    # u_h.name = "u_mpc"

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
    # # Backsubstitution to full solution vector
    uh_numpy = np.dot(K, d)
    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)

    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

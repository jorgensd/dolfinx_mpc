# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from petsc4py import PETSc
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl

dolfinx_mpc.utils.cache_numba(matrix=True, vector=True, backsubstitution=True)


def test_surface_integrals():
    N = 4
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, N, N)
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Fixed Dirichlet BC on the left wall
    def left_wall(x):
        return np.isclose(x[0], np.finfo(float).eps)

    fdim = mesh.topology.dim - 1
    left_facets = dolfinx.mesh.compute_marked_boundary_entities(mesh,
                                                                fdim,
                                                                left_wall)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, 1, left_facets)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = dolfinx.fem.DirichletBC(u_bc, bc_dofs)
    bcs = [bc]

    # Traction on top of domain
    def top(x):
        return np.isclose(x[1], 1)
    mf = dolfinx.MeshFunction("size_t", mesh, fdim, 0)
    mf.mark(top, 3)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
    g = dolfinx.Constant(mesh, (0, -9.81e2))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create MPC (y-displacement of right wall is constant)
    dof_at = dolfinx_mpc.dof_close_to

    def slave_locater(i, N):
        return lambda x: dof_at(x, [1, float(i/N)])

    s_m_c = {}
    for i in range(1, N):
        s_m_c[slave_locater(i, N)] = {lambda x: dof_at(x, [1, 1]): 1}

    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c,
                                                           1, 1)
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc"
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "uh.xdmf").write(u_h)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

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
    # Back substitution to full solution vector
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    if uh.owner_range[0] < masters[0] and masters[0] < uh.owner_range[1]:
        print("MASTER DOF, MPC {0:.4e}"
              .format(uh.array[masters[0]-uh.owner_range[0]]))
        print("Slave (given as master*coeff) {0:.4e}".
              format(uh.array[masters[0]-uh.owner_range[0]]*coeffs[0]))

    if uh.owner_range[0] < slaves[0] and slaves[0] < uh.owner_range[1]:
        print("SLAVE  DOF, MPC {0:.4e}"
              .format(uh.array[slaves[0]-uh.owner_range[0]]))
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

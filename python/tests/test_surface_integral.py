# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

import dolfinx.fem as fem
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx import default_scalar_type
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx_mpc
import dolfinx_mpc.utils
from dolfinx_mpc.utils import get_assemblers  # noqa: F401


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
def test_surface_integrals(get_assemblers):  # noqa: F811

    assemble_matrix, assemble_vector = get_assemblers

    N = 4
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))

    # Fixed Dirichlet BC on the left wall
    def left_wall(x):
        return np.isclose(x[0], 0, atol=100 * np.finfo(x.dtype).resolution)

    fdim = mesh.topology.dim - 1
    left_facets = locate_entities_boundary(mesh, fdim, left_wall)
    bc_dofs = fem.locate_dofs_topological(V, 1, left_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0
    bc = fem.dirichletbc(u_bc, bc_dofs)
    bcs = [bc]

    # Traction on top of domain
    def top(x):
        return np.isclose(x[1], 1, atol=100 * np.finfo(x.dtype).resolution)
    top_facets = locate_entities_boundary(mesh, 1, top)
    arg_sort = np.argsort(top_facets)
    mt = meshtags(mesh, fdim, top_facets[arg_sort], np.full(len(top_facets), 3, dtype=np.int32))

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)
    g = fem.Constant(mesh, default_scalar_type((0, -9.81e2)))

    # Elasticity parameters
    E = 1.0e2
    nu = 0.0
    mu = fem.Constant(mesh, default_scalar_type(E / (2.0 * (1.0 + nu))))
    lmbda = fem.Constant(mesh, default_scalar_type(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v))
                + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(fem.Constant(mesh, default_scalar_type((0, 0))), v) * ufl.dx\
        + ufl.inner(g, v) * ds
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Setup LU solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Setup multipointconstraint
    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()
    s_m_c = {}
    for i in range(1, N):
        s_m_c[l2b([1, i / N])] = {l2b([1, 1]): 0.8}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, 1, 1)
    mpc.finalize()

    with Timer("~TEST: Assemble matrix old"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    with Timer("~TEST: Assemble vector"):
        b = assemble_vector(linear_form, mpc)

    dolfinx_mpc.apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver.setOperators(A)
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    A_org = fem.petsc.assemble_matrix(bilinear_form, bcs)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(L_org, bcs)

    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
        # Gather LHS, RHS and solution on one process
        is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
        scipy_dtype = np.complex128 if is_complex else np.float64
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.vector, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
            reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K.astype(scipy_dtype) @ d
            assert np.allclose(uh_numpy.astype(u_mpc.dtype), u_mpc, rtol=500
                               * np.finfo(default_scalar_type).resolution)

    L_org.destroy()
    b.destroy()
    list_timings(comm, [TimingType.wall])


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
def test_surface_integral_dependency(get_assemblers):  # noqa: F811

    assemble_matrix, assemble_vector = get_assemblers
    N = 10
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    def top(x):
        return np.isclose(x[1], 1)
    fdim = mesh.topology.dim - 1
    top_facets = locate_entities_boundary(mesh, fdim, top)

    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)
    markers = {3: top_facets}
    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key, dtype=np.intc))
    sort = np.argsort(indices)
    mt = meshtags(mesh, mesh.topology.dim - 1, np.array(indices[sort], dtype=np.intc),
                  np.array(values[sort], dtype=np.intc))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    g = fem.Constant(mesh, default_scalar_type((2, 1)))
    h = fem.Constant(mesh, default_scalar_type((3, 2)))
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ds(3) + ufl.inner(ufl.grad(u), ufl.grad(v)) * ds
    rhs = ufl.inner(g, v) * ds + ufl.inner(h, v) * ds(3)
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Create multipoint constraint and assemble system
    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()
    s_m_c = {}
    for i in range(1, N):
        s_m_c[l2b([1, i / N])] = {l2b([1, 1]): 0.3}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, 1, 1)
    mpc.finalize()
    with Timer("~TEST: Assemble matrix"):
        A = assemble_matrix(bilinear_form, mpc)
    with Timer("~TEST: Assemble vector"):
        b = assemble_vector(linear_form, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = fem.petsc.assemble_matrix(bilinear_form)
    A_org.assemble()

    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)
    L_org.destroy()
    b.destroy()
    list_timings(comm, [TimingType.wall])

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import numpy as np
import numpy.testing as nt
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx import default_scalar_type
from dolfinx.common import Timer, list_timings
from dolfinx.mesh import create_unit_square

import dolfinx_mpc
import dolfinx_mpc.utils


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("Nx", [4])
@pytest.mark.parametrize("Ny", [2, 3])
@pytest.mark.parametrize("slave_space", [0, 1])
@pytest.mark.parametrize("master_space", [0, 1])
def test_vector_possion(Nx, Ny, slave_space, master_space, get_assemblers):  # noqa: F811
    assemble_matrix, assemble_vector = get_assemblers
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny)

    V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    def boundary(x):
        return np.isclose(x.T, [0, 0, 0], atol=500 * np.finfo(x.dtype).resolution).all(axis=1)

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = fem.Function(V)
    with u_bc.x.petsc_vec.localForm() as u_local:
        u_local.set(0.0)
    u_bc.x.petsc_vec.destroy()

    bdofsV = fem.locate_dofs_geometrical(V, boundary)
    bc = fem.dirichletbc(u_bc, bdofsV)
    bcs = [bc]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((-5 * x[1], 7 * x[0]))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(f, v) * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Setup LU solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # Create multipoint constraint
    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

    s_m_c = {l2b([1, 0]): {l2b([1, 1]): 0.1, l2b([0.5, 1]): 0.3}}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, slave_space, master_space)
    mpc.finalize()

    with Timer("~TEST: Assemble matrix"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    with Timer("~TEST: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(linear_form, mpc)

    dolfinx_mpc.apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver.setOperators(A)
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)

    # Generate reference matrices for unconstrained problem
    A_org = fem.petsc.assemble_matrix(bilinear_form, bcs)
    A_org.assemble()

    L_org = fem.petsc.assemble_vector(linear_form)
    fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(L_org, bcs)

    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
        scipy_dtype = np.complex128 if is_complex else np.float64
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
            reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K.astype(scipy_dtype) @ d
            nt.assert_allclose(
                uh_numpy.astype(u_mpc.dtype),
                u_mpc,
                rtol=500 * np.finfo(default_scalar_type).resolution,
            )

    b.destroy()
    L_org.destroy()
    solver.destroy()
    list_timings(comm)

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as fem
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import create_unit_square
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc


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
        return np.isclose(x.T, [0, 0, 0]).all(axis=1)

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    u_bc.vector.destroy()

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

    solver.solve(b, uh.vector)
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
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.vector, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)
    b.destroy()
    L_org.destroy()
    list_timings(comm, [TimingType.wall])

# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from dolfinx import fem
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import CellType, create_unit_square
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1,
                    reason="This test should only be run in serial.")
@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
def test_lifting(get_assemblers):  # noqa: F811
    """
    Test MPC lifting operation on a single cell
    """
    assemble_matrix, assemble_vector = get_assemblers

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.quadrilateral)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = x[1] * ufl.sin(2 * ufl.pi * x[0])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(f, v) * ufl.dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Create Dirichlet boundary condition
    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(2.3)
    u_bc.vector.destroy()

    def dirichletboundary(x):
        return np.isclose(x[0], 1)

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = fem.locate_dofs_geometrical(V, dirichletboundary)
    bc = fem.dirichletbc(u_bc, geometrical_dofs)
    bcs = [bc]

    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(L_org, bcs)

    # Create multipoint constraint

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([0, 0]): {l2b([0, 1]): 1}}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    b = assemble_vector(linear_form, mpc)
    dolfinx_mpc.apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    fem.petsc.set_bc(b, bcs)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    # Solve
    uh = fem.Function(mpc.function_space)
    uh.x.set(0)
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)

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
        # constants = dolfinx_mpc.utils.gather_contants(mpc, root=root)
        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ (L_np)  # - constants)
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vecto
            uh_numpy = K @ (d)  # + constants)
            assert np.allclose(uh_numpy, u_mpc)

    list_timings(comm, [TimingType.wall])

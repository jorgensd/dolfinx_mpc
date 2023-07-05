# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT


from dolfinx import fem
import dolfinx_mpc
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import compute_midpoints, create_unit_square, meshtags
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
def test_cell_domains(get_assemblers):  # noqa: F811
    """
    Periodic MPC conditions over integral with different cell subdomains
    """
    assemble_matrix, assemble_vector = get_assemblers
    N = 5
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 15, N)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    def left_side(x):
        return x[0] < 0.5

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = compute_midpoints(mesh, tdim, range(num_cells))
    values = np.ones(num_cells, dtype=np.intc)
    # All cells on right side marked one, all other with 1
    values += left_side(cell_midpoints.T)
    ct = meshtags(mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32), values)

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    c1 = fem.Constant(mesh, PETSc.ScalarType(2))
    c2 = fem.Constant(mesh, PETSc.ScalarType(10))

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    a = c1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(1) +\
        c2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(2)\
        + 0.01 * ufl.inner(u, v) * dx(1)

    rhs = ufl.inner(x[1], v) * dx(1) + \
        ufl.inner(fem.Constant(mesh, PETSc.ScalarType(1)), v) * dx(2)
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(bilinear_form)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()

    s_m_c = {}
    for i in range(0, N + 1):
        s_m_c[l2b([1, i / N])] = {l2b([0, i / N]): 1}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    # Setup MPC system
    with Timer("~TEST: Assemble matrix old"):
        A = assemble_matrix(bilinear_form, mpc)
    with Timer("~TEST: Assemble vector"):
        b = assemble_vector(linear_form, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0
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

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)
    uh.vector.destroy()
    list_timings(comm, [TimingType.wall])

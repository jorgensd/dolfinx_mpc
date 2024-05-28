# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import numpy.testing as nt
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import compute_midpoints, create_unit_square, meshtags

import dolfinx_mpc
from dolfinx_mpc.utils import get_assemblers  # noqa: F401


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
def test_cell_domains(get_assemblers):  # noqa: F811
    """
    Periodic MPC conditions over integral with different cell subdomains
    """
    assemble_matrix, assemble_vector = get_assemblers
    N = 5
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 15, N)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    def left_side(x):
        return x[0] < 0.5

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    mesh.topology.create_connectivity(tdim, tdim)
    cell_midpoints = compute_midpoints(mesh, tdim, cells)
    values = np.ones_like(cells)
    # All cells on right side marked one, all other with 1
    values += left_side(cell_midpoints.T)
    ct = meshtags(mesh, mesh.topology.dim, cells, values)

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    c1 = fem.Constant(mesh, default_scalar_type(2))
    c2 = fem.Constant(mesh, default_scalar_type(10))

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    a = (
        c1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(1)
        + c2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(2)
        + 0.87 * ufl.inner(u, v) * dx(1)
    )

    rhs = ufl.inner(x[1], v) * dx(1) + ufl.inner(fem.Constant(mesh, default_scalar_type(1)), v) * dx(2)
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Generate reference matrices
    A_org = fem.petsc.assemble_matrix(bilinear_form)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

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

    solver = PETSc.KSP().create(mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)

    root = 0
    comm = mesh.comm
    is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process (work with high precision )
        scipy_dtype = np.complex128 if is_complex else np.float64
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
            reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK.astype(scipy_dtype), reduced_L.astype(scipy_dtype))
            # Back substitution to full solution vector
            uh_numpy = K.astype(scipy_dtype) @ d.astype(scipy_dtype)
            nt.assert_allclose(
                uh_numpy.astype(u_mpc.dtype),
                u_mpc,
                rtol=500 * np.finfo(default_scalar_type).resolution,
            )
    solver.destroy()
    b.destroy()
    del uh
    A.destroy()
    pc.destroy()
    L_org.destroy()
    A_org.destroy()
    list_timings(comm, [TimingType.wall])

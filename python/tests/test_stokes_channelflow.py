# Copyright (C) 2026 Stein K.F. Stoter
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, fem, la, mesh
from dolfinx.mesh import CellType, create_unit_cube

import dolfinx_mpc


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("els", [2, 4, 8])
@pytest.mark.parametrize("order", [2, 3])
def test_stokes_channelflow(cell_type, els, order):
    """Test 3D Stokes channel flow with periodic BCs and nested assembly and Taylor-Hood elements.

    The analytical solution for Poiseuille flow between parallel plates is:
    u_x = (dp/dx) * (1/(2*mu)) * y * (H - y)
    where H is the channel height, dp/dx is the pressure gradient enforced as a body force, mu is viscosity.
    """

    # Cube mesh
    rtype = default_real_type
    domain = create_unit_cube(MPI.COMM_WORLD, els, els, els, cell_type=cell_type, dtype=rtype)

    # Function spaces: Taylor-Hood
    V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), order, shape=(3,), dtype=rtype))
    Q = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), order - 1, dtype=rtype))

    # Wall BCs (top/bottom y={0,1})
    wall_facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
    )
    wall_dofs = fem.locate_dofs_topological(V, 2, wall_facets)
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), wall_dofs, V)

    # Periodicity definition (x and z directions)
    def periodic_boundary(x):
        return np.logical_or(np.isclose(x[0], 1), np.isclose(x[2], 1))

    def periodic_map(x):
        out = x.copy()
        out[0][np.isclose(x[0], 1).nonzero()] -= 1
        out[2][np.isclose(x[2], 1).nonzero()] -= 1
        return out

    # Create MPCs for velocity and pressure
    mpc_u = dolfinx_mpc.MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_map, [bc])
    mpc_u.finalize()
    mpc_p = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_p.create_periodic_constraint_geometrical(Q, periodic_boundary, periodic_map, [])
    mpc_p.finalize()

    # Stokes weak form
    u = ufl.TrialFunction(V)
    p = ufl.TrialFunction(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    # Viscosity and forcing
    mu = fem.Constant(domain, default_scalar_type(1.0))
    dp_dx = fem.Constant(domain, default_scalar_type(1.0))  # Pressure gradient in x-direction
    f = ufl.as_vector([dp_dx, 0.0, 0.0])

    # Weak statement
    a00 = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a01 = -ufl.inner(p, ufl.div(v)) * ufl.dx
    a10 = -ufl.inner(ufl.div(u), q) * ufl.dx
    L0 = ufl.inner(f, v) * ufl.dx
    L1 = ufl.ZeroBaseForm((q,))

    forms = fem.form([[a00, a01], [a10, None]])
    rhs_forms = fem.form([L0, L1])

    # Assemble - THIS IS WHERE THE BUG OCCURS with MPI and hexahedral cells
    A = dolfinx_mpc.create_matrix_nest(forms, constraints=[mpc_u, mpc_p])
    dolfinx_mpc.assemble_matrix_nest(A, forms, constraints=[mpc_u, mpc_p], bcs=[bc])
    A.assemble()

    b = dolfinx_mpc.create_vector_nest(rhs_forms, constraints=[mpc_u, mpc_p])
    dolfinx_mpc.assemble_vector_nest(b, rhs_forms, constraints=[mpc_u, mpc_p])

    # Apply lifting for boundary conditions
    bcs_block = fem.bcs_by_block(fem.extract_function_spaces(forms, 1), [bc])
    dolfinx_mpc.apply_lifting(b, forms, bcs_block, constraint=[mpc_u, mpc_p])

    # Ghost update after lifting
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set BC values in RHS
    bcs0 = fem.bcs_by_block(fem.extract_function_spaces(rhs_forms), [bc])
    fem.petsc.set_bc(b, bcs0)

    # Setup Minres solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setErrorIfNotConverged(True)
    ksp.setType("minres")
    rtol = 100 * np.finfo(rtype).eps
    atol = 2 * np.finfo(rtype).eps
    ksp.setTolerances(rtol=rtol, atol=atol)

    # Setup preconditioning
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    nested_IS = A.getNestISs()
    pc.setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))
    ksp_u, ksp_p = pc.getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("lu")
    ksp_u.getPC().setFactorSolverType("mumps")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Finalize KSP setup
    ksp.setFromOptions()

    # Create solution functions
    uh = fem.Function(mpc_u.function_space)
    ph = fem.Function(mpc_p.function_space)

    # Solve the system
    x = b.duplicate()
    ksp.solve(b, x)
    la.petsc._ghost_update(
        x,
        insert_mode=PETSc.InsertMode.INSERT,  # type: ignore
        scatter_mode=PETSc.ScatterMode.FORWARD,  # type: ignore
    )  # type: ignore
    fem.petsc.assign(x, [uh, ph])

    # Apply backsubstitution for MPCs
    mpc_u.backsubstitution(uh)
    mpc_p.backsubstitution(ph)

    # Verify solution: Poiseuille flow u_x = (dp/dx) * (1/(2*mu)) * y * (H - y)
    # For our parameters: mu=1, dp/dx=-1, H=1
    # u_x = -(-1) * (1/2) * y * (1 - y) = 0.5 * y * (1 - y)
    # Maximum velocity at y=0.5: u_max = 0.5 * 0.5 * 0.5 = 0.125

    # Create analytical solution function
    def u_exact_expr(x):
        """Exact solution for Poiseuille flow"""
        values = np.zeros((3, x.shape[1]))
        values[0] = 0.5 * x[1] * (1.0 - x[1])  # u_x component
        return values

    u_exact = fem.Function(V)
    u_exact.interpolate(u_exact_expr)

    # Compute error
    error_u = uh.x.array - u_exact.x.array
    error_norm = np.linalg.norm(error_u)
    error_norm_global = domain.comm.allreduce(error_norm**2, op=MPI.SUM) ** 0.5

    assert error_norm_global < np.sqrt(rtol), f"Velocity error too high: {error_norm_global}"

    # Cleanup
    ksp.destroy()
    A.destroy()
    b.destroy()
    x.destroy()

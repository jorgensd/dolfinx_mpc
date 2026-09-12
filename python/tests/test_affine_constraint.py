# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Tests for affine multi point constraints, i.e. x = K x_red + g."""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import numpy.testing as nt
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import create_unit_square

import dolfinx_mpc
import dolfinx_mpc.utils


def _l2b(li, mesh):
    return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()


def _poisson_forms(V, mesh):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(x[1] * ufl.sin(2 * ufl.pi * x[0]) + 1, v) * ufl.dx
    return fem.form(a), fem.form(rhs)


def _solve_mpc(bilinear_form, linear_form, mpc, bcs, mesh):
    """Assemble and solve the reduced system, then backsubstitute."""
    A = dolfinx_mpc.assemble_matrix(bilinear_form, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(linear_form, mpc)
    dolfinx_mpc.apply_lifting(b, [bilinear_form], [bcs], mpc)
    dolfinx_mpc.apply_mpc_lifting(b, [bilinear_form], constraint=mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = fem.Function(mpc.function_space)
    uh.x.array[:] = 0
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)
    solver.destroy()
    return A, b, uh


def _reference(bilinear_form, linear_form, mpc, bcs, uh, root=0):
    """Compare against an explicit K^T A K x_red = K^T (b - A g) solve."""
    A_org = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A_org.assemble()
    L_org = fem.petsc.assemble_vector(linear_form)
    fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(L_org, bcs)

    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
    g = dolfinx_mpc.utils.gather_constants(mpc, root=root)
    u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)

    if MPI.COMM_WORLD.rank == root:
        KTAK = np.conj(K.T) * A_csr * K
        reduced = np.conj(K.T) @ (L_np - A_csr @ g)
        d = scipy.sparse.linalg.spsolve(KTAK, reduced)
        nt.assert_allclose(K @ d + g, u_mpc, rtol=1e-6, atol=1e-10)
    A_org.destroy()
    L_org.destroy()


def test_dirichlet_master_is_eliminated():
    """A master constrained by a Dirichlet condition is folded into the offset."""
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 3.0
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bc = fem.dirichletbc(u_bc, dofs)

    # The master at (1, 0) lies on the Dirichlet boundary x = 1
    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=[bc])
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    assert mpc.has_inhomogeneity
    for slave in mpc.slaves[: mpc.num_local_slaves]:
        # The only master was Dirichlet constrained, so it has been removed and
        # its contribution 2.0 * 3.0 folded into the offset
        assert len(mpc.masters.links(slave)) == 0
        nt.assert_allclose(mpc.constants[slave], 6.0)


def test_update_constants_tracks_time_dependent_data():
    """Changing the value of a Dirichlet condition is picked up by update_constants."""
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 3.0
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bc = fem.dirichletbc(u_bc, dofs)

    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=[bc])
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    for value in (5.0, -1.5, 0.0):
        u_bc.x.array[:] = value
        mpc.update_constants()
        for slave in mpc.slaves[: mpc.num_local_slaves]:
            nt.assert_allclose(mpc.constants[slave], 2.0 * value)


def test_homogeneous_constraint_is_unchanged():
    """Without bcs or rhs_coeffs the constraint is exactly the linear one."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    assert not mpc.has_inhomogeneity
    nt.assert_allclose(mpc.constants, 0.0)


def test_affine_solve_with_dirichlet_master():
    """End-to-end solve where a master carries a Dirichlet condition."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    bilinear_form, linear_form = _poisson_forms(V, mesh)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 2.3
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=bcs)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    A, b, uh = _solve_mpc(bilinear_form, linear_form, mpc, bcs, mesh)
    _reference(bilinear_form, linear_form, mpc, bcs, uh)
    A.destroy()
    b.destroy()


def test_affine_solve_with_rhs_coeffs():
    """End-to-end solve for an explicitly inhomogeneous constraint u_s = c u_m + g_s."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    bilinear_form, linear_form = _poisson_forms(V, mesh)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 1.0
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    # Slave and master are both away from the Dirichlet boundary
    s_m_c = {_l2b([0, 0], mesh): {_l2b([0, 1], mesh): 0.5}}

    g = fem.Function(V)
    g.x.array[:] = 0.0
    slave_dof = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    g.x.array[slave_dof] = 0.75
    g.x.scatter_forward()

    mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    assert mpc.has_inhomogeneity
    A, b, uh = _solve_mpc(bilinear_form, linear_form, mpc, bcs, mesh)
    _reference(bilinear_form, linear_form, mpc, bcs, uh)
    A.destroy()
    b.destroy()


def test_linear_problem_affine():
    """LinearProblem applies the constraint offset without an explicit lifting call."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(x[1] * ufl.sin(2 * ufl.pi * x[0]) + 1, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 2.3
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=bcs)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    problem = dolfinx_mpc.LinearProblem(
        a,
        rhs,
        mpc,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    bilinear_form, linear_form = fem.form(a), fem.form(rhs)
    _reference(bilinear_form, linear_form, mpc, bcs, uh)

    # The constraint must hold in the solution: u_slave = 2.0 * u_bc = 4.6
    for slave in mpc.slaves[: mpc.num_local_slaves]:
        nt.assert_allclose(uh.x.array[slave], 4.6, rtol=1e-6)


@pytest.mark.skipif(default_scalar_type != np.float64, reason="Numba assemblers are only built for float64 here")
def test_numba_rejects_inhomogeneous_constraint():
    """The numba assemblers refuse an affine constraint rather than silently ignoring g."""
    numba = pytest.importorskip("numba")  # noqa: F841
    from dolfinx_mpc.numba import assemble_matrix as numba_assemble_matrix

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    bilinear_form, _ = _poisson_forms(V, mesh)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 3.0
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=bcs)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    with pytest.raises(NotImplementedError):
        numba_assemble_matrix(bilinear_form, mpc, bcs=bcs)


def test_update_constants_tracks_rhs_coeffs():
    """Changing the rhs_coeffs Function is picked up by update_constants."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    slave_dof = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    g = fem.Function(V)
    g.x.array[:] = 0.0
    g.x.array[slave_dof] = 0.75
    g.x.scatter_forward()

    s_m_c = {_l2b([0, 0], mesh): {_l2b([0, 1], mesh): 0.5}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()

    for slave in mpc.slaves[: mpc.num_local_slaves]:
        nt.assert_allclose(mpc.constants[slave], 0.75)

    g.x.array[slave_dof] = -2.25
    g.x.scatter_forward()
    mpc.update_constants()
    for slave in mpc.slaves[: mpc.num_local_slaves]:
        nt.assert_allclose(mpc.constants[slave], -2.25)


def test_affine_blocked_problem():
    """A blocked problem where one block carries an affine constraint."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    Q = fem.functionspace(mesh, ("Lagrange", 1))

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    x = ufl.SpatialCoordinate(mesh)

    a = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(p, v) * ufl.dx],
        [ufl.inner(u, q) * ufl.dx, ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx],
    ]
    L = [ufl.inner(x[1] + 1, v) * ufl.dx, ufl.inner(x[0] + 1, q) * ufl.dx]

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 1.7
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    # The master at (1, 0) lies on the Dirichlet boundary of V
    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc_u = dolfinx_mpc.MultiPointConstraint(V, bcs=bcs)
    mpc_u.create_general_constraint(s_m_c)
    mpc_u.finalize()

    mpc_p = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_p.finalize()

    assert mpc_u.has_inhomogeneity
    assert not mpc_p.has_inhomogeneity

    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        [mpc_u, mpc_p],
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    # The constraint must hold in the solution: u_slave = 2.0 * 1.7 = 3.4
    for slave in mpc_u.slaves[: mpc_u.num_local_slaves]:
        nt.assert_allclose(uh[0].x.array[slave], 3.4, rtol=1e-6)

    # Because the only master is Dirichlet constrained, the relation collapses to the
    # fixed value u_slave = 3.4. Solving the same blocked problem with that stated as an
    # ordinary Dirichlet condition and no constraint must give the same solution, which
    # exercises the -K^T A g term and the pairing of each block with its column
    # constraint.
    slave_dof = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    u_pin = fem.Function(V)
    u_pin.x.array[:] = 3.4
    bcs_ref = [fem.dirichletbc(u_bc, dofs), fem.dirichletbc(u_pin, slave_dof)]
    ref_problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=bcs_ref,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="affine_blocked_reference_",
    )
    u_ref = ref_problem.solve()

    # Compare the owned blocks only: the constraint's function space carries extra
    # ghosts for off-process masters, so `uh` is longer than `u_ref` in parallel.
    # Reduce the error before asserting, so that every process reaches the same
    # verdict rather than one failing while the others wait in a collective.
    n_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    n_p = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    err_u = np.max(np.abs(uh[0].x.array[:n_u] - u_ref[0].x.array[:n_u])) if n_u else 0.0
    err_p = np.max(np.abs(uh[1].x.array[:n_p] - u_ref[1].x.array[:n_p])) if n_p else 0.0
    err = mesh.comm.allreduce(max(float(err_u), float(err_p)), op=MPI.MAX)
    assert err < 1e-9, f"blocked affine solution differs from the reference by {err}"


def test_slave_that_is_also_dirichlet_is_rejected():
    """A dof cannot be prescribed by both the constraint and a Dirichlet condition."""
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 1.0
    # The Dirichlet boundary x = 0 contains the slave at (0, 0)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    bcs = [fem.dirichletbc(u_bc, dofs)]

    s_m_c = {_l2b([0, 0], mesh): {_l2b([1, 0], mesh): 2.0}}
    mpc = dolfinx_mpc.MultiPointConstraint(V, bcs=bcs)
    mpc.create_general_constraint(s_m_c)
    with pytest.raises(Exception, match="both a slave"):
        mpc.finalize()


@pytest.mark.parametrize("degree", [1, 2])
def test_dirichletbc_as_pure_mpc(degree):
    """A Dirichlet condition expressed entirely as an affine constraint.

    A constrained dof is added as a slave with an *empty* master list and an offset equal
    to the boundary value, so the relation degenerates to :math:`u_s = g_s`. No
    `DirichletBC` is given to the assembler at all. The result must match what plain
    DOLFINx produces for the same problem with an ordinary `DirichletBC`.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", degree))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    rhs = ufl.inner(x[1] * ufl.sin(2 * ufl.pi * x[0]) + 1, v) * ufl.dx

    # A spatially varying condition on x = 0 and x = 1
    def boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: 1.0 + x[0] + 2.0 * x[1] ** 2)
    bc_dofs = fem.locate_dofs_geometrical(V, boundary)

    # Reference: ordinary DOLFINx with a DirichletBC
    ref = dolfinx.fem.petsc.LinearProblem(
        a,
        rhs,
        bcs=[fem.dirichletbc(u_bc, bc_dofs)],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="dirichlet_as_mpc_reference_",
    )
    u_ref = ref.solve()

    # The same condition as a constraint. Ghost slaves must be declared too: a process
    # that only ghosts a constrained dof still assembles cells touching it, and would
    # otherwise not eliminate it. `locate_dofs_geometrical` returns owned and ghost dofs,
    # which is exactly what is needed here.
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    slaves = np.sort(bc_dofs).astype(np.int32)

    g = fem.Function(V)
    g.x.array[:] = 0.0
    g.x.array[slaves] = u_bc.x.array[slaves]
    g.x.scatter_forward()

    mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
    mpc.add_constraint(
        V,
        slaves,
        np.array([], dtype=np.int64),  # no masters
        np.array([], dtype=default_scalar_type),
        np.array([], dtype=np.int32),
        np.zeros(len(slaves) + 1, dtype=np.int32),
    )
    mpc.finalize()

    # Every slave is a pure offset, and no Dirichlet condition reaches the assembler
    assert mpc.has_inhomogeneity
    for slave in mpc.slaves[: mpc.num_local_slaves]:
        assert len(mpc.masters.links(slave)) == 0

    problem = dolfinx_mpc.LinearProblem(
        a,
        rhs,
        mpc,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    # The constraint must reproduce the boundary data exactly
    nt.assert_allclose(uh.x.array[slaves], u_bc.x.array[slaves], rtol=1e-10, atol=1e-12)

    # ... and the whole solution must match the ordinary DirichletBC solve. Compare the
    # owned block only, since the constraint's space carries extra ghosts in parallel,
    # and reduce before asserting so that every process reaches the same verdict.
    err = np.max(np.abs(uh.x.array[:num_owned] - u_ref.x.array[:num_owned])) if num_owned else 0.0
    err = mesh.comm.allreduce(float(err), op=MPI.MAX)
    assert err < 1e-9, f"MPC-as-DirichletBC solution differs from DOLFINx by {err}"


@pytest.mark.skipif(
    np.issubdtype(default_scalar_type, np.complexfloating),
    reason="The nonlinear residual used here is real valued.",
)
@pytest.mark.parametrize("degree", [1, 2])
def test_nonlinear_dirichletbc_as_pure_mpc(degree):
    """A Dirichlet condition on a *nonlinear* problem, expressed purely as a constraint.

    For :math:`F(u) = 0` the reduced problem is :math:`K^T F(K \\hat{u} + g) = 0` with
    Jacobian :math:`K^T J K`. Because the residual is assembled at an iterate that already
    satisfies :math:`u = K \\hat{u} + g`, the offset enters through :math:`F` itself and no
    separate lifting of :math:`g` is required — unlike the linear path. This test pins that
    down by solving the same problem with an ordinary `DirichletBC` and comparing.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", degree))

    def boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: 0.5 + 0.25 * x[1])
    bc_dofs = fem.locate_dofs_geometrical(V, boundary)

    x = ufl.SpatialCoordinate(mesh)
    source = 2.0 + ufl.sin(2 * ufl.pi * x[1])

    tol = 1e-10
    petsc_options = {
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_atol": tol,
        "snes_rtol": tol,
        "snes_linesearch_type": "none",
        "ksp_error_if_not_converged": True,
        "snes_error_if_not_converged": True,
    }

    # Reference: ordinary DOLFINx with a DirichletBC
    u_ref = fem.Function(V)
    u_ref.interpolate(lambda x: 0.3 * np.ones_like(x[0]))
    v = ufl.TestFunction(V)
    F_ref = ufl.inner((1 + u_ref**2) * ufl.grad(u_ref), ufl.grad(v)) * ufl.dx - ufl.inner(source, v) * ufl.dx
    J_ref = ufl.derivative(F_ref, u_ref, ufl.TrialFunction(V))
    ref_problem = dolfinx.fem.petsc.NonlinearProblem(
        F_ref,
        u_ref,
        J=J_ref,
        bcs=[fem.dirichletbc(u_bc, bc_dofs)],
        petsc_options_prefix="nonlinear_dirichlet_reference_",
        petsc_options=petsc_options,
    )
    # NOTE: DOLFINx returns the Function, dolfinx_mpc returns (u, converged, iterations)
    ref_problem.solve()
    assert ref_problem.solver.getConvergedReason() > 0

    # The same condition as a constraint: slaves with no masters, offset = boundary value.
    # Ghost slaves are declared too, so every process that assembles a cell touching a
    # constrained dof knows to eliminate it.
    slaves = np.sort(bc_dofs).astype(np.int32)
    g = fem.Function(V)
    g.x.array[:] = 0.0
    g.x.array[slaves] = u_bc.x.array[slaves]
    g.x.scatter_forward()

    mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
    mpc.add_constraint(
        V,
        slaves,
        np.array([], dtype=np.int64),
        np.array([], dtype=default_scalar_type),
        np.array([], dtype=np.int32),
        np.zeros(len(slaves) + 1, dtype=np.int32),
    )
    mpc.finalize()
    assert mpc.has_inhomogeneity

    # The unknown lives in the MPC space, the test function in the original space
    uh = fem.Function(mpc.function_space)
    uh.interpolate(lambda x: 0.3 * np.ones_like(x[0]))
    v = ufl.TestFunction(V)
    F = ufl.inner((1 + uh**2) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(source, v) * ufl.dx
    J = ufl.derivative(F, uh, ufl.TrialFunction(V))

    problem = dolfinx_mpc.NonlinearProblem(F, uh, mpc=mpc, bcs=[], J=J, petsc_options=petsc_options)
    _, converged, _ = problem.solve()
    assert converged

    # The constraint must reproduce the boundary data exactly
    nt.assert_allclose(uh.x.array[slaves], u_bc.x.array[slaves], rtol=1e-10, atol=1e-12)

    # ... and the whole solution must match the ordinary DirichletBC solve
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    err = np.max(np.abs(uh.x.array[:num_owned] - u_ref.x.array[:num_owned])) if num_owned else 0.0
    err = mesh.comm.allreduce(float(err), op=MPI.MAX)
    assert err < 1e-8, f"nonlinear MPC-as-DirichletBC solution differs from DOLFINx by {err}"

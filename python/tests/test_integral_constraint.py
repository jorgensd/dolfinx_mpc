# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Tests for expressing a scalar integral condition as a multi point constraint."""

from __future__ import annotations

from mpi4py import MPI

import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.mesh import (
    create_unit_square,
    exterior_facet_indices,
    locate_entities_boundary,
    meshtags,
)

import dolfinx_mpc

_eps = np.finfo(default_real_type).eps
_tol = max(1e-12, 1000 * _eps)
_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def _poisson(V, u_ex):
    """Pure Neumann Poisson forms, with data differentiated out of ``u_ex``."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(V.mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(-ufl.div(ufl.grad(u_ex)), v) * ufl.dx + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ufl.ds
    return a, L


@pytest.mark.parametrize("degree", [1, 2])
def test_mean_value_constraint(degree):
    """int_Omega u dx = gamma selects the solution of a singular problem."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 12, 12)
    V = fem.functionspace(mesh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(mesh)
    u_ex = x[0] ** 2 - x[0] + 3.0
    a, L = _poisson(V, u_ex)
    gamma = comm.allreduce(fem.assemble_scalar(fem.form(u_ex * ufl.dx)), op=MPI.SUM)

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_integral_constraint(ufl.conj(ufl.TestFunction(V)) * ufl.dx, gamma)
    mpc.finalize()

    uh = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=[], petsc_options=_options).solve()
    mean = comm.allreduce(fem.assemble_scalar(fem.form(uh * ufl.dx)), op=MPI.SUM)
    assert abs(mean - gamma) < _tol


def test_facet_constraint_matches_real_space():
    """A boundary average matches the equivalent Lagrange multiplier solve."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 12, 12)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    exterior = exterior_facet_indices(mesh.topology)
    gamma_facets = locate_entities_boundary(mesh, tdim - 1, lambda p: np.isclose(p[0], 0.0))
    values = np.full(len(exterior), 2, dtype=np.int32)
    values[np.isin(exterior, gamma_facets)] = 1
    mt = meshtags(mesh, tdim - 1, exterior, values)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)

    V = fem.functionspace(mesh, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    u_ex = x[0] ** 2 / 2 + x[0] + 3.0
    n = ufl.FacetNormal(mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(-ufl.div(ufl.grad(u_ex)), v) * ufl.dx + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ds(2)
    target = comm.allreduce(fem.assemble_scalar(fem.form(u_ex * ds(1))), op=MPI.SUM)

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_integral_constraint(ufl.conj(ufl.TestFunction(V)) * ds(1), target)
    mpc.finalize()
    uh = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=[], petsc_options=_options).solve()

    integral = comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(1))), op=MPI.SUM)
    assert abs(integral - target) < _tol
    # u_ex is a second order polynomial, so it lies in the discrete space
    error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM))
    assert error < max(1e-11, 1e5 * _eps)


def test_standalone_matches_method():
    """The standalone builder and the convenience method agree."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    form = ufl.conj(ufl.TestFunction(V)) * ufl.dx

    slaves, masters, coeffs, owners, offsets, rhs = dolfinx_mpc.create_integral_constraint(V, form, 0.5)
    manual = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=rhs)
    manual.add_constraint(V, slaves, masters, coeffs, owners, offsets)
    manual.finalize()

    viaclass = dolfinx_mpc.MultiPointConstraint(V)
    viaclass.add_integral_constraint(form, 0.5)
    viaclass.finalize()

    np.testing.assert_allclose(manual.constants, viaclass.constants)
    for s in manual.slaves[: manual.num_local_slaves]:
        np.testing.assert_allclose(manual.masters.links(s), viaclass.masters.links(s))


def test_rejects_form_from_another_space():
    """The test function must live in the space of the constraint."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    W = fem.functionspace(mesh, ("Lagrange", 2))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(ValueError, match="function space of the constraint"):
        mpc.add_integral_constraint(ufl.TestFunction(W) * ufl.dx, 1.0)


def test_rejects_bilinear_form():
    """A form with a trial function is not a functional."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    bilinear = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(ValueError, match="single test function"):
        mpc.add_integral_constraint(bilinear, 1.0)


def test_rejects_vanishing_functional():
    """A functional with no admissible slave is reported, not silently used."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    zero = fem.Constant(mesh, default_scalar_type(0.0))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(RuntimeError, match="No admissible slave"):
        mpc.add_integral_constraint(ufl.inner(zero, ufl.TestFunction(V)) * ufl.dx, 1.0)


def test_dirichlet_dofs_are_not_chosen_as_slave():
    """A constrained dof may be a master, but never the slave."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 8, 8)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Constrain everything except the left edge, so the slave has to come from it
    facets = locate_entities_boundary(mesh, tdim - 1, lambda p: ~np.isclose(p[0], 0.0))
    g = fem.Function(V)
    g.x.array[:] = 0.0
    bc = fem.dirichletbc(g, fem.locate_dofs_topological(V, tdim - 1, facets))

    slaves, _, _, _, _, _ = dolfinx_mpc.create_integral_constraint(
        V, ufl.conj(ufl.TestFunction(V)) * ufl.dx, 1.0, bcs=[bc]
    )
    constrained, num_owned_bc = bc.dof_indices()
    owned_constrained = set(constrained[:num_owned_bc].tolist())
    assert all(int(s) not in owned_constrained for s in slaves)

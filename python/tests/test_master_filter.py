# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Tests for the master filter of MultiPointConstraint.finalize."""

from __future__ import annotations

from mpi4py import MPI

import numpy as np
import numpy.testing as nt
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, la
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc

_eps = np.finfo(default_real_type).eps


def _integral_constraint(V, weight_form, value, filter=None):
    """Enforce ``L(u) = value`` as an affine MPC, keeping every master.

    No weights are discarded while the constraint is built, so the only pruning
    is whatever ``finalize`` is asked to do.
    """
    comm = V.mesh.comm
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    num_owned = imap.size_local * bs
    dtype = default_scalar_type
    mpi_scalar = MPI._typedict[np.dtype(dtype).char]

    w = fem.assemble_vector(fem.form(weight_form, dtype=dtype))
    w.scatter_reverse(la.InsertMode.add)
    w_owned = w.array[:num_owned]

    # Owned dofs are a contiguous global range; local_range is in blocks
    global_dofs = np.arange(imap.local_range[0] * bs, imap.local_range[1] * bs, dtype=np.int64)
    counts = np.array(comm.allgather(global_dofs.size), dtype=np.int32)
    displ = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    total = int(counts.sum())
    all_dofs = np.empty(total, dtype=np.int64)
    all_weights = np.empty(total, dtype=dtype)
    comm.Allgatherv(global_dofs, (all_dofs, counts, displ, MPI.INT64_T))
    comm.Allgatherv(np.ascontiguousarray(w_owned, dtype=dtype), (all_weights, counts, displ, mpi_scalar))
    all_owners = np.repeat(np.arange(comm.size, dtype=np.int32), counts)

    slave = int(np.argmax(np.abs(all_weights)))
    w_slave = all_weights[slave]
    slave_global = int(all_dofs[slave])

    masters = np.delete(all_dofs, slave).astype(np.int64)
    coeffs = (-np.delete(all_weights, slave) / w_slave).astype(dtype)
    owners = np.delete(all_owners, slave).astype(np.int32)

    g = fem.Function(V, dtype=dtype)
    g.x.array[:] = 0.0
    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype, rhs_coeffs=g)
    slave_block = int(imap.global_to_local(np.array([slave_global // bs], dtype=np.int64))[0])
    if slave_block != -1:
        slave_local = np.int32(slave_block * bs + slave_global % bs)
        g.x.array[slave_local] = dtype(value) / w_slave
        mpc.add_constraint(
            V,
            np.array([slave_local], dtype=np.int32),
            masters,
            coeffs,
            owners,
            np.array([0, masters.size], dtype=np.int32),
        )
    g.x.scatter_forward()
    mpc.finalize(filter=filter)
    return mpc, g, slave_global


def _num_masters_global(mpc, comm):
    """Total number of masters of the owned slaves."""
    masters = mpc.masters
    local = sum(len(masters.links(s)) for s in mpc.slaves[: mpc.num_local_slaves])
    return comm.allreduce(local, op=MPI.SUM)


def test_filter_removes_zero_coefficients():
    """A facet functional gives zero weights off the facet; those must drop out."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 8, 8)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    gamma = np.sort(locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0)))
    mt = meshtags(mesh, tdim - 1, gamma, np.full(len(gamma), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    weight_form = ufl.conj(ufl.TestFunction(V)) * ds(1)

    unfiltered, _, _ = _integral_constraint(V, weight_form, 1.0, filter=None)
    filtered, _, _ = _integral_constraint(V, weight_form, 1.0, filter=1e-14)

    n_all = _num_masters_global(unfiltered, comm)
    n_kept = _num_masters_global(filtered, comm)
    # Every dof is offered as a master, but only those on Gamma carry a weight
    num_dofs = V.dofmap.index_map.size_global
    assert n_all == num_dofs - 1
    assert n_kept < n_all
    # P1 on a 8x8 square: 9 vertices on Gamma, minus the slave
    assert n_kept == 8


def test_filter_preserves_solution():
    """Dropping zero coefficients must not change the solution."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 10, 10)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    gamma = np.sort(locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0)))
    mt = meshtags(mesh, tdim - 1, gamma, np.full(len(gamma), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    weight_form = ufl.conj(ufl.TestFunction(V)) * ds(1)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    L = ufl.inner(fem.Constant(mesh, default_scalar_type(1.0)), v) * ufl.dx
    options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

    solutions = []
    for filter in (None, 1e-14):
        mpc, _, _ = _integral_constraint(V, weight_form, 0.7, filter=filter)
        problem = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=[], petsc_options=options)
        uh = problem.solve()
        num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        solutions.append(uh.x.array[:num_owned].copy())
        integral = comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(1))), op=MPI.SUM)
        assert abs(integral - 0.7) < max(1e-12, 1000 * _eps)

    diff = np.max(np.abs(solutions[0] - solutions[1])) if solutions[0].size else 0.0
    assert comm.allreduce(diff, op=MPI.MAX) < max(1e-12, 1000 * _eps)


def test_filter_is_relative_to_each_slave():
    """The threshold is relative to the largest coefficient of the same slave."""
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        pytest.skip("Hand-built two-slave constraint is written for serial")
    mesh = create_unit_square(comm, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    imap = V.dofmap.index_map

    # Two slaves whose coefficients live on wildly different scales. With a
    # relative filter the same fraction survives in both rows.
    slaves = np.array([0, 1], dtype=np.int32)
    masters = imap.local_to_global(np.array([4, 5, 6, 7], dtype=np.int32)).astype(np.int64)
    coeffs = np.array([1.0, 1e-20, 1e6, 1e-14], dtype=default_scalar_type)
    owners = np.zeros(4, dtype=np.int32)
    offsets = np.array([0, 2, 4], dtype=np.int32)

    g = fem.Function(V)
    g.x.array[:] = 0.0
    mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
    mpc.add_constraint(V, slaves, masters, coeffs, owners, offsets)
    mpc.finalize(filter=1e-10)

    # Slave 0: max is 1.0, so 1e-20 is dropped. Slave 1: max is 1e6, so 1e-14
    # is dropped even though it is 6 orders of magnitude *larger* than the
    # coefficient kept for slave 0. That is what "relative to each slave" means.
    coeff_values, coeff_offsets = mpc.coefficients()
    assert len(mpc.masters.links(0)) == 1
    assert len(mpc.masters.links(1)) == 1
    kept = {s: coeff_values[coeff_offsets[s] : coeff_offsets[s + 1]] for s in (0, 1)}
    nt.assert_allclose(np.abs(kept[0]), [1.0])
    nt.assert_allclose(np.abs(kept[1]), [1e6])


def test_filter_none_keeps_everything():
    """The default must reproduce the unfiltered behaviour exactly."""
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    weight_form = ufl.conj(ufl.TestFunction(V)) * ufl.dx

    default, _, _ = _integral_constraint(V, weight_form, 1.0)
    explicit_zero, _, _ = _integral_constraint(V, weight_form, 1.0, filter=0.0)
    assert _num_masters_global(default, comm) == V.dofmap.index_map.size_global - 1
    assert _num_masters_global(explicit_zero, comm) == _num_masters_global(default, comm)


def test_filter_rejects_negative():
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(ValueError, match="non-negative"):
        mpc.finalize(filter=-1.0)

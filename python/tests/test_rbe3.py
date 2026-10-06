# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Flexible spiders (RBE3): the dofs of a spider tied to the least-squares rigid fit of its feet.

The tests that only build constraints run for every scalar type; the one that solves uses the
scalar type of PETSc.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, la, mesh

import dolfinx_mpc

scalar_types = [np.float32, np.float64, np.complex64, np.complex128]


def _real_type(dtype):
    """The coordinate type of a mesh carrying a constraint of `dtype`."""
    return np.finfo(dtype).dtype.type


def _tol(dtype):
    """The tolerance of the coefficients, which come from solving the normal equations of the fit."""
    return 1e3 * np.finfo(dtype).eps


def _on(x, value):
    """Whether the coordinates `x` equal `value`, to the rounding of their type."""
    return np.isclose(x, value, atol=100 * np.finfo(x.dtype).eps)


def _spiders(points, real_type=default_real_type):
    """A spider mesh of `points`, given on every process. The input index of `points[k]` is `k`."""
    points = np.asarray(points, dtype=real_type).reshape(-1, 3)
    return dolfinx_mpc.create_spider_mesh(MPI.COMM_WORLD, points)


def _body_space(spiders, num_components):
    element = basix.ufl.element("DG", "point", 0, shape=(num_components,), dtype=spiders.geometry.x.dtype)
    return fem.functionspace(spiders, element)


def _rigid_map(r, gdim, num_body):
    """B with (B @ [t, theta])_j = (t + theta x r)_j."""
    B = np.zeros((gdim, num_body))
    B[:, :gdim] = np.eye(gdim)
    if num_body > gdim and gdim == 3:
        B[:, 3:] = [[0, r[2], -r[1]], [-r[2], 0, r[0]], [r[1], -r[0], 0]]
    elif num_body > gdim:
        B[:, 2] = [-r[1], r[0]]
    return B


def _gather_feet(V, dofs, weights):
    """Global dof, coordinate and weight of every foot, on every process."""
    imap = V.dofmap.index_map
    owned = dofs[dofs < imap.size_local]
    x = V.tabulate_dof_coordinates()
    data = V.mesh.comm.allgather((imap.local_range[0] + owned, x[owned], weights(x[owned].T)))
    return (np.concatenate([d[0] for d in data]), np.vstack([d[1] for d in data]), np.concatenate([d[2] for d in data]))


def _global_rows(mpc_body, mpcs):
    """Per global slave of the spider space: {(block, global master): coefficient}."""
    cpp = mpc_body._cpp_object
    W = mpc_body.function_space
    bs_W = W.dofmap.index_map_bs
    rows = {}
    coeffs = cpp.coefficients()[0]
    assert coeffs.dtype == mpc_body.dtype
    for slave in cpp.slaves[: cpp.num_local_slaves]:
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        row = {}
        for k in range(start, end):
            block = cpp.master_blocks[k]
            V = mpcs[block].function_space
            bs = V.dofmap.index_map_bs
            m = cpp.masters.array[k]
            g = V.dofmap.index_map.local_to_global(np.array([m // bs], dtype=np.int32))[0] * bs + m % bs
            row[(int(block), int(g))] = coeffs[k]
        g_slave = W.dofmap.index_map.local_to_global(np.array([slave // bs_W], dtype=np.int32))[0]
        rows[(int(g_slave), int(slave % bs_W))] = row
    return rows


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("gdim, rotations", [(3, True), (3, False), (2, True), (2, False)])
def test_rbe3_coefficients(gdim, rotations, dtype):
    """The rows of the spider are A^{-1} w_i B_i^T, from all feet, with position-dependent weights."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    if gdim == 3:
        domain = mesh.create_unit_cube(comm, 3, 3, 3, dtype=real_type)
        centre = np.array([1.5, 0.4, 0.6], dtype=real_type)
    else:
        domain = mesh.create_unit_square(comm, 4, 4, dtype=real_type)
        centre = np.array([1.5, 0.4, 0.0], dtype=real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    num_body = (6 if gdim == 3 else 3) if rotations else gdim
    W = _body_space(_spiders([centre], real_type), num_body)

    def weights(x):
        return 1.0 + x[1]

    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    mpc_body.add_rbe3_geometrical(V, lambda x: _on(x[0], 1.0), weights)
    mpcs = [mpc, mpc_body]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    feet, x, w = _gather_feet(V, fem.locate_dofs_geometrical(V, lambda x: _on(x[0], 1.0)), weights)
    B = [_rigid_map(x_i[:gdim] - centre[:gdim], gdim, num_body) for x_i in x]
    A = sum(w_i * B_i.T @ B_i for w_i, B_i in zip(w, B))
    C = [np.linalg.solve(A, w_i * B_i.T) for w_i, B_i in zip(w, B)]

    rows = _global_rows(mpc_body, mpcs)
    num_rows = comm.allreduce(len(rows), op=MPI.SUM)
    assert num_rows == num_body
    for (_, c), row in rows.items():
        expected = {(0, int(f * gdim + j)): C_i[c, j] for f, C_i in zip(feet, C) for j in range(gdim)}
        assert set(row) == set(expected)
        for key, value in expected.items():
            assert np.isclose(row[key], value, atol=_tol(dtype))


@pytest.mark.parametrize("dtype", scalar_types)
def test_rbe3_feet_on_one_line(dtype):
    """Feet on one line do not determine the rotation about it: an error on every process."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    V = fem.functionspace(mesh.create_unit_cube(comm, 2, 2, 2, dtype=real_type), ("Lagrange", 1, (3,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.5]], real_type), 6)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    mpc_body.add_rbe3_geometrical(V, lambda x: _on(x[0], 1.0) & _on(x[1], 0.0))
    with pytest.raises(RuntimeError, match="on one line"):
        dolfinx_mpc.finalize_multipointconstraints([dolfinx_mpc.MultiPointConstraint(V, dtype=dtype), mpc_body])
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


@pytest.mark.parametrize("dtype", scalar_types)
def test_rbe3_invalid_input(dtype):
    """Negative weights, and feet in a space not finalized together, raise on every process."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    V = fem.functionspace(mesh.create_unit_square(comm, 2, 2, dtype=real_type), ("Lagrange", 1, (2,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.0]], real_type), 3)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    mpc_body.add_rbe3_geometrical(V, lambda x: _on(x[0], 1.0), -1.0)
    with pytest.raises(ValueError, match="non-negative weight"):
        dolfinx_mpc.finalize_multipointconstraints([dolfinx_mpc.MultiPointConstraint(V, dtype=dtype), mpc_body])
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    mpc_body.add_rbe3_geometrical(V, lambda x: _on(x[0], 1.0))
    with pytest.raises(ValueError, match="exactly one of the constraints"):
        dolfinx_mpc.finalize_multipointconstraints([mpc_body])
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("gdim", [2, 3])
def test_update_rbe3(gdim, dtype):
    """After the meshes move, the updated coefficients are those of a constraint built anew."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    if gdim == 3:
        domain = mesh.create_unit_cube(comm, 3, 3, 3, dtype=real_type)
        points = [[1.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
    else:
        domain = mesh.create_unit_square(comm, 4, 4, dtype=real_type)
        points = [[1.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = _body_space(_spiders(points, real_type), 6 if gdim == 3 else 3)

    # The feet are found before the mesh moves, so that both constraints tie the same dofs. The
    # weights are evaluated when the feet are added, so they must not change with the motion
    # (z is unchanged; in 2D they are constant).
    def weights(x):
        return 1.0 + x[2] ** 2 if gdim == 3 else np.full(x.shape[1], 2.0)

    fdim = gdim - 1
    facets = [mesh.locate_entities_boundary(domain, fdim, lambda x, s=s: _on(x[0], s)) for s in (1.0, 0.0)]

    def build():
        mpcs = [dolfinx_mpc.MultiPointConstraint(V, dtype=dtype), dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)]
        mpcs[1].add_rbe3_topological(V, fdim, facets, weights)
        dolfinx_mpc.finalize_multipointconstraints(mpcs)
        return mpcs

    mpcs = build()
    before = _global_rows(mpcs[1], mpcs)
    x = domain.geometry.x
    x[:, 0] += 0.2 * x[:, 1] ** 2
    x[:, 1] += 0.1 * x[:, 0]
    W.mesh.geometry.x[:, :gdim] += np.array([0.05, -0.1, 0.2][:gdim], dtype=real_type)
    mpcs[1].update_rbe3()
    updated = _global_rows(mpcs[1], mpcs)
    # Compared by global master: two constraints may number their ghosts differently
    reference = build()
    expected = _global_rows(reference[1], reference)
    assert set(updated) == set(expected)
    change = 0.0
    for key, row in updated.items():
        assert set(row) == set(expected[key])
        for m, value in row.items():
            assert np.isclose(value, expected[key][m], atol=_tol(dtype))
            change = max(change, abs(value - before[key][m]))
    assert comm.allreduce(change, op=MPI.MAX) > 0.01


def _reaction(V, a, u, dofs):
    """The force with which the condition on `dofs` holds the solution `u`."""
    u_ref = fem.Function(V, dtype=default_scalar_type)
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_ref.x.array[:n] = u.x.array[:n]
    u_ref.x.scatter_forward()
    residual = fem.assemble_vector(fem.form(ufl.action(a, u_ref), dtype=default_scalar_type))
    residual.scatter_reverse(la.InsertMode.add)
    owned = dofs[dofs < V.dofmap.index_map.size_local]
    return V.mesh.comm.allreduce(residual.array.reshape(-1, 3)[owned].sum(axis=0), op=MPI.SUM)


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_rbe3_joins_two_meshes(kind):
    """A spider between two clamped cubes, with feet on both, carries a force.

    The spider moves with the least-squares rigid fit of its feet, and the clamps balance the force.
    """
    comm = MPI.COMM_WORLD
    tol = 1e4 * np.finfo(default_real_type).eps
    cube_a = mesh.create_box(
        comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 3], mesh.CellType.tetrahedron, dtype=default_real_type
    )
    cube_b = mesh.create_box(
        comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [3, 3, 3], mesh.CellType.hexahedron, dtype=default_real_type
    )
    V_a = fem.functionspace(cube_a, ("Lagrange", 1, (3,)))
    V_b = fem.functionspace(cube_b, ("Lagrange", 1, (3,)))
    centre = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
    spiders = _spiders([centre])
    W = _body_space(spiders, 6)

    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=default_scalar_type)
    mpc_body.add_rbe3_geometrical(V_a, lambda x: _on(x[0], 1.0))
    mpc_body.add_rbe3_geometrical(V_b, lambda x: _on(x[0], 1.2))
    mpcs = [
        dolfinx_mpc.MultiPointConstraint(V_a, dtype=default_scalar_type),
        dolfinx_mpc.MultiPointConstraint(V_b, dtype=default_scalar_type),
        mpc_body,
    ]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    def a(V):
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    force = np.array([0.1, -0.2, -1.0])
    load = fem.Constant(spiders, np.concatenate([force, [0.05, 0.1, 0.0]]).astype(default_scalar_type))
    clamped_a = fem.locate_dofs_geometrical(V_a, lambda x: _on(x[0], 0.0))
    clamped_b = fem.locate_dofs_geometrical(V_b, lambda x: _on(x[0], 2.2))
    zero = np.zeros(3, dtype=default_scalar_type)
    bcs = [fem.dirichletbc(zero, clamped_a, V_a), fem.dirichletbc(zero, clamped_b, V_b)]
    problem = dolfinx_mpc.LinearProblem(
        [[a(V_a), None, None], [None, a(V_b), None], [None, None, None]],
        [
            ufl.ZeroBaseForm((ufl.TestFunction(V_a),)),
            ufl.ZeroBaseForm((ufl.TestFunction(V_b),)),
            ufl.inner(load, ufl.TestFunction(W)) * ufl.dx(spiders),
        ],
        mpcs,
        bcs=bcs,
        kind=kind,
        petsc_options_prefix=f"test_rbe3_{kind}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    u_a, u_b, body = problem.solve()

    # The spider's motion is the least-squares rigid fit of the feet of both cubes
    feet = []
    for V, u, side in ((V_a, u_a, 1.0), (V_b, u_b, 1.2)):
        dofs = fem.locate_dofs_geometrical(V, lambda x: _on(x[0], side))
        dofs = dofs[dofs < V.dofmap.index_map.size_local]
        feet.extend(comm.allgather((V.tabulate_dof_coordinates()[dofs], u.x.array.reshape(-1, 3)[dofs])))
    x = np.vstack([f[0] for f in feet])
    u = np.vstack([f[1] for f in feet])
    B = np.vstack([_rigid_map(x_i - centre, 3, 6) for x_i in x])
    fit = np.linalg.lstsq(B, u.reshape(-1), rcond=None)[0]
    np.testing.assert_allclose(dolfinx_mpc.spider_values(body, 0), fit, atol=tol)
    assert np.abs(fit[:3]).max() > 1e-3

    reaction = _reaction(V_a, a(V_a), u_a, clamped_a) + _reaction(V_b, a(V_b), u_b, clamped_b)
    np.testing.assert_allclose(reaction, -force, atol=100 * tol)

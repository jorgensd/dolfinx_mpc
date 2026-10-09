# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Tests for resolving chained constraints, where a master is itself a slave."""

from __future__ import annotations

from mpi4py import MPI

import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary

import dolfinx_mpc

_dtypes = [np.float64, np.complex128] if default_real_type == np.float64 else [np.float32, np.complex64]
N = 4


def _point(i, j):
    """The coordinate of grid vertex (i, j) of the unit square with N x N cells."""
    return np.array([i / N, j / N], dtype=default_real_type).tobytes()


def _space():
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    return fem.functionspace(mesh, ("Lagrange", 1))


def _constraint(V, rows, dtype, resolve_chains=False, round_limit=None, rhs_coeffs=None):
    """A constraint with one general constraint per slave, rows given as
    {slave: {master: coefficient}} on grid vertices."""
    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype, rhs_coeffs=rhs_coeffs)
    for slave, masters in rows.items():
        mpc.create_general_constraint({_point(*slave): {_point(*m): c for m, c in masters.items()}})
    mpc.finalize(resolve_chains=resolve_chains, round_limit=round_limit)
    return mpc


def _global_rows(mpc):
    """{global slave: ({global master: coefficient}, offset)} of the owned slaves of every process."""
    V = mpc.function_space
    imap, bs = V.dofmap.index_map, V.dofmap.index_map_bs

    def to_global(local):
        local = np.asarray(local, dtype=np.int32)
        return imap.local_to_global(local // bs) * bs + local % bs

    coeffs, offsets = mpc.coefficients()
    rows = {}
    for s in mpc.slaves[mpc.slaves < imap.size_local * bs]:
        masters = to_global(mpc.masters.links(s))
        rows[int(to_global([s])[0])] = (
            {int(m): coeffs[k] for m, k in zip(masters, range(offsets[s], offsets[s + 1]))},
            mpc.constants[s],
        )
    gathered = {}
    for r in V.mesh.comm.allgather(rows):
        gathered.update(r)
    return gathered


def _assert_same_rows(mpc, reference, dtype):
    rows, ref = _global_rows(mpc), _global_rows(reference)
    assert rows.keys() == ref.keys()
    tol = 100 * np.finfo(dtype).eps
    for s, (masters, offset) in rows.items():
        ref_masters, ref_offset = ref[s]
        assert masters.keys() == ref_masters.keys()
        for m, c in masters.items():
            assert np.isclose(c, ref_masters[m], atol=tol)
        assert np.isclose(offset, ref_offset, atol=tol)


# u(1, 1/4) = 0.5 u(0, 1/4) + 0.5 u(1, 1/2), u(1, 1/2) = 2 u(0, 1/2)
_chain = {(N, 1): {(0, 1): 0.5, (N, 2): 0.5}, (N, 2): {(0, 2): 2.0}}
_composed = {(N, 1): {(0, 1): 0.5, (0, 2): 1.0}, (N, 2): {(0, 2): 2.0}}


@pytest.mark.parametrize("dtype", _dtypes)
def test_chain_raises_by_default(dtype):
    V = _space()
    with pytest.raises(ValueError, match="is also a slave"):
        _constraint(V, _chain, dtype)


@pytest.mark.parametrize("dtype", _dtypes)
def test_chain_resolved(dtype):
    V = _space()
    mpc = _constraint(V, _chain, dtype, resolve_chains=True)
    _assert_same_rows(mpc, _constraint(V, _composed, dtype), dtype)


@pytest.mark.parametrize("dtype", _dtypes)
def test_no_chain_unchanged(dtype):
    """With nothing to resolve, the constraint is the one built without resolution."""
    V = _space()
    _assert_same_rows(_constraint(V, _composed, dtype, resolve_chains=True), _constraint(V, _composed, dtype), dtype)


def _chain_of_length(length):
    """u(1, j/N) = u(1, (j+1)/N) for j < length, and u(1, length/N) = u(0, 0): `length` slaves after the
    first."""
    rows = {(N, j): {(N, j + 1): 1.0} for j in range(length)}
    rows[(N, length)] = {(0, 0): 1.0}
    return rows


@pytest.mark.parametrize("length", [1, 3])
def test_round_limit(length):
    V = _space()
    rows = _chain_of_length(length)
    mpc = _constraint(V, rows, default_scalar_type, resolve_chains=True, round_limit=length)
    composed = {s: {(0, 0): 1.0} for s in rows}
    _assert_same_rows(mpc, _constraint(V, composed, default_scalar_type), default_scalar_type)
    with pytest.raises(ValueError, match="round_limit"):
        _constraint(V, rows, default_scalar_type, resolve_chains=True, round_limit=length - 1)


@pytest.mark.parametrize("cycle_length", [1, 2, 3])
@pytest.mark.parametrize("round_limit", [None, 2])
def test_cycle_raises(cycle_length, round_limit):
    V = _space()
    rows = {(N, j): {(N, (j + 1) % cycle_length): 0.5, (0, j): 0.5} for j in range(cycle_length)}
    with pytest.raises(ValueError, match="cycle"):
        _constraint(V, rows, default_scalar_type, resolve_chains=True, round_limit=round_limit)


def _expected_offsets(V, g):
    """g at each slave, plus 2 g(1, 2/N) for the slave (1, 1/N), as {global dof: value}."""
    x = V.tabulate_dof_coordinates()
    imap = V.dofmap.index_map
    owned = np.arange(imap.size_local, dtype=np.int32)
    values = {
        (round(x[d, 0] * N), round(x[d, 1] * N)): (int(global_dof), g.x.array[d])
        for d, global_dof in zip(owned, imap.local_to_global(owned))
    }
    gathered = {}
    for v in V.mesh.comm.allgather(values):
        gathered.update(v)
    a, b = gathered[(N, 1)], gathered[(N, 2)]
    return {a[0]: a[1] + 2 * b[1], b[0]: b[1]}


@pytest.mark.parametrize("dtype", _dtypes)
def test_chain_offsets(dtype):
    """The offset of a slave in the chain is carried along: u_a = 2 u_b + g_a, u_b = u_m + g_b gives
    u_a = 2 u_m + 2 g_b + g_a, also after the offsets change."""
    V = _space()
    g = fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: 1 + x[0] + 2 * x[1])
    mpc = _constraint(V, {(N, 1): {(N, 2): 2.0}, (N, 2): {(0, 2): 1.0}}, dtype, resolve_chains=True, rhs_coeffs=g)
    for scale in (1, 3):
        g.interpolate(lambda x: scale * (1 + x[0] + 2 * x[1]))
        mpc.update_constants()
        expected = _expected_offsets(V, g)
        rows = _global_rows(mpc)
        assert rows.keys() == expected.keys()
        for s, (_, offset) in rows.items():
            assert np.isclose(offset, expected[s])


@pytest.mark.parametrize("dtype", _dtypes)
def test_update_coefficients_resolves_again(dtype):
    """The layout of the updates is the rows supplied: changing them equals a fresh build."""
    V = _space()
    mpc = _constraint(V, _chain, dtype, resolve_chains=True)
    coeffs, offsets = mpc.all_coefficients()
    mpc.update_coefficients(2 * coeffs)
    doubled = {s: {m: 2 * c for m, c in masters.items()} for s, masters in _chain.items()}
    _assert_same_rows(mpc, _constraint(V, doubled, dtype, resolve_chains=True), dtype)

    mpc.scale_coefficients(3)
    scaled = {s: {m: 3 * c for m, c in masters.items()} for s, masters in doubled.items()}
    _assert_same_rows(mpc, _constraint(V, scaled, dtype, resolve_chains=True), dtype)


def test_chain_solution():
    """The solution with a resolved chain equals the one with the constraint composed by hand."""
    V = _space()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
    L = (1 + x[0] * x[1]) * v * ufl.dx
    options = {"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
    solutions = []
    for rows, resolve in [(_chain, True), (_composed, False)]:
        mpc = _constraint(V, rows, default_scalar_type, resolve_chains=resolve)
        problem = dolfinx_mpc.LinearProblem(
            a, L, mpc, petsc_options_prefix=f"test_chain_solution_{resolve}_", petsc_options=options
        )
        uh = problem.solve()
        assert isinstance(uh, fem.Function)
        problem.solver.destroy()
        solutions.append(uh.x.array[: V.dofmap.index_map.size_local].copy())
    assert np.allclose(solutions[0], solutions[1], atol=1e3 * np.finfo(default_real_type).eps)


def _to_global(space, local):
    """Global (unrolled) index of local dofs of `space`."""
    imap, bs = space.dofmap.index_map, space.dofmap.index_map_bs
    local = np.asarray(local, dtype=np.int32)
    return imap.local_to_global(local // bs) * bs + local % bs


def _rows_of_blocks(mpcs, supplied):
    """{(block, global slave): {(block, global master): coefficient}} of the owned slaves of every block
    on every process: the resolved rows, or the rows `supplied` before resolution."""
    rows = {}
    for b, mpc in enumerate(mpcs):
        cpp = mpc._cpp_object
        bs = mpc.function_space.dofmap.index_map_bs
        if supplied:
            (coeffs, offsets), masters, blocks = cpp.all_coefficients(), cpp.all_masters(), cpp.all_master_blocks()
        else:
            (coeffs, offsets), masters, blocks = cpp.coefficients(), cpp.masters.array, cpp.master_blocks
        for s in cpp.slaves[cpp.slaves < mpc.function_space.dofmap.index_map.size_local * bs]:
            row = {}
            for k in range(offsets[s], offsets[s + 1]):
                key = (int(blocks[k]), int(_to_global(mpcs[blocks[k]].function_space, [masters[k]])[0]))
                row[key] = row.get(key, 0) + coeffs[k]
            rows[(b, int(_to_global(mpc.function_space, [s])[0]))] = row
    gathered = {}
    for r in MPI.COMM_WORLD.allgather(rows):
        gathered.update(r)
    return gathered


def _compose(rows):
    """Substitute every row wherever its slave is a master, by recursion."""
    done = {}

    def expand(slave):
        if slave not in done:
            out = {}
            for m, c in rows[slave].items():
                for n, d in expand(m).items() if m in rows else [(m, 1)]:
                    out[n] = out.get(n, 0) + c * d
            done[slave] = out
        return done[slave]

    return {s: expand(s) for s in rows}


def _assert_composed(mpcs, dtype):
    """The resolved rows of every block are the rows supplied, composed by hand."""
    composed, resolved = _compose(_rows_of_blocks(mpcs, True)), _rows_of_blocks(mpcs, False)
    assert composed.keys() == resolved.keys()
    tol = 100 * np.finfo(dtype).eps
    for s, row in composed.items():
        for m in row.keys() | resolved[s].keys():
            assert np.isclose(resolved[s].get(m, 0), row.get(m, 0), atol=tol)


def _tie_to_other_space(V, Q, dtype):
    """Constraint on V: the dofs on x = 1 equal 2 times those of Q at the same points. V and Q are the same
    element on the same mesh, so a dof has the same local index, and owner, in both. Ghosts included."""
    imap = Q.dofmap.index_map
    x = V.tabulate_dof_coordinates()[: imap.size_local + imap.num_ghosts]
    slaves = np.flatnonzero(np.isclose(x[:, 0], 1.0)).astype(np.int32)
    owners = np.full(len(slaves), MPI.COMM_WORLD.rank, dtype=np.int32)
    ghost = slaves >= imap.size_local
    owners[ghost] = imap.owners[slaves[ghost] - imap.size_local]
    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc.add_constraint(
        V,
        slaves,
        imap.local_to_global(slaves).astype(np.int64),
        np.full(len(slaves), 2, dtype=dtype),
        owners,
        np.arange(len(slaves) + 1, dtype=np.int32),
        master_space=Q,
    )
    return mpc


@pytest.mark.parametrize("dtype", _dtypes)
def test_cross_block_chain(dtype):
    """V on x = 1 is tied to Q there, which is periodic in x: a chain through the slaves of another block.
    An update of Q reaches the resolved rows of V."""
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    Q = fem.functionspace(mesh, ("Lagrange", 1))

    def build():
        mpc_q = dolfinx_mpc.MultiPointConstraint(Q, dtype=dtype)
        mpc_q.create_periodic_constraint_geometrical(
            Q, lambda x: np.isclose(x[0], 1.0), lambda x: np.vstack((x[0] - 1.0, x[1], x[2])), []
        )
        return [_tie_to_other_space(V, Q, dtype), mpc_q]

    with pytest.raises(ValueError, match="is also a slave"):
        dolfinx_mpc.finalize_multipointconstraints(build())
    mpcs = build()
    dolfinx_mpc.finalize_multipointconstraints(mpcs, resolve_chains=True)
    _assert_composed(mpcs, dtype)
    for (b, _), row in _rows_of_blocks(mpcs, False).items():
        if b == 0:
            assert all(m[0] == 1 and np.isclose(c, 2) for m, c in row.items())

    mpcs[1].scale_coefficients(3)
    _assert_composed(mpcs, dtype)
    for (b, _), row in _rows_of_blocks(mpcs, False).items():
        assert all(np.isclose(c, 6 if b == 0 else 3) for c in row.values())


def _spider_space(points, num_components):
    import basix.ufl

    spiders = dolfinx_mpc.create_spider_mesh(MPI.COMM_WORLD, np.asarray(points, dtype=default_real_type))
    element = basix.ufl.element("DG", "point", 0, shape=(num_components,), dtype=default_real_type)
    return fem.functionspace(spiders, element)


def _spider_blocks(W, points):
    """The global block of each spider, found by its point, and the process owning it."""
    imap = W.dofmap.index_map
    x = W.tabulate_dof_coordinates()[: imap.size_local]
    owned = imap.local_range[0] + np.arange(imap.size_local, dtype=np.int64)
    found = {}
    for xs, blocks, rank in MPI.COMM_WORLD.allgather((x, owned, MPI.COMM_WORLD.rank)):
        for k, p in enumerate(points):
            hit = np.flatnonzero(np.linalg.norm(xs - np.asarray(p), axis=1) < 1e-10)
            if len(hit) > 0:
                found[k] = (int(blocks[hit[0]]), rank)
    return [found[k] for k in range(len(points))]


def _hinge(W, spiders, r, dtype):
    """Constraint on W: the translation of spider 1 follows the rigid motion of spider 0, with lever arm r
    from 0 to 1, and its rotation is free."""
    (a, owner_a), (b, owner_b) = spiders
    mpc = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    if owner_b == MPI.COMM_WORLD.rank:
        local_b = b - W.dofmap.index_map.local_range[0]
        slaves = np.array([3 * local_b, 3 * local_b + 1], dtype=np.int32)
        # u_b = u_a - theta_a r_y, v_b = v_a + theta_a r_x
        masters = np.array([3 * a, 3 * a + 2, 3 * a + 1, 3 * a + 2], dtype=np.int64)
        coeffs = np.array([1, -r[1], 1, r[0]], dtype=dtype)
        owners = np.full(4, owner_a, dtype=np.int32)
        offsets = np.array([0, 2, 4], dtype=np.int32)
    else:
        slaves, masters = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
        coeffs, owners, offsets = np.zeros(0, dtype=dtype), np.zeros(0, dtype=np.int32), np.zeros(1, dtype=np.int32)
    mpc.add_constraint(W, slaves, masters, coeffs, owners, offsets)
    return mpc


@pytest.mark.parametrize("dtype", _dtypes)
def test_hinge_update_rbe2(dtype):
    """Two edges of a square tied to two spiders by RBE2, the spiders joined by a hinge: the feet of the
    second spider chain through its translation. After the meshes move, the updated constraint is the
    one built anew."""
    domain = create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    points = [[-0.5, 0.25, 0.0], [1.5, 0.75, 0.0]]
    W = _spider_space(points, 3)
    facets = [locate_entities_boundary(domain, 1, lambda x, s=s: np.isclose(x[0], s)) for s in (0.0, 1.0)]
    # The hinge is fixed: its spiders and lever arm are those before the meshes move
    spiders, r = _spider_blocks(W, points), np.subtract(points[1], points[0])

    def build():
        mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
        mpc.add_rbe2_topological(1, facets, W)
        mpcs = [mpc, _hinge(W, spiders, r, dtype)]
        dolfinx_mpc.finalize_multipointconstraints(mpcs, resolve_chains=True)
        return mpcs

    mpcs = build()
    _assert_composed(mpcs, dtype)

    x = domain.geometry.x
    x[:, 0] += 0.2 * x[:, 1] ** 2
    x[:, 1] += 0.1 * x[:, 0]
    W.mesh.geometry.x[:, :2] += np.array([0.05, -0.1], dtype=default_real_type)
    mpcs[0].update_rbe2()
    _assert_composed(mpcs, dtype)
    reference = build()
    tol = 100 * np.finfo(dtype).eps
    for mpc, ref in zip(mpcs, reference):
        np.testing.assert_array_equal(mpc.masters.array, ref.masters.array)
        np.testing.assert_allclose(mpc.coefficients()[0], ref.coefficients()[0], atol=tol)

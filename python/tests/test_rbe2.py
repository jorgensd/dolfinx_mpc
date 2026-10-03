# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Rigid spiders (RBE2): dofs tied to the rigid-body motion of a point of a spider mesh."""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh

import dolfinx_mpc

_atol = 1e4 * np.finfo(default_real_type).eps


def _spiders(points):
    """A spider mesh of `points`, split in contiguous chunks over the processes.

    The input index of `points[k]` is then `k`.
    """
    comm = MPI.COMM_WORLD
    points = np.asarray(points, dtype=default_real_type).reshape(-1, 3)
    return dolfinx_mpc.create_spider_mesh(comm, np.array_split(points, comm.size)[comm.rank])


def _body_space(spiders, num_components):
    return fem.functionspace(
        spiders, basix.ufl.element("DG", "point", 0, shape=(num_components,), dtype=default_real_type)
    )


def _global_blocks(W, points):
    """The global block of `W` at each of `points`, by gathering all of `W`. For checks only."""
    imap = W.dofmap.index_map
    x = W.tabulate_dof_coordinates()[: imap.size_local]
    blocks = imap.local_range[0] + np.arange(imap.size_local, dtype=np.int64)
    gathered = W.mesh.comm.allgather((x, blocks))
    x_all = np.vstack([g[0] for g in gathered])
    blocks_all = np.concatenate([g[1] for g in gathered])
    nearest = np.linalg.norm(x_all[None, :, :] - np.asarray(points)[:, None, :], axis=2).argmin(axis=1)
    return blocks_all[nearest]


def _expected_rows(V, dofs, centre, rotations):
    """Per foot dof component, the expected {body component: coefficient} of u = t + theta x r.

    Every rotation term is present, also with a zero coefficient.
    """
    gdim = V.mesh.geometry.dim
    r = V.tabulate_dof_coordinates()[dofs, :gdim] - centre[:gdim]
    rows = []
    for i in range(len(dofs)):
        for j in range(gdim):
            row = {j: 1.0}
            if rotations and gdim == 3:
                terms = {0: ((4, 2, 1), (5, 1, -1)), 1: ((5, 0, 1), (3, 2, -1)), 2: ((3, 1, 1), (4, 0, -1))}[j]
            elif rotations:
                terms = {0: ((2, 1, -1),), 1: ((2, 0, 1),)}[j]
            else:
                terms = ()
            for rot, comp, sign in terms:
                row[rot] = sign * r[i, comp]
            rows.append(row)
    return rows


def _check_constraint(mpc, num_body, dofs, centre, rotations):
    """The masters of every foot are the body dofs of the spider, with the rigid-motion coefficients.

    There is one spider, so a master's body component is its local index modulo the block size.
    """
    cpp = mpc._cpp_object
    bs = cpp.function_space.dofmap.index_map_bs
    masters, (coeffs, _), blocks = cpp.masters, cpp.coefficients(), cpp.master_blocks
    expected = _expected_rows(mpc.function_space, dofs, centre, rotations)
    for i, dof in enumerate(dofs):
        for j in range(bs):
            slave = bs * dof + j
            start, end = masters.offsets[slave], masters.offsets[slave + 1]
            assert (blocks[start:end] == 1).all()
            got = {int(masters.array[k] % num_body): coeffs[k] for k in range(start, end)}
            exp = expected[i * bs + j]
            assert set(got) == set(exp)
            for component, value in exp.items():
                assert np.isclose(got[component], value, atol=_atol)


@pytest.mark.parametrize("gdim, rotations", [(3, True), (3, False), (2, True), (2, False)])
def test_rbe2_coefficients(gdim, rotations):
    """Each foot component is t_j + (theta x r)_j, with r from the spider's point."""
    comm = MPI.COMM_WORLD
    if gdim == 3:
        domain = mesh.create_unit_cube(comm, 3, 3, 3)
        centre = np.array([1.5, 0.5, 0.5])
    else:
        domain = mesh.create_unit_square(comm, 4, 4)
        centre = np.array([1.5, 0.5, 0.0])
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    num = (6 if gdim == 3 else 3) if rotations else gdim
    W = _body_space(_spiders([centre]), num)

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])
    assert mpc.has_cross_block_masters

    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
    _check_constraint(mpc, num, dofs, centre, rotations)


def test_topological_equals_geometrical():
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.5]]), 6)

    def build(add):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        add(mpc)
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W)])
        return mpc._cpp_object

    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.0))
    topo = build(lambda m: m.add_rbe2_topological(2, facets, W))
    geom = build(lambda m: m.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W))
    np.testing.assert_array_equal(topo.slaves, geom.slaves)
    np.testing.assert_array_equal(topo.masters.offsets, geom.masters.offsets)
    np.testing.assert_array_equal(topo.masters.array, geom.masters.array)
    np.testing.assert_allclose(topo.coefficients()[0], geom.coefficients()[0])


def _check_spider_blocks(mpc, mpc_body, num_body, spider_of_dof, blocks):
    """Each foot's masters are the body dofs of its own spider."""
    cpp = mpc._cpp_object
    body_map = mpc_body.function_space.dofmap.index_map
    for slave in cpp.slaves:
        dof = slave // cpp.function_space.dofmap.index_map_bs
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        local_blocks = cpp.masters.array[start:end] // num_body
        global_blocks = body_map.local_to_global(local_blocks.astype(np.int32))
        assert (global_blocks == blocks[spider_of_dof(dof)]).all()


def test_two_spiders_on_one_mesh():
    """Two faces of a cube tied to two spiders, whose points may be owned by different processes."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    points = np.array([[-0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    W = _body_space(_spiders(points), 3)

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical([lambda x: np.isclose(x[0], 0.0), lambda x: np.isclose(x[0], 1.0)], W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])

    # Without rotations, each foot component has one master: the same component of its spider
    cpp = mpc._cpp_object
    for slave in cpp.slaves:
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        assert end - start == 1
        assert cpp.masters.array[start] % 3 == slave % 3
    x = V.tabulate_dof_coordinates()
    _check_spider_blocks(mpc, mpc_body, 3, lambda dof: 0 if x[dof, 0] < 0.5 else 1, _global_blocks(W, points))


@pytest.mark.parametrize("num_spiders", [1, 3, 7, 11])
def test_many_spiders(num_spiders):
    """Spiders spread over several post offices; with fewer spiders than processes, some hold none."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 6, 6)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    points = np.zeros((num_spiders, 3))
    points[:, 0] = 2.0 + np.arange(num_spiders)
    W = _body_space(_spiders(points), 3)

    def spider_of(x):
        return np.round(6 * x[0] + 7 * 6 * x[1]).astype(np.int64) % num_spiders

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical([lambda x, k=k: spider_of(x) == k for k in range(num_spiders)], W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])

    x = V.tabulate_dof_coordinates()
    _check_spider_blocks(mpc, mpc_body, 3, lambda dof: spider_of(x[dof : dof + 1].T)[0], _global_blocks(W, points))
    # The rotation coefficient uses each foot's own spider
    cpp = mpc._cpp_object
    coeffs = cpp.coefficients()[0]
    for slave in cpp.slaves:
        dof, j = divmod(slave, 2)
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        r = x[dof, :2] - points[spider_of(x[dof : dof + 1].T)[0], :2]
        rotation = [c for m, c in zip(cpp.masters.array[start:end], coeffs[start:end]) if m % 3 == 2]
        assert np.allclose(rotation, [-r[1] if j == 0 else r[0]], atol=_atol)


def test_spiders_in_input_order():
    """A spider's index is its position among the points of all processes, process 0 first."""
    comm = MPI.COMM_WORLD
    # A different number of points per process, in an order unrelated to the coordinates
    mine = np.array([[10.0 * comm.rank + 3.0 - i, comm.rank, i] for i in range(comm.rank % 3 + 1)])
    W = _body_space(dolfinx_mpc.create_spider_mesh(comm, mine), 3)
    x_W = fem.Function(W)
    x_W.x.array[:] = W.tabulate_dof_coordinates().reshape(-1)
    all_points = np.vstack(comm.allgather(mine))
    for k, point in enumerate(all_points):
        np.testing.assert_allclose(dolfinx_mpc.spider_values(x_W, k), point)


def test_spider_index_out_of_range():
    """A spider index past the points of the spider mesh raises on every process."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.0]]), 2)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(IndexError, match="outside the 1 points"):
        mpc.add_rbe2_geometrical([None, lambda x: np.isclose(x[0], 1.0)], W)
    with pytest.raises(IndexError, match="no point with input index 3"):
        dolfinx_mpc.spider_values(fem.Function(W), 3)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


def test_foot_of_two_spiders():
    """A dof tied to two spiders is a slave constrained twice."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]), 3)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc.add_rbe2_geometrical([None, lambda x: np.isclose(x[0], 1.0)], W)
    with pytest.raises(ValueError, match="more than one constraint"):
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W)])


def test_spider_mesh():
    """Coinciding points are merged."""
    merged = _spiders([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.7, 0.5, 0.5]])
    assert merged.topology.index_map(0).size_global == 2


@pytest.mark.parametrize("gdim", [2, 3])
def test_update_rbe2(gdim):
    """After the meshes move, the updated coefficients are those of a constraint built anew."""
    comm = MPI.COMM_WORLD
    if gdim == 3:
        domain = mesh.create_unit_cube(comm, 3, 3, 3)
        points = [[1.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
    else:
        domain = mesh.create_unit_square(comm, 4, 4)
        points = [[1.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = _body_space(_spiders(points), 6 if gdim == 3 else 3)
    # Every foot on one plane through each spider: some rotation coefficients start at zero. The
    # feet are found before the mesh moves, so that both constraints tie the same dofs.
    fdim = gdim - 1
    facets = [mesh.locate_entities_boundary(domain, fdim, lambda x, s=s: np.isclose(x[0], s)) for s in (1.0, 0.0)]

    def build():
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.add_rbe2_topological(fdim, facets, W)
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W)])
        return mpc

    mpc = build()
    before = mpc._cpp_object.all_coefficients()[0].copy()

    # Shear the domain and move the spiders off their planes
    x = domain.geometry.x
    x[:, 0] += 0.2 * x[:, 1] ** 2
    x[:, 1] += 0.1 * x[:, 0]
    W.mesh.geometry.x[:, :gdim] += np.array([0.05, -0.1, 0.2][:gdim])
    mpc.update_rbe2()
    updated = mpc._cpp_object.all_coefficients()[0]
    reference = build()._cpp_object
    np.testing.assert_array_equal(mpc._cpp_object.all_masters(), reference.all_masters())
    np.testing.assert_allclose(updated, reference.all_coefficients()[0], atol=_atol)
    moved = comm.allreduce(np.abs(updated - before).max(initial=0.0), op=MPI.MAX)
    assert moved > 0.01


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_spider_joins_two_meshes(kind):
    """Two cubes joined by a spider alone: the feet move rigidly and the load reaches the clamp."""
    comm = MPI.COMM_WORLD
    cube_a = mesh.create_box(comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 3], mesh.CellType.tetrahedron)
    cube_b = mesh.create_box(comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [3, 3, 3], mesh.CellType.hexahedron)
    V_a = fem.functionspace(cube_a, ("Lagrange", 1, (3,)))
    V_b = fem.functionspace(cube_b, ("Lagrange", 1, (3,)))
    centre = np.array([1.1, 0.5, 0.5])
    W = _body_space(_spiders([centre]), 6)

    mpc_a = dolfinx_mpc.MultiPointConstraint(V_a)
    mpc_a.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc_b = dolfinx_mpc.MultiPointConstraint(V_b)
    mpc_b.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.2), W)
    mpcs = [mpc_a, mpc_b, dolfinx_mpc.MultiPointConstraint(W)]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    def a(V):
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    load = fem.Constant(cube_b, np.array([0.0, 0.0, -1.0], dtype=default_scalar_type))
    clamped = fem.locate_dofs_geometrical(V_a, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V_a)
    problem = dolfinx_mpc.LinearProblem(
        [[a(V_a), None, None], [None, a(V_b), None], [None, None, None]],
        [
            ufl.ZeroBaseForm((ufl.TestFunction(V_a),)),
            ufl.inner(load, ufl.TestFunction(V_b)) * ufl.dx,
            None,
        ],
        mpcs,
        bcs=[bc],
        kind=kind,
        petsc_options_prefix=f"test_spider_{kind}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    u_a, u_b, body = problem.solve()
    values = dolfinx_mpc.spider_values(body, 0)
    t, theta = values[:3], values[3:]
    assert abs(t[2]) > 0

    for u, V, side in ((u_a, V_a, 1.0), (u_b, V_b, 1.2)):
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], side))
        r = V.tabulate_dof_coordinates()[dofs] - centre
        expected = t + np.cross(theta, r) if len(dofs) > 0 else np.zeros((0, 3))
        np.testing.assert_allclose(u.x.array.reshape(-1, 3)[dofs], expected, atol=_atol)

# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Rigid spiders (RBE2): dofs tied to the rigid-body motion of a point of a spider mesh.

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
from dolfinx import default_real_type, default_scalar_type, fem, mesh
from dolfinx.common import local_range

import dolfinx_mpc

scalar_types = [np.float32, np.float64, np.complex64, np.complex128]


def _real_type(dtype):
    """The coordinate type of a mesh carrying a constraint of `dtype`."""
    return np.finfo(dtype).dtype.type


def _tol(dtype):
    return 100 * np.finfo(dtype).eps


def _spiders(points, real_type=default_real_type):
    """A spider mesh of `points`, given on every process. The input index of `points[k]` is `k`."""
    points = np.asarray(points, dtype=real_type).reshape(-1, 3)
    return dolfinx_mpc.create_spider_mesh(MPI.COMM_WORLD, points)


def _body_space(spiders, num_components):
    element = basix.ufl.element("DG", "point", 0, shape=(num_components,), dtype=spiders.geometry.x.dtype)
    return fem.functionspace(spiders, element)


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
    assert coeffs.dtype == mpc.dtype
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
                assert np.isclose(got[component], value, atol=_tol(mpc.dtype))


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("gdim, rotations", [(3, True), (3, False), (2, True), (2, False)])
def test_rbe2_coefficients(gdim, rotations, dtype):
    """Each foot component is t_j + (theta x r)_j, with r from the spider's point."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    if gdim == 3:
        domain = mesh.create_unit_cube(comm, 3, 3, 3, dtype=real_type)
        centre = np.array([1.5, 0.5, 0.5], dtype=real_type)
    else:
        domain = mesh.create_unit_square(comm, 4, 4, dtype=real_type)
        centre = np.array([1.5, 0.5, 0.0], dtype=real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    num = (6 if gdim == 3 else 3) if rotations else gdim
    W = _body_space(_spiders([centre], real_type), num)

    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])
    assert mpc.has_cross_block_masters

    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
    _check_constraint(mpc, num, dofs, centre, rotations)


@pytest.mark.parametrize("dtype", scalar_types)
def test_topological_equals_geometrical(dtype):
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    domain = mesh.create_unit_cube(comm, 3, 3, 3, dtype=real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.5]], real_type), 6)

    def build(add):
        mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
        add(mpc)
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)])
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


@pytest.mark.parametrize("dtype", scalar_types)
def test_two_spiders_on_one_mesh(dtype):
    """Two faces of a cube tied to two spiders, whose points may be owned by different processes."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    domain = mesh.create_unit_cube(comm, 3, 3, 3, dtype=real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    points = np.array([[-0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], dtype=real_type)
    W = _body_space(_spiders(points, real_type), 3)

    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc.add_rbe2_geometrical([lambda x: np.isclose(x[0], 0.0), lambda x: np.isclose(x[0], 1.0)], W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])

    # Without rotations, each foot component has one master: the same component of its spider
    cpp = mpc._cpp_object
    for slave in cpp.slaves:
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        assert end - start == 1
        assert cpp.masters.array[start] % 3 == slave % 3
    x = V.tabulate_dof_coordinates()
    _check_spider_blocks(mpc, mpc_body, 3, lambda dof: 0 if x[dof, 0] < 0.5 else 1, _global_blocks(W, points))


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("num_spiders", [1, 3, 7, 11])
def test_many_spiders(num_spiders, dtype):
    """Spiders spread over several post offices; with fewer spiders than processes, some hold none."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    domain = mesh.create_unit_square(comm, 6, 6, dtype=real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    points = np.zeros((num_spiders, 3), dtype=real_type)
    points[:, 0] = 2.0 + np.arange(num_spiders)
    W = _body_space(_spiders(points, real_type), 3)

    def spider_of(x):
        return np.round(6 * x[0] + 7 * 6 * x[1]).astype(np.int64) % num_spiders

    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc.add_rbe2_geometrical([lambda x, k=k: spider_of(x) == k for k in range(num_spiders)], W)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)
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
        assert np.allclose(rotation, [-r[1] if j == 0 else r[0]], atol=_tol(dtype))


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("others", ["same", "empty"])
def test_spiders_in_input_order(dtype, others):
    """Spider `k` is row `k` of the first process's points, whichever way the others pass them."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    # More points than processes, in an order unrelated to the coordinates
    points = np.array([[3.0 - i, (5 * i) % 7, i] for i in range(2 * comm.size + 1)], dtype=real_type)
    given = points if (comm.rank == 0 or others == "same") else np.zeros((0, 3), dtype=real_type)
    W = _body_space(dolfinx_mpc.create_spider_mesh(comm, given), 3)
    assert W.mesh.topology.index_map(0).size_global == len(points)
    x_W = fem.Function(W, dtype=dtype)
    x_W.x.array[:] = W.tabulate_dof_coordinates().reshape(-1)
    for k, point in enumerate(points):
        values = dolfinx_mpc.spider_values(x_W, k)
        assert values.dtype == dtype
        np.testing.assert_allclose(values, point, rtol=_tol(dtype))


def test_spider_mesh_input_must_agree():
    """Points on another process that differ from the first process's raise on every process."""
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        pytest.skip("Needs a second process")
    points = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], dtype=default_real_type)
    given = points if comm.rank == 0 else points + 0.1
    with pytest.raises(ValueError, match="points of the first process"):
        dolfinx_mpc.create_spider_mesh(comm, given)


def test_spider_mesh_dtype_from_first_process():
    """The coordinate type is that of the first process's points, on every process."""
    comm = MPI.COMM_WORLD
    given = np.array([[0.5, 0.5, 0.5]], dtype=np.float32) if comm.rank == 0 else np.zeros((0, 3))
    spiders = dolfinx_mpc.create_spider_mesh(comm, given)
    assert comm.allreduce(spiders.geometry.x.dtype == np.float32, op=MPI.LAND)
    explicit = dolfinx_mpc.create_spider_mesh(comm, given, dtype=np.float64)
    assert explicit.geometry.x.dtype == np.float64


def test_spider_index_out_of_range():
    """A spider index past the points of the spider mesh raises on every process."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 2, 2, dtype=default_real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.0]]), 2)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(IndexError, match="outside the 1 points"):
        mpc.add_rbe2_geometrical([None, lambda x: np.isclose(x[0], 1.0)], W)
    with pytest.raises(IndexError, match="no point with input index 3"):
        dolfinx_mpc.spider_values(fem.Function(W, dtype=default_scalar_type), 3)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


def test_foot_of_two_spiders():
    """A dof tied to two spiders is a slave constrained twice."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2, dtype=default_real_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = _body_space(_spiders([[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]), 3)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc.add_rbe2_geometrical([None, lambda x: np.isclose(x[0], 1.0)], W)
    with pytest.raises(ValueError, match="more than one constraint"):
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W)])


def test_scalar_type():
    """The scalar type defaults to the precision of the mesh, and must match it."""
    comm = MPI.COMM_WORLD
    is_complex = np.issubdtype(default_scalar_type, np.complexfloating)
    for real_type, complex_type in ((np.float32, np.complex64), (np.float64, np.complex128)):
        domain = mesh.create_unit_square(comm, 2, 2, dtype=real_type)
        V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
        assert dolfinx_mpc.MultiPointConstraint(V).dtype == (complex_type if is_complex else real_type)
        other = np.float64 if real_type == np.float32 else np.complex64
        with pytest.raises(ValueError, match="needs a mesh of"):
            dolfinx_mpc.MultiPointConstraint(V, dtype=other)
    with pytest.raises(ValueError, match="Unsupported scalar type"):
        dolfinx_mpc.MultiPointConstraint(V, dtype=np.int32)


@pytest.mark.parametrize("real_type", [np.float32, np.float64])
def test_spider_mesh(real_type):
    """Coinciding points are distinct spiders, and the mesh has the points' precision."""
    points = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.7, 0.5, 0.5]]
    spiders = _spiders(points, real_type)
    assert spiders.topology.index_map(0).size_global == 3
    assert spiders.geometry.x.dtype == real_type
    x = fem.Function(_body_space(spiders, 3), dtype=real_type)
    x.x.array[:] = x.function_space.tabulate_dof_coordinates().reshape(-1)
    for k, point in enumerate(points):
        np.testing.assert_allclose(dolfinx_mpc.spider_values(x, k), point)


def _input_indices(spiders):
    imap = spiders.topology.index_map(0)
    return np.asarray(spiders.topology.original_cell_index[: imap.size_local + imap.num_ghosts])


def test_spider_mesh_partition():
    """Of M points, a process owns those in its local range, as the post office splits an index range,
    so spider k of two meshes of M points is on the same process."""
    comm = MPI.COMM_WORLD
    for num in (1, comm.size, 3 * comm.size + 1):
        spiders = _spiders(np.column_stack([np.arange(num), np.zeros(num), np.zeros(num)]))
        start, end = local_range(comm.rank, num, comm.size)
        np.testing.assert_array_equal(np.sort(_input_indices(spiders)), np.arange(start, end))


def test_spider_pair():
    """Spider k of one spider mesh is related to spider k of the other."""
    comm = MPI.COMM_WORLD
    num = 3 * comm.size + 1
    points = np.column_stack([np.arange(num), np.zeros(num), np.zeros(num)])
    spiders_A, spiders_B = _spiders(points), _spiders(points[::-1])
    emap = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)
    local_B = _input_indices(spiders_B)
    mapped = emap.sub_topology_to_topology(np.arange(len(local_B), dtype=np.int32), False)
    np.testing.assert_array_equal(_input_indices(spiders_A)[mapped], local_B)


def test_spider_pair_errors():
    """Meshes of different numbers of spiders, or with spider k on different processes, raise on every
    process."""
    comm = MPI.COMM_WORLD
    num = 3 * comm.size
    points = np.column_stack([np.arange(num), np.zeros(num), np.zeros(num)]).astype(default_real_type)
    spiders_A = _spiders(points)
    with pytest.raises(ValueError, match="same number of spiders"):
        dolfinx_mpc.create_spider_pair(spiders_A, _spiders(points[:-1]))
    if comm.size > 1:
        # Every point on the first process
        on_first = mesh.create_point_mesh(comm, points if comm.rank == 0 else np.zeros((0, 3), dtype=points.dtype))
        with pytest.raises(ValueError, match="different processes"):
            dolfinx_mpc.create_spider_pair(spiders_A, on_first)


@pytest.mark.parametrize("dtype", scalar_types)
def test_move_mesh(dtype):
    """Every geometry node moves by the displacement at the node, given as a function, an
    expression or a callable."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)

    def displacement(x):
        return np.vstack([0.1 * x[0] * x[1], 0.2 * x[0] - 0.05 * x[1] ** 2])

    for kind in ("function", "expression", "callable"):
        domain = mesh.create_unit_square(comm, 4, 3, mesh.CellType.quadrilateral, dtype=real_type)
        x0 = domain.geometry.x.copy()
        if kind == "function":
            V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
            u = fem.Function(V, dtype=dtype)
            u.interpolate(displacement)
        elif kind == "expression":
            x = ufl.SpatialCoordinate(domain)
            u = ufl.as_vector([0.1 * x[0] * x[1], 0.2 * x[0] - 0.05 * x[1] ** 2])
        else:
            u = displacement
        dolfinx_mpc.spider.move(domain, u)
        expected = x0[:, :2] + displacement(x0.T).T
        np.testing.assert_allclose(domain.geometry.x[:, :2], expected, atol=_tol(real_type))
        np.testing.assert_array_equal(domain.geometry.x[:, 2], x0[:, 2])


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("num_components", [3, 6])
def test_move_spiders(dtype, num_components):
    """Each spider moves by its own translation, the first components of its dofs, whatever the
    order of the dofs."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    points = np.array([[0.5 * i, 1.0, -0.5 * i] for i in range(2 * comm.size + 1)], dtype=real_type)
    spiders = _spiders(points, real_type)
    W = _body_space(spiders, num_components)
    w = fem.Function(W, dtype=dtype)
    # A translation per spider, from its coordinate; the rotations, if any, are not used
    x_W = W.tabulate_dof_coordinates()
    values = np.zeros((len(x_W), num_components), dtype=dtype)
    values[:, :3] = np.column_stack([x_W[:, 0] + 1, 2 * x_W[:, 2], -x_W[:, 0]])
    if num_components == 6:
        values[:, 3:] = 7.0
    w.x.array[:] = values.reshape(-1)
    expected = [dolfinx_mpc.spider_values(w, k) for k in range(len(points))]
    dolfinx_mpc.spider.move(spiders, w)
    x_moved = fem.Function(_body_space(spiders, 3), dtype=real_type)
    x_moved.x.array[:] = x_moved.function_space.tabulate_dof_coordinates().reshape(-1)
    for k, point in enumerate(points):
        np.testing.assert_allclose(
            dolfinx_mpc.spider_values(x_moved, k), point + expected[k][:3].real, atol=_tol(real_type)
        )


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("gdim", [2, 3])
def test_update_rbe2(gdim, dtype):
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
    # Every foot on one plane through each spider: some rotation coefficients start at zero. The
    # feet are found before the mesh moves, so that both constraints tie the same dofs.
    fdim = gdim - 1
    facets = [mesh.locate_entities_boundary(domain, fdim, lambda x, s=s: np.isclose(x[0], s)) for s in (1.0, 0.0)]

    def build():
        mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
        mpc.add_rbe2_topological(fdim, facets, W)
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W, dtype=dtype)])
        return mpc

    mpc = build()
    before = mpc._cpp_object.all_coefficients()[0].copy()

    # Shear the domain and move the spiders off their planes
    x = domain.geometry.x
    x[:, 0] += 0.2 * x[:, 1] ** 2
    x[:, 1] += 0.1 * x[:, 0]
    W.mesh.geometry.x[:, :gdim] += np.array([0.05, -0.1, 0.2][:gdim], dtype=real_type)
    mpc.update_rbe2()
    updated = mpc._cpp_object.all_coefficients()[0]
    assert updated.dtype == dtype
    reference = build()._cpp_object
    np.testing.assert_array_equal(mpc._cpp_object.all_masters(), reference.all_masters())
    np.testing.assert_allclose(updated, reference.all_coefficients()[0], atol=_tol(dtype))
    moved = comm.allreduce(np.abs(updated - before).max(initial=0.0), op=MPI.MAX)
    assert moved > 0.01


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_spider_joins_two_meshes(kind):
    """Two cubes joined by a spider alone: the feet move rigidly and the load reaches the clamp.

    Solved with PETSc, so in its scalar type."""
    comm = MPI.COMM_WORLD
    cube_a = mesh.create_box(
        comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 3], mesh.CellType.tetrahedron, dtype=default_real_type
    )
    cube_b = mesh.create_box(
        comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [3, 3, 3], mesh.CellType.hexahedron, dtype=default_real_type
    )
    V_a = fem.functionspace(cube_a, ("Lagrange", 1, (3,)))
    V_b = fem.functionspace(cube_b, ("Lagrange", 1, (3,)))
    centre = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
    W = _body_space(_spiders([centre]), 6)

    mpc_a = dolfinx_mpc.MultiPointConstraint(V_a, dtype=default_scalar_type)
    mpc_a.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W)
    mpc_b = dolfinx_mpc.MultiPointConstraint(V_b, dtype=default_scalar_type)
    mpc_b.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.2), W)
    mpcs = [mpc_a, mpc_b, dolfinx_mpc.MultiPointConstraint(W, dtype=default_scalar_type)]
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

    # The rigid motion is exact up to the rounding of the solve
    tol = 1e4 * np.finfo(default_real_type).eps
    for u, V, side in ((u_a, V_a, 1.0), (u_b, V_b, 1.2)):
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], side))
        r = V.tabulate_dof_coordinates()[dofs] - centre
        expected = t + np.cross(theta, r) if len(dofs) > 0 else np.zeros((0, 3))
        np.testing.assert_allclose(u.x.array.reshape(-1, 3)[dofs], expected, atol=tol)

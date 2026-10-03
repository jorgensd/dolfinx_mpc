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

_tol = 1e3 * np.finfo(default_real_type).eps
_atol = 1e4 * np.finfo(default_real_type).eps


def _spiders(points_by_rank):
    """A spider mesh with `points_by_rank[r]` on process r (modulo the number of processes)."""
    comm = MPI.COMM_WORLD
    mine = [p for r, p in enumerate(points_by_rank) if r % comm.size == comm.rank]
    points = np.array(mine, dtype=default_real_type).reshape(-1, 3)
    return dolfinx_mpc.create_spider_mesh(comm, points)


def _body_space(spiders, num_components):
    return fem.functionspace(
        spiders, basix.ufl.element("DG", "point", 0, shape=(num_components,), dtype=default_real_type)
    )


def _expected_rows(V, dofs, centre, rotations):
    """Per foot dof component, the expected {body component: coefficient} of u = t + theta x r."""
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
                if sign * r[i, comp] != 0:
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
    spiders = _spiders([centre])
    num = (6 if gdim == 3 else 3) if rotations else gdim
    W = _body_space(spiders, num)
    index = dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[0], 1.5))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W, index)
    mpc_body = dolfinx_mpc.MultiPointConstraint(W)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])
    assert mpc.has_cross_block_masters

    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
    _check_constraint(mpc, num, dofs, centre, rotations)


def test_topological_equals_geometrical():
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    spiders = _spiders([[1.5, 0.5, 0.5]])
    W = _body_space(spiders, 6)

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


def test_two_spiders_on_one_mesh():
    """Two faces of a cube tied to two spiders, whose points may be owned by different processes."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    left, right = np.array([-0.5, 0.5, 0.5]), np.array([1.5, 0.5, 0.5])
    spiders = _spiders([left, right])
    W = _body_space(spiders, 3)
    i_left = dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[0], -0.5))
    i_right = dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[0], 1.5))
    assert {i_left, i_right} == {0, 1}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(
        lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
        W,
        lambda x: np.where(x[0] < 0.5, i_left, i_right),
    )
    mpc_body = dolfinx_mpc.MultiPointConstraint(W)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_body])

    # Each foot's master is the translation of the spider on its own side
    cpp = mpc._cpp_object
    body_map = mpc_body.function_space.dofmap.index_map
    x = V.tabulate_dof_coordinates()
    for slave in cpp.slaves:
        dof, j = divmod(slave, 3)
        start, end = cpp.masters.offsets[slave], cpp.masters.offsets[slave + 1]
        assert end - start == 1
        m = cpp.masters.array[start]
        block = body_map.local_to_global(np.array([m // 3], dtype=np.int32))[0]
        assert m % 3 == j
        # The body blocks are numbered with the points of the spider mesh
        assert block == (i_left if x[dof, 0] < 0.5 else i_right)


def test_foot_of_two_spiders():
    """A dof tied to two spiders is a slave constrained twice."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    spiders = _spiders([[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    W = _body_space(spiders, 3)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W, 0)
    mpc.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0), W, 1)
    with pytest.raises(ValueError, match="more than one constraint"):
        dolfinx_mpc.finalize_multipointconstraints([mpc, dolfinx_mpc.MultiPointConstraint(W)])


def test_spider_mesh():
    """Coinciding points are merged, and a marker that finds no point or two raises everywhere."""
    comm = MPI.COMM_WORLD
    merged = _spiders([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.7, 0.5, 0.5]])
    assert merged.topology.index_map(0).size_global == 2
    assert dolfinx_mpc.locate_spider(merged, lambda x: np.isclose(x[0], 0.5)) == 0
    spiders = _spiders([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="selects 0"):
        dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[0], 3.0))
    with pytest.raises(ValueError, match="selects 2"):
        dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[1], 0.0))
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


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
    spiders = _spiders([centre])
    W = _body_space(spiders, 6)

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

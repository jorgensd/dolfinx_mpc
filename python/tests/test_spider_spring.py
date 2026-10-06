# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""0-D springs between spiders of two spider meshes, related by :func:`dolfinx_mpc.create_spider_pair`.

Spring `k` joins spider `k` of mesh A to spider `k` of mesh B, which start at the same point. The
spring is a form, :math:`\\int K (w_A - w_B) \\cdot (v_A - v_B)` over mesh A, with a symmetric 6x6
stiffness coupling translations and rotations. The tests that assemble through PETSc use its
scalar type.

A spring between spiders a fixed offset apart, a rigid link free to turn about an axis, is tested
through its energy, for every scalar type.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import numpy as np
import pytest
import scipy.sparse
import ufl
from dolfinx import default_real_type, default_scalar_type, fem

import dolfinx_mpc
import dolfinx_mpc.utils

_tol = 1e3 * np.finfo(default_real_type).eps


def _springs():
    """Two spider meshes of the same points, the spaces on them, the pairing and the spring form."""
    comm = MPI.COMM_WORLD
    # More springs than processes, so that the spiders are spread over the processes
    num = 2 * comm.size + 1
    points = np.column_stack([np.linspace(1.0, 2.0, num), np.full(num, 0.5), np.full(num, 0.5)])
    points = points.astype(default_real_type)
    spiders_A = dolfinx_mpc.create_spider_mesh(comm, points)
    spiders_B = dolfinx_mpc.create_spider_mesh(comm, points)
    emap = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)

    element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type)
    W_A = fem.functionspace(spiders_A, element)
    W_B = fem.functionspace(spiders_B, element)

    # Symmetric positive definite, with translation-rotation coupling
    M = np.random.default_rng(3).standard_normal((6, 6))
    K = (M @ M.T + 6 * np.eye(6)).astype(default_scalar_type)

    W = ufl.MixedFunctionSpace(W_A, W_B)
    u_A, u_B = ufl.TrialFunctions(W)
    v_A, v_B = ufl.TestFunctions(W)
    a = ufl.inner(fem.Constant(spiders_A, K) * (u_A - u_B), v_A - v_B) * ufl.dx(spiders_A)
    return num, W_A, W_B, emap, K, a


def _constraints(W_A, W_B):
    mpcs = [
        dolfinx_mpc.MultiPointConstraint(W_A, dtype=default_scalar_type),
        dolfinx_mpc.MultiPointConstraint(W_B, dtype=default_scalar_type),
    ]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)
    return mpcs


def _dense(A):
    """The assembled matrix on the first process, `None` elsewhere."""
    if A.getType() == "nest":
        gather = dolfinx_mpc.utils.gather_PETScMatrix
        blocks = [[gather(A.getNestSubMatrix(i, j), root=0) for j in range(2)] for i in range(2)]
        return scipy.sparse.bmat(blocks).toarray() if MPI.COMM_WORLD.rank == 0 else None
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=0)
    return A_csr.toarray() if MPI.COMM_WORLD.rank == 0 else None


def test_spring_operator():
    """Both layouts assemble the operator DOLFINx does, which applies K (u_A - u_B) and its negative
    to the two ends of every spring, and leaves the joint motion of each pair free."""
    comm = MPI.COMM_WORLD
    num, W_A, W_B, emap, K, a = _springs()
    a_blocks = fem.form(ufl.extract_blocks(a), entity_maps=[emap], dtype=default_scalar_type)
    A_ref = dolfinx.fem.petsc.assemble_matrix(a_blocks, kind="nest")
    A_ref.assemble()
    mpcs = _constraints(W_A, W_B)
    A = {}
    for kind in ("nest", "mpi"):
        A[kind] = dolfinx_mpc.assemble_matrix(a_blocks, mpcs, kind=kind)
        A[kind].assemble()

    dense = [_dense(M) for M in (A_ref, A["nest"], A["mpi"])]
    if comm.rank == 0:
        values = [np.linalg.svd(D, compute_uv=False) for D in dense]
        scale = values[0].max()
        for v in values[1:]:
            np.testing.assert_allclose(v, values[0], atol=_tol * scale)
        assert np.linalg.matrix_rank(dense[0], tol=_tol * scale) == 6 * num

    # The action on random motions of the spiders, spring by spring
    rng = np.random.default_rng(5)
    u = [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]
    y = [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]
    for f in u:
        f.x.array[:] = rng.standard_normal(f.x.array.shape).astype(default_scalar_type)
        f.x.scatter_forward()
    x_nest = PETSc.Vec().createNest([f.x.petsc_vec for f in u])
    y_nest = PETSc.Vec().createNest([f.x.petsc_vec for f in y])
    A["nest"].mult(x_nest, y_nest)
    for f in y:
        f.x.scatter_forward()
    for k in range(num):
        u_A, u_B = (dolfinx_mpc.spider_values(f, k) for f in u)
        y_A, y_B = (dolfinx_mpc.spider_values(f, k) for f in y)
        expected = K @ (u_A - u_B)
        np.testing.assert_allclose(y_A, expected, atol=_tol * np.abs(K).max())
        np.testing.assert_allclose(y_B, -expected, atol=_tol * np.abs(K).max())


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_spring_solve(kind):
    """With every spider of A held fixed and a force and moment on each spider of B, spring k
    stretches by u_B = K^{-1} f_k."""
    num, W_A, W_B, emap, K, a = _springs()
    held = fem.locate_dofs_geometrical(W_A, lambda x: np.full(x.shape[1], True))
    bc = fem.dirichletbc(np.zeros(6, dtype=default_scalar_type), held, W_A)

    # A different load on each spring
    load = fem.Function(W_B, dtype=default_scalar_type)
    x_B = W_B.tabulate_dof_coordinates()
    values = np.zeros((len(x_B), 6), dtype=default_scalar_type)
    values[:, 1] = 2.0 * (x_B[:, 0] - 1.5)
    values[:, 2] = -1.0
    values[:, 3] = 0.5 * x_B[:, 0]
    load.x.array[:] = values.reshape(-1)
    v_A, v_B = ufl.TestFunction(W_A), ufl.TestFunction(W_B)
    L = [ufl.ZeroBaseForm((v_A,)), ufl.inner(load, v_B) * ufl.dx(W_B.mesh)]

    problem = dolfinx_mpc.LinearProblem(
        ufl.extract_blocks(a),
        L,
        _constraints(W_A, W_B),
        bcs=[bc],
        kind=kind,
        entity_maps=[emap],
        petsc_options_prefix=f"test_spider_spring_{kind}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    w_A, w_B = problem.solve()
    for k in range(num):
        expected = np.linalg.solve(K, dolfinx_mpc.spider_values(load, k))
        np.testing.assert_allclose(dolfinx_mpc.spider_values(w_A, k), 0.0, atol=_tol)
        np.testing.assert_allclose(dolfinx_mpc.spider_values(w_B, k), expected, atol=_tol * np.abs(expected).max())


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
def test_stiff_spring_hinge():
    """Two clamped boxes tied to coincident spiders, joined by a spring stiff in all but the rotation
    about the x axis: as k grows the gap between the spiders closes like 1/k, the turn about the axis
    settles, and no moment about the axis reaches the first box's clamp for any k."""
    comm = MPI.COMM_WORLD
    tol = 1e3 * np.finfo(default_real_type).eps
    box_A = dolfinx.mesh.create_box(
        comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 3], dolfinx.mesh.CellType.tetrahedron, dtype=default_real_type
    )
    box_B = dolfinx.mesh.create_box(
        comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [3, 3, 3], dolfinx.mesh.CellType.hexahedron, dtype=default_real_type
    )
    V_A = fem.functionspace(box_A, ("Lagrange", 1, (3,)))
    V_B = fem.functionspace(box_B, ("Lagrange", 1, (3,)))
    x_c = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
    spiders_A = dolfinx_mpc.create_spider_mesh(comm, x_c.reshape(1, 3))
    spiders_B = dolfinx_mpc.create_spider_mesh(comm, x_c.reshape(1, 3))
    pair = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)
    element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type)
    W_A, W_B = fem.functionspace(spiders_A, element), fem.functionspace(spiders_B, element)

    # The facing faces follow the spiders; their bottom edges are clamped, so they are not feet
    mpc_A = dolfinx_mpc.MultiPointConstraint(V_A, dtype=default_scalar_type)
    mpc_A.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0, atol=tol) & (x[2] > tol), W_A)
    mpc_B = dolfinx_mpc.MultiPointConstraint(V_B, dtype=default_scalar_type)
    mpc_B.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.2, atol=tol) & (x[2] > tol), W_B)
    mpcs = [mpc_A, mpc_B] + [dolfinx_mpc.MultiPointConstraint(W_s, dtype=default_scalar_type) for W_s in (W_A, W_B)]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    E, nu = 1.0e3, 0.3
    mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def elasticity(u, v):
        eps = ufl.sym(ufl.grad(u))
        return ufl.inner(2 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(3), ufl.sym(ufl.grad(v)))

    K_unit = np.zeros((6, 6))
    K_unit[:3, :3] = np.eye(3)
    K_unit[4, 4] = K_unit[5, 5] = 1.0  # no stiffness for the rotation about x
    K = fem.Constant(spiders_A, K_unit.astype(default_scalar_type))
    W = ufl.MixedFunctionSpace(V_A, V_B, W_A, W_B)
    u_A, u_B, w_A, w_B = ufl.TrialFunctions(W)
    v_A, v_B, z_A, z_B = ufl.TestFunctions(W)
    a = elasticity(u_A, v_A) * ufl.dx(box_A) + elasticity(u_B, v_B) * ufl.dx(box_B)
    a += ufl.inner(K * (w_A - w_B), z_A - z_B) * ufl.dx(spiders_A)

    # A sideways traction on top of box B, which turns it about the axis
    top = dolfinx.mesh.locate_entities_boundary(box_B, 2, lambda x: np.isclose(x[2], 1.0, atol=tol))
    top_tags = dolfinx.mesh.meshtags(box_B, 2, top, np.full(len(top), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=box_B, subdomain_data=top_tags)
    g = fem.Constant(box_B, np.array([0.0, 1.0, 0.0], dtype=default_scalar_type))
    L = [ufl.ZeroBaseForm((v_A,)), ufl.inner(g, v_B) * ds(1), ufl.ZeroBaseForm((z_A,)), ufl.ZeroBaseForm((z_B,))]
    zero = np.zeros(3, dtype=default_scalar_type)
    # Coordinates are only exact to the precision, so every locator has a tolerance
    clamped_A = fem.locate_dofs_geometrical(V_A, lambda x: np.isclose(x[2], 0.0, atol=tol))
    clamped_B = fem.locate_dofs_geometrical(V_B, lambda x: np.isclose(x[2], 0.0, atol=tol))
    bcs = [fem.dirichletbc(zero, clamped_A, V_A), fem.dirichletbc(zero, clamped_B, V_B)]
    problem = dolfinx_mpc.LinearProblem(
        ufl.extract_blocks(a),
        L,
        mpcs,
        bcs=bcs,
        kind="mpi",
        entity_maps=[pair],
        petsc_options_prefix="test_stiff_spring_hinge_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )

    def moment_about_axis_at_clamp_A(u):
        """The moment about the x axis through x_c that the clamp exerts on box A."""
        u_ref = fem.Function(V_A, dtype=default_scalar_type)
        # The ghosts of the constraint's extended space are ordered differently: copy the owned values
        n_owned = V_A.dofmap.index_map.size_local * 3
        u_ref.x.array[:n_owned] = u.x.array[:n_owned]
        u_ref.x.scatter_forward()
        residual = fem.assemble_vector(
            fem.form(elasticity(u_ref, ufl.TestFunction(V_A)) * ufl.dx, dtype=default_scalar_type)
        )
        residual.scatter_reverse(dolfinx.la.InsertMode.add)
        owned = clamped_A[clamped_A < V_A.dofmap.index_map.size_local]
        f = residual.array.reshape(-1, 3)[owned].real
        r = V_A.tabulate_dof_coordinates()[owned] - x_c
        return comm.allreduce(np.cross(r, f).sum(axis=0)[0], op=MPI.SUM)

    gaps, turns = [], []
    for k in (1e2, 1e3, 1e4):
        K.value[:] = k * K_unit
        u_A_h, _, w_A_h, w_B_h = problem.solve()
        difference = dolfinx_mpc.spider_values(w_B_h, 0).real - dolfinx_mpc.spider_values(w_A_h, 0).real
        turns.append(difference[3])
        gaps.append(np.linalg.norm(np.delete(difference, 3)))
        # The applied moment about the axis is 0.5; none of it passes the spring
        assert abs(moment_about_axis_at_clamp_A(u_A_h)) < 1e-4
    rates = [np.log10(g0 / g1) for g0, g1 in zip(gaps[:-1], gaps[1:])]
    assert rates[-1] > 0.9
    assert abs(turns[-1]) > 100 * gaps[-1]
    assert abs(turns[-1] - turns[-2]) < 0.5 * abs(turns[-2] - turns[-3])


# The spring of two spiders a fixed offset `d` apart: stiff against spider B leaving the rigid
# extension of spider A, t_B = t_A + theta_A x d and theta_B = theta_A, except for turning about the
# axis `a`. The tests evaluate its energy as a functional, so they need no PETSc and run for every
# scalar type.
_scalar_types = [np.float32, np.float64, np.complex64, np.complex128]
_axis = np.array([0.0, 1.0, 0.0])
_offset = np.array([0.1, 0.3, -0.2])


def _link_gap(w_A, w_B, d):
    """The motion of spider B away from the rigid extension of spider A."""
    t_A, theta_A = ufl.as_vector([w_A[i] for i in range(3)]), ufl.as_vector([w_A[i] for i in range(3, 6)])
    t_B, theta_B = ufl.as_vector([w_B[i] for i in range(3)]), ufl.as_vector([w_B[i] for i in range(3, 6)])
    translation = t_B - t_A - ufl.cross(theta_A, d)
    rotation = theta_B - theta_A
    return ufl.as_vector([translation[i] for i in range(3)] + [rotation[i] for i in range(3)])


def _link_stiffness(k, k_t):
    K = np.zeros((6, 6))
    K[:3, :3] = k * np.eye(3)
    K[3:, 3:] = k * (np.eye(3) - np.outer(_axis, _axis)) + k_t * np.outer(_axis, _axis)
    return K


def _rigid_links(dtype, k_t):
    """Springs `k` from spider `k` of A to spider `k` of B, at a skew offset, and their energy."""
    comm = MPI.COMM_WORLD
    real_type = np.finfo(dtype).dtype.type
    num = 2 * comm.size + 1
    points_A = np.column_stack([np.linspace(1.0, 2.0, num), np.full(num, 0.5), np.full(num, 0.5)])
    points_A = points_A.astype(real_type)
    points_B = (points_A + _offset).astype(real_type)
    spiders_A = dolfinx_mpc.create_spider_mesh(comm, points_A, dtype=real_type)
    spiders_B = dolfinx_mpc.create_spider_mesh(comm, points_B, dtype=real_type)
    pair = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)
    element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=real_type)
    W_A, W_B = fem.functionspace(spiders_A, element), fem.functionspace(spiders_B, element)
    w_A, w_B = fem.Function(W_A, dtype=dtype), fem.Function(W_B, dtype=dtype)

    K = _link_stiffness(1e2, k_t)
    d = fem.Constant(spiders_A, _offset.astype(dtype))
    gap = _link_gap(w_A, w_B, d)
    energy_form = fem.form(
        0.5 * ufl.inner(fem.Constant(spiders_A, K.astype(dtype)) * gap, gap) * ufl.dx(spiders_A),
        entity_maps=[pair],
        dtype=dtype,
    )

    def energy() -> float:
        return comm.allreduce(fem.assemble_scalar(energy_form), op=MPI.SUM).real

    return num, points_A, w_A, w_B, K, energy


def _rigid_motion(t, theta, centre):
    """The spider dofs (t + theta x (x - centre), theta) of a rigid motion, at points `x`."""
    return lambda x: np.vstack(
        [t[:, None] + np.cross(theta, (x - centre[:, None]).T).T, np.tile(theta[:, None], x.shape[1])]
    )


@pytest.mark.parametrize("dtype", _scalar_types)
@pytest.mark.parametrize("k_t", [0.0, 3.0])
def test_rigid_link_spring_rigid_motion(dtype, k_t):
    """A rigid motion of the springs and the spiders stores no energy, and turning spider B about
    the axis by phi stores k_t phi^2 / 2 per spring, none for a free pin."""
    num, _, w_A, w_B, _, energy = _rigid_links(dtype, k_t)
    t, theta, centre = np.array([0.2, -0.1, 0.3]), np.array([0.05, -0.2, 0.1]), np.array([1.5, 0.0, 0.2])
    w_A.interpolate(_rigid_motion(t, theta, centre))
    w_B.interpolate(_rigid_motion(t, theta, centre))
    scale = 1e2 * (np.abs(t).max() + np.abs(theta).max()) ** 2 * num
    tol = 1e3 * np.finfo(dtype).eps * scale
    assert abs(energy()) < tol

    # Spider B turned by phi about the axis, about its own point, so its translation is unchanged
    phi = 0.3
    turn = np.concatenate([np.zeros(3), phi * _axis])
    w_B.interpolate(lambda x: _rigid_motion(t, theta, centre)(x) + turn[:, None])
    assert np.isclose(energy(), 0.5 * k_t * phi**2 * num, atol=tol)


@pytest.mark.parametrize("dtype", _scalar_types)
def test_rigid_link_spring_energy(dtype):
    """For any motion of the spiders the energy is that of the gaps delta computed spring by spring,
    delta^H K delta / 2."""
    num, points_A, w_A, w_B, K, energy = _rigid_links(dtype, k_t=3.0)
    imaginary = 0.5j if np.issubdtype(dtype, np.complexfloating) else 0.0

    def motion(shift):
        return lambda x: np.vstack(
            [np.sin(shift + (i + 1) * x[0]) + imaginary * np.cos(shift + x[0] + i) for i in range(6)]
        )

    w_A.interpolate(motion(0.0))
    w_B.interpolate(motion(1.0))
    expected = 0.0
    for k in range(num):
        value_A, value_B = dolfinx_mpc.spider_values(w_A, k), dolfinx_mpc.spider_values(w_B, k)
        delta = np.concatenate([value_B[:3] - value_A[:3] - np.cross(value_A[3:], _offset), value_B[3:] - value_A[3:]])
        expected += 0.5 * np.vdot(delta, K @ delta).real
    tol = 1e3 * np.finfo(dtype).eps * np.abs(K).max() * num
    assert np.isclose(energy(), expected, rtol=tol, atol=tol)

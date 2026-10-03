# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Tests for changing the coefficients of a finalized multi point constraint."""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import numpy.testing as nt
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.mesh import create_unit_square

import dolfinx_mpc

_eps = np.finfo(default_real_type).eps
_atol = 500 * _eps


def _slave_boundary(x):
    return np.isclose(x[0], 1, atol=_atol)


def _master_boundary(x):
    return np.isclose(x[0], 0, atol=_atol)


def _periodic_relation(x):
    out = x.copy()
    out[0] = x[0] - 1
    return out


def _space(mesh, element):
    family, degree, shape = element
    return fem.functionspace(mesh, (family, degree, shape))


def _create_mpc(V, scale, dtype, bcs=None):
    """Periodic constraint in x, keeping every master so that it can be rescaled."""
    bcs = [] if bcs is None else bcs
    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(
        V, _slave_boundary, _periodic_relation, bcs, scale=np.dtype(dtype).type(scale), tol=None
    )
    mpc.finalize()
    return mpc


def _master_bc(V, dtype):
    """Dirichlet condition on the master side only, so that masters are eliminated."""
    dofs = fem.locate_dofs_geometrical(V, _master_boundary)
    g = fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: np.tile(1 + x[1], (V.dofmap.index_map_bs, 1)))
    return fem.dirichletbc(g, dofs)


_elements = [("Lagrange", 1, None), ("Lagrange", 2, None), ("Lagrange", 1, (2,))]
_dtypes = [np.float64, np.complex128] if default_real_type == np.float64 else [np.float32, np.complex64]


@pytest.mark.parametrize("element", _elements)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("with_bc", [False, True])
def test_scale_matches_creation(element, dtype, with_bc):
    """Scaling a scale-1 constraint by s equals creating it with scale s."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 4)
    V = _space(mesh, element)
    s = 0.7 - 0.4j if np.issubdtype(dtype, np.complexfloating) else 0.7
    bcs = [_master_bc(V, dtype)] if with_bc else []

    mpc = _create_mpc(V, 1.0, dtype, bcs)
    ref = _create_mpc(V, s, dtype, bcs)
    if with_bc:
        assert mpc.has_inhomogeneity
    mpc.scale_coefficients(np.dtype(dtype).type(s))

    tol = 50 * np.finfo(dtype).eps
    c, o = mpc.all_coefficients()
    c_ref, o_ref = ref.all_coefficients()
    nt.assert_array_equal(o, o_ref)
    nt.assert_allclose(c, c_ref, atol=tol)
    nt.assert_array_equal(mpc.all_masters(), ref.all_masters())
    nt.assert_allclose(mpc.coefficients()[0], ref.coefficients()[0], atol=tol)
    nt.assert_allclose(mpc.constants, ref.constants, atol=tol)

    # Backsubstitution sees the new coefficients and offsets
    u = fem.Function(mpc.function_space, dtype=dtype)
    u_ref = fem.Function(ref.function_space, dtype=dtype)
    u.x.array[:] = np.arange(len(u.x.array))
    u_ref.x.array[:] = u.x.array
    mpc.backsubstitution(u)
    ref.backsubstitution(u_ref)
    nt.assert_allclose(u.x.array, u_ref.x.array, atol=tol * len(u.x.array))


@pytest.mark.parametrize("element", _elements)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("kind", ["ufl", "expression", "function"])
def test_scale_by_expression(element, dtype, kind):
    """A spatially varying factor multiplies the masters of each slave by its value at the slave."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 4)
    V = _space(mesh, element)
    mpc = _create_mpc(V, 1.0, dtype)
    c0, offsets = mpc.all_coefficients()
    c0 = c0.copy()

    k = fem.Constant(mesh, dtype(2.0))
    x = ufl.SpatialCoordinate(mesh)
    f = k * (1 + x[1])
    if V.dofmap.index_map_bs > 1:
        f = ufl.as_vector([f, 2 * f])
    if np.issubdtype(dtype, np.complexfloating):
        f = f * (1 + 1j)

    def factor_at(xs, comp):
        val = 2.0 * (1 + xs[:, 1]) * (comp + 1)
        return val * (1 + 1j) if np.issubdtype(dtype, np.complexfloating) else val

    if kind == "ufl":
        scale = f
    elif kind == "expression":
        scale = fem.Expression(f, V.element.interpolation_points, dtype=dtype)
    else:
        scale = fem.Function(V, dtype=dtype)
        scale.interpolate(fem.Expression(f, V.element.interpolation_points, dtype=dtype))
    mpc.scale_coefficients(scale)

    bs = V.dofmap.index_map_bs
    xs = V.tabulate_dof_coordinates()
    slaves = mpc.slaves
    expected = c0.copy()
    for slave in slaves:
        expected[offsets[slave] : offsets[slave + 1]] *= factor_at(xs[slave // bs][None, :], slave % bs)[0]
    tol = 100 * np.finfo(dtype).eps
    nt.assert_allclose(mpc.all_coefficients()[0], expected, rtol=tol, atol=tol)

    # Repeated calls compound
    if kind == "expression":
        k.value = 0.5
        mpc.scale_coefficients(scale)
        for slave in slaves:
            expected[offsets[slave] : offsets[slave + 1]] *= factor_at(xs[slave // bs][None, :], slave % bs)[0] / 4
        nt.assert_allclose(mpc.all_coefficients()[0], expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("with_bc", [False, True])
def test_update_coefficients(dtype, with_bc):
    """Replacing all coefficients, eliminated masters included, recomputes the offsets."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 6)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    s = 1.5 + 0.5j if np.issubdtype(dtype, np.complexfloating) else -1.5
    bcs = [_master_bc(V, dtype)] if with_bc else []
    mpc = _create_mpc(V, 1.0, dtype, bcs)
    ref = _create_mpc(V, s, dtype, bcs)

    c, _ = mpc.all_coefficients()
    mpc.update_coefficients(s * c)
    tol = 50 * np.finfo(dtype).eps
    nt.assert_allclose(mpc.all_coefficients()[0], ref.all_coefficients()[0], atol=tol)
    nt.assert_allclose(mpc.coefficients()[0], ref.coefficients()[0], atol=tol)
    nt.assert_allclose(mpc.constants, ref.constants, atol=tol)

    with pytest.raises(ValueError):
        mpc.update_coefficients(np.zeros(len(c) + 1, dtype=dtype))
    with pytest.raises(ValueError):
        mpc._cpp_object.scale_coefficients(np.ones(1, dtype=dtype))


def test_tol_none_keeps_masters():
    """With tol=None no basis value is cut, so there are at least as many masters."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    mpcs = []
    for tol in [None, _atol]:
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(V, _slave_boundary, _periodic_relation, [], tol=tol)
        mpc.finalize()
        mpcs.append(mpc)
    n_all, n_cut = (mesh.comm.allreduce(len(m.all_masters()), op=MPI.SUM) for m in mpcs)
    assert n_all >= n_cut > 0


def _solve(a, L, mpc, bcs):
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(L, mpc)
    dolfinx_mpc.apply_lifting(b, [a], [bcs], mpc)
    dolfinx_mpc.apply_mpc_lifting(b, [a], constraint=mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    solver = PETSc.KSP().create(mpc.function_space.mesh.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType("mumps")
    solver.setOperators(A)
    uh = fem.Function(mpc.function_space)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.backsubstitution(uh)
    norm_A = A.norm(PETSc.NormType.FROBENIUS)
    for obj in (solver, A, b):
        obj.destroy()
    return uh, norm_A


@pytest.mark.parametrize("with_bc", [False, True])
def test_assembly_after_scaling(with_bc):
    """Assembly and solve with a rescaled constraint equals those with a freshly created one."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 5)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    dtype = default_scalar_type
    s = 0.6
    bcs = [_master_bc(V, dtype)]
    if not with_bc:
        # Fix the solution on y=0 only, so no master is eliminated
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0, atol=_atol) & (x[0] < 1 - _atol))
        bcs = [fem.dirichletbc(dtype(0), dofs, V)]

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx)
    L = fem.form(ufl.inner(1 + x[0] * x[1], v) * ufl.dx)

    mpc = _create_mpc(V, 1.0, dtype, bcs)
    sk = fem.Constant(mesh, dtype(s))
    mpc.scale_coefficients(sk)
    ref = _create_mpc(V, s, dtype, bcs)

    uh, norm_A = _solve(a, L, mpc, bcs)
    uh_ref, norm_A_ref = _solve(a, L, ref, bcs)
    tol = 1e4 * _eps
    assert np.isclose(norm_A, norm_A_ref, rtol=tol)
    nt.assert_allclose(uh.x.array, uh_ref.x.array, atol=tol)

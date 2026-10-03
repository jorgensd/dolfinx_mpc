# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Monolithic (``kind="mpi"``) layout of a system with one constraint per block.

The nest layout is the reference: both layouts must represent the same reduced operator and
right-hand side. They number the degrees of freedom differently, so they are compared through the
functions the vectors are assigned to, restricted to the dofs that are not slaves.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.la.petsc import _ghost_update
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square, locate_entities_boundary

import dolfinx_mpc

_atol = 5e3 * np.finfo(default_real_type).eps


def _stokes(cell_type, els, order):
    """Periodic Stokes channel flow: vector velocity, a pressure block with no pressure-pressure form."""
    domain = create_unit_cube(MPI.COMM_WORLD, els, els, els, cell_type=cell_type, dtype=default_real_type)
    V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), order, shape=(3,), dtype=default_real_type))
    Q = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), order - 1, dtype=default_real_type))

    wall_facets = locate_entities_boundary(domain, 2, lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), fem.locate_dofs_topological(V, 2, wall_facets), V)

    def periodic_boundary(x):
        return np.logical_or(np.isclose(x[0], 1), np.isclose(x[2], 1))

    def periodic_map(x):
        out = x.copy()
        out[0][np.isclose(x[0], 1).nonzero()] -= 1
        out[2][np.isclose(x[2], 1).nonzero()] -= 1
        return out

    mpc_u = dolfinx_mpc.MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_map, [bc])
    mpc_p = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_p.create_periodic_constraint_geometrical(Q, periodic_boundary, periodic_map, [])
    dolfinx_mpc.finalize_multipointconstraints([mpc_u, mpc_p])

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    f = ufl.as_vector([fem.Constant(domain, default_scalar_type(1.0)), 0.0, 0.0])
    a = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, -ufl.inner(p, ufl.div(v)) * ufl.dx],
        [-ufl.inner(ufl.div(u), q) * ufl.dx, None],
    ]
    L = [ufl.inner(f, v) * ufl.dx, ufl.ZeroBaseForm((q,))]
    return fem.form(a), fem.form(L), [mpc_u, mpc_p], bc


def _random_functions(mpcs, seed):
    rng = np.random.default_rng(seed + MPI.COMM_WORLD.rank)
    functions = []
    for mpc in mpcs:
        f = fem.Function(mpc.function_space)
        imap, bs = f.function_space.dofmap.index_map, f.function_space.dofmap.index_map_bs
        f.x.array[: imap.size_local * bs] = rng.random(imap.size_local * bs)
        f.x.scatter_forward()
        functions.append(f)
    return functions


def _free_owned(mpc, array):
    """Entries of `array` owned by this process that are not slaves."""
    imap, bs = mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs
    n = imap.size_local * bs
    return array[:n][mpc.is_slave[:n] == 0]


def _assemble_nest_system(a, L, mpcs, bc, kind="nest"):
    A = dolfinx_mpc.assemble_matrix(a, mpcs, bcs=[bc], kind=kind)
    b = dolfinx_mpc.assemble_vector(L, mpcs, kind="nest")
    bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), [bc])
    dolfinx_mpc.apply_lifting(b, a, bcs1, constraint=mpcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, fem.bcs_by_block(fem.extract_function_spaces(L), [bc]))
    return A, b


def _assemble_block_system(a, L, mpcs, bc, kind=None):
    A = dolfinx_mpc.assemble_matrix(a, mpcs, bcs=[bc], kind=kind)
    b = dolfinx_mpc.assemble_vector(L, mpcs, kind=kind)
    bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), [bc])
    dolfinx_mpc.apply_lifting(b, a, bcs1, constraint=mpcs)
    dolfinx_mpc.apply_mpc_lifting(b, a, constraint=mpcs)
    _ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, fem.bcs_by_block(fem.extract_function_spaces(L), [bc]))
    return A, b


def _to_functions(vec, mpcs):
    _ghost_update(vec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    out = [fem.Function(mpc.function_space) for mpc in mpcs]
    fem.petsc.assign(vec, out)
    return out


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("order", [2, 3])
def test_monolithic_matches_nest(cell_type, order):
    """The monolithic operator and right-hand side are those of the nest layout.

    The velocity has block size 3, so the blocked indices of the element tensors have to be expanded
    for the local sub-matrix of the monolithic matrix. The pressure block has no form but carries
    slaves, and still needs its diagonal.
    """
    a, L, mpcs, bc = _stokes(cell_type, 3, order)
    A_nest, b_nest = _assemble_nest_system(a, L, mpcs, bc)
    A_block, b_block = _assemble_block_system(a, L, mpcs, bc)

    # Operator: compare the action on the same random functions
    x_functions = _random_functions(mpcs, seed=1)
    x_nest, x_block = b_nest.duplicate(), b_block.duplicate()
    fem.petsc.assign(x_functions, x_nest)
    fem.petsc.assign(x_functions, x_block)
    y_nest, y_block = b_nest.duplicate(), b_block.duplicate()
    A_nest.mult(x_nest, y_nest)
    A_block.mult(x_block, y_block)
    y_nest_f, y_block_f = _to_functions(y_nest, mpcs), _to_functions(y_block, mpcs)
    for mpc, f_nest, f_block in zip(mpcs, y_nest_f, y_block_f):
        np.testing.assert_allclose(
            _free_owned(mpc, f_block.x.array), _free_owned(mpc, f_nest.x.array), atol=_atol, rtol=_atol
        )

    # Right-hand side, including lifting of the Dirichlet condition
    b_nest_f, b_block_f = _to_functions(b_nest, mpcs), _to_functions(b_block, mpcs)
    for mpc, f_nest, f_block in zip(mpcs, b_nest_f, b_block_f):
        np.testing.assert_allclose(
            _free_owned(mpc, f_block.x.array), _free_owned(mpc, f_nest.x.array), atol=_atol, rtol=_atol
        )

    # Every slave row of the monolithic matrix is the identity row of the diagonal, also in the
    # block that has no form
    for mpc, f_x, f_y in zip(mpcs, x_functions, y_block_f):
        imap, bs = mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs
        n = imap.size_local * bs
        slave = mpc.is_slave[:n] == 1
        np.testing.assert_allclose(f_y.x.array[:n][slave], f_x.x.array[:n][slave], atol=_atol)


def _coupled(kind, options):
    """A non-singular periodic problem with a vector and a scalar block, solved with `kind`."""
    domain = create_unit_square(MPI.COMM_WORLD, 9, 7, dtype=default_real_type)
    V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 2, shape=(2,), dtype=default_real_type))
    Q = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type))
    wall = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
    bc = fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), fem.locate_dofs_topological(V, 1, wall), V)

    def right_edge(x):
        return np.isclose(x[0], 1.0)

    def to_left(x):
        out = x.copy()
        out[0] -= 1.0
        return out

    mpcs = []
    for space, bcs in ((V, [bc]), (Q, [])):
        mpc = dolfinx_mpc.MultiPointConstraint(space)
        mpc.create_periodic_constraint_geometrical(space, right_edge, to_left, bcs)
        mpcs.append(mpc)
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    x = ufl.SpatialCoordinate(domain)
    f = ufl.as_vector([ufl.sin(2 * ufl.pi * x[0]), x[1]])
    a = [
        [
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx,
            0.3 * ufl.inner(p, ufl.div(v)) * ufl.dx,
        ],
        [
            0.3 * ufl.inner(ufl.div(u), q) * ufl.dx,
            ufl.inner(p, q) * ufl.dx + ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx,
        ],
    ]
    L = [ufl.inner(f, v) * ufl.dx, ufl.inner(ufl.cos(2 * ufl.pi * x[0]), q) * ufl.dx]
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpcs,
        bcs=[bc],
        kind=kind,
        petsc_options_prefix=f"test_coupled_{kind}_",
        petsc_options=options,
    )
    return problem, mpcs


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
def test_linear_problem_monolithic():
    """`LinearProblem(kind="mpi")` agrees with `kind="nest"`, both solved directly.

    A direct solver keeps the comparison meaningful in single precision, where an iterative one
    converges too slowly to compare solutions.
    """
    direct = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    atol = 1e3 * np.finfo(default_real_type).eps

    block, mpcs = _coupled("mpi", direct)
    assert block.A.getType() != "nest"
    nest, _ = _coupled("nest", direct)
    assert nest.A.getType() == "nest"
    solution_block, solution_nest = block.solve(), nest.solve()

    for mpc, f_block, f_nest in zip(mpcs, solution_block, solution_nest):
        assert np.linalg.norm(f_block.x.array) > 0
        np.testing.assert_allclose(_free_owned(mpc, f_block.x.array), _free_owned(mpc, f_nest.x.array), atol=atol)


def _nonlinear(kind, options):
    """A periodic, nonlinear problem with a scalar block each for `u` and `p`, solved with `kind`."""
    domain = create_unit_square(MPI.COMM_WORLD, 8, 6, dtype=default_real_type)
    V = fem.functionspace(domain, ("Lagrange", 2))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    wall = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
    bc = fem.dirichletbc(default_scalar_type(0.0), fem.locate_dofs_topological(V, 1, wall), V)

    def right_edge(x):
        return np.isclose(x[0], 1.0)

    def to_left(x):
        out = x.copy()
        out[0] -= 1.0
        return out

    mpcs = []
    for space, bcs in ((V, [bc]), (Q, [])):
        mpc = dolfinx_mpc.MultiPointConstraint(space)
        mpc.create_periodic_constraint_geometrical(space, right_edge, to_left, bcs)
        mpcs.append(mpc)
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    # The unknowns live in the space of the constraints, the arguments in the original spaces
    uh, ph = (fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    x = ufl.SpatialCoordinate(domain)
    F = [
        (1 + uh**2) * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        + 0.3 * ufl.inner(ph, v) * ufl.dx
        - ufl.inner(ufl.sin(2 * ufl.pi * x[0]), v) * ufl.dx,
        ufl.inner(ph - uh**2, q) * ufl.dx + ufl.inner(ufl.grad(ph), ufl.grad(q)) * ufl.dx,
    ]
    unknowns, trials = [uh, ph], [ufl.TrialFunction(V), ufl.TrialFunction(Q)]
    J = [[ufl.derivative(F_i, u_j, du_j) for u_j, du_j in zip(unknowns, trials)] for F_i in F]
    tol = 100 * np.finfo(default_real_type).eps
    options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": tol,
        "snes_rtol": tol,
        "snes_error_if_not_converged": True,
        "ksp_error_if_not_converged": True,
        **options,
    }
    problem = dolfinx_mpc.NonlinearProblem(
        F,
        unknowns,
        mpcs,
        bcs=[bc],
        J=J,
        kind=kind,
        petsc_options_prefix=f"test_nonlinear_{kind}_",
        petsc_options=options,
    )
    return problem, mpcs


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
def test_nonlinear_problem_monolithic():
    """`NonlinearProblem` with the monolithic layout reaches the solution of the nest layout."""
    direct = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    block, mpcs = _nonlinear(None, direct)
    assert block.A.getType() != "nest"
    assert block.b.getAttr("_blocks") is not None
    nest, _ = _nonlinear("nest", direct)
    assert nest.A.getType() == "nest"

    solution_block, reason_block, iterations_block = block.solve()
    solution_nest, reason_nest, iterations_nest = nest.solve()
    assert reason_block > 0 and reason_nest > 0
    assert iterations_block == iterations_nest

    atol = 1e3 * np.finfo(default_real_type).eps
    for mpc, f_block, f_nest in zip(mpcs, solution_block, solution_nest):
        assert np.linalg.norm(f_block.x.array) > 0
        np.testing.assert_allclose(_free_owned(mpc, f_block.x.array), _free_owned(mpc, f_nest.x.array), atol=atol)


def test_kind_selects_the_layout():
    """`kind` of the unified functions picks nest or monolithic, with the types of a nest as given."""
    a, L, mpcs, bc = _stokes(CellType.tetrahedron, 2, 2)
    assert dolfinx_mpc.create_matrix(a, mpcs, kind="nest").getType() == "nest"
    assert dolfinx_mpc.create_matrix(a, mpcs).getType() != "nest"
    assert dolfinx_mpc.create_matrix(a, mpcs, kind="mpi").getType() != "nest"
    assert dolfinx_mpc.create_vector(L, mpcs, kind="nest").getType() == "nest"
    assert dolfinx_mpc.create_vector(L, mpcs).getAttr("_blocks") is not None

    # A nested sequence gives a nest whose blocks have the given types, none for a block without a form
    A = dolfinx_mpc.create_matrix(a, mpcs, kind=[["aij", "aij"], ["aij", None]])
    assert A.getType() == "nest"
    assert A.getNestSubMatrix(0, 0).getType() in ("seqaij", "mpiaij")

    # A matrix type for a single form
    form = fem.form(
        ufl.inner(ufl.TrialFunction(mpcs[1].function_space), ufl.TestFunction(mpcs[1].function_space)) * ufl.dx
    )
    assert dolfinx_mpc.create_matrix(form, mpcs[1], kind="aij").getType() in ("seqaij", "mpiaij")


def test_assemble_into_a_given_matrix_follows_its_type():
    """`assemble_matrix` with `A` and `assemble_vector` with `b` take the layout from them, not from `kind`."""
    a, L, mpcs, bc = _stokes(CellType.tetrahedron, 2, 2)
    A_block = dolfinx_mpc.create_matrix(a, mpcs)
    assert dolfinx_mpc.assemble_matrix(a, mpcs, bcs=[bc], A=A_block, kind="nest") is A_block
    b_nest = dolfinx_mpc.create_vector(L, mpcs, kind="nest")
    assert dolfinx_mpc.assemble_vector(L, mpcs, b_nest) is b_nest


def test_nest_wrappers_are_deprecated():
    """The `*_nest` functions still work, and say they are deprecated."""
    a, L, mpcs, bc = _stokes(CellType.tetrahedron, 2, 2)
    with pytest.warns(DeprecationWarning, match="create_matrix_nest"):
        A = dolfinx_mpc.create_matrix_nest(a, mpcs)
    with pytest.warns(DeprecationWarning, match="assemble_matrix_nest"):
        dolfinx_mpc.assemble_matrix_nest(A, a, mpcs, bcs=[bc])
    with pytest.warns(DeprecationWarning, match="create_vector_nest"):
        b = dolfinx_mpc.create_vector_nest(L, mpcs)
    with pytest.warns(DeprecationWarning, match="assemble_vector_nest"):
        dolfinx_mpc.assemble_vector_nest(b, L, mpcs)
    # They agree with the unified functions
    A_new, b_new = _assemble_nest_system(a, L, mpcs, bc)
    for i, j in ((0, 0), (0, 1), (1, 0)):
        assert np.isclose(A.getNestSubMatrix(i, j).norm(), A_new.getNestSubMatrix(i, j).norm())
    assert np.isclose(b.norm(), dolfinx_mpc.assemble_vector(L, mpcs, kind="nest").norm())


def test_kind_is_checked():
    """The kind is validated before anything is compiled."""
    V = fem.functionspace(create_unit_cube(MPI.COMM_WORLD, 2, 2, 2), ("Lagrange", 1))
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.finalize()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = ufl.inner(u, v) * ufl.dx
    with pytest.raises(ValueError, match="Unsupported kind"):
        dolfinx_mpc.LinearProblem(form, form, mpc, kind=3)
    with pytest.raises(ValueError, match="sequence of constraints"):
        dolfinx_mpc.LinearProblem(form, form, mpc, kind="nest")
    with pytest.raises(ValueError, match="sequence of constraints"):
        dolfinx_mpc.LinearProblem(form, form, mpc, kind=[["aij"]])

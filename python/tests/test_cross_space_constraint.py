# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Constraints whose masters are in another block of a blocked system.

Two spaces on one mesh, with a block-diagonal form and no coupling between the blocks. The slaves on
the right edge of the first space are tied to the dofs at the same points of the second, which then
also carries a periodic constraint of its own. The reduced operator `K^H A K` has off-diagonal
blocks although the form has none, and the assembled system must equal it, in both layouts.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pytest
import scipy.sparse
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.la.petsc import _ghost_update
from dolfinx.mesh import create_unit_square, locate_entities_boundary

import dolfinx_mpc
from dolfinx_mpc.utils import gather_PETScMatrix, gather_PETScVector

_tol = 500 * np.finfo(default_real_type).eps
_atol = 5e3 * np.finfo(default_real_type).eps
_coeff = default_scalar_type(2.0)


def _tie_to_other_space(V, Q, mixed=False):
    """Constraint on V: the slaves on x=1 (below the top corner) equal `_coeff` times Q there.

    V and Q are the same element on the same mesh, so a dof has the same local index, and owner, in
    both. Ghost slaves are included, as a cell owned here may hold a slave owned elsewhere. With
    `mixed`, every slave also has a master in its own block, an interior dof of V owned by process 0,
    and the blocks are given per master.
    """
    comm = MPI.COMM_WORLD
    np.testing.assert_array_equal(V.dofmap.list, Q.dofmap.list)
    imap = Q.dofmap.index_map
    num_local = imap.size_local + imap.num_ghosts
    x = V.tabulate_dof_coordinates()[:num_local]
    slaves = np.flatnonzero(np.isclose(x[:, 0], 1.0) & (x[:, 1] < 1.0 - _tol)).astype(np.int32)
    masters = imap.local_to_global(slaves).astype(np.int64)
    ghost = slaves >= imap.size_local
    owners = np.full(len(slaves), comm.rank, dtype=np.int32)
    owners[ghost] = imap.owners[slaves[ghost] - imap.size_local]
    coeffs = np.full(len(slaves), _coeff, dtype=default_scalar_type)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    if not mixed:
        mpc.add_constraint(
            V, slaves, masters, coeffs, owners, np.arange(len(slaves) + 1, dtype=np.int32), master_space=Q
        )
        return mpc

    # An interior dof of V, owned by process 0, as a second master of every slave
    interior = None
    if comm.rank == 0:
        x_owned = x[: imap.size_local]
        candidates = np.flatnonzero((x_owned[:, 0] > 0.3) & (x_owned[:, 0] < 0.7))
        interior = int(V.dofmap.index_map.local_to_global(candidates[:1].astype(np.int32))[0])
    interior = comm.bcast(interior, root=0)
    n = len(slaves)
    pairs_masters = np.empty(2 * n, dtype=np.int64)
    pairs_masters[0::2], pairs_masters[1::2] = masters, interior
    pairs_coeffs = np.empty(2 * n, dtype=default_scalar_type)
    pairs_coeffs[0::2], pairs_coeffs[1::2] = coeffs, default_scalar_type(0.5)
    pairs_owners = np.empty(2 * n, dtype=np.int32)
    pairs_owners[0::2], pairs_owners[1::2] = owners, 0
    blocks = np.tile(np.array([1, 0], dtype=np.int32), n)
    mpc.add_constraint(
        V,
        slaves,
        pairs_masters,
        pairs_coeffs,
        pairs_owners,
        np.arange(0, 2 * n + 1, 2, dtype=np.int32),
        master_blocks=blocks,
    )
    return mpc


def _problem(degree=1, mixed=False):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 4)
    V = fem.functionspace(mesh, ("Lagrange", degree))
    Q = fem.functionspace(mesh, ("Lagrange", degree))

    left = locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(default_scalar_type(0.7), fem.locate_dofs_topological(V, 1, left), V)

    mpc_v = _tie_to_other_space(V, Q, mixed)
    mpc_q = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_q.create_periodic_constraint_geometrical(
        Q,
        lambda x: np.isclose(x[1], 1.0),
        lambda x: np.vstack((x[0], x[1] - 1.0, x[2])),
        [],
    )
    mpcs = [mpc_v, mpc_q]
    dolfinx_mpc.finalize_multipointconstraints(mpcs)

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    x = ufl.SpatialCoordinate(mesh)
    a = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx, None],
        [None, ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx + 2 * ufl.inner(p, q) * ufl.dx],
    ]
    L = [ufl.inner(x[1], v) * ufl.dx, ufl.inner(ufl.sin(x[0]), q) * ufl.dx]
    return a, L, mpcs, [bc], (V, Q)


def _block_offsets(spaces):
    """Start of each block in the block-major global numbering, all of block 0 first."""
    sizes = [V.dofmap.index_map.size_global * V.dofmap.index_map_bs for V in spaces]
    return np.concatenate(([0], np.cumsum(sizes)))


def _monolithic_to_block_major(spaces):
    """Permutation from the global numbering of a monolithic system to the block-major one."""
    comm = MPI.COMM_WORLD
    offsets = _block_offsets(spaces)
    local = []
    for k, V in enumerate(spaces):
        imap, bs = V.dofmap.index_map, V.dofmap.index_map_bs
        start = imap.local_range[0] * bs
        local.append(offsets[k] + start + np.arange(imap.size_local * bs, dtype=np.int64))
    # A monolithic system lists the owned dofs of every block, process by process
    return np.concatenate(comm.allgather(np.concatenate(local)))


def _gather_K(mpcs, spaces, root=0):
    """`K` of `x = K x_red` over every block, in block-major numbering, without the slave columns."""
    comm = MPI.COMM_WORLD
    offsets = _block_offsets(spaces)
    rows, cols, vals = [], [], []
    for k, mpc in enumerate(mpcs):
        cpp = mpc._cpp_object
        imap = cpp.function_space.dofmap.index_map
        bs = cpp.function_space.dofmap.index_map_bs
        masters = cpp.masters
        coeffs, _ = cpp.coefficients()
        blocks = cpp.master_blocks
        for s in cpp.slaves[: cpp.num_local_slaves]:
            row = offsets[k] + imap.local_to_global(np.array([s // bs], dtype=np.int32))[0] * bs + s % bs
            for j in range(masters.offsets[s], masters.offsets[s + 1]):
                b = blocks[j]
                m = masters.array[j]
                space_b = mpcs[b]._cpp_object.function_space
                bs_b = space_b.dofmap.index_map_bs
                m_global = space_b.dofmap.index_map.local_to_global(np.array([m // bs_b], dtype=np.int32))[0]
                rows.append(row)
                cols.append(offsets[b] + m_global * bs_b + m % bs_b)
                vals.append(coeffs[j])
    rows_all = comm.gather(np.array(rows, dtype=np.int64), root=root)
    cols_all = comm.gather(np.array(cols, dtype=np.int64), root=root)
    vals_all = comm.gather(np.array(vals, dtype=default_scalar_type), root=root)
    if comm.rank != root:
        return None, None
    rows, cols, vals = np.concatenate(rows_all), np.concatenate(cols_all), np.concatenate(vals_all)
    N = offsets[-1]
    slaves = np.unique(rows)
    free = np.setdiff1d(np.arange(N), slaves)
    position = np.full(N, -1)
    position[free] = np.arange(len(free))
    assert (position[cols] >= 0).all(), "a master is a slave"
    K = scipy.sparse.coo_matrix(
        (
            np.concatenate((np.ones(len(free), dtype=default_scalar_type), vals)),
            (np.concatenate((free, rows)), np.concatenate((np.arange(len(free)), position[cols]))),
        ),
        shape=(N, len(free)),
    ).tocsr()
    return K, free


def _gather_nest_matrix(A):
    nr, nc = A.getNestSize()
    blocks = [[None] * nc for _ in range(nr)]
    for k in range(nr):
        for col in range(nc):
            A_kl = A.getNestSubMatrix(k, col)
            if A_kl.handle != 0:
                blocks[k][col] = gather_PETScMatrix(A_kl)
    if MPI.COMM_WORLD.rank == 0:
        return scipy.sparse.bmat(blocks, format="csr")
    return None


def _gather_block_major_matrix(A, spaces):
    if A.getType() == "nest":
        return _gather_nest_matrix(A)
    perm = _monolithic_to_block_major(spaces)
    A_csr = gather_PETScMatrix(A)
    if MPI.COMM_WORLD.rank == 0:
        P = scipy.sparse.coo_matrix((np.ones(len(perm)), (perm, np.arange(len(perm)))), shape=A_csr.shape).tocsr()
        return P @ A_csr @ P.T
    return None


def _gather_block_major_vector(b, spaces):
    if b.getType() == "nest":
        parts = [gather_PETScVector(b_k) for b_k in b.getNestSubVecs()]
        return np.concatenate(parts) if MPI.COMM_WORLD.rank == 0 else None
    perm = _monolithic_to_block_major(spaces)
    b_all = gather_PETScVector(b)
    if MPI.COMM_WORLD.rank == 0:
        out = np.zeros_like(b_all)
        out[perm] = b_all
        return out
    return None


def _lift(b, a, L, mpcs, bcs):
    bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs)
    dolfinx_mpc.apply_lifting(b, a, bcs1, constraint=mpcs)
    _ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, fem.bcs_by_block(fem.extract_function_spaces(L), bcs))


@pytest.mark.parametrize("kind", ["nest", "mpi"])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
def test_cross_space_constraint(kind, degree, mixed):
    """The assembled matrix and vector are `K^H A K` and `K^H b`, off-diagonal blocks included.

    With `mixed`, every slave has a master in each block, given by a block per master.
    """
    a, L, mpcs, bcs, spaces = _problem(degree, mixed)
    a, L = fem.form(a), fem.form(L)
    assert mpcs[0].has_cross_block_masters
    assert not mpcs[1].has_cross_block_masters

    A = dolfinx_mpc.assemble_matrix(a, mpcs, bcs, kind=kind)
    b = dolfinx_mpc.assemble_vector(L, mpcs, kind=kind)
    _lift(b, a, L, mpcs, bcs)

    # Reference without the constraint, and the same Dirichlet condition
    A_ref = dolfinx.fem.petsc.assemble_matrix(a, bcs=bcs, kind="nest")
    A_ref.assemble()
    b_ref = dolfinx.fem.petsc.assemble_vector(L, kind="nest")
    dolfinx.fem.petsc.apply_lifting(b_ref, a, bcs=fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs))
    for b_k in b_ref.getNestSubVecs():
        b_k.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_ref, fem.bcs_by_block(fem.extract_function_spaces(L), bcs))

    K, free = _gather_K(mpcs, spaces)
    A_mpc = _gather_block_major_matrix(A, spaces)
    A_full = _gather_nest_matrix(A_ref)
    b_mpc = _gather_block_major_vector(b, spaces)
    b_full = _gather_block_major_vector(b_ref, spaces)
    if MPI.COMM_WORLD.rank == 0:
        KAK = (K.conj().T @ A_full @ K).toarray()
        # The constraint couples the blocks although the form does not
        n0 = spaces[0].dofmap.index_map.size_global * spaces[0].dofmap.index_map_bs
        free0 = np.count_nonzero(free < n0)
        assert np.abs(KAK[:free0, free0:]).max() > 0
        np.testing.assert_allclose(A_mpc[free][:, free].toarray(), KAK, atol=_atol)
        np.testing.assert_allclose(b_mpc[free], K.conj().T @ b_full, atol=_atol)


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"), reason="PETSc was not built with MUMPS")
@pytest.mark.parametrize("kind", ["nest", "mpi"])
def test_cross_space_solution(kind):
    """A solve back-substitutes each slave from the function of the other block."""
    a, L, mpcs, bcs, spaces = _problem()
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpcs,
        bcs=bcs,
        kind=kind,
        petsc_options_prefix=f"test_cross_space_{kind}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    uh, ph = problem.solve()
    slaves = mpcs[0].slaves
    assert MPI.COMM_WORLD.allreduce(len(slaves), op=MPI.SUM) > 0
    np.testing.assert_allclose(uh.x.array[slaves], _coeff * ph.x.array[slaves], atol=_atol)
    assert np.linalg.norm(ph.x.array) > 0


def test_unknown_master_space():
    """A master space that is not one of the blocks finalized together is rejected."""
    a, L, mpcs, bcs, (V, Q) = _problem()
    mpc = _tie_to_other_space(V, Q)
    with pytest.raises(ValueError, match="master space"):
        dolfinx_mpc.finalize_multipointconstraints([mpc])


def test_single_matrix_rejects_cross_block_masters():
    """The matrix of one form cannot hold entries in another block."""
    a, L, mpcs, bcs, spaces = _problem()
    with pytest.raises(ValueError, match="another block"):
        dolfinx_mpc.assemble_matrix(fem.form(a[0][0]), mpcs[0], bcs)

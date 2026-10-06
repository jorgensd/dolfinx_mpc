# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Assembly of forms whose test and trial spaces live on different meshes.

A space on a submesh coupled to one on its parent (mortar / Lagrange multiplier
on a boundary, HDG facet spaces) gives a bilinear form whose two argument
spaces have unrelated cell numberings, and whose kernel asks for the facet
permutation. Both were mishandled before.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import GhostMode, create_submesh, create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc
import dolfinx_mpc.utils


def _mortar_spaces(N, ghost_mode, degree):
    """A P-`degree` space on a unit square and a P-`degree` space on its left edge."""
    msh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=ghost_mode)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)

    left = np.sort(locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)))
    submesh, cell_map, _, _ = create_submesh(msh, fdim, left)
    ft = meshtags(msh, fdim, left, np.full(len(left), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    V = fem.functionspace(msh, ("Lagrange", degree))
    Vbar = fem.functionspace(submesh, ("Lagrange", degree))
    # The parent mesh is reachable as `V.mesh`, so it is not returned separately
    return cell_map, ds, V, Vbar


def _mortar_setup(N, ghost_mode, degree):
    """The mortar coupling form, with test space on the mesh and trial on the submesh."""
    cell_map, ds, V, Vbar = _mortar_spaces(N, ghost_mode, degree)
    a = ufl.inner(ufl.TrialFunction(Vbar), ufl.TestFunction(V)) * ds(1)
    return V, Vbar, fem.form(a, entity_maps=[cell_map])


def _edge_periodic(space, atol):
    """Tie the top of the left edge to its bottom, giving the space real slaves."""
    mpc = dolfinx_mpc.MultiPointConstraint(space)
    mpc.create_periodic_constraint_geometrical(
        space,
        lambda x: np.isclose(x[1], 1.0, atol=atol),
        lambda x: np.vstack([x[0], 1.0 - x[1], x[2]]),
        [],
    )
    mpc.finalize()
    return mpc


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("degree", [1, 2])
def test_submesh_coupling_matches_dolfinx(ghost_mode, degree):
    """With no constraints, the MPC assembler must reproduce plain DOLFINx exactly.

    Guards two separate defects: the sparsity pattern used one cell index for
    both argument dofmaps, and the kernel was handed a null facet permutation.
    """
    V, Vbar, form = _mortar_setup(8, ghost_mode, degree)

    A_ref = dolfinx.fem.petsc.create_matrix(form)
    dolfinx.fem.petsc.assemble_matrix(A_ref, form)
    A_ref.assemble()

    mpc_V = dolfinx_mpc.MultiPointConstraint(V)
    mpc_V.finalize()
    mpc_bar = dolfinx_mpc.MultiPointConstraint(Vbar)
    mpc_bar.finalize()

    A = dolfinx_mpc.cpp.mpc.create_matrix(form._cpp_object, mpc_V._cpp_object, mpc_bar._cpp_object)
    dolfinx_mpc.assemble_matrix(form, (mpc_V, mpc_bar), A=A)

    assert np.isclose(A.norm(), A_ref.norm(), rtol=1e-13)
    A_ref.axpy(-1.0, A)
    assert A_ref.norm() < 1e-12 * max(A.norm(), 1.0)

    A.destroy()
    A_ref.destroy()


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_submesh_coupling_with_slaves(ghost_mode):
    """The row space carries a constraint, so masters are inserted into the pattern.

    PETSc raises on a new nonzero, so a pattern missing the master entries
    fails loudly here rather than silently.
    """
    V, Vbar, form = _mortar_setup(8, ghost_mode, 1)
    atol = 500 * np.finfo(dolfinx.default_real_type).eps

    mpc_V = dolfinx_mpc.MultiPointConstraint(V)
    mpc_V.create_periodic_constraint_geometrical(
        V,
        lambda x: np.isclose(x[0], 1.0, atol=atol),
        lambda x: np.vstack([x[0] - 1.0, x[1], x[2]]),
        [],
    )
    mpc_V.finalize()
    mpc_bar = dolfinx_mpc.MultiPointConstraint(Vbar)
    mpc_bar.finalize()

    A = dolfinx_mpc.cpp.mpc.create_matrix(form._cpp_object, mpc_V._cpp_object, mpc_bar._cpp_object)
    dolfinx_mpc.assemble_matrix(form, (mpc_V, mpc_bar), A=A)
    assert np.isfinite(A.norm())
    A.destroy()


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("degree", [1, 2])
def test_submesh_vector_matches_dolfinx(ghost_mode, degree):
    """A linear form whose test function lives on the submesh.

    `_assemble_entities_impl` took the cell index from the *integration* mesh
    and used it for the test space's dofmap, constraint and dof transformation.
    Those are the same cell only when both live on one mesh; here it indexed the
    submesh with a parent cell index and read out of range.
    """
    cell_map, ds, V, Vbar = _mortar_spaces(8, ghost_mode, degree)
    x = ufl.SpatialCoordinate(V.mesh)
    L = fem.form(ufl.inner(ufl.sin(2 * ufl.pi * x[1]), ufl.TestFunction(Vbar)) * ds(1), entity_maps=[cell_map])

    mpc = dolfinx_mpc.MultiPointConstraint(Vbar)
    mpc.finalize()
    b = dolfinx_mpc.assemble_vector(L, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    b_ref = dolfinx.fem.petsc.assemble_vector(L)
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert b.norm() > 0.0
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc)
    b.destroy()
    b_ref.destroy()


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_submesh_vector_with_slaves(ghost_mode):
    """As above, but the submesh space carries slaves.

    This is what exercises `cell_to_slaves` being looked up with the test
    space's own cell rather than the integration mesh's.
    """
    atol = 500 * np.finfo(dolfinx.default_real_type).eps
    cell_map, ds, V, Vbar = _mortar_spaces(8, ghost_mode, 1)
    x = ufl.SpatialCoordinate(V.mesh)
    L = fem.form(ufl.inner(ufl.sin(2 * ufl.pi * x[1]), ufl.TestFunction(Vbar)) * ds(1), entity_maps=[cell_map])

    mpc = _edge_periodic(Vbar, atol)
    assert mpc.num_local_slaves >= 0  # the constraint exists on at least one rank

    b = dolfinx_mpc.assemble_vector(L, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    b_ref = dolfinx.fem.petsc.assemble_vector(L)
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc)
    b.destroy()
    b_ref.destroy()


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_submesh_lifting_with_slaves(ghost_mode):
    """Lifting a Dirichlet condition out of a form whose spaces live on two meshes.

    `lift_bc_entities` looked the row constraint's slaves up with the *column*
    cell, while everything else in that branch (`dmap0`, `is_slave`, `masters`)
    belongs to the row space.
    """
    atol = 500 * np.finfo(dolfinx.default_real_type).eps
    cell_map, ds, V, Vbar = _mortar_spaces(8, ghost_mode, 1)
    a = fem.form(ufl.inner(ufl.TrialFunction(Vbar), ufl.TestFunction(V)) * ds(1), entity_maps=[cell_map])
    L = fem.form(ufl.inner(fem.Constant(V.mesh, dolfinx.default_scalar_type(0.0)), ufl.TestFunction(V)) * ufl.dx)

    u_bc = fem.Function(Vbar)
    u_bc.x.array[:] = 1.5
    bdofs = fem.locate_dofs_geometrical(Vbar, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(u_bc, bdofs)

    mpc = _edge_periodic(V, atol)

    b = dolfinx_mpc.assemble_vector(L, mpc)
    dolfinx_mpc.apply_lifting(b, [a], [[bc]], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    b_ref = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b_ref, [a], bcs=[[bc]])
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert b_ref.norm() > 0.0
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc)
    b.destroy()
    b_ref.destroy()


@pytest.mark.parametrize("block_size", [1, 3])
def test_zero_form_reserves_slave_diagonal(block_size):
    """A diagonal block with no integrals still needs its slave diagonal reserved.

    `insert_slave_diagonal` writes `(s, s)` for every owned slave of a diagonal
    block whatever the form contains, so `create_sparsity_pattern` must reserve
    those entries; a `ufl.ZeroBaseForm` block has no integral to reserve them
    incidentally. `slaves()` is unrolled while the pattern is indexed by blocks,
    which is why `block_size > 1` is covered: getting that wrong corrupts the
    heap inside `SparsityPattern::finalize`.
    """
    atol = 500 * np.finfo(dolfinx.default_real_type).eps
    msh = create_unit_square(MPI.COMM_WORLD, 8, 8)
    element = ("Lagrange", 1) if block_size == 1 else ("Lagrange", 1, (block_size,))
    V = fem.functionspace(msh, element)
    mpc = _edge_periodic(V, atol)

    zero = fem.form(ufl.ZeroBaseForm((ufl.TrialFunction(V), ufl.TestFunction(V))))
    A = dolfinx_mpc.assemble_matrix(zero, mpc, [])

    diagonal = A.getDiagonal()
    owned_slaves = np.asarray(mpc.slaves, dtype=np.int32)[: mpc.num_local_slaves]
    expected = np.zeros(diagonal.local_size, dtype=diagonal.array_r.dtype)
    expected[owned_slaves] = 1.0
    assert np.allclose(diagonal.array_r, expected)

    diagonal.destroy()
    A.destroy()

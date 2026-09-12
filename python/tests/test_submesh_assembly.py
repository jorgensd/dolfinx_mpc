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

import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import GhostMode, create_submesh, create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc


def _mortar_setup(N, ghost_mode, degree):
    """A P-`degree` space on a unit square, a P-`degree` space on its left edge."""
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
    a = ufl.inner(ufl.TrialFunction(Vbar), ufl.TestFunction(V)) * ds(1)
    return V, Vbar, fem.form(a, entity_maps=[cell_map])


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
        tol=atol,
    )
    mpc_V.finalize()
    mpc_bar = dolfinx_mpc.MultiPointConstraint(Vbar)
    mpc_bar.finalize()

    A = dolfinx_mpc.cpp.mpc.create_matrix(form._cpp_object, mpc_V._cpp_object, mpc_bar._cpp_object)
    dolfinx_mpc.assemble_matrix(form, (mpc_V, mpc_bar), A=A)
    assert np.isfinite(A.norm())
    A.destroy()

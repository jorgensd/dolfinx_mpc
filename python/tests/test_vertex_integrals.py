# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Vertex (`dP`) and ridge (`dr`) integrals under a multi point constraint.

Both are integrals over one entity of a cell, assembled like exterior facets. They were
previously skipped by the assemblers, so their contributions were silently lost.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh

import dolfinx_mpc
import dolfinx_mpc.utils

_atol = 5e3 * np.finfo(default_real_type).eps


def _tags(domain, dim, marker):
    """The entities of dimension `dim` satisfying `marker`, tagged 1."""
    domain.topology.create_entities(dim)
    entities = mesh.locate_entities(domain, dim, marker)
    return mesh.meshtags(domain, dim, entities, np.full(len(entities), 1, dtype=np.int32))


def _compare(a, L, mpc, bcs):
    """Matrix, vector and lifting of `a` and `L` against `K^H A K` and `K^H b`."""
    a_form, L_form = fem.form(a), fem.form(L)
    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(L_form, mpc)
    dolfinx_mpc.apply_lifting(b, [a_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    A_ref = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A_ref.assemble()
    b_ref = dolfinx.fem.petsc.assemble_vector(L_form)
    dolfinx.fem.petsc.apply_lifting(b_ref, [a_form], [bcs])
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b_ref, bcs)

    assert mpc.V.mesh.comm.allreduce(mpc.num_local_slaves, op=MPI.SUM) > 0
    dolfinx_mpc.utils.compare_mpc_lhs(A_ref, A, mpc, atol=_atol)
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc)
    for obj in (A, b, A_ref, b_ref):
        obj.destroy()


def _periodic(V, bcs):
    """Tie the dofs on `x = 1` to those on `x = 0`."""
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(
        V,
        lambda x: np.isclose(x[0], 1.0, atol=_atol),
        lambda x: np.vstack([x[0] - 1.0, x[1], x[2]]),
        bcs,
    )
    mpc.finalize()
    return mpc


@pytest.mark.parametrize("element", [("Lagrange", 2), ("Lagrange", 1, (2,))])
def test_vertex_integrals(element):
    """`dP` terms on slave and master vertices, in the matrix, the vector and the lifting."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(domain, element)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    vector = V.dofmap.index_map_bs > 1

    # Vertices on the slave edge, the master edge and the Dirichlet edge
    tags = _tags(
        domain,
        0,
        lambda x: (
            np.isclose(x[0], 0.0, atol=_atol) | np.isclose(x[0], 1.0, atol=_atol) | np.isclose(x[1], 0.0, atol=_atol)
        ),
    )
    dP = ufl.Measure("dP", domain=domain, subdomain_data=tags)
    weight = 2 + x[1]
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + weight * ufl.inner(u, v) * dP(1)
    if vector:
        f = ufl.as_vector((1 + x[1], x[0] * x[1]))
        g = ufl.as_vector((x[0], 1 + x[0]))
    else:
        f = 1 + x[0] * x[1]
        g = 1 + x[0]
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(f, v) * dP(1)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g, V.element.interpolation_points))
    bcs = [fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0)))]
    _compare(a, L, _periodic(V, bcs), bcs)


def test_point_forces():
    """A point force on a master vertex reaches the reduced system in full."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    tags = _tags(domain, 0, lambda x: np.isclose(x[0], 0.0, atol=_atol) & np.isclose(x[1], 0.5, atol=_atol))
    dP = ufl.Measure("dP", domain=domain, subdomain_data=tags)
    force = fem.Constant(domain, np.array([2.0, 3.0], dtype=default_scalar_type))
    mpc = _periodic(V, [])
    b = dolfinx_mpc.assemble_vector(fem.form(ufl.inner(force, ufl.TestFunction(V)) * dP(1)), mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    total = domain.comm.allreduce(b.array[: V.dofmap.index_map.size_local * 2].sum(), op=MPI.SUM)
    assert np.isclose(total, 5.0, atol=_atol)
    b.destroy()


@pytest.mark.parametrize("cell_type", [mesh.CellType.tetrahedron, mesh.CellType.hexahedron])
@pytest.mark.parametrize("element", [("Lagrange", 2), ("Lagrange", 1, (3,))])
def test_ridge_integrals(cell_type, element):
    """`dr` terms on edges through slaves, masters and Dirichlet dofs, in the matrix, the vector
    and the lifting."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, cell_type=cell_type)
    V = fem.functionspace(domain, element)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    vector = V.dofmap.index_map_bs > 1

    # The lines along y on the slave face x = 1 and the master face x = 0 at z = 0, and the
    # line y = z = 0 on the Dirichlet face y = 0
    def on_lines(x):
        along_y = (np.isclose(x[0], 0.0, atol=_atol) | np.isclose(x[0], 1.0, atol=_atol)) & np.isclose(
            x[2], 0.0, atol=_atol
        )
        return along_y | (np.isclose(x[1], 0.0, atol=_atol) & np.isclose(x[2], 0.0, atol=_atol))

    tags = _tags(domain, 1, on_lines)
    dr = ufl.Measure("dr", domain=domain, subdomain_data=tags)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + (1 + x[1]) * ufl.inner(u, v) * dr(1)
    if vector:
        f = ufl.as_vector((1 + x[1], x[0] * x[1], x[2]))
        g = ufl.as_vector((x[0], 1 + x[0], x[0] * x[2]))
    else:
        f = 1 + x[0] * x[1]
        g = 1 + x[0]
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(f, v) * dr(1)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g, V.element.interpolation_points))
    bcs = [fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0, atol=_atol)))]
    _compare(a, L, _periodic(V, bcs), bcs)


def test_line_forces():
    """A force per unit length on a line of slaves reaches the reduced system in full."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    tags = _tags(domain, 1, lambda x: np.isclose(x[0], 1.0, atol=_atol) & np.isclose(x[2], 1.0 / 3.0, atol=_atol))
    dr = ufl.Measure("dr", domain=domain, subdomain_data=tags)
    force = fem.Constant(domain, np.array([2.0, 3.0, 4.0], dtype=default_scalar_type))
    mpc = _periodic(V, [])
    b = dolfinx_mpc.assemble_vector(fem.form(ufl.inner(force, ufl.TestFunction(V)) * dr(1)), mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # The line, of length 1, is on the slave face: the force moves to the masters, whose
    # coefficients sum to one
    total = domain.comm.allreduce(b.array[: V.dofmap.index_map.size_local * 3].sum(), op=MPI.SUM)
    assert np.isclose(total, 9.0, atol=_atol), total
    b.destroy()

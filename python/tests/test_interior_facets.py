# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Interior facet integrals under a multi point constraint.

The element tensor of an interior facet spans the two cells sharing it, so a
slave on the facet appears twice in it, once per cell. On the interface between
two subdomains an argument may exist on one side only.
"""

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.graph import adjacencylist
from dolfinx.mesh import GhostMode, create_mesh, create_submesh, create_unit_square, locate_entities

import dolfinx_mpc
import dolfinx_mpc.utils
from dolfinx_mpc.utils.test import _gather_slaves_global

_atol = 5e3 * np.finfo(default_real_type).eps


def _periodic(V, indicator, relation, bcs=()):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, indicator, relation, list(bcs), tol=_atol)
    mpc.finalize()
    return mpc


def _compare_rectangular(A_ref, A, mpc0, mpc1, root=0):
    """`A` restricted to free rows and columns must equal `K0^H A_ref K1`."""
    K0 = dolfinx_mpc.utils.gather_transformation_matrix(mpc0, root=root)
    K1 = dolfinx_mpc.utils.gather_transformation_matrix(mpc1, root=root)
    A_ref_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_ref, root=root)
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    slaves0 = _gather_slaves_global(mpc0)
    slaves1 = _gather_slaves_global(mpc1)
    if MPI.COMM_WORLD.rank == root:
        KAK = np.conj(K0.T) @ A_ref_csr @ K1
        free0 = np.flatnonzero(np.isin(np.arange(A_csr.shape[0]), slaves0, invert=True))
        free1 = np.flatnonzero(np.isin(np.arange(A_csr.shape[1]), slaves1, invert=True))
        diff = np.abs(KAK - A_csr[free0, :][:, free1])
        assert diff.max() < _atol


@pytest.mark.parametrize(
    "element, ipdg",
    [(("Lagrange", 2), False), (("Lagrange", 1, (2,)), False), (("DG", 1), True)],
)
def test_interior_facets_periodic(element, ipdg):
    """Matrix, vector and lifting of `dS` forms against `K^H A K`.

    The constraint is periodic in `x`, so slaves on `x = 1` sit on interior
    facets along that edge and on both cells of each.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 6, ghost_mode=GhostMode.shared_facet)
    V = fem.functionspace(mesh, element)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    h_avg = ufl.avg(h)

    if V.dofmap.index_map_bs == 1:
        f = ufl.sin(2 * ufl.pi * x[1]) * x[0]
        g = 1 + x[1]
    else:
        f = ufl.as_vector((ufl.sin(2 * ufl.pi * x[1]), x[0]))
        g = ufl.as_vector((1 + x[1], x[1]))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    if ipdg:
        # Symmetric interior penalty, with Nitsche's method on the boundary
        alpha = 8.0
        a += (
            -ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS
            - ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
            + alpha / h_avg * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS
            + alpha / h * ufl.inner(u, v) * ufl.ds
        )
    else:
        # Continuous interior penalty on the gradient jump
        a += h_avg**2 * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * ufl.dS
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(ufl.avg(f), ufl.avg(v)) * ufl.dS
    a_form, L_form = fem.form(a), fem.form(L)

    # A Dirichlet condition on the bottom edge, lifted through the dS terms. A
    # DG space carries its boundary data weakly instead.
    bcs = []
    if not ipdg:
        u_bc = fem.Function(V)
        u_bc.interpolate(fem.Expression(g, V.element.interpolation_points))
        bcs = [fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0)))]

    def periodic_relation(x):
        return np.vstack([x[0] - 1.0, x[1], x[2]])

    mpc = _periodic(V, lambda x: np.isclose(x[0], 1.0, atol=_atol), periodic_relation, bcs)

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

    assert mesh.comm.allreduce(mpc.num_local_slaves, op=MPI.SUM) > 0
    dolfinx_mpc.utils.compare_mpc_lhs(A_ref, A, mpc, atol=_atol)
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc)

    for obj in (A, b, A_ref, b_ref):
        obj.destroy()


def _interface_entities(msh, interface_facets, plus_cells):
    """Integration entities for the interface, with `plus_cells` on the "+" side."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    entities = fem.compute_integration_domains(fem.IntegralType.interior_facet, msh.topology, interface_facets)
    entities = entities.reshape(-1, 4)
    num_cells = msh.topology.index_map(tdim).size_local + msh.topology.index_map(tdim).num_ghosts
    is_plus = np.zeros(num_cells, dtype=bool)
    is_plus[plus_cells] = True
    swap = ~is_plus[entities[:, 0]]
    entities[swap] = entities[swap][:, [2, 3, 0, 1]]
    return entities.ravel()


def _split_unit_square(comm, n):
    """Unit square with the cells left of x = 0.5 on rank 0 and the rest on rank 1.

    Every interface facet then has its two cells on different processes, so
    the ghost cell path is exercised, which the default partitioner does not
    guarantee. Cells sharing a facet across the split are ghosted on the
    other rank, as `GhostMode.shared_facet` would. Ranks beyond the second
    are left empty.
    """
    if comm.rank == 0:
        xs = np.linspace(0.0, 1.0, n + 1)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        x = np.column_stack([X.ravel(), Y.ravel()]).astype(default_real_type)
        v0 = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
        v1, v2, v3 = v0 + n + 1, v0 + n + 2, v0 + 1
        cells = np.vstack([np.column_stack([v0, v1, v2]), np.column_stack([v0, v2, v3])]).astype(np.int64)
    else:
        x = np.empty((0, 2), dtype=default_real_type)
        cells = np.empty((0, 3), dtype=np.int64)

    def partitioner(*args):
        # Destination of each cell: its owner, then the ranks ghosting it
        owner = np.where(x[cells].mean(axis=1)[:, 0] < 0.5, 0, 1 % comm.size).astype(np.int32)
        cells_of_edge = {}
        for c, cell in enumerate(cells):
            for edge in ((0, 1), (1, 2), (0, 2)):
                cells_of_edge.setdefault(tuple(sorted(cell[list(edge)])), []).append(c)
        ghosts = [set() for _ in range(len(cells))]
        for pair in cells_of_edge.values():
            if len(pair) == 2 and owner[pair[0]] != owner[pair[1]]:
                ghosts[pair[0]].add(owner[pair[1]])
                ghosts[pair[1]].add(owner[pair[0]])
        dest = [[owner[c], *sorted(ghosts[c])] for c in range(len(cells))]
        offsets = np.cumsum([0] + [len(d) for d in dest]).astype(np.int32)
        return adjacencylist(np.array([r for d in dest for r in d], dtype=np.int32), offsets)

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=default_real_type))
    return create_mesh(comm, cells, domain, x, partitioner)


@pytest.mark.parametrize("partition", ["default", "split"])
def test_interface_between_subdomains(partition):
    """Test and trial functions on either side of an interface.

    Each argument exists on one side of every interface facet only, so the
    joint element tensor has a missing cell for each of them. Both spaces
    carry a periodic constraint that puts a slave on the interface.
    """
    n = 8  # even, so no cell straddles x = 0.5
    if partition == "default":
        msh = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=GhostMode.shared_facet)
    else:
        msh = _split_unit_square(MPI.COMM_WORLD, n)
    tdim = msh.topology.dim
    left = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    right = locate_entities(msh, tdim, lambda x: x[0] >= 0.5)
    smsh_l, emap_l = create_submesh(msh, tdim, left)[0:2]
    smsh_r, emap_r = create_submesh(msh, tdim, right)[0:2]

    V_l = fem.functionspace(smsh_l, ("Lagrange", 2))
    V_r = fem.functionspace(smsh_r, ("Lagrange", 1))

    interface = locate_entities(msh, tdim - 1, lambda x: np.isclose(x[0], 0.5))
    dS = ufl.Measure("dS", domain=msh, subdomain_data=[(1, _interface_entities(msh, interface, left))])

    f_l = fem.Function(V_l)
    f_l.interpolate(lambda x: 1 + x[0] * x[1])
    u_l, v_r = ufl.TrialFunction(V_l), ufl.TestFunction(V_r)
    a = fem.form(ufl.inner(f_l("+") * u_l("+"), v_r("-")) * dS(1), entity_maps=[emap_l, emap_r])
    L = fem.form(ufl.inner(f_l("+"), v_r("-")) * dS(1), entity_maps=[emap_l, emap_r])

    # Ties top to bottom on each subdomain; the slave at (0.5, 1) lies on the
    # interface
    def top(x):
        return np.isclose(x[1], 1.0, atol=_atol)

    def to_bottom(x):
        return np.vstack([x[0], x[1] - 1.0, x[2]])

    # A condition on the upper part of the interface, short of the slave at
    # (0.5, 1), so that lifted values reach the row of the right-hand slave
    bc_dofs = fem.locate_dofs_geometrical(V_l, lambda x: np.isclose(x[0], 0.5) & (x[1] > 0.6) & (x[1] < 1.0 - _atol))
    bc = fem.dirichletbc(default_scalar_type(2.5), bc_dofs, V_l)

    mpc_l = _periodic(V_l, top, to_bottom, [bc])
    mpc_r = _periodic(V_r, top, to_bottom)
    for mpc in (mpc_l, mpc_r):
        assert msh.comm.allreduce(mpc.num_local_slaves, op=MPI.SUM) > 0

    # Matrix: test space on the right, trial on the left
    A = dolfinx_mpc.cpp.mpc.create_matrix(a._cpp_object, mpc_r._cpp_object, mpc_l._cpp_object)
    dolfinx_mpc.assemble_matrix(a, (mpc_r, mpc_l), A=A)
    A_ref = dolfinx.fem.petsc.create_matrix(a)
    dolfinx.fem.petsc.assemble_matrix(A_ref, a)
    A_ref.assemble()
    _compare_rectangular(A_ref, A, mpc_r, mpc_l)

    # Vector
    b = dolfinx_mpc.assemble_vector(L, mpc_r)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    b_ref = dolfinx.fem.petsc.assemble_vector(L)
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc_r)

    # Lifting alone, from zero
    with b.localForm() as b_local:
        b_local.set(0.0)
    dolfinx_mpc.apply_lifting(b, [a], [[bc]], mpc_r)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    with b_ref.localForm() as b_local:
        b_local.set(0.0)
    dolfinx.fem.petsc.apply_lifting(b_ref, [a], [[bc]])
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    assert b_ref.norm() > 1e-3
    dolfinx_mpc.utils.compare_mpc_rhs(b_ref, b, mpc_r)

    for obj in (A, A_ref, b, b_ref):
        obj.destroy()

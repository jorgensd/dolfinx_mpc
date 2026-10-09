# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Hanging nodes: the facets of a refined patch are tied to the coarse facets they subdivide
# with an inelastic contact condition.
from __future__ import annotations

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl

import dolfinx_mpc

COARSE, FINE = 2, 3


def hanging_node_mesh(comm: MPI.Comm) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
    """A 3x3 quadrilateral grid on [0, 2]x[0, 1] whose centre cell is split into 2x2 cells.

    The four coarse facets around the centre each face two fine facets, with a hanging node at
    their midpoint. Coarse facets are tagged `COARSE`, fine facets `FINE`.
    """
    xs, ys = np.array([0, 2 / 3, 4 / 3, 2]), np.array([0, 1 / 3, 2 / 3, 1])
    if comm.rank == 0:
        grid = np.array([[x, y] for y in ys for x in xs])
        hanging = np.array([[1, 1 / 3], [2 / 3, 1 / 2], [1, 2 / 3], [4 / 3, 1 / 2], [1, 1 / 2]])
        nodes = np.vstack([grid, hanging])
        coarse = [[i + 4 * j, i + 1 + 4 * j, i + 4 * (j + 1), i + 1 + 4 * (j + 1)] for j in range(3) for i in range(3)]
        coarse.pop(4)
        fine = [[5, 16, 17, 20], [16, 6, 20, 19], [17, 20, 9, 18], [20, 19, 18, 10]]
        cells = np.array(coarse + fine, dtype=np.int64)
    else:
        nodes = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 4), dtype=np.int64)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    mesh = dolfinx.mesh.create_mesh(comm, cells, domain, nodes)

    # The interface facets each have a single cell: the facet is coarse if that cell is outside
    # the refined patch
    tol = 500 * np.finfo(mesh.geometry.x.dtype).eps

    def on_interface(x):
        in_x = (x[0] > xs[1] - tol) & (x[0] < xs[2] + tol)
        in_y = (x[1] > ys[1] - tol) & (x[1] < ys[2] + tol)
        on_x = np.isclose(x[0], xs[1], atol=tol) | np.isclose(x[0], xs[2], atol=tol)
        on_y = np.isclose(x[1], ys[1], atol=tol) | np.isclose(x[1], ys[2], atol=tol)
        return (on_x & in_y) | (on_y & in_x)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, on_interface)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    interface_cells = np.array([f_to_c.links(f)[0] for f in facets], dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, interface_cells)
    in_patch = (
        (midpoints[:, 0] > xs[1]) & (midpoints[:, 0] < xs[2]) & (midpoints[:, 1] > ys[1]) & (midpoints[:, 1] < ys[2])
    )
    values = np.where(in_patch, FINE, COARSE).astype(np.int32)
    return mesh, dolfinx.mesh.meshtags(mesh, tdim - 1, facets, values)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_hanging_nodes(degree):
    mesh, ft = hanging_node_mesh(MPI.COMM_WORLD)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_contact_inelastic_condition(ft, FINE, COARSE)
    mpc.finalize()

    # The endpoints of the coarse facets are shared by both sides and not constrained. The
    # slaves are the hanging nodes and the interior dofs of the eight fine facets.
    fdim = mesh.topology.dim - 1
    shared = np.intersect1d(
        dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(FINE)),
        dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(COARSE)),
    )
    num_owned = V.dofmap.index_map.size_local
    owned_slaves = mpc.slaves[mpc.slaves < num_owned]
    assert not np.isin(shared[shared < num_owned], owned_slaves).any()
    num_slaves = mesh.comm.allreduce(mpc.num_local_slaves, op=MPI.SUM)
    assert num_slaves == 4 + 8 * (degree - 1)

    # The constrained space is conforming and contains Q_p, so the solution is exact
    x = ufl.SpatialCoordinate(mesh)
    u_ex = x[0] ** degree + x[1] ** degree + x[0] * x[1]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = -ufl.div(ufl.grad(u_ex)) * v * ufl.dx

    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(dolfinx.fem.Expression(u_ex, V.element.interpolation_points))

    # The interface facets are exterior facets, so the outer boundary is located geometrically
    def outer(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], 2) | np.isclose(x[1], 0) | np.isclose(x[1], 1)

    bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_geometrical(V, outer))
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpc,
        bcs=[bc],
        petsc_options_prefix=f"test_hanging_nodes_{degree}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    uh = problem.solve()
    problem.solver.destroy()

    error = np.max(np.abs(uh.x.array[:num_owned] - u_bc.x.array[:num_owned]), initial=0)
    error = mesh.comm.allreduce(error, op=MPI.MAX)
    assert error < 1e3 * np.finfo(dolfinx.default_scalar_type).eps

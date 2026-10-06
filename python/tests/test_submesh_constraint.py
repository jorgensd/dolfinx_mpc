# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Constraints between a space on a submesh and a space on its parent, through the entity map
from __future__ import annotations

from mpi4py import MPI

import numpy as np
import pytest
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.mesh import (
    CellType,
    create_submesh,
    create_unit_cube,
    create_unit_square,
    locate_entities,
    locate_entities_boundary,
)

import dolfinx_mpc

tol = 100 * np.finfo(default_real_type).eps


def _polynomial(degree, shape=()):
    """A polynomial of the given degree, so a Lagrange space of that degree holds it exactly."""

    def f(x):
        p = 1 + x[0] - 2 * x[1] + 0.5 * x[2]
        if degree > 1:
            p = p + 3 * x[0] * x[1] - x[1] ** 2
        return p if shape == () else np.vstack([p, 2 - p])

    return f


def _mesh_and_submesh(cell_type, codim):
    """A mesh, a submesh of codimension `codim` on part of it, the entity map and the entities."""
    if cell_type in (CellType.triangle, CellType.quadrilateral):
        mesh = create_unit_square(MPI.COMM_WORLD, 6, 5, cell_type, dtype=default_real_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 4, 3, cell_type, dtype=default_real_type)
    tdim = mesh.topology.dim
    if codim == 1:
        entities = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0) | np.isclose(x[1], 1))
    else:
        entities = locate_entities(mesh, tdim, lambda x: x[0] <= 0.5 + 1e-6)
    submesh, entity_map, _, _ = create_submesh(mesh, tdim - codim, entities)
    return mesh, submesh, entity_map, entities


def _tie(slave, master, entity_map, slave_root=None, master_root=None, **kwargs):
    """Tie `slave` to `master`, finalized together, and return the two constraints."""
    mpc_s = dolfinx_mpc.MultiPointConstraint(slave if slave_root is None else slave_root)
    mpc_m = dolfinx_mpc.MultiPointConstraint(master if master_root is None else master_root)
    mpc_s.create_submesh_constraint(slave, master, entity_map, **kwargs)
    dolfinx_mpc.finalize_multipointconstraints([mpc_s, mpc_m])
    return mpc_s, mpc_m


def _check_slaves(mpc_s, mpc_m, f_master, f_expected, component=None):
    """Back-substitute the master function `f_master`; every slave must equal `f_expected` at its
    coordinate. Returns the number of slaves summed over the processes."""
    u_m = fem.Function(mpc_m.function_space, dtype=default_scalar_type)
    u_m.interpolate(f_master)
    # Masters owned elsewhere are ghosts of the extended space, in no local cell
    u_m.x.scatter_forward()
    u_s = fem.Function(mpc_s.function_space, dtype=default_scalar_type)
    mpc_s.backsubstitution([u_s, u_m])

    V = mpc_s.function_space
    bs = V.dofmap.index_map_bs
    slaves = np.asarray(mpc_s.slaves)
    x = V.tabulate_dof_coordinates()[slaves // bs]
    expected = np.atleast_2d(f_expected(x.T))
    expected = expected[slaves % bs if expected.shape[0] > 1 else 0, np.arange(len(slaves))]
    np.testing.assert_allclose(u_s.x.array[slaves], expected, rtol=tol, atol=tol)
    if component is not None:
        assert np.all(slaves % bs == component)
    return MPI.COMM_WORLD.allreduce(mpc_s.num_local_slaves, op=MPI.SUM)


@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron])
@pytest.mark.parametrize("codim", [0, 1])
@pytest.mark.parametrize("slave_on", ["submesh", "parent"])
@pytest.mark.parametrize("degrees", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_submesh_tie(cell_type, codim, slave_on, degrees):
    """Every slave equals the master at its coordinate, and the slaves are exactly the dofs of the
    slave space on the submesh."""
    mesh, submesh, entity_map, entities = _mesh_and_submesh(cell_type, codim)
    slave_degree, master_degree = degrees
    if slave_on == "submesh":
        slave = fem.functionspace(submesh, ("Lagrange", slave_degree))
        master = fem.functionspace(mesh, ("Lagrange", master_degree))
    else:
        slave = fem.functionspace(mesh, ("Lagrange", slave_degree))
        master = fem.functionspace(submesh, ("Lagrange", master_degree))
    mpc_s, mpc_m = _tie(slave, master, entity_map)
    f = _polynomial(master_degree)
    num_slaves = _check_slaves(mpc_s, mpc_m, f, f)

    if slave_on == "submesh":
        expected = slave.dofmap.index_map.size_global
    else:
        tdim = mesh.topology.dim
        dofs = fem.locate_dofs_topological(slave, tdim - codim, entities)
        expected = MPI.COMM_WORLD.allreduce(np.count_nonzero(dofs < slave.dofmap.index_map.size_local), op=MPI.SUM)
    assert num_slaves == expected


@pytest.mark.parametrize("slave_on", ["submesh", "parent"])
def test_vector_tie(slave_on):
    """Component `b` is tied to component `b`."""
    mesh, submesh, entity_map, _ = _mesh_and_submesh(CellType.triangle, 1)
    sub = fem.functionspace(submesh, ("Lagrange", 2, (2,)))
    parent = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    slave, master = (sub, parent) if slave_on == "submesh" else (parent, sub)
    mpc_s, mpc_m = _tie(slave, master, entity_map)
    f = _polynomial(2, (2,))
    _check_slaves(mpc_s, mpc_m, f, f)


def test_subspaces():
    """A subspace on either side: only its component is slaved, and the master block is the space
    containing the master subspace."""
    mesh, submesh, entity_map, _ = _mesh_and_submesh(CellType.quadrilateral, 1)
    sub = fem.functionspace(submesh, ("Lagrange", 1, (2,)))
    parent = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    mpc_s, mpc_m = _tie(sub.sub(0), parent.sub(1), entity_map, slave_root=sub, master_root=parent)
    f = _polynomial(1, (2,))
    num_slaves = _check_slaves(mpc_s, mpc_m, f, lambda x: f(x)[1], component=0)
    assert num_slaves == sub.dofmap.index_map.size_global


def test_dirichlet_dofs_are_not_slaves():
    mesh, submesh, entity_map, entities = _mesh_and_submesh(CellType.triangle, 1)
    parent = fem.functionspace(mesh, ("Lagrange", 1))
    sub = fem.functionspace(submesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    top = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 1))
    bc = fem.dirichletbc(default_scalar_type(0), fem.locate_dofs_topological(parent, tdim - 1, top), parent)
    mpc_s, mpc_m = _tie(parent, sub, entity_map, bcs=[bc])
    f = _polynomial(1)
    num_slaves = _check_slaves(mpc_s, mpc_m, f, f)
    assert not np.any(np.isin(mpc_s.slaves, bc.dof_indices()[0]))

    on_submesh = fem.locate_dofs_topological(parent, tdim - 1, entities)
    free = np.setdiff1d(on_submesh, bc.dof_indices()[0])
    expected = MPI.COMM_WORLD.allreduce(np.count_nonzero(free < parent.dofmap.index_map.size_local), op=MPI.SUM)
    assert num_slaves == expected


def test_invalid_input():
    """Each error is raised on every process, before any communication."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 5, dtype=default_real_type)
    tdim = mesh.topology.dim
    facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
    submesh, entity_map, vertex_map, _ = create_submesh(mesh, tdim - 1, facets)
    parent = fem.functionspace(mesh, ("Lagrange", 1))
    sub = fem.functionspace(submesh, ("Lagrange", 1))
    other = create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=default_real_type)
    unrelated = fem.functionspace(other, ("Lagrange", 1))
    vector = fem.functionspace(mesh, ("Lagrange", 1, (2,)))

    with pytest.raises(ValueError, match="must relate the meshes"):
        dolfinx_mpc.MultiPointConstraint(sub).create_submesh_constraint(sub, unrelated, entity_map)
    with pytest.raises(ValueError, match="same number of components"):
        dolfinx_mpc.MultiPointConstraint(sub).create_submesh_constraint(sub, vector, entity_map)
    with pytest.raises(ValueError, match="cells of the submesh"):
        dolfinx_mpc.MultiPointConstraint(sub).create_submesh_constraint(sub, parent, vertex_map)
    with pytest.raises(ValueError, match="subspace"):
        dolfinx_mpc.MultiPointConstraint(parent).create_submesh_constraint(sub, parent, entity_map)

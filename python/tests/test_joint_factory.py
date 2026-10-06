# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Joint finalization of the constraints of several function spaces.

The constraints of independent meshes that do not refer to each other must come out exactly as
if each had been finalized on its own, and the checks that need the communicator must give the
same verdict on every process.
"""

from __future__ import annotations

from mpi4py import MPI

import numpy as np
import pytest
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import CellType, create_unit_square, locate_entities_boundary

import dolfinx_mpc


def _space(n_x, n_y, cell_type, comm=MPI.COMM_WORLD):
    mesh = create_unit_square(comm, n_x, n_y, cell_type=cell_type)
    return fem.functionspace(mesh, ("Lagrange", 1))


def _periodic(V):
    """Tie the right edge of the unit square to the left edge, without finalizing."""
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(
        V,
        lambda x: np.isclose(x[0], 1.0),
        lambda x: np.vstack((x[0] - 1.0, x[1], x[2])),
        [],
    )
    return mpc


def _assert_same_constraint(a, b):
    a, b = a._cpp_object, b._cpp_object
    np.testing.assert_array_equal(a.slaves, b.slaves)
    np.testing.assert_array_equal(a.is_slave, b.is_slave)
    assert a.num_local_slaves == b.num_local_slaves
    np.testing.assert_array_equal(a.masters.array, b.masters.array)
    np.testing.assert_array_equal(a.masters.offsets, b.masters.offsets)
    np.testing.assert_array_equal(a.owners.array, b.owners.array)
    np.testing.assert_array_equal(a.cell_to_slaves.array, b.cell_to_slaves.array)
    (data_a, offsets_a), (data_b, offsets_b) = a.coefficients(), b.coefficients()
    np.testing.assert_allclose(data_a, data_b)
    np.testing.assert_array_equal(offsets_a, offsets_b)
    map_a, map_b = a.function_space.dofmap.index_map, b.function_space.dofmap.index_map
    assert map_a.size_local == map_b.size_local
    np.testing.assert_array_equal(map_a.ghosts, map_b.ghosts)


def _raised_everywhere(comm, fn, exception=ValueError, match=None):
    """Assert that `fn` raises on every process, as a throw on only some would deadlock."""
    raised = 0
    try:
        fn()
    except exception as e:
        raised = 1
        if match is not None:
            assert match in str(e), str(e)
    assert comm.allreduce(raised, op=MPI.SUM) == comm.size


def test_joint_equals_separate():
    """Two meshes, of different cell type and resolution, each tied to itself."""
    V0 = _space(9, 9, CellType.triangle)
    V1 = _space(6, 7, CellType.quadrilateral)

    separate = [_periodic(V0), _periodic(V1)]
    for mpc in separate:
        mpc.finalize()

    joint = [_periodic(V0), _periodic(V1)]
    dolfinx_mpc.finalize_multipointconstraints(joint)

    for s, j in zip(separate, joint):
        assert j.finalized
        _assert_same_constraint(s, j)


def test_block_without_constraint():
    """A block with nothing to constrain still takes part in the collective calls."""
    V0 = _space(8, 8, CellType.triangle)
    V1 = _space(3, 3, CellType.triangle)

    separate = _periodic(V0)
    separate.finalize()

    joint = [_periodic(V0), dolfinx_mpc.MultiPointConstraint(V1)]
    dolfinx_mpc.finalize_multipointconstraints(joint)

    _assert_same_constraint(separate, joint[0])
    assert len(joint[1].slaves) == 0
    assert not joint[1].is_slave.any()
    assert not joint[1].has_inhomogeneity


def test_foreign_dirichlet_condition_names_block():
    """A condition on another block's space is rejected, and the block is named."""
    V0 = _space(5, 5, CellType.triangle)
    V1 = _space(5, 5, CellType.triangle)
    facets = locate_entities_boundary(V0.mesh, 1, lambda x: np.isclose(x[0], 0.0))
    dofs = fem.locate_dofs_topological(V0, 1, facets)
    bc0 = fem.dirichletbc(default_scalar_type(0), dofs, V0)

    mpcs = [dolfinx_mpc.MultiPointConstraint(V0), dolfinx_mpc.MultiPointConstraint(V1, bcs=[bc0])]
    _raised_everywhere(MPI.COMM_WORLD, lambda: dolfinx_mpc.finalize_multipointconstraints(mpcs), match="block 1")
    assert not any(mpc.finalized for mpc in mpcs)


def test_master_that_is_a_slave():
    """Chained constraints are not resolved by `backsubstitution`, so they are rejected."""
    V = _space(8, 8, CellType.triangle)
    imap = V.dofmap.index_map
    assert imap.size_local >= 2
    first = imap.local_range[0]
    rank = MPI.COMM_WORLD.rank

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    # Local dof 0 is tied to dof 1, and dof 1 to dof 0
    mpc.add_constraint(
        V,
        np.array([0, 1], dtype=np.int32),
        np.array([first + 1, first], dtype=np.int64),
        np.array([1, 1], dtype=default_scalar_type),
        np.array([rank, rank], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
    )
    _raised_everywhere(MPI.COMM_WORLD, mpc.finalize, match="also a slave")


def test_master_that_is_a_slave_on_another_process():
    """A chain that closes through a ghost is rejected, which only the owner of the master can see.

    Every process ties its local dof 0 to dof 0 of the next process, so the master of each slave is
    the slave of its neighbour. On one process the master is the slave itself.
    """
    comm = MPI.COMM_WORLD
    V = _space(8, 8, CellType.triangle)
    imap = V.dofmap.index_map
    firsts = comm.allgather(imap.local_range[0])
    target = (comm.rank + 1) % comm.size

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_constraint(
        V,
        np.array([0], dtype=np.int32),
        np.array([firsts[target]], dtype=np.int64),
        np.array([1], dtype=default_scalar_type),
        np.array([target], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    )
    _raised_everywhere(comm, mpc.finalize, match="also a slave")


def test_slave_constrained_twice_on_one_process():
    """A repeated slave is rejected on every process, though only one of them sees it."""
    comm = MPI.COMM_WORLD
    V = _space(8, 8, CellType.triangle)
    first = V.dofmap.index_map.local_range[0]
    rank = comm.rank

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_constraint(
        V,
        np.array([0], dtype=np.int32),
        np.array([first + 1], dtype=np.int64),
        np.array([1], dtype=default_scalar_type),
        np.array([rank], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    )
    if rank == 0:
        # Dof 0 again, with another master
        mpc.add_constraint(
            V,
            np.array([0], dtype=np.int32),
            np.array([first + 2], dtype=np.int64),
            np.array([1], dtype=default_scalar_type),
            np.array([rank], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )
    _raised_everywhere(comm, mpc.finalize, match="more than one constraint")


def test_doubly_periodic_from_two_conditions():
    """Two periodic conditions on one space both claim the corner, which is rejected."""
    V = _space(8, 8, CellType.triangle)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    for axis in range(2):

        def relation(x, axis=axis):
            y = x.copy()
            y[axis] -= 1.0
            return y

        mpc.create_periodic_constraint_geometrical(V, lambda x, axis=axis: np.isclose(x[axis], 1.0), relation, [])
    _raised_everywhere(MPI.COMM_WORLD, mpc.finalize, match="more than one constraint")


def test_incongruent_communicators():
    """Meshes on communicators of different size cannot share a constraint."""
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        pytest.skip("A communicator of one process is congruent to itself")
    V0 = _space(4, 4, CellType.triangle)
    V1 = _space(4, 4, CellType.triangle, comm=MPI.COMM_SELF)
    mpcs = [dolfinx_mpc.MultiPointConstraint(V0), dolfinx_mpc.MultiPointConstraint(V1)]
    _raised_everywhere(comm, lambda: dolfinx_mpc.finalize_multipointconstraints(mpcs), match="communicators")


def test_argument_checks():
    V = _space(3, 3, CellType.triangle)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    with pytest.raises(ValueError, match="At least one"):
        dolfinx_mpc.finalize_multipointconstraints([])
    with pytest.raises(ValueError, match="more than once"):
        dolfinx_mpc.finalize_multipointconstraints([mpc, mpc])
    mpc.finalize()
    with pytest.raises(Exception, match="inalized"):
        dolfinx_mpc.finalize_multipointconstraints([mpc])

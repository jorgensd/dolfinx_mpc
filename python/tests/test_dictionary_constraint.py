# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Constraints given as a dictionary from the coordinates of slaves to those of their masters."""

from __future__ import annotations

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import dolfinx_mpc
from dolfinx_mpc.dictcondition import create_dictionary_constraint

scalar_types = [np.float32, np.float64, np.complex64, np.complex128]


def _real_type(dtype):
    """The coordinate type of a mesh carrying a constraint of `dtype`."""
    return np.finfo(dtype).dtype.type


def _key(point, real_type):
    return np.asarray(point, dtype=real_type).tobytes()


def _right_to_left(real_type, coefficient=1.0):
    """Each dof on the right edge of a unit square of 4 x 4 cells tied to the one on the left edge."""
    return {_key([1.0, y], real_type): {_key([0.0, y], real_type): coefficient} for y in np.linspace(0.0, 1.0, 5)}


@pytest.mark.parametrize("dtype", scalar_types)
@pytest.mark.parametrize("cell_type", (dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.CellType.tetrahedron))
def test_ghost_slaves_get_all_masters(cell_type, dtype):
    """Every copy of a slave, owned or ghost, must receive the same, fully resolved masters.

    A master owned by a process on which the slave is only a ghost has to reach the ghost copies
    of that slave on the other processes as well.
    """
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    N = 6
    mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N, cell_type=cell_type, dtype=real_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    tol = 100 * np.finfo(real_type).eps

    # Periodic relation u(x) = u(x - s), s_i = 1 if x_i = 1, on the faces x_i = 1 (the corner (1, 1, 1) excluded)
    x = V.tabulate_dof_coordinates()
    on_faces = np.isclose(x, 1.0, atol=tol).any(axis=1) & ~np.isclose(x, 1.0, atol=tol).all(axis=1)
    # A dof shared by processes has coordinates that may differ in the last digits between them
    decimals = int(-np.log10(tol))
    slave_points = np.unique(np.round(np.vstack(comm.allgather(x[on_faces])), decimals), axis=0)
    slave_master_dict = {
        _key(p, real_type): {_key(np.where(np.isclose(p, 1.0, atol=tol), 0.0, p), real_type): 1.0} for p in slave_points
    }

    slaves, masters, coeffs, owners, offsets = create_dictionary_constraint(V, slave_master_dict, 0, 0, dtype=dtype)
    assert len(np.unique(slaves)) == len(slaves)
    assert coeffs.dtype == dtype

    # All masters are resolved
    assert comm.allreduce(int(np.sum(masters < 0)) + int(np.sum(owners < 0)), op=MPI.SUM) == 0

    # Each master is the dof of the same component at the image of its slave
    bs = V.dofmap.index_map_bs
    imap = V.dofmap.index_map
    x_global = np.vstack(comm.allgather(x[: imap.size_local]))
    for i, slave in enumerate(slaves):
        image = np.where(np.isclose(x[slave // bs], 1.0, atol=tol), 0.0, x[slave // bs])
        for master in masters[offsets[i] : offsets[i + 1]]:
            assert master % bs == slave % bs
            assert np.allclose(x_global[master // bs], image, atol=tol)

    # Owned and ghost copies of each slave have the same masters
    blocks = imap.local_to_global(slaves // bs)
    global_slaves = blocks * bs + slaves % bs
    local = {int(s): tuple(sorted(masters[offsets[i] : offsets[i + 1]])) for i, s in enumerate(global_slaves)}
    reference = {}
    for part in comm.allgather(local):
        for s, m in part.items():
            assert reference.setdefault(s, m) == m


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="The input of one process cannot differ from itself")
@pytest.mark.parametrize("change", ["coefficient", "order", "subspace"])
def test_input_must_agree(change):
    """A dictionary, order or sub space that differs on one process raises on every process."""
    comm = MPI.COMM_WORLD
    real_type = dolfinx.default_real_type
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 4, dtype=real_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    slave_master_dict = _right_to_left(real_type)
    subspace = 0
    if comm.rank == comm.size - 1:
        if change == "coefficient":
            slave_master_dict[_key([1.0, 0.5], real_type)] = {_key([0.0, 0.5], real_type): 2.0}
        elif change == "order":
            slave_master_dict = dict(reversed(list(slave_master_dict.items())))
        else:
            subspace = 1
    with pytest.raises(ValueError, match="same dictionary"):
        create_dictionary_constraint(V, slave_master_dict, subspace, subspace)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


@pytest.mark.parametrize("dtype", scalar_types)
def test_missing_master(dtype):
    """A master point that no process has raises on every process, rather than giving the master -1."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    V = dolfinx.fem.functionspace(dolfinx.mesh.create_unit_square(comm, 4, 4, dtype=real_type), ("Lagrange", 1))
    slave_master_dict = _right_to_left(real_type)
    slave_master_dict[_key([1.0, 0.5], real_type)] = {_key([2.0, 0.5], real_type): 1.0}
    with pytest.raises(ValueError, match="No process has a degree of freedom at a master"):
        create_dictionary_constraint(V, slave_master_dict, dtype=dtype)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


@pytest.mark.parametrize("dtype", scalar_types)
def test_invalid_input(dtype):
    """Keys that are not bytes of one to three coordinates, and complex coefficients of a real
    constraint, raise on every process."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    V = dolfinx.fem.functionspace(dolfinx.mesh.create_unit_square(comm, 4, 4, dtype=real_type), ("Lagrange", 1))
    with pytest.raises(TypeError, match="as bytes"):
        create_dictionary_constraint(V, {(1.0, 0.5): {(0.0, 0.5): 1.0}}, dtype=dtype)
    four = {_key([1.0, 0.5, 0.0, 0.0], real_type): {_key([0.0, 0.5], real_type): 1.0}}
    with pytest.raises(ValueError, match="1 to 3 values"):
        create_dictionary_constraint(V, four, dtype=dtype)
    complex_coefficient = _right_to_left(real_type, coefficient=0.5 + 0.5j)
    if np.issubdtype(dtype, np.complexfloating):
        coeffs = create_dictionary_constraint(V, complex_coefficient, dtype=dtype)[2]
        assert np.allclose(coeffs, 0.5 + 0.5j)
    else:
        with pytest.raises(ValueError, match="complex coefficient"):
            create_dictionary_constraint(V, complex_coefficient, dtype=dtype)
    assert comm.allreduce(1, op=MPI.SUM) == comm.size


@pytest.mark.parametrize("dtype", scalar_types)
def test_coefficients_in_scalar_type_of_constraint(dtype):
    """`create_general_constraint` gives the coefficients the scalar type of the constraint."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    V = dolfinx.fem.functionspace(dolfinx.mesh.create_unit_square(comm, 4, 4, dtype=real_type), ("Lagrange", 1))
    coefficient = 0.5 + 0.25j if np.issubdtype(dtype, np.complexfloating) else 0.5
    mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype)
    mpc.create_general_constraint(_right_to_left(real_type, coefficient))
    mpc.finalize()
    coeffs, _ = mpc.coefficients()
    assert coeffs.dtype == dtype
    assert np.allclose(coeffs, coefficient)
    num_slaves = comm.allreduce(mpc.num_local_slaves, op=MPI.SUM)
    assert num_slaves == 5


@pytest.mark.parametrize("dtype", scalar_types)
def test_blocked_space_needs_subspace(dtype):
    """A point of a blocked space holds a dof of every component, so the sub spaces must be given."""
    comm = MPI.COMM_WORLD
    real_type = _real_type(dtype)
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 4, dtype=real_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    with pytest.raises(RuntimeError, match="sub-space locators"):
        create_dictionary_constraint(V, _right_to_left(real_type), dtype=dtype)
    slaves = create_dictionary_constraint(V, _right_to_left(real_type), 1, 1, dtype=dtype)[0]
    assert (slaves % 2 == 1).all()
    assert comm.allreduce(1, op=MPI.SUM) == comm.size

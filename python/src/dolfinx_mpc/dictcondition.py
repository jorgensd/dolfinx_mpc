# Copyright (C) 2020-2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""A multi-point constraint given as a dictionary from the coordinates of slaves to those of their masters."""

from __future__ import annotations

import hashlib
import typing

from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import numpy as np
import numpy.typing as npt
from dolfinx import default_scalar_type

from .container import _deprecated, _tolerance


def close_to(
    point: np.typing.NDArray[np.float64 | np.float32],
    atol: float | None = None,
    *,
    distance_tol: float | None = None,
):
    """
    Convenience function for locating a point [x,y,z]
    within an array x [[x0,...,xN],[y0,...,yN], [z0,...,zN]].

    Args:
        point: The point should be padded to 3D
        atol: Deprecated, use `distance_tol`.
        distance_tol: Every component of a located point is within ``distance_tol + 1e-5 |point|``
            of the point's. Defaults to `500` machine epsilon of the type of `point`, or of
            ``dolfinx.default_real_type`` if it is not a floating point type.
    """
    if atol is not None:
        _deprecated("atol", "`distance_tol`")
        distance_tol = atol if distance_tol is None else distance_tol
    point = np.asarray(point)
    real_type = point.dtype if np.issubdtype(point.dtype, np.floating) else dolfinx.default_real_type
    tol = _tolerance(distance_tol, real_type)
    return lambda x: np.isclose(x, point, atol=tol).all(axis=0)


def _close_pairs(
    points: npt.NDArray[np.float64], candidate_points: npt.NDArray[np.float64], atol: float, rtol: float = 1e-5
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The pairs `(i, k)` for which row `k` of `candidate_points` is close to point `i`, as in
    :func:`close_to`: every component within ``atol + rtol * |point|``.

    The rows are sorted by their projection on a generic direction. The rows that can be close to a
    point are then a window of the sorted rows, found for all points at once by a binary search,
    and only those are compared with the point.
    """
    direction = np.array([1.0, 1.0 / np.pi, 1.0 / np.e])
    projection = candidate_points @ direction
    order = np.argsort(projection, kind="stable")
    sorted_projection = projection[order]
    tol = atol + rtol * np.abs(points)
    # A row with every component within tol of the point projects within tol @ |direction| of it,
    # widened by the rounding of the projections
    centre = points @ direction
    width = tol @ np.abs(direction) + 1e-12 * (1.0 + np.abs(centre))
    lo = np.searchsorted(sorted_projection, centre - width, side="left")
    hi = np.searchsorted(sorted_projection, centre + width, side="right")
    counts = hi - lo
    # Expand the windows into one candidate (point, row) per slot, the slots of all points one after
    # the other. For counts = [2, 0, 3] and lo = [5, 9, 1]: point = [0, 0, 2, 2, 2], first = [0, 2, 2],
    # within = [0, 1, 0, 1, 2] and position = [5, 6, 1, 2, 3].
    # The point of each slot
    point = np.repeat(np.arange(len(points)), counts)
    # The first slot of each point
    first = np.cumsum(counts) - counts
    # The place of each slot in the window of its point: 0, 1, ...
    within = np.arange(counts.sum()) - np.repeat(first, counts)
    # The position of each slot in the sorted rows, and the row there
    position = np.repeat(lo, counts) + within
    row = order[position]
    close = (np.abs(candidate_points[row] - points[point]) <= tol[point]).all(axis=1)
    return point[close], row[close]


def _digest(
    slave_master_dict: dict[bytes, dict[bytes, typing.Any]],
    subspace_slave: int | None,
    subspace_master: int | None,
) -> bytes | None:
    """A fingerprint of the input, equal on processes given the same input, or None if the keys are not bytes."""
    h = hashlib.sha256(repr((subspace_slave, subspace_master)).encode())
    try:
        for slave, masters in slave_master_dict.items():
            h.update(len(slave).to_bytes(8, "little") + slave)
            h.update(len(masters).to_bytes(8, "little"))
            for master, coeff in masters.items():
                h.update(len(master).to_bytes(8, "little") + master)
                h.update(np.complex128(coeff).tobytes())
    except (TypeError, ValueError):
        return None
    return h.digest()


def _check_input(
    comm: MPI.Comm,
    slave_master_dict: dict[bytes, dict[bytes, typing.Any]],
    subspace_slave: int | None,
    subspace_master: int | None,
    real_type: np.dtype,
    scalar_type: np.dtype,
) -> None:
    """Raise on every process unless the input is the same on every process, and valid.

    The slaves and masters are matched across processes by their position in the dictionary, and
    every process reads the coefficients from its own copy, so all must have the same.
    """
    digest = _digest(slave_master_dict, subspace_slave, subspace_master)
    if not comm.allreduce(digest is not None, op=MPI.LAND):
        raise TypeError("The coordinates of the slaves and masters must be given as bytes")
    if not comm.allreduce(digest == comm.bcast(digest, root=0), op=MPI.LAND):
        raise ValueError("Every process must pass the same dictionary, in the same order, with the same sub spaces")
    # The input is the same everywhere, so the checks below give the same verdict on every process
    itemsize = real_type.itemsize
    for key in (k for slave, masters in slave_master_dict.items() for k in (slave, *masters)):
        if len(key) % itemsize != 0 or not 0 < len(key) // itemsize <= 3:
            raise ValueError(f"A coordinate must be 1 to 3 values of the mesh's coordinate type, {real_type}")
    if not np.issubdtype(scalar_type, np.complexfloating):
        for masters in slave_master_dict.values():
            if any(np.imag(c) != 0 for c in masters.values()):
                raise ValueError(f"A complex coefficient cannot be used for a constraint of scalar type {scalar_type}")


@typing.no_type_check
def create_dictionary_constraint(
    V: fem.functionspace,
    slave_master_dict: dict[bytes, dict[bytes, float]],
    subspace_slave: int | None = None,
    subspace_master: int | None = None,
    dtype: npt.DTypeLike | None = None,
    distance_tol: float | None = None,
):
    """
    Returns a multi point constraint for a given function space
    and dictionary constraint.

    Every process must pass the same dictionary, in the same order, and the same sub spaces: the
    slaves and masters are matched across processes by their position in it. This is checked.

    Args:
        V: The function space
        slave_master_dict: The dictionary
        subspace_slave: If using mixed or vector space, and only want to use dofs from
            a sub space as slave add index here.
        subspace_master: Subspace index for mixed or vector spaces
        dtype: The scalar type of the coefficients. Defaults to the default scalar type.
        distance_tol: Every component of the coordinate of a dof is within
            ``distance_tol + 1e-5 |x|`` of that of its key. Defaults to `500` machine epsilon of the
            coordinate type of the mesh.

    Returns:
        The slaves on this process, owned first and then ghosts, in the order of the dictionary,
        and their masters, coefficients, the owners of the masters and the offsets.

    Raises:
        ValueError: On every process, if the processes are not given the same input, or if no
            process has a degree of freedom at a master of a slave.

    Examples:
        If the dof `D` located at `[d0,d1]` should be constrained to the dofs `E` and
        F at `[e0,e1]` and `[f0,f1]` as :math:`D = \\alpha E + \\beta F`
        the dictionary should be:

        .. highlight:: python
        .. code-block:: python

            {np.array([d0, d1], dtype=mesh.geometry.x.dtype).tobytes():
                {numpy.array([e0, e1], dtype=mesh.geometry.x.dtype).tobytes(): alpha,
                numpy.array([f0, f1], dtype=mesh.geometry.x.dtype).tobytes(): beta}}

    Note:
        Collective.
    """
    comm = V.mesh.comm
    real_type = np.dtype(V.mesh.geometry.x.dtype)
    scalar_type = np.dtype(default_scalar_type if dtype is None else dtype)
    _check_input(comm, slave_master_dict, subspace_slave, subspace_master, real_type, scalar_type)

    bs = V.dofmap.index_map_bs
    index_map = V.dofmap.index_map
    local_size = index_map.size_local * bs
    atol = _tolerance(distance_tol, real_type)

    def dof_table(subspace):
        """The coordinates of the dofs of `V`, or of its sub space, and their indices in `V`, local
        to the process, ghosts included. A blocked space has a row per component, all at the
        coordinate of the block, so a point finds all of them."""
        if subspace is None:
            coordinates, table_bs = V.tabulate_dof_coordinates(), bs
            dofs = np.arange(len(coordinates) * bs, dtype=np.int64)
        else:
            # One map per dofmap, that is per cell type of the mesh
            V_sub, sub_to_V = V.sub(subspace).collapse()
            if len(sub_to_V) != 1:
                raise NotImplementedError("A dictionary constraint on a mesh of several cell types")
            coordinates, table_bs = V_sub.tabulate_dof_coordinates(), V_sub.dofmap.index_map_bs
            dofs = np.asarray(sub_to_V[0], dtype=np.int64)
        return np.repeat(coordinates.astype(np.float64), table_bs, axis=0), dofs

    def points(keys):
        """The coordinates in `keys`, padded to 3D."""
        out = np.zeros((len(keys), 3), dtype=np.float64)
        for k, key in enumerate(keys):
            coordinates = np.frombuffer(key, dtype=real_type)
            out[k, : len(coordinates)] = coordinates
        return out

    # Master j of slave i is entry starts[i] + j of the flat layout, the same on every process
    slave_keys = list(slave_master_dict.keys())
    master_keys = [master for key in slave_keys for master in slave_master_dict[key]]
    num_slaves = len(slave_keys)
    starts = np.zeros(num_slaves + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(slave_master_dict[key]) for key in slave_keys])
    num_entries = int(starts[-1])

    # One reduction carries everything: the global index and the owner of each master, filled by
    # the process owning it, whether each slave is on some process, and the errors of locating
    reduced = np.full(2 * num_entries + num_slaves + 2, -1, dtype=np.int64)
    masters_all = reduced[:num_entries]
    owners_all = reduced[num_entries : 2 * num_entries]
    found = reduced[2 * num_entries : 2 * num_entries + num_slaves]
    errors = reduced[2 * num_entries + num_slaves :]

    # The slaves, owned or ghost, at their points
    slave_coordinates, slave_table = dof_table(subspace_slave)
    i, k = _close_pairs(points(slave_keys), slave_coordinates, atol)
    count = np.bincount(i, minlength=num_slaves)
    single = count[i] == 1
    slave_dofs = np.full(num_slaves, -1, dtype=np.int64)
    slave_dofs[i[single]] = slave_table[k[single]]
    found[:] = slave_dofs >= 0
    errors[0] = (count > 1).any()

    # The masters this process owns, as only the owner of a master fills it in
    master_coordinates, master_table = dof_table(subspace_master)
    owned = master_table < local_size
    e, k = _close_pairs(points(master_keys), master_coordinates[owned], atol)
    count = np.bincount(e, minlength=num_entries)
    single = count[e] == 1
    blocks, components = np.divmod(master_table[owned][k[single]], bs)
    masters_all[e[single]] = index_map.local_to_global(blocks.astype(np.int32)) * bs + components
    owners_all[e[single]] = comm.rank
    errors[1] = (count > 1).any()
    comm.Allreduce(MPI.IN_PLACE, reduced, op=MPI.MAX)

    # The reduced data is the same on every process, and so is every verdict on it
    if errors[0]:
        raise RuntimeError("Multiple slaves found at same point. You should use sub-space locators.")
    if errors[1]:
        raise RuntimeError("Multiple masters found at same point. You should use sub-space locators.")
    unresolved = [i for i in np.flatnonzero(found) if (masters_all[starts[i] : starts[i + 1]] < 0).any()]
    if len(unresolved) > 0:
        point = np.frombuffer(slave_keys[unresolved[0]], dtype=real_type)
        raise ValueError(
            f"No process has a degree of freedom at a master of {len(unresolved)} slave(s), the first at {point}"
        )

    coeffs_all = np.array([c for key in slave_keys for c in slave_master_dict[key].values()])
    if not np.issubdtype(scalar_type, np.complexfloating):
        coeffs_all = coeffs_all.real
    coeffs_all = coeffs_all.astype(scalar_type)
    # The slaves on this process, owned first, then ghosts, each in the order of the dictionary
    held = np.flatnonzero(slave_dofs >= 0)
    order = np.concatenate([held[slave_dofs[held] < local_size], held[slave_dofs[held] >= local_size]])
    counts = starts[order + 1] - starts[order]
    entries = np.repeat(starts[order] - (np.cumsum(counts) - counts), counts) + np.arange(counts.sum())
    offsets = np.zeros(len(order) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return (
        slave_dofs[order].astype(np.int32),
        masters_all[entries].copy(),
        coeffs_all[entries],
        owners_all[entries].astype(np.int32),
        offsets,
    )

# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Spiders: points whose dofs are tied to many dofs of another space (RBE2, RBE3).

A spider has a *body*, a point of a point mesh, and *feet*, the dofs it is tied to. The bodies of
all spiders of a problem are the points of one point mesh, made by :func:`create_spider_mesh`, and
a spider is named by the index of its point, found with :func:`locate_spider`.
"""

from __future__ import annotations

from typing import Callable

from mpi4py import MPI

from .container import MPCData
from dolfinx import default_scalar_type
import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy as np
import numpy.typing as npt

__all__ = ["create_spider_mesh", "locate_spider", "spider_values"]


def create_spider_mesh(comm: MPI.Comm, points: npt.ArrayLike, tol: float | None = None) -> _mesh.Mesh:
    """Create the point mesh holding the bodies of spiders.

    Each process passes the points it owns, so the bodies' dofs can be spread over the processes.
    A point is numbered by process, the points of process 0 first: the index that
    :func:`locate_spider` returns.

    Coinciding points are merged: the first, in that numbering, is kept and the others dropped,
    so a point given twice, on one process or on several, is one spider.

    Args:
        comm: The communicator of the meshes the spiders join
        points: The points owned by this process, shape `(num_points, 3)`. May be empty.
        tol: Points closer than this coincide. Defaults to a multiple of the machine precision
            of the points' type, relative to the extent of all points.

    Returns:
        A mesh of points, from :func:`dolfinx.mesh.create_point_mesh`.

    Note:
        Collective.
    """
    points = np.asarray(points)
    dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float64
    points = points.astype(dtype).reshape(-1, 3)

    # Spiders are few, so every process sees all points and drops the same duplicates
    gathered = comm.allgather(points)
    all_points = np.vstack(gathered)
    if tol is None:
        extent = np.abs(all_points).max() if len(all_points) > 0 else 1.0
        tol = 1e3 * np.finfo(dtype).eps * max(extent, 1.0)
    distance = np.linalg.norm(all_points[:, None, :] - all_points[None, :, :], axis=2)
    # A point is a duplicate if it coincides with an earlier one
    duplicate = np.tril(distance < tol, k=-1).any(axis=1)
    start = sum(len(p) for p in gathered[: comm.rank])
    keep = ~duplicate[start : start + len(points)]
    return _mesh.create_point_mesh(comm, points[keep])


def locate_spider(spider_mesh: _mesh.Mesh, marker: Callable[[np.ndarray], np.ndarray]) -> int:
    """The index of the spider whose point `marker` selects.

    The point is located on the process that owns it and its global index broadcast, so every
    process gets the same index.

    Args:
        spider_mesh: A mesh from :func:`create_spider_mesh`
        marker: Marks the point, given coordinates of shape `(3, num_points)`

    Returns:
        The index of the point, as taken by the `map` argument of
        :meth:`dolfinx_mpc.MultiPointConstraint.add_rbe2_topological`.

    Raises:
        ValueError: If `marker` selects no point or more than one, on every process.

    Note:
        Collective.
    """
    if not spider_mesh.topology.dim == 0:
        raise ValueError("The spider mesh must be a point mesh")
    imap = spider_mesh.topology.index_map(0)
    local = _mesh.locate_entities(spider_mesh, 0, marker)
    local = local[local < imap.size_local]
    found = np.concatenate(spider_mesh.comm.allgather(imap.local_range[0] + local.astype(np.int64)))
    if len(found) != 1:
        raise ValueError(f"The marker selects {len(found)} spider points, expected one")
    return int(found[0])


def spider_values(u: _fem.Function, index: int) -> np.ndarray:
    """The values of a function on a spider mesh at one spider, on every process.

    A process only holds a spider's values if it owns the point or has feet tied to it, so they are
    gathered from the owner.

    Args:
        u: A function on a space of a mesh from :func:`create_spider_mesh`
        index: The index of the spider, see :func:`locate_spider`

    Returns:
        The values at the point, one per component of the space.

    Note:
        Collective.
    """
    V = u.function_space
    bs = V.dofmap.index_map_bs
    # The point is owned where its index falls in the point mesh's range; its cell is the point
    start, end = V.mesh.topology.index_map(0).local_range
    values = None
    if start <= index < end:
        block = V.dofmap.cell_dofs(index - start)[0]
        values = u.x.array[bs * block : bs * (block + 1)].copy()
    gathered = [v for v in V.mesh.comm.allgather(values) if v is not None]
    return gathered[0]


def create_rbe2(
    V: _fem.FunctionSpace,
    dofs: npt.NDArray[np.int32],
    points: npt.NDArray[np.int64],
    W: _fem.FunctionSpace,
    dtype: np.dtype | None = None,
) -> MPCData:
    """Tie every component of the blocked `dofs` to the rigid-body motion of `points` of `W`."""
    comm = V.mesh.comm
    gdim = V.mesh.geometry.dim
    bs = V.dofmap.index_map_bs
    if bs != gdim:
        raise ValueError(f"The tied space must have one component per dimension ({gdim}), it has {bs}")
    if W.mesh.topology.cell_type != _mesh.CellType.point:
        raise ValueError("The body of a spider must be a space on a point mesh")
    num_body = W.dofmap.index_map_bs
    num_ridig_body_motions = 6 if gdim == 3 else 3
    rotations = {gdim: False, num_ridig_body_motions: True}
    if num_body not in rotations:
        raise ValueError(
            f"The body space must have {gdim} components (translations) or {num_ridig_body_motions} "
            f"(translations and rotations), it has {num_body}"
        )
    with_rotations = rotations[num_body]

    # Every process learns, for every point of the point mesh: its owner, the global index of
    # its block of dofs in W, and its coordinate. Spiders are few, so all points are shared.
    pmesh = W.mesh
    num_points = pmesh.topology.index_map(0).size_local
    imap_W = W.dofmap.index_map
    assert num_points == imap_W.size_local, "The point mesh must have one dof per point"
    blocks = imap_W.local_to_global(np.arange(num_points, dtype=np.int32))
    coords = W.tabulate_dof_coordinates().reshape(num_points, 3)
    gathered = comm.allgather((blocks, coords))
    point_owner = np.concatenate([np.full(len(b), r, dtype=np.int32) for r, (b, _) in enumerate(gathered)])
    point_block = np.concatenate([b for b, _ in gathered]).astype(np.int64)
    point_x = np.vstack([x for _, x in gathered]) if gathered else np.zeros((0, 3))
    if len(points) > 0 and (points.min() < 0 or points.max() >= len(point_block)):
        raise IndexError(f"A spider point is outside the {len(point_block)} points of the point mesh")

    # u_j = t_j + (theta x r)_j: per component, the translation and the rotations with their
    # coefficients. In 3D (theta x r)_x = theta_y r_z - theta_z r_y and cyclically; in 2D
    # (theta x r) = (-theta r_y, theta r_x).
    x = V.tabulate_dof_coordinates()[dofs, :gdim]
    r = x - point_x[points][:, :gdim]
    rotation_terms: dict[int, tuple[tuple[int, int, float], ...]]
    if gdim == 3:
        rotation_terms = {
            0: ((4, 2, 1.0), (5, 1, -1.0)),
            1: ((5, 0, 1.0), (3, 2, -1.0)),
            2: ((3, 1, 1.0), (4, 0, -1.0)),
        }
    else:
        rotation_terms = {0: ((2, 1, -1.0),), 1: ((2, 0, 1.0),)}

    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    for i, (dof, point) in enumerate(zip(dofs, points)):
        base = point_block[point] * num_body
        for j in range(gdim):
            slaves.append(bs * dof + j)
            masters.append(base + j)
            coeffs.append(1.0)
            owners.append(point_owner[point])
            if with_rotations:
                for rot, comp, sign in rotation_terms[j]:
                    c = sign * r[i, comp]
                    if c != 0:
                        masters.append(base + rot)
                        coeffs.append(c)
                        owners.append(point_owner[point])
            offsets.append(len(masters))
    _dtype = dtype if dtype is not None else default_scalar_type
    return MPCData(
        np.array(slaves, dtype=np.int32),
        np.array(masters, dtype=np.int64),
        np.array(coeffs, dtype=_dtype),
        np.array(owners, dtype=np.int32),
        np.array(offsets, dtype=np.int32),
    )

# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Spiders: points whose dofs are tied to many dofs of another space (RBE2, RBE3).

A spider has a *body*, a point of a point mesh, and *feet*, the dofs it is tied to. The bodies of
all spiders of a problem are the points of one point mesh, made by :func:`create_spider_mesh`. A
spider is named by its input index, the original cell index of its point, and its coordinate is
the coordinate of its dofs in the space on the point mesh: moving the point mesh moves the spider.
"""

from __future__ import annotations

from mpi4py import MPI

import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy as np
import numpy.typing as npt
from dolfinx import default_scalar_type

from .container import _mpc_data_classes
from .cpp import mpc as _cpp_mpc

__all__ = ["create_rbe2", "create_spider_mesh", "spider_values"]


def create_spider_mesh(comm: MPI.Comm, points: npt.ArrayLike, tol: float | None = None) -> _mesh.Mesh:
    """Create the point mesh holding the bodies of spiders.

    Each process passes the points it owns, so the bodies' dofs can be spread over the processes.
    A spider is named by its input index: its position among the points of all processes, those of
    process 0 first. This is the original cell index of its point.

    Coinciding points are merged: the first, in that numbering, is kept and the others dropped,
    so a point given twice, on one process or on several, is one spider. The input index counts the
    kept points only.

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


_type_names = {
    np.float32: "float",
    np.float64: "double",
    np.complex64: "complex_float",
    np.complex128: "complex_double",
}


def create_rbe2(
    V: _fem.FunctionSpace,
    dofs: npt.NDArray[np.int32],
    spiders: npt.NDArray[np.int64],
    W: _fem.FunctionSpace,
    dtype: npt.DTypeLike | None = None,
    x: npt.NDArray[np.floating] | None = None,
) -> _mpc_data_classes:
    r"""The constraint tying every component of the blocked `dofs` to the rigid-body motion of a spider.

    Foot `i`, at `x`, follows spider `spiders[i]` at :math:`x_c`:
    :math:`u_j = t_j + (\theta \times (x - x_c))_j`. The translation :math:`t` is the first
    `gdim` components of the spider's block of dofs in `W`, the rotation :math:`\theta` the rest
    (components 3, 4, 5 in 3D, component 2 in 2D), and :math:`x_c` the coordinate of that block.
    Every rotation term is kept, also where its coefficient is zero, so that the masters do not
    depend on the configuration. Wraps `dolfinx_mpc::create_rbe2`.

    Args:
        V: The space of the feet, with one component per dimension
        dofs: The feet, blocked dofs of `V` local to the process, ghosts included
        spiders: The spider of each foot, its input index (see :func:`create_spider_mesh`)
        W: The space on the spider mesh
        dtype: The scalar type of the coefficients. Defaults to the default scalar type.
        x: The coordinates of all dofs of `V` local to the process, from
            `V.tabulate_dof_coordinates()`, if already at hand

    Returns:
        The slaves, masters (global dofs of `W`), coefficients, owners and offsets.

    Note:
        Collective.
    """
    _dtype = np.dtype(default_scalar_type if dtype is None else dtype).type
    create = getattr(_cpp_mpc, f"create_rbe2_{_type_names[_dtype]}")
    return create(
        V._cpp_object,
        np.ascontiguousarray(dofs, dtype=np.int32),
        np.ascontiguousarray(spiders, dtype=np.int64),
        W._cpp_object,
        None if x is None else np.ascontiguousarray(x),
    )


def spider_values(u: _fem.Function, spider: int) -> np.ndarray:
    """The values of a function on a spider mesh at one spider, on every process.

    A process only holds a spider's values if it owns the spider's dofs or has feet tied to it, so
    the owner broadcasts them.

    Args:
        u: A function on a space of a mesh from :func:`create_spider_mesh`
        spider: The input index of the spider

    Returns:
        The values at the spider, one per component of the space.

    Note:
        Collective.
    """
    V = u.function_space
    comm = V.mesh.comm
    topology = V.mesh.topology
    num_cells = topology.index_map(topology.dim).size_local
    cell = np.flatnonzero(np.asarray(topology.original_cell_index[:num_cells]) == spider)
    owner = comm.allreduce(comm.rank if len(cell) > 0 else -1, op=MPI.MAX)
    if owner < 0:
        raise IndexError(f"The spider mesh has no point with input index {spider}")
    values = None
    if comm.rank == owner:
        bs = V.dofmap.index_map_bs
        dof = V.dofmap.list[cell[0], 0]
        values = u.x.array[bs * dof : bs * (dof + 1)].copy()
    return comm.bcast(values, root=owner)

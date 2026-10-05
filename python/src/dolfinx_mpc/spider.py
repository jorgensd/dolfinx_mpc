# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Spider meshes: the point meshes holding the bodies of spiders (RBE2, RBE3).

A spider has a *body*, a point of a point mesh, and *feet*, the dofs it is tied to. The bodies of
the spiders of a problem are the points of a point mesh made by :func:`create_spider_mesh`. A
spider is named by its input index, the original cell index of its point, and its coordinate is
the coordinate of its dofs in the space on the point mesh: moving the point mesh, with
:func:`move`, moves the spider.
"""

from __future__ import annotations

import typing

from mpi4py import MPI

import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import default_real_type
from dolfinx.common import local_range

__all__ = ["create_spider_mesh", "create_spider_pair", "move", "spider_values"]


def create_spider_mesh(comm: MPI.Comm, points: npt.ArrayLike, dtype: npt.DTypeLike | None = None) -> _mesh.Mesh:
    """Create the point mesh holding the bodies of spiders.

    The points are given on the first process. Every other process passes either no points or the
    same points. Spider `k` is the point in row `k`, its input index; coinciding points are
    distinct spiders, for instance two spiders joined by a spring. The points are spread over the
    processes as the post office splits an index range: of `M` points, process `r` owns those in
    :func:`dolfinx.common.local_range(r, M, size) <dolfinx.common.local_range>`. Spider `k` of any
    two spider meshes of `M` points is therefore on the same process.

    Args:
        comm: The communicator of the meshes the spiders join
        points: The points, shape `(num_points, 3)`, on the first process. On the others, no
            points or the same points.
        dtype: The coordinate type of the mesh. Defaults to the type of the first process's
            points, or the default real type if they are not floating point.

    Returns:
        A mesh of points, from :func:`dolfinx.mesh.create_point_mesh`.

    Raises:
        ValueError: If a process passes points other than those of the first process. Raised on
            every process.

    Note:
        Collective.
    """
    points = np.asarray(points)
    if dtype is None:
        own = points.dtype if np.issubdtype(points.dtype, np.floating) else np.dtype(default_real_type)
        dtype = comm.bcast(own if comm.rank == 0 else None, root=0)
    dtype = np.dtype(dtype)
    local = points.astype(dtype).reshape(-1, 3)
    reference = comm.bcast(local if comm.rank == 0 else None, root=0)

    # The same points up to rounding in the coordinate type
    eps = 100 * np.finfo(dtype).eps
    scale = max(float(np.abs(reference).max(initial=0.0)), 1.0)
    same = len(local) == 0 or (
        local.shape == reference.shape and np.allclose(local, reference, rtol=eps, atol=eps * scale)
    )
    if not comm.allreduce(same, op=MPI.LAND):
        raise ValueError("Every process must pass no points or the points of the first process")
    start, end = local_range(comm.rank, len(reference), comm.size)
    return _mesh.create_point_mesh(comm, reference[start:end])


def _local_input_indices(spiders: _mesh.Mesh) -> npt.NDArray[np.int64]:
    """The input index of each point of a spider mesh local to the process."""
    imap = spiders.topology.index_map(0)
    return np.asarray(spiders.topology.original_cell_index[: imap.size_local + imap.num_ghosts], dtype=np.int64)


def create_spider_pair(spiders_A: _mesh.Mesh, spiders_B: _mesh.Mesh) -> _mesh.EntityMap:
    r"""Relate spider `k` of one spider mesh to spider `k` of another, for every `k`.

    The pairs are, for instance, the two ends of springs: order the points of the two meshes so that
    row `k` of each holds the two ends of spring `k`. A form coupling the spaces on the two meshes,
    such as a spring :math:`\int K (w_A - w_B) \cdot (v_A - v_B)`, is integrated over `spiders_A`
    with the returned map in `entity_maps`. Each mesh keeps its own geometry, so the two spiders of
    a pair may start at the same point and move apart.

    Both meshes must have the same number of spiders. :func:`create_spider_mesh` then puts spider
    `k` of both on the same process.

    Args:
        spiders_A: The spider mesh integrated over
        spiders_B: The other spider mesh

    Returns:
        The entity map with `spiders_A` as topology and `spiders_B` as sub-topology.

    Raises:
        ValueError: If the meshes have different numbers of spiders, or spider `k` of the two is on
            different processes, as for a point mesh not made by :func:`create_spider_mesh`. Raised
            on every process.

    Note:
        Collective.
    """
    num_A = spiders_A.topology.index_map(0).size_global
    num_B = spiders_B.topology.index_map(0).size_global
    if num_A != num_B:
        raise ValueError(f"The spider meshes must have the same number of spiders, not {num_A} and {num_B}")

    # The local point of spiders_A with the input index of each local point of spiders_B
    local_A = {k: i for i, k in enumerate(_local_input_indices(spiders_A).tolist())}
    b_to_a = np.array([local_A.get(k, -1) for k in _local_input_indices(spiders_B).tolist()], dtype=np.int32)
    if not spiders_A.comm.allreduce(bool((b_to_a >= 0).all()), op=MPI.LAND):
        raise ValueError("Spider k of the two meshes is on different processes. Create both with create_spider_mesh.")
    return _mesh.entity_map(spiders_A.topology, spiders_B.topology, 0, b_to_a)


def move(
    mesh: _mesh.Mesh,
    u: _fem.Function | ufl.core.expr.Expr | typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.inexact]],
) -> None:
    """Move the geometry nodes of a mesh by a displacement.

    The displacement is interpolated into the space of the mesh's coordinate element, whose dofs
    in each cell are the cell's geometry nodes, and added node by node. No relation between the
    dofs of `u` and the geometry nodes is assumed. As in `scifem.move`.

    Args:
        mesh: The mesh to move
        u: The displacement: a function on `mesh`, an expression, or a callable of the
            coordinates, shape `(3, num_points)`. A function with more components than the
            geometric dimension, such as one on a spider space with rotations, moves the mesh by
            its first components, the translation. A complex displacement moves it by its real part.
    """
    gdim = mesh.geometry.dim
    V_x = _fem.functionspace(mesh, mesh.ufl_domain().ufl_coordinate_element())
    if isinstance(u, _fem.Function):
        dtype = u.x.array.dtype
        if int(np.prod(u.ufl_shape)) > gdim:
            u = ufl.as_vector([u[i] for i in range(gdim)])
    else:
        dtype = mesh.geometry.x.dtype
    if isinstance(u, ufl.core.expr.Expr) and not isinstance(u, _fem.Function):
        dtype = np.result_type(dtype, np.dtype(mesh.geometry.x.dtype))
        u = _fem.Expression(u, V_x.element.interpolation_points, dtype=dtype)
    u_x = _fem.Function(V_x, dtype=dtype)
    u_x.interpolate(u)

    # Each node once: a node in several cells has the same value in each
    nodes = mesh.geometry.dofmaps[0].reshape(-1)
    displacement = np.zeros((mesh.geometry.x.shape[0], gdim), dtype=mesh.geometry.x.dtype)
    displacement[nodes] = u_x.x.array.reshape(-1, gdim)[V_x.dofmap.list.reshape(-1)].real
    mesh.geometry.x[:, :gdim] += displacement


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

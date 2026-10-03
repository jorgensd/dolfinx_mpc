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
import dolfinx.la as _la
import dolfinx.mesh as _mesh
import numpy as np
import numpy.typing as npt
from dolfinx import default_real_type, default_scalar_type

from .container import MPCData
from .cpp import mpc as _cpp_mpc

__all__ = ["create_rbe2", "create_spider_mesh", "spider_values", "update_rbe2_coefficients"]


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


def _locate_spiders(
    W: _fem.FunctionSpace, needed: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.floating]]:
    """The global block of `W`, the owning process and the coordinate of each spider in `needed`.

    Found through a post office keyed on the input index, see `dolfinx_mpc.cpp.mpc.locate_spiders`.

    Note:
        Collective.
    """
    return _cpp_mpc.locate_spiders(W._cpp_object, np.ascontiguousarray(needed, dtype=np.int64))


def _rotation_table(gdim: int) -> tuple[np.ndarray, np.ndarray]:
    r"""The sign and the component of `r` of each (foot component, body component) term.

    :math:`(\theta \times r)_j = \sum_c \mathrm{sign}[j, c]\, r_{\mathrm{comp}[j, c]}\, \theta_c`. In 3D
    :math:`(\theta \times r)_x = \theta_y r_z - \theta_z r_y` and cyclically, in 2D
    :math:`(-\theta r_y, \theta r_x)`. A zero sign marks a body component without a term.
    """
    if gdim == 3:
        terms = [[(4, 2, 1), (5, 1, -1)], [(5, 0, 1), (3, 2, -1)], [(3, 1, 1), (4, 0, -1)]]
        num_body = 6
    else:
        terms = [[(2, 1, -1)], [(2, 0, 1)]]
        num_body = 3
    sign = np.zeros((gdim, num_body), dtype=np.int64)
    comp = np.zeros((gdim, num_body), dtype=np.int64)
    for j, row in enumerate(terms):
        for c, k, sg in row:
            sign[j, c], comp[j, c] = sg, k
    return sign, comp


def _rbe2_coefficients(
    j: npt.NDArray[np.int64], c: npt.NDArray[np.int64], r: npt.NDArray[np.floating], gdim: int
) -> np.ndarray:
    """The coefficient of body component `c` in foot component `j`, at `r = x - x_c` from the body."""
    coeff = (c == j).astype(r.dtype)
    rotation = c >= gdim
    if rotation.any():
        sign, comp = _rotation_table(gdim)
        jr, cr = j[rotation], c[rotation]
        coeff[rotation] = sign[jr, cr] * r[np.flatnonzero(rotation), comp[jr, cr]]
    return coeff


def _check_spaces(V: _fem.FunctionSpace, W: _fem.FunctionSpace) -> tuple[int, int, bool]:
    gdim = V.mesh.geometry.dim
    bs = V.dofmap.index_map_bs
    if bs != gdim:
        raise ValueError(f"The tied space must have one component per dimension ({gdim}), it has {bs}")
    if W.mesh.topology.cell_type != _mesh.CellType.point:
        raise ValueError("The body of a spider must be a space on a point mesh")
    num_body = W.dofmap.index_map_bs
    num_rigid_body_motions = 6 if gdim == 3 else 3
    if num_body not in (gdim, num_rigid_body_motions):
        raise ValueError(
            f"The body space must have {gdim} components (translations) or {num_rigid_body_motions} "
            f"(translations and rotations), it has {num_body}"
        )
    return gdim, num_body, num_body == num_rigid_body_motions


def create_rbe2(
    V: _fem.FunctionSpace,
    dofs: npt.NDArray[np.int32],
    spiders: npt.NDArray[np.int64],
    W: _fem.FunctionSpace,
    dtype: npt.DTypeLike | None = None,
    x: npt.NDArray[np.floating] | None = None,
) -> MPCData:
    r"""The constraint tying every component of the blocked `dofs` to the rigid-body motion of a spider.

    Foot `i`, at `x`, follows spider `spiders[i]` at :math:`x_c`:
    :math:`u_j = t_j + (\theta \times (x - x_c))_j`. The translation :math:`t` is the first
    `gdim` components of the spider's block of dofs in `W`, the rotation :math:`\theta` the rest
    (components 3, 4, 5 in 3D, component 2 in 2D), and :math:`x_c` the coordinate of that block.

    Every rotation term is kept, also where its coefficient is zero, so that the masters do not
    depend on the configuration and :func:`update_rbe2_coefficients` can recompute the
    coefficients after the meshes move.

    The global block, owner and coordinate of each spider are found through a post office: no
    process gathers the spiders of all others.

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
    gdim, num_body, with_rotations = _check_spaces(V, W)
    bs = gdim
    spiders = np.asarray(spiders, dtype=np.int64)
    dofs = np.asarray(dofs, dtype=np.int32)

    needed, spider_position = np.unique(spiders, return_inverse=True)
    needed_block, needed_owner, needed_x = _locate_spiders(W, needed)
    base = needed_block[spider_position] * num_body
    owner = needed_owner[spider_position]

    # The body components in each foot component: its translation, then its rotation terms
    if with_rotations:
        sign, _ = _rotation_table(gdim)
        components = np.array([[j, *np.flatnonzero(sign[j])] for j in range(gdim)], dtype=np.int64)
    else:
        components = np.arange(gdim, dtype=np.int64).reshape(gdim, 1)
    num_terms = components.shape[1]

    n = len(dofs)
    if x is None:
        x = V.tabulate_dof_coordinates()
    r = x[dofs, :gdim] - needed_x[spider_position, :gdim]
    j = np.broadcast_to(np.arange(gdim, dtype=np.int64)[None, :, None], (n, gdim, num_terms))
    c = np.broadcast_to(components[None], (n, gdim, num_terms))
    r_term = np.broadcast_to(r[:, None, None, :], (n, gdim, num_terms, gdim))
    _dtype = default_scalar_type if dtype is None else dtype
    coeffs = _rbe2_coefficients(j.reshape(-1), c.reshape(-1), r_term.reshape(-1, gdim), gdim).astype(_dtype)
    return MPCData(
        (bs * dofs[:, None] + np.arange(gdim, dtype=np.int32)).reshape(-1),
        (base[:, None, None] + c).reshape(-1),
        coeffs,
        np.repeat(owner, gdim * num_terms).astype(np.int32),
        np.arange(0, n * gdim * num_terms + 1, num_terms, dtype=np.int32),
    )


def update_rbe2_coefficients(
    V: _fem.FunctionSpace,
    W: _fem.FunctionSpace,
    W_extended: _fem.FunctionSpace,
    block: int,
    masters: npt.NDArray[np.int32],
    blocks: npt.NDArray[np.int32],
    coeffs: np.ndarray,
    offsets: npt.NDArray[np.int32],
) -> None:
    """Recompute, in place, the coefficients of the masters in `W` from the current coordinates.

    The foot coordinates are the dof coordinates of `V`, the spider coordinates those of `W`, both
    tabulated now; the coordinate of a spider owned elsewhere arrives by a forward scatter over
    the extended index map, which already holds it as a ghost.

    Args:
        V: The space of the feet, as given to the constraint
        W: The space on the spider mesh, as given to the constraint
        W_extended: `W` with the index map extended by finalization
        block: The block of `W` among the constraints finalized together
        masters: Masters in the layout of `all_coefficients`, local to the space of their block
        blocks: The block of each master
        coeffs: The coefficients, in the same layout. Updated in place.
        offsets: The offsets per local dof of the layout

    Note:
        Collective.
    """
    gdim, num_body, _ = _check_spaces(V, W)
    imap = W_extended.dofmap.index_map
    x_c = _la.vector(imap, 3, dtype=default_real_type)
    num_owned = W.dofmap.index_map.size_local
    x_c.array[: 3 * num_owned] = W.tabulate_dof_coordinates()[:num_owned].reshape(-1)
    x_c.scatter_forward()
    x_c = x_c.array.reshape(-1, 3)

    in_W = np.flatnonzero(blocks == block)
    slave = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))[in_W]
    m = masters[in_W].astype(np.int64)
    x = V.tabulate_dof_coordinates()
    r = x[slave // gdim, :gdim] - x_c[m // num_body, :gdim]
    coeffs[in_W] = _rbe2_coefficients(slave % gdim, m % num_body, r, gdim)


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

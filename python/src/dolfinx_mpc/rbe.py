# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Constraints tying dofs to spiders (RBE2, RBE3): points of a spider mesh, see :mod:`dolfinx_mpc.spider`."""

from __future__ import annotations

import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt

from .container import _cpp_function, _mpc_data_classes, _scalar_type

__all__ = ["create_rbe2", "create_rbe3"]


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
        dtype: The scalar type of the coefficients. Defaults to the default scalar type of
            DOLFINx, real or complex, at the precision of the meshes.
        x: The coordinates of all dofs of `V` local to the process, from
            `V.tabulate_dof_coordinates()`, if already at hand

    Returns:
        The slaves, masters (global dofs of `W`), coefficients, owners and offsets.

    Note:
        Collective.
    """
    real_type = V.mesh.geometry.x.dtype
    if W.mesh.geometry.x.dtype != real_type:
        raise ValueError("The mesh of the feet and the spider mesh must have the same coordinate type")
    create = _cpp_function("create_rbe2", _scalar_type(real_type, dtype))
    return create(
        V._cpp_object,
        np.ascontiguousarray(dofs, dtype=np.int32),
        np.ascontiguousarray(spiders, dtype=np.int64),
        W._cpp_object,
        None if x is None else np.ascontiguousarray(x, dtype=real_type),
    )


def create_rbe3(
    W: _fem.FunctionSpace,
    V: list[_fem.FunctionSpace],
    dofs: list[npt.NDArray[np.int32]],
    spiders: list[npt.NDArray[np.int64]],
    weights: list[npt.NDArray[np.floating]],
    dtype: npt.DTypeLike | None = None,
) -> tuple[_mpc_data_classes, npt.NDArray[np.int32]]:
    r"""The constraint tying the dofs of spiders to the motion of their feet (RBE3).

    Each spider moves with the rigid motion that best fits its feet,
    :math:`\min_{t, \theta} \sum_i w_i |u_i - t - \theta \times (x_i - x_c)|^2`. Without rotations,
    :math:`t` is the weighted mean of the feet. Wraps `dolfinx_mpc::create_rbe3`.

    Args:
        W: The space on the spider mesh, holding the slaves
        V: The spaces of the feet, with one component per dimension
        dofs: The feet in each space, blocked dofs local to the process
        spiders: The spider of each foot, its input index (see :func:`create_spider_mesh`)
        weights: The weight of each foot, non-negative
        dtype: The scalar type of the coefficients. Defaults to the default scalar type of
            DOLFINx, real or complex, at the precision of the meshes.

    Returns:
        The slaves, masters, coefficients, owners and offsets, and the position in `V` of the
        space of each master.

    Note:
        Collective.
    """
    real_type = W.mesh.geometry.x.dtype
    if any(V_s.mesh.geometry.x.dtype != real_type for V_s in V):
        raise ValueError("The meshes of the feet and the spider mesh must have the same coordinate type")
    create = _cpp_function("create_rbe3", _scalar_type(real_type, dtype))
    return create(
        W._cpp_object,
        [V_s._cpp_object for V_s in V],
        [np.ascontiguousarray(d, dtype=np.int32) for d in dofs],
        [np.ascontiguousarray(k, dtype=np.int64) for k in spiders],
        [np.ascontiguousarray(w, dtype=real_type) for w in weights],
    )

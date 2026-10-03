# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Interpretation of the ``kind`` argument of the assemblers, as in :mod:`dolfinx.fem.petsc`."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Optional, Union

# A PETSc matrix type, "nest", "mpi" or a nested sequence of matrix types for a nest matrix
Kind = Union[str, Sequence[Sequence[Optional[str]]], None]


def blocked_layout(kind: Kind) -> tuple[str, Union[str, Sequence[Sequence[Optional[str]]], None]]:
    """Layout of a system of several blocks selected by ``kind``.

    Args:
        kind: ``"nest"`` or a nested sequence of matrix types selects a nest matrix, the types being
            those of its blocks. ``None``, ``"mpi"`` or a PETSc matrix type selects a single
            monolithic matrix.

    Returns:
        ``("nest", types)``, with `types` ``None`` for the default type of every block, or
        ``("block", type)``, with `type` ``None`` for the default type.
    """
    if isinstance(kind, str):
        if kind == "nest":
            return "nest", None
        return "block", None if kind == "mpi" else kind
    if kind is None:
        return "block", None
    return "nest", kind


def single_type(kind: Kind) -> Optional[str]:
    """PETSc matrix type of a single, non-blocked matrix selected by ``kind``, `None` for the default."""
    if kind is None or kind == "mpi":
        return None
    if not isinstance(kind, str) or kind == "nest":
        raise ValueError(f"A single form needs a PETSc matrix type as kind, got {kind!r}")
    return kind


def deprecated(old: str, new: str) -> None:
    """Warn that the function `old` is deprecated in favour of `new`, attributed to the caller of `old`."""
    warnings.warn(f"{old} is deprecated, use {new}", DeprecationWarning, stacklevel=3)

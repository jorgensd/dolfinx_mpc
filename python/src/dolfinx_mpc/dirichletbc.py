# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Dirichlet condition data shared by the assemblers.

To avoid re-computing the marked degrees of freedom that
gets its rows and columns zeroed in assembly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt

__all__ = ["BCData"]


class BCData:
    """Dof markers, diagonal rows and lifting values for a set of Dirichlet conditions.

    Markers and rows depend only on a function space and the conditions, not on
    the condition values. We cache these computations so that multiple
    assembly calls inside {py:class}`dolfinx_mpc.LinearProblem` or
    {py:class}`dolfinx_mpc.NonlinearProblem` do not repeat them.
    """

    def __init__(self, bcs: Optional[Sequence[_fem.DirichletBC]] = None):
        self._bcs = list(bcs) if bcs else []
        self._markers: dict[int, npt.NDArray[np.int8]] = {}
        self._rows: dict[int, npt.NDArray[np.int32]] = {}

    def markers(
        self, V0: _fem.FunctionSpace, V1: _fem.FunctionSpace
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        """Constrained dof markers for the test and trial spaces of a form."""
        return self._marker(V0), self._marker(V1)

    def rows(self, V: _fem.FunctionSpace) -> npt.NDArray[np.int32]:
        """Locally owned rows of `V` carrying a Dirichlet condition."""
        key = id(V._cpp_object)
        if key not in self._rows:
            owned = [bc.dof_indices()[0][: bc.dof_indices()[1]] for bc in self._bcs if V.contains(bc.function_space)]
            self._rows[key] = np.concatenate(owned) if owned else np.empty(0, dtype=np.int32)
        return self._rows[key]

    def lifting(
        self, V: _fem.FunctionSpace, dtype: npt.DTypeLike
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.generic]]:
        """Constrained dof markers and Dirichlet values on the trial space `V`.

        The markers are cached; the values are re-read from the conditions on
        every call, so a time-dependent condition is picked up. Both are empty
        when no condition applies to `V`.
        """
        markers = self._marker(V)
        values = np.zeros(markers.size, dtype=dtype)
        if markers.size > 0:
            for bc in self._bcs:
                if V.contains(bc.function_space):
                    bc.set(values, None, 1)
        return markers, values

    def _marker(self, V: _fem.FunctionSpace) -> npt.NDArray[np.int8]:
        key = id(V._cpp_object)
        if key not in self._markers:
            self._markers[key] = _fem.assemble._bc_dof_markers(V, self._bcs)
        return self._markers[key]

# Copyright (C) 2021 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Numba extension for dolfinx_mpc"""

# flake8: noqa


try:
    import numba
except ModuleNotFoundError:
    raise ModuleNotFoundError("Numba is required to use numba assembler")

from .assemble_matrix import assemble_matrix
from .assemble_vector import assemble_vector

# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINX_MPC"""

# flake8: noqa

from .multipointconstraint import MultiPointConstraint
from .assemble_matrix import assemble_matrix_mpc
import dolfinx.cpp
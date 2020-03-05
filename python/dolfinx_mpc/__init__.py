# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINX_MPC"""

# flake8: noqa

from .multipointconstraint import backsubstitution, slave_master_structure, dof_close_to, facet_normal_approximation
from .assemble_matrix import assemble_matrix_numba, assemble_matrix
from .assemble_vector import assemble_vector_numba, assemble_vector
import dolfinx_mpc.cpp

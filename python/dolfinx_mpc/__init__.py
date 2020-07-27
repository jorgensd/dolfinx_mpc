# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINX_MPC"""

# flake8: noqa

from .multipointconstraint import (backsubstitution, slave_master_structure, dof_close_to, facet_normal_approximation,
                                   create_collision_constraint)
from .assemble_matrix import assemble_matrix
from .assemble_vector import assemble_vector

import dolfinx_mpc.cpp

# New local assemblies
from .assemble_matrix_new import assemble_matrix_local
from .assemble_vector_new import assemble_vector_local
from .backsubstitution_new import backsubstitution_local
from .contactcondition import create_contact_condition
from .slipcondition import create_slip_condition

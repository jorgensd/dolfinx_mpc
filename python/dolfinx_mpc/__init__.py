# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINX_MPC"""

# flake8: noqa

import dolfinx_mpc.cpp

# New local assemblies
from .assemble_matrix import assemble_matrix
from .assemble_vector import assemble_vector
from .backsubstitution import backsubstitution
from .contactcondition import create_contact_condition
from .slipcondition import create_slip_condition
from .dictcondition import create_dictionary_constraint
from .periodic_condition import create_periodic_condition

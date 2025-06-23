# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Main module for DOLFINX_MPC"""

# flake8: noqa
from __future__ import annotations

import dolfinx_mpc.cpp

# New local assemblies
from .assemble_matrix import (
    assemble_matrix,
    assemble_matrix_nest,
    create_matrix_nest,
    create_sparsity_pattern,
)
from .assemble_vector import (
    apply_lifting,
    assemble_vector,
    assemble_vector_nest,
    create_vector_nest,
)
from .multipointconstraint import MultiPointConstraint
from .problem import LinearProblem, NonlinearProblem

__all__ = [
    "assemble_matrix",
    "create_matrix_nest",
    "assemble_matrix_nest",
    "assemble_vector",
    "apply_lifting",
    "assemble_vector_nest",
    "create_vector_nest",
    "MultiPointConstraint",
    "LinearProblem",
    "create_sparsity_pattern",
    "NonlinearProblem"
]

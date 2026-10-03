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
from .dirichletbc import BCData
from .assemble_matrix import (
    assemble_matrix,
    assemble_matrix_nest,
    create_matrix,
    create_matrix_nest,
    create_sparsity_pattern,
)
from .assemble_vector import (
    apply_lifting,
    apply_mpc_lifting,
    assemble_vector,
    assemble_vector_nest,
    create_vector,
    create_vector_nest,
)
from .integralcondition import create_integral_constraint
from .multipointconstraint import MultiPointConstraint, finalize_multipointconstraints
from .problem import LinearProblem, NonlinearProblem, assemble_jacobian_mpc, assemble_residual_mpc
from .rbe import create_spider_mesh, locate_spider, spider_values

__all__ = [
    "BCData",
    "assemble_matrix",
    "create_matrix",
    "create_matrix_nest",
    "assemble_matrix_nest",
    "assemble_vector",
    "create_vector",
    "apply_lifting",
    "apply_mpc_lifting",
    "assemble_vector_nest",
    "create_vector_nest",
    "MultiPointConstraint",
    "finalize_multipointconstraints",
    "create_integral_constraint",
    "LinearProblem",
    "create_sparsity_pattern",
    "NonlinearProblem",
    "assemble_jacobian_mpc",
    "assemble_residual_mpc",
    "create_spider_mesh",
    "locate_spider",
    "spider_values",
]

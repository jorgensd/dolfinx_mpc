# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import List

import dolfinx
import dolfinx.common
import dolfinx.log

import ufl
from dolfinx_mpc import cpp
from petsc4py import PETSc
from .multipointconstraint import MultiPointConstraint, cpp_dirichletbc

Timer = dolfinx.common.Timer


def assemble_matrix(form: ufl.form.Form, constraint: MultiPointConstraint, bcs: List[dolfinx.fem.DirichletBC] = [],
                    diagval: PETSc.ScalarType = 1, A: PETSc.Mat = None,
                    form_compiler_parameters={}, jit_parameters={}) -> PETSc.Mat:
    """
    Assemble a bi-linear form into a PETSc matrix with corresponding multi point constraints and Dirichlet boundary
    conditions.

    Parameters
    ----------
    form
        The bilinear variational form
    constraint
        The multi point constraint
    bcs
        List of Dirichlet boundary conditions
    diagval
        Value to set on the diagonal of the matrix (Default 1)
    A
        PETSc matrix to assemble into (optional)
    form_compiler_parameters
        Parameters used in FFCx compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINx.

    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCx.
        See `dolfinx/python/dolfinx/jit.py` for all available parameters.
        Takes priority over all other parameter values.

    Returns
    -------
    PETSc.Mat
        The assembled bi-linear form
    """
    assert(form.arguments()[0].ufl_function_space() == form.arguments()[1].ufl_function_space())

    cpp_form = dolfinx.Form(form, form_compiler_parameters=form_compiler_parameters,
                            jit_parameters=jit_parameters)._cpp_object

    # Generate matrix with MPC sparsity pattern
    if A is None:
        A = cpp.mpc.create_matrix(cpp_form, constraint._cpp_object)
    A.zeroEntries()

    # Assemble matrix in C++
    cpp_bc = cpp_dirichletbc(bcs)
    cpp.mpc.assemble_matrix(A, cpp_form, constraint._cpp_object, cpp_bc, diagval)

    # Add one on diagonal for Dirichlet boundary conditions
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, cpp_form.function_spaces[0], cpp_bc, diagval)

    A.assemble()
    return A

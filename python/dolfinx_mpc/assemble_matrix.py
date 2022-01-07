# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from typing import List

import dolfinx.fem as _fem
import dolfinx.cpp as _cpp

from dolfinx_mpc import cpp
from petsc4py import PETSc as _PETSc
from .multipointconstraint import MultiPointConstraint


def assemble_matrix(form: _fem.FormMetaClass, constraint: MultiPointConstraint,
                    bcs: List[_fem.DirichletBCMetaClass] = [], diagval: _PETSc.ScalarType = 1,
                    A: _PETSc.Mat = None) -> _PETSc.Mat:
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix with corresponding multi point constraints
    and Dirichlet boundary conditions.

    Parameters
    ----------
    form
        The compiled bilinear variational form
    constraint
        The multi point constraint
    bcs
        List of Dirichlet boundary conditions
    diagval
        Value to set on the diagonal of the matrix (Default 1)
    A
        PETSc matrix to assemble into (optional)

    Returns
    -------
    _PETSc.Mat
        The assembled bi-linear form
    """
    assert(form.function_spaces[0] == form.function_spaces[1])

    # Generate matrix with MPC sparsity pattern
    if A is None:
        A = cpp.mpc.create_matrix(form, constraint._cpp_object)
    A.zeroEntries()

    # Assemble matrix in C++
    cpp.mpc.assemble_matrix(A, form, constraint._cpp_object, bcs, diagval)

    # Add one on diagonal for Dirichlet boundary conditions
    if form.function_spaces[0].id == form.function_spaces[1].id:
        A.assemblyBegin(_PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(_PETSc.Mat.AssemblyType.FLUSH)
        _cpp.fem.petsc.insert_diagonal(A, form.function_spaces[0], bcs, diagval)

    A.assemble()
    return A

# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from typing import Optional, Sequence, Union

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
from petsc4py import PETSc as _PETSc

from dolfinx_mpc import cpp

from .multipointconstraint import MultiPointConstraint


def assemble_matrix(form: _fem.Form,
                    constraint: Union[MultiPointConstraint,
                                      Sequence[MultiPointConstraint]],
                    bcs: Optional[Sequence[_fem.DirichletBC]] = None,
                    diagval: _PETSc.ScalarType = 1,  # type: ignore
                    A: Optional[_PETSc.Mat] = None) -> _PETSc.Mat:  # type: ignore
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix with corresponding multi point constraints
    and Dirichlet boundary conditions.

    Args:
        form: The compiled bilinear variational form
        constraint: The multi point constraint
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix
        A: PETSc matrix to assemble into

    Returns:
        _PETSc.Mat: The matrix with the assembled bi-linear form  #type: ignore
    """
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    if not isinstance(constraint, Sequence):
        assert form.function_spaces[0] == form.function_spaces[1]
        constraint = (constraint, constraint)

    # Generate matrix with MPC sparsity pattern
    if A is None:
        A = cpp.mpc.create_matrix(form._cpp_object, constraint[0]._cpp_object,
                                  constraint[1]._cpp_object)
    A.zeroEntries()

    # Assemble matrix in C++
    cpp.mpc.assemble_matrix(A, form._cpp_object, constraint[0]._cpp_object,
                            constraint[1]._cpp_object, bcs, diagval)

    # Add one on diagonal for Dirichlet boundary conditions
    if form.function_spaces[0] is form.function_spaces[1]:
        A.assemblyBegin(_PETSc.Mat.AssemblyType.FLUSH)  # type: ignore
        A.assemblyEnd(_PETSc.Mat.AssemblyType.FLUSH)  # type: ignore
        _cpp.fem.petsc.insert_diagonal(A, form.function_spaces[0], bcs, diagval)

    A.assemble()
    return A


def create_sparsity_pattern(form: _fem.Form,
                            mpc: Union[MultiPointConstraint,
                                       Sequence[MultiPointConstraint]]):
    """
    Create sparsity-pattern for MPC given a compiled DOLFINx form

    Args:
        form: The form
        mpc: For square forms, the MPC. For rectangular forms a list of 2 MPCs on
            axis 0 & 1, respectively
    """
    if isinstance(mpc, list):
        assert len(mpc) == 2
        for mpc_ in mpc:
            mpc_._not_finalized()  # type: ignore
            return cpp.mpc.create_sparsity_pattern(form._cpp_object, mpc[0]._cpp_object,
                                                   mpc[1]._cpp_object)
    else:
        mpc._not_finalized()  # type: ignore
        return cpp.mpc.create_sparsity_pattern(form._cpp_object, mpc._cpp_object,  # type: ignore
                                               mpc._cpp_object)  # type: ignore


def create_matrix_nest(
        a: Sequence[Sequence[_fem.Form]],
        constraints: Sequence[MultiPointConstraint]):
    """
    Create a PETSc matrix of type "nest" with appropriate sparsity pattern
    given the provided multi points constraints

    Args:
       a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
    """
    assert len(constraints) == len(a)

    A_ = [[None for _ in range(len(a[0]))] for _ in range(len(a))]

    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a[i][j] is None:
                continue
            A_[i][j] = cpp.mpc.create_matrix(
                a[i][j]._cpp_object, constraints[i]._cpp_object, constraints[j]._cpp_object)

    A = _PETSc.Mat().createNest(  # type: ignore
        A_, comm=constraints[0].function_space.mesh.comm)
    return A


def assemble_matrix_nest(
        A: _PETSc.Mat,  # type: ignore
        a: Sequence[Sequence[_fem.Form]],
        constraints: Sequence[MultiPointConstraint],
        bcs: Sequence[_fem.DirichletBC] = [],
        diagval: _PETSc.ScalarType = 1):  # type: ignore
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix of type
    "nest" with corresponding multi point constraints and Dirichlet boundary
    conditions.

    Args:
        a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix (Default 1)
        A: PETSc matrix to assemble into
    """
    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                assemble_matrix(
                    a_block, (constraints[i], constraints[j]),
                    bcs=bcs, diagval=diagval, A=Asub)

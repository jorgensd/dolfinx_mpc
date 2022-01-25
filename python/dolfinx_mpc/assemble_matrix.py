# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from typing import Sequence, Union
import collections

import dolfinx.fem as _fem
import dolfinx.cpp as _cpp

from dolfinx_mpc import cpp
from petsc4py import PETSc as _PETSc
from .multipointconstraint import MultiPointConstraint


def assemble_matrix(form: _fem.FormMetaClass,
                    constraint: Union[MultiPointConstraint,
                                      Sequence[MultiPointConstraint]],
                    bcs: Sequence[_fem.DirichletBCMetaClass] = [],
                    diagval: _PETSc.ScalarType = 1,
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
        Sequence of Dirichlet boundary conditions
    diagval
        Value to set on the diagonal of the matrix (Default 1)
    A
        PETSc matrix to assemble into (optional)

    Returns
    -------
    _PETSc.Mat
        The assembled bi-linear form
    """
    if not isinstance(constraint, collections.Sequence):
        assert(form.function_spaces[0] == form.function_spaces[1])
        constraint = (constraint, constraint)

    # Generate matrix with MPC sparsity pattern
    if A is None:
        A = cpp.mpc.create_matrix(form, constraint[0]._cpp_object,
                                  constraint[1]._cpp_object)
    A.zeroEntries()

    # Assemble matrix in C++
    cpp.mpc.assemble_matrix(A, form, constraint[0]._cpp_object,
                            constraint[1]._cpp_object, bcs, diagval)

    # Add one on diagonal for Dirichlet boundary conditions
    if form.function_spaces[0].id == form.function_spaces[1].id:
        A.assemblyBegin(_PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(_PETSc.Mat.AssemblyType.FLUSH)
        _cpp.fem.petsc.insert_diagonal(A, form.function_spaces[0], bcs, diagval)

    A.assemble()
    return A


def create_sparsity_pattern(form: _fem.FormMetaClass,
                            mpc: Union[MultiPointConstraint,
                                       Sequence[MultiPointConstraint]]):
    """
    Create sparsity-pattern for MPC given a compiled DOLFINx form

    Parameters
    ----------
    form
        The form
    mpc
        For square forms, the MPC. For rectangular forms a list of 2 MPCs on
        axis 0 & 1, respectively
    """
    if not isinstance(mpc, list):
        mpc = [mpc, mpc]
    assert len(mpc) == 2
    for mpc_ in mpc:
        mpc_._not_finalized()
    return cpp.mpc.create_sparsity_pattern(form, mpc[0]._cpp_object,
                                           mpc[1]._cpp_object)


def create_matrix_nest(
        a: Sequence[Sequence[_fem.FormMetaClass]],
        constraint: Sequence[MultiPointConstraint]):
    assert len(constraint) == len(a)

    A_ = [[None for _ in range(len(a[0]))] for _ in range(len(a))]

    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a[i][j] is None:
                continue
            A_[i][j] = cpp.mpc.create_matrix(
                a[i][j], constraint[i]._cpp_object, constraint[j]._cpp_object)

    A = _PETSc.Mat().createNest(
        A_, comm=constraint[0].function_space.mesh.comm)
    return A


def assemble_matrix_nest(
        A: _PETSc.Mat,
        a: Sequence[Sequence[_fem.FormMetaClass]],
        constraint: Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
        bcs: Sequence[_fem.DirichletBCMetaClass] = [],
        diagval: _PETSc.ScalarType = 1):
    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                assemble_matrix(
                    a_block, (constraint[i], constraint[j]),
                    bcs=bcs, diagval=diagval, A=Asub)
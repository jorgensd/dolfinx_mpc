# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem

from dolfinx_mpc import cpp

from .multipointconstraint import MultiPointConstraint


def _assemble_form(
    A: _PETSc.Mat,  # type: ignore
    form: _fem.Form,
    constraint: Sequence[MultiPointConstraint],
    bcs: Sequence,
    num_threads: Optional[int] = 1,
):
    """
    Assemble one compiled form into a matrix.

    Additive: `A` is not zeroed. Does not finalise or add any diagonal entry —
    those belong to the system rather than to a single form, see
    :func:`_finalize_matrix`.
    """
    cpp.mpc.assemble_matrix(A, form._cpp_object, constraint[0]._cpp_object, constraint[1]._cpp_object, bcs, num_threads)


def _finalize_matrix(
    A: _PETSc.Mat,  # type: ignore
    slave_blocks: Sequence,
    bc_blocks: Sequence,
    bcs: Sequence,
    diagval: _PETSc.ScalarType = 1,  # type: ignore
):
    """
    Add the diagonal entries and assemble `A`.

    A diagonal entry belongs to a block of the system, not to a form: a block
    may carry slaves without having a diagonal bilinear form to assemble, and a
    block appearing in several forms must still receive exactly one entry. So
    this runs once, after every form has been assembled into `A`.

    Args:
        A: The matrix, with every form already assembled into it
        slave_blocks: `(sub-matrix, constraint)` pairs whose slave rows get `diagval`
        bc_blocks: `(sub-matrix, function space)` pairs whose Dirichlet rows get `diagval`
        bcs: Sequence of C++ Dirichlet boundary conditions
        diagval: Value to place on the diagonal
    """
    for A_sub, mpc in slave_blocks:
        cpp.mpc.insert_diagonal_slaves(A_sub, mpc._cpp_object, diagval)

    # The slave diagonal is added, the Dirichlet diagonal is inserted, so the
    # additive contributions have to be communicated before switching mode.
    if bc_blocks:
        A.assemblyBegin(_PETSc.Mat.AssemblyType.FLUSH)  # type: ignore
        A.assemblyEnd(_PETSc.Mat.AssemblyType.FLUSH)  # type: ignore
        for A_sub, V in bc_blocks:
            _cpp.fem.petsc.insert_diagonal(A_sub, V, bcs, diagval)

    A.assemble()


def assemble_matrix(
    form: _fem.Form,
    constraint: Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    bcs: Optional[Sequence[_fem.DirichletBC]] = None,
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    A: Optional[_PETSc.Mat] = None,  # type: ignore
    num_threads: Optional[int] = 1,
) -> _PETSc.Mat:  # type: ignore
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix with corresponding multi point constraints
    and Dirichlet boundary conditions.

    Args:
        form: The compiled bilinear variational form
        constraint: The multi point constraint
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix
        A: PETSc matrix to assemble into. Assembly is additive, so `A` is not
            zeroed; call `A.zeroEntries()` first to discard its contents. If not
            supplied a new matrix is created, which is already zeroed.
        num_threads: The number of threads to use for certain operations
    Returns:
        _PETSc.Mat: The matrix with the assembled bi-linear form  #type: ignore
    """
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    if not isinstance(constraint, Sequence):
        assert form.function_spaces[0] == form.function_spaces[1]
        constraint = (constraint, constraint)

    # Generate matrix with MPC sparsity pattern. A freshly created matrix is
    # already zeroed; an `A` supplied by the caller is added into, following the
    # additive convention of the DOLFINx assemblers.
    if A is None:
        A = cpp.mpc.create_matrix(form._cpp_object, constraint[0]._cpp_object, constraint[1]._cpp_object)

    _assemble_form(A, form, constraint, bcs, num_threads)

    slave_blocks = [(A, constraint[0])] if constraint[0] is constraint[1] else []
    bc_blocks = [(A, form.function_spaces[0])] if form.function_spaces[0] is form.function_spaces[1] else []
    _finalize_matrix(A, slave_blocks, bc_blocks, bcs, diagval)
    return A


def create_sparsity_pattern(form: _fem.Form, mpc: Union[MultiPointConstraint, Sequence[MultiPointConstraint]]):
    """
    Create sparsity-pattern for MPC given a compiled DOLFINx form

    Args:
        form: The form
        mpc: For square forms, the MPC. For rectangular forms a list of 2 MPCs on
            axis 0 & 1, respectively
    """
    if isinstance(mpc, Sequence):
        assert len(mpc) == 2
        for mpc_ in mpc:
            mpc_._not_finalized()  # type: ignore
            return cpp.mpc.create_sparsity_pattern(form._cpp_object, mpc[0]._cpp_object, mpc[1]._cpp_object)
    else:
        mpc._not_finalized()  # type: ignore
        return cpp.mpc.create_sparsity_pattern(
            form._cpp_object,
            mpc._cpp_object,  # type: ignore
            mpc._cpp_object,  # type: ignore
        )  # type: ignore


def create_matrix_nest(a: Sequence[Sequence[_fem.Form]], constraints: Sequence[MultiPointConstraint]):
    """
    Create a PETSc matrix of type "nest" with appropriate sparsity pattern
    given the provided multi points constraints

    Args:
       a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
    """
    assert len(constraints) == len(a)

    A_: list[list[_PETSc.Mat | None]] = [[None for _ in range(len(a[0]))] for _ in range(len(a))]

    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a[i][j] is None:
                continue
            A_[i][j] = cpp.mpc.create_matrix(
                a[i][j]._cpp_object, constraints[i]._cpp_object, constraints[j]._cpp_object
            )

    A = _PETSc.Mat().createNest(
        A_,  # type: ignore
        comm=constraints[0].function_space.mesh.comm,
    )
    return A


def assemble_matrix_nest(
    A: _PETSc.Mat,  # type: ignore
    a: Sequence[Sequence[_fem.Form]],
    constraints: Sequence[MultiPointConstraint],
    bcs: Sequence[_fem.DirichletBC] = [],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    num_threads: Optional[int] = 1,
):
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix of type
    "nest" with corresponding multi point constraints and Dirichlet boundary
    conditions.

    Args:
        A: PETSc matrix to assemble into. Assembly is additive, so `A` is not
            zeroed; call `A.zeroEntries()` first to discard its contents.
        a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix (Default 1)
        num_threads: The number of threads to use for certain operations
    """
    _bcs = [bc._cpp_object for bc in bcs]

    for i, a_row in enumerate(a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                _assemble_form(
                    A.getNestSubMatrix(i, j),
                    a_block,
                    (constraints[i], constraints[j]),
                    _bcs,
                    num_threads,
                )

    # The diagonal is a property of a block, so it is added once per diagonal
    # block after every form has been assembled.
    slave_blocks = []
    bc_blocks = []
    for i, a_row in enumerate(a):
        a_ii = a_row[i] if i < len(a_row) else None
        if a_ii is None:
            continue
        A_ii = A.getNestSubMatrix(i, i)
        slave_blocks.append((A_ii, constraints[i]))
        if a_ii.function_spaces[0] is a_ii.function_spaces[1]:
            bc_blocks.append((A_ii, a_ii.function_spaces[0]))

    _finalize_matrix(A, slave_blocks, bc_blocks, _bcs, diagval)

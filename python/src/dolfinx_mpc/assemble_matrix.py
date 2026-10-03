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
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import numpy.typing as npt
from dolfinx import default_scalar_type

from dolfinx_mpc import cpp

from ._kind import Kind, blocked_layout, deprecated, single_type
from .dirichletbc import BCData
from .multipointconstraint import MultiPointConstraint


def _assemble_form(
    A: _PETSc.Mat,  # type: ignore
    form: _fem.Form,
    constraint: Sequence[MultiPointConstraint],
    bc_data: BCData,
    num_threads: Optional[int] = 1,
):
    """
    Assemble one compiled form into a matrix.

    Additive: `A` is not zeroed. Does not finalise or add any diagonal entry —
    those belong to the system rather than to a single form, see
    :func:`_finalize_matrix`.
    """
    # Markers come from the cache rather than being rebuilt here, so a caller
    # that reassembles the same form pays for them once; see `BCData`.
    dof_marker0, dof_marker1 = bc_data.markers(*form.function_spaces)
    cpp.mpc.assemble_matrix(
        A,
        form._cpp_object,
        constraint[0]._cpp_object,
        constraint[1]._cpp_object,
        dof_marker0,
        dof_marker1,
        num_threads,
    )


def _add_diagonals(
    slave_blocks: Sequence,
    bc_blocks: Sequence[tuple[_PETSc.Mat, npt.NDArray[np.int32]]],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
):
    """
    Add the diagonal entries of the slave and Dirichlet rows. Does not assemble.

    A diagonal entry belongs to a block of the system, not to a form: a block
    may carry slaves without having a diagonal bilinear form to assemble, and a
    block appearing in several forms must still receive exactly one entry. So
    this runs once, after every form has been assembled.

    Args:
        slave_blocks: `(sub-matrix, constraint)` pairs whose slave rows get `diagval`
        bc_blocks: `(sub-matrix, rows)` pairs whose Dirichlet rows get `diagval`
        diagval: Value to place on the diagonal
    """
    for A_sub, mpc in slave_blocks:
        cpp.mpc.insert_diagonal_slaves(A_sub, mpc._cpp_object, diagval)

    # Both diagonals are added rather than inserted, so no flush is needed to
    # take the matrix out of add mode. Adding is equivalent here because
    # assembly leaves a Dirichlet row empty: the element matrix has its
    # constrained rows and columns zeroed, slaves are rejected at finalize if
    # they carry a condition, and a master that carries one is eliminated into
    # the constraint offset rather than kept in the master list.
    for A_sub, rows in bc_blocks:
        _cpp.fem.petsc.set_diagonal(A_sub, rows, default_scalar_type(diagval), _PETSc.InsertMode.ADD_VALUES)  # type: ignore


def _finalize_matrix(
    A: _PETSc.Mat,  # type: ignore
    slave_blocks: Sequence,
    bc_blocks: Sequence[tuple[_PETSc.Mat, npt.NDArray[np.int32]]],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
):
    """
    Add the diagonal entries, see :func:`_add_diagonals`, and assemble `A`.

    Args:
        A: The matrix, with every form already assembled into it
        slave_blocks: `(sub-matrix, constraint)` pairs whose slave rows get `diagval`
        bc_blocks: `(sub-matrix, rows)` pairs whose Dirichlet rows get `diagval`
        diagval: Value to place on the diagonal
    """
    _add_diagonals(slave_blocks, bc_blocks, diagval)
    A.assemble()


def assemble_matrix(
    form: Union[_fem.Form, Sequence[Sequence[Optional[_fem.Form]]]],
    constraint: Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    bcs: Optional[Sequence[_fem.DirichletBC]] = None,
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    A: Optional[_PETSc.Mat] = None,  # type: ignore
    num_threads: Optional[int] = 1,
    bc_data: Optional[BCData] = None,
    kind: Kind = None,
) -> _PETSc.Mat:  # type: ignore
    """
    Assemble a compiled DOLFINx bilinear form, or an array of them, into a PETSc matrix with
    corresponding multi point constraints and Dirichlet boundary conditions.

    As in :func:`dolfinx.fem.petsc.assemble_matrix`, the kind of matrix is selected by `kind`, or by
    the type of `A` if it is supplied.

    Args:
        form: The compiled bilinear variational form, or a rank 2 list of them with `None` for a
            block without a form
        constraint: For a single form, its multi point constraint, or for a rectangular form a
            list of 2 constraints on axis 0 & 1. For an array of forms, the constraint of each
            block, which is used for the rows and the columns.
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix
        A: PETSc matrix to assemble into. Assembly is additive, so `A` is not
            zeroed; call `A.zeroEntries()` first to discard its contents. If not
            supplied a new matrix is created, which is already zeroed.
        num_threads: The number of threads to use for certain operations
        bc_data: A :class:`BCData` cache. Built from `bcs` when not supplied;
            pass one to share it with the other assemblies of the same system.
        kind: The kind of matrix to create when `A` is not supplied, see :func:`create_matrix`.
    Returns:
        _PETSc.Mat: The matrix with the assembled bi-linear form  #type: ignore
    """
    if bc_data is None:
        bc_data = BCData(bcs)

    if isinstance(form, Sequence):
        if not isinstance(constraint, Sequence):
            raise ValueError("An array of forms needs one multi point constraint per block")
        if A is None:
            A = create_matrix(form, constraint, kind)
        if A.getType() == "nest":
            _assemble_matrix_nest(A, form, constraint, diagval=diagval, num_threads=num_threads, bc_data=bc_data)
        else:
            _assemble_matrix_block(A, form, constraint, diagval=diagval, num_threads=num_threads, bc_data=bc_data)
        return A

    if not isinstance(constraint, Sequence):
        assert form.function_spaces[0] == form.function_spaces[1]
        constraint = (constraint, constraint)

    # Generate matrix with MPC sparsity pattern. A freshly created matrix is
    # already zeroed; an `A` supplied by the caller is added into.
    if A is None:
        A = create_matrix(form, constraint, kind)

    _assemble_form(A, form, constraint, bc_data, num_threads)

    V0, V1 = form.function_spaces
    slave_blocks = [(A, constraint[0])] if constraint[0] is constraint[1] else []
    bc_blocks = [(A, bc_data.rows(V0))] if V0 is V1 else []
    _finalize_matrix(A, slave_blocks, bc_blocks, diagval)

    return A


def create_matrix(
    a: Union[_fem.Form, Sequence[Sequence[Optional[_fem.Form]]]],
    constraint: Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    kind: Kind = None,
) -> _PETSc.Mat:  # type: ignore
    """
    Create a PETSc matrix with the sparsity pattern of a bilinear form, or an array of them, under
    multi point constraints.

    As in :func:`dolfinx.fem.petsc.create_matrix`, three cases are supported:

    1. A single form gives a matrix of the PETSc type `kind`, the default if `None`.
    2. An array of forms with `kind` ``"nest"``, or a nested sequence of PETSc matrix types of the
       same shape as the array, gives a matrix of type ``nest`` whose blocks have those types.
    3. An array of forms with any other `kind` gives a single, monolithic matrix of PETSc type
       `kind`, the default if `None` or ``"mpi"``, arranged as
       :math:`A = [a_{ij}]` with the dofs of each block ordered ``[owned, ghosts]`` and the blocks
       one after another. The ghosts include the masters added by the constraint of the block.
       A diagonal block that has no form still reserves the diagonal entry of each of its slaves.

    Args:
        a: The compiled bilinear form, or a rank 2 list of them with `None` for a block without a form
        constraint: As in :func:`assemble_matrix`
        kind: The kind of matrix, as above

    Returns:
        The matrix, to assemble into with :func:`assemble_matrix`.
    """
    if not isinstance(a, Sequence):
        if not isinstance(constraint, Sequence):
            constraint = (constraint, constraint)
        for mpc in constraint:
            mpc._raise_if_not_finalized()
        return cpp.mpc.create_matrix(
            a._cpp_object, constraint[0]._cpp_object, constraint[1]._cpp_object, single_type(kind)
        )

    if not isinstance(constraint, Sequence):
        raise ValueError("An array of forms needs one multi point constraint per block")
    layout, types = blocked_layout(kind)
    if layout == "nest":
        return _create_matrix_nest(a, constraint, types)
    return _create_matrix_block(a, constraint, matrix_type=types)  # type: ignore[arg-type]


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
            mpc_._raise_if_not_finalized()  # type: ignore
            return cpp.mpc.create_sparsity_pattern(form._cpp_object, mpc[0]._cpp_object, mpc[1]._cpp_object)
    else:
        mpc._raise_if_not_finalized()  # type: ignore
        return cpp.mpc.create_sparsity_pattern(
            form._cpp_object,
            mpc._cpp_object,  # type: ignore
            mpc._cpp_object,  # type: ignore
        )  # type: ignore


def _create_matrix_nest(
    a: Sequence[Sequence[Optional[_fem.Form]]],
    constraints: Sequence[MultiPointConstraint],
    types: Optional[Sequence[Sequence[Optional[str]]]] = None,
):
    """
    Create a PETSc matrix of type "nest" with the blocks of the types in `types`, if given.

    A block has a matrix if it has a form or is a diagonal block, which holds the diagonal of its
    slaves. If a constraint has masters in another block, every block has one, since those masters
    put entries in blocks without a form.
    """
    assert len(constraints) == len(a)
    for mpc in constraints:
        mpc._raise_if_not_finalized()
    forms = [[None if a_ij is None else a_ij._cpp_object for a_ij in a_i] for a_i in a]
    mpcs = [mpc._cpp_object for mpc in constraints]
    _types = None if types is None else [list(t) for t in types]
    return cpp.mpc.create_matrix_nest(forms, mpcs, mpcs, _types)


def create_matrix_nest(a: Sequence[Sequence[_fem.Form | None]], constraints: Sequence[MultiPointConstraint]):
    """
    Create a PETSc matrix of type "nest" with appropriate sparsity pattern
    given the provided multi points constraints

    .. deprecated::
        Use :func:`create_matrix` with ``kind="nest"``.

    Args:
       a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
    """
    deprecated("create_matrix_nest", "create_matrix(a, constraints, kind='nest')")
    return _create_matrix_nest(a, constraints)


def _block_spaces(a: Sequence[Sequence[Optional[_fem.Form]]], num_blocks: int) -> list[Optional[_fem.FunctionSpace]]:
    """The space of each diagonal block, taken from the forms, or `None` if no form has it."""
    spaces: list[Optional[_fem.FunctionSpace]] = [None] * num_blocks
    for i, a_row in enumerate(a):
        for j, a_ij in enumerate(a_row):
            if a_ij is None:
                continue
            if spaces[i] is None:
                spaces[i] = a_ij.function_spaces[0]
            if j < num_blocks and spaces[j] is None:
                spaces[j] = a_ij.function_spaces[1]
    return spaces


def _assemble_blocks(
    blocks: Sequence[Sequence[Optional[_PETSc.Mat]]],  # type: ignore
    a: Sequence[Sequence[Optional[_fem.Form]]],
    constraints: Sequence[MultiPointConstraint],
    bc_data: BCData,
    diagval: _PETSc.ScalarType,  # type: ignore
    num_threads: Optional[int],
):
    """
    Assemble an array of forms into the matrices of the blocks of a system, then add the diagonal
    of the slave and Dirichlet rows of each diagonal block.

    The entries of a master go to the block of the master, which may have no form.
    """
    for i, a_row in enumerate(a):
        for j, a_ij in enumerate(a_row):
            if a_ij is None:
                continue
            dof_marker0, dof_marker1 = bc_data.markers(*a_ij.function_spaces)
            cpp.mpc.assemble_matrix_blocks(
                blocks,
                i,
                j,
                a_ij._cpp_object,
                constraints[i]._cpp_object,
                constraints[j]._cpp_object,
                dof_marker0,
                dof_marker1,
                num_threads,
            )

    # The diagonal is a property of a block, so it is added once per diagonal block after
    # every form has been assembled, including for a block that has no form of its own
    spaces = _block_spaces(a, len(constraints))
    for i, mpc in enumerate(constraints):
        V = spaces[i]
        has_bcs = V is not None and bc_data.markers(V, V)[0].size > 0
        if a[i][i] is None and has_bcs:
            raise RuntimeError(
                f"Diagonal block ({i}, {i}) cannot be 'None' and have a Dirichlet condition applied."
                " Consider assembling a zero block."
            )
        A_ii = blocks[i][i]
        if A_ii is None:
            continue
        bc_blocks = [(A_ii, bc_data.rows(V))] if (has_bcs and V is not None) else []
        _add_diagonals([(A_ii, mpc)], bc_blocks, diagval)


def _assemble_matrix_nest(
    A: _PETSc.Mat,  # type: ignore
    a: Sequence[Sequence[Optional[_fem.Form]]],
    constraints: Sequence[MultiPointConstraint],
    bcs: Sequence[_fem.DirichletBC] = [],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    num_threads: Optional[int] = 1,
    bc_data: Optional[BCData] = None,
):
    """Assemble an array of forms into a PETSc matrix of type "nest"."""
    if bc_data is None:
        bc_data = BCData([bc for bc in bcs])
    nr, nc = A.getNestSize()
    blocks: list[list[Optional[_PETSc.Mat]]] = []  # type: ignore
    for k in range(nr):
        row = []
        for col in range(nc):
            A_kl = A.getNestSubMatrix(k, col)
            row.append(None if A_kl.handle == 0 else A_kl)
        blocks.append(row)
    _assemble_blocks(blocks, a, constraints, bc_data, diagval, num_threads)
    A.assemble()


def assemble_matrix_nest(
    A: _PETSc.Mat,  # type: ignore
    a: Sequence[Sequence[_fem.Form]],
    constraints: Sequence[MultiPointConstraint],
    bcs: Sequence[_fem.DirichletBC] = [],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    num_threads: Optional[int] = 1,
    bc_data: Optional[BCData] = None,
):
    """
    Assemble a compiled DOLFINx bilinear form into a PETSc matrix of type
    "nest" with corresponding multi point constraints and Dirichlet boundary
    conditions.

    .. deprecated::
        Use :func:`assemble_matrix`, which selects the layout from `A`.

    Args:
        A: PETSc matrix to assemble into. Assembly is additive, so `A` is not
            zeroed; call `A.zeroEntries()` first to discard its contents.
        a: The compiled bilinear variational form provided in a rank 2 list
        constraints: An ordered list of multi point constraints
        bcs: Sequence of Dirichlet boundary conditions
        diagval: Value to set on the diagonal of the matrix (Default 1)
        num_threads: The number of threads to use for certain operations
        bc_data: A :class:`BCData` cache. Built from `bcs` when not supplied;
            pass one to share it with the other assemblies of the same system.
    """
    deprecated("assemble_matrix_nest", "assemble_matrix(a, constraints, A=A)")
    _assemble_matrix_nest(A, a, constraints, bcs, diagval, num_threads, bc_data)


def _block_index_sets(constraints: Sequence[MultiPointConstraint]):
    """Index sets selecting each block of a monolithic matrix, in the local numbering of the matrix."""
    return _cpp.la.petsc.create_index_sets(
        [
            (mpc.function_space.dofmap.index_map._cpp_object, mpc.function_space.dofmap.index_map_bs)
            for mpc in constraints
        ]
    )


def _create_matrix_block(
    a: Sequence[Sequence[Optional[_fem.Form]]],
    constraints: Sequence[MultiPointConstraint],
    constraints1: Optional[Sequence[MultiPointConstraint]] = None,
    matrix_type: Optional[str] = None,
):
    """Create a monolithic PETSc matrix, see :func:`create_matrix`. The columns use `constraints1` if given."""
    cols = constraints if constraints1 is None else constraints1
    for mpc in (*constraints, *cols):
        mpc._raise_if_not_finalized()
    forms = [[None if a_ij is None else a_ij._cpp_object for a_ij in a_i] for a_i in a]
    return cpp.mpc.create_matrix_block(
        forms,
        [mpc._cpp_object for mpc in constraints],
        [mpc._cpp_object for mpc in cols],
        matrix_type,
    )


def _assemble_matrix_block(
    A: _PETSc.Mat,  # type: ignore
    a: Sequence[Sequence[Optional[_fem.Form]]],
    constraints: Sequence[MultiPointConstraint],
    bcs: Sequence[_fem.DirichletBC] = [],
    diagval: _PETSc.ScalarType = 1,  # type: ignore
    num_threads: Optional[int] = 1,
    bc_data: Optional[BCData] = None,
):
    """
    Assemble an array of forms into a monolithic PETSc matrix made by :func:`create_matrix`.

    Raises:
        RuntimeError: If a diagonal block has no form while a Dirichlet condition applies to it,
            as it would have no diagonal entry to set.
    """
    if bc_data is None:
        bc_data = BCData(list(bcs))
    is_ = _block_index_sets(constraints)
    # The local sub-matrix of every block, as a master may put entries in any of them
    blocks = [[A.getLocalSubMatrix(is_k, is_l) for is_l in is_] for is_k in is_]
    try:
        _assemble_blocks(blocks, a, constraints, bc_data, diagval, num_threads)
    finally:
        for is_k, row in zip(is_, blocks):
            for is_l, A_kl in zip(is_, row):
                A.restoreLocalSubMatrix(is_k, is_l, A_kl)
    A.assemble()

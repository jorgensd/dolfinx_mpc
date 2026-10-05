# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

from mpi4py import MPI
from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.fem.petsc  # noqa: F401
import dolfinx.la
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

    V0, V1 = form.function_spaces
    for mpc, V in {id(c): (c, W) for c, W in zip(constraint, (V0, V1))}.values():
        if not mpc._cpp_object.has_cross_block_masters:
            _raise_if_constrained_masters([mpc], [V], bc_data)
    _assemble_form(A, form, constraint, bc_data, num_threads)

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


def _block_spaces(
    a: Sequence[Sequence[Optional[_fem.Form]]], constraints: Sequence[MultiPointConstraint]
) -> list[_fem.FunctionSpace]:
    """The space of each diagonal block, from the forms, or from its constraint if no form has it.

    A block without a form still has a space: its constraint's, which may carry Dirichlet
    conditions, and whose dofs may be masters of other blocks.
    """
    spaces: list[Optional[_fem.FunctionSpace]] = [None] * len(constraints)
    for i, a_row in enumerate(a):
        for j, a_ij in enumerate(a_row):
            if a_ij is None:
                continue
            if spaces[i] is None:
                spaces[i] = a_ij.function_spaces[0]
            if j < len(constraints) and spaces[j] is None:
                spaces[j] = a_ij.function_spaces[1]
    return [mpc.input_space if V is None else V for V, mpc in zip(spaces, constraints)]


def _raise_if_constrained_masters(
    constraints: Sequence[MultiPointConstraint], spaces: Sequence[_fem.FunctionSpace], bc_data: BCData
):
    """Raise if a Dirichlet condition of the assembly constrains a master the constraints kept.

    Assembly adds the entries of a slave's row and column to those of its masters after the
    constrained rows and columns are zeroed, so a constrained master must have been eliminated, by
    giving the condition to the constraint of the master's block before finalizing it. The verdict
    is cached in `bc_data`.

    Args:
        constraints: The constraint of each block, in the order they were finalized in if a
            constraint has masters in another block
        spaces: The space of each block, on which the conditions are stated
        bc_data: The Dirichlet conditions of the assembly

    Raises:
        ValueError: On every process, if a kept master is constrained.

    Note:
        Collective.
    """
    key = tuple(id(mpc._cpp_object) for mpc in constraints)
    if key in bc_data._checked:
        return
    # The markers of each block on its extended space: the owned dofs are numbered as in the
    # space of the conditions, the ghosts are the extended space's own
    markers: list[Optional[npt.NDArray[np.bool_]]] = []
    for mpc, V in zip(constraints, spaces):
        # Decided by the spaces of the conditions, the same on every process, as the scatter is
        # collective; the markers of a process without dofs are empty
        if not any(V.contains(bc.function_space) for bc in bc_data._bcs):
            markers.append(None)
            continue
        owned_markers = bc_data.markers(V, V)[0]
        dofmap = mpc.function_space.dofmap
        marker = dolfinx.la.vector(dofmap.index_map, dofmap.index_map_bs, dtype=np.float64)
        num_owned = dofmap.index_map.size_local * dofmap.index_map_bs
        marker.array[:num_owned] = owned_markers[:num_owned]
        marker.scatter_forward()
        markers.append(marker.array > 0)

    constrained = False
    if any(marker is not None for marker in markers):
        for k, mpc in enumerate(constraints):
            masters = mpc._cpp_object.masters.array
            blocks = np.asarray(mpc._cpp_object.master_blocks)
            if blocks.size == 0:
                blocks = np.full(masters.size, k, dtype=np.int32)
            else:
                # A master in the constraint's own block is in block `k` of the system, also when
                # the constraint was finalized on its own; any other block is the master's
                blocks = np.where(blocks == mpc._cpp_object.block, k, blocks)
            for j, marker in enumerate(markers):
                if marker is not None and marker[masters[blocks == j]].any():
                    constrained = True
    if constraints[0].function_space.mesh.comm.allreduce(constrained, op=MPI.LOR):
        raise ValueError(
            "A Dirichlet condition of the assembly constrains a master of a multi point constraint. "
            "Give the condition to the constraint of the master's space as well, "
            "MultiPointConstraint(V, bcs=...), before finalizing, so that the master is eliminated."
        )
    bc_data._checked.add(key)


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

    Raises:
        ValueError: On every process, if a Dirichlet condition constrains a master that the
            constraints kept.
    """
    spaces = _block_spaces(a, constraints)
    _raise_if_constrained_masters(constraints, spaces, bc_data)
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
    # every form has been assembled, including for a block that has no form of its own, whose
    # pattern reserves its whole diagonal
    for i, mpc in enumerate(constraints):
        A_ii = blocks[i][i]
        if A_ii is None:
            continue
        rows = bc_data.rows(spaces[i])
        _add_diagonals([(A_ii, mpc)], [(A_ii, rows)] if rows.size > 0 else [], diagval)


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
    a: Sequence[Sequence[Optional[_fem.Form]]],
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
        ValueError: On every process, if a Dirichlet condition constrains a master that the
            constraints kept.
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

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

import contextlib
from typing import Iterable, Optional, Sequence, Union

from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import numpy
import ufl
from dolfinx import default_scalar_type
from dolfinx.common import Timer
from dolfinx.la.petsc import create_vector

import dolfinx_mpc.cpp

from .multipointconstraint import MultiPointConstraint, _float_classes


def apply_lifting(
    b: _PETSc.Vec,  # type: ignore
    form: Union[Iterable[Sequence[_fem.Form]], Iterable[_fem.Form]],  # type: ignore
    bcs: Union[Sequence[_fem.DirichletBC], Sequence[Sequence[_fem.DirichletBC]]],
    constraint: Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    x0: Optional[Sequence[_PETSc.Vec]] = None,  # type: ignore
    scale: _float_classes = default_scalar_type(1.0),
):  # type: ignore
    """
    Apply lifting to vector b, i.e.
    :math:`b = b - scale \\cdot K^T (A_j (g_j - x0_j))`

    Args:
        b: PETSc vector to assemble into
        form: The linear form
        bcs: List of Dirichlet boundary conditions
        constraint: The multi point constraint
        x0: List of vectors
        scale: Scaling for lifting
    """
    t = Timer("~MPC: Apply lifting (C++)")
    if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
        scale = scale.item()  # type: ignore

    if b.getType() == "nest":
        try:
            bcs = _fem.bcs_by_block(_fem.extract_function_spaces(form, 1), bcs)  # type: ignore
        except AttributeError:
            pass
        x0 = [] if x0 is None else x0.getNestSubVecs()  # type: ignore
        assert isinstance(form, Sequence) and isinstance(constraint, Sequence)
        for b_sub, a_sub, mpc_i in zip(b.getNestSubVecs(), form, constraint):
            _a = [None if form is None else form._cpp_object for form in a_sub]  # type:ignore
            _bcs = [[bc._cpp_object for bc in bcs0] for bcs0 in bcs]  # type: ignore
            dolfinx_mpc.cpp.mpc.apply_lifting(b_sub.array_w, _a, _bcs, x0, scale, mpc_i._cpp_object)
    else:
        with contextlib.ExitStack() as stack:
            if x0 is None:
                x0 = []
            else:
                x0 = [stack.enter_context(x.localForm()) for x in x0]
            x0_r = [x.array_r for x in x0]
            b_local = stack.enter_context(b.localForm())
            _forms = [f._cpp_object for f in form]  # type: ignore
            _bcs = [[bc._cpp_object for bc in bcs0] for bcs0 in bcs]  # type: ignore
            assert isinstance(constraint, MultiPointConstraint)
            dolfinx_mpc.cpp.mpc.apply_lifting(b_local.array_w, _forms, _bcs, x0_r, scale, constraint._cpp_object)
    t.stop()


def assemble_vector(
    form: ufl.form.Form,
    constraint: MultiPointConstraint,
    b: Optional[_PETSc.Vec] = None,  # type: ignore
) -> _PETSc.Vec:  # type: ignore
    """
    Assemble a linear form into vector `b` with corresponding multi point constraint

    Args:
        form: The linear form
        constraint: The multi point constraint
        b: PETSc vector to assemble

    Returns:
        The vector with the assembled linear form (`b` if supplied)
    """

    if b is None:
        b = create_vector([(constraint.function_space.dofmap.index_map, constraint.function_space.dofmap.index_map_bs)])
    t = Timer("~MPC: Assemble vector (C++)")
    with b.localForm() as b_local:
        b_local.set(0.0)
        dolfinx_mpc.cpp.mpc.assemble_vector(b_local.array_w, form._cpp_object, constraint._cpp_object)
    t.stop()
    return b


def create_vector_nest(L: Sequence[_fem.Form], constraints: Sequence[MultiPointConstraint]) -> _PETSc.Vec:  # type: ignore
    """
    Create a PETSc vector of type "nest" appropriate for the provided multi
    point constraints

    Args:
        L: A sequence of linear forms
        constraints: An ordered list of multi point constraints

    Returns:
        PETSc.Vec: A PETSc vector of type "nest"  #type: ignore
    """
    assert len(constraints) == len(L)

    maps = [
        (constraint.function_space.dofmap.index_map, constraint.function_space.dofmap.index_map_bs)
        for constraint in constraints
    ]
    return _cpp.fem.petsc.create_vector_nest(maps)


def assemble_vector_nest(
    b: _PETSc.Vec,  # type: ignore
    L: Sequence[_fem.Form],
    constraints: Sequence[MultiPointConstraint],
):
    """
    Assemble a linear form into a PETSc vector of type "nest"

    Args:
        b: A PETSc vector of type "nest"
        L: A sequence of linear forms
        constraints: An ordered list of multi point constraints
    """
    assert len(constraints) == len(L)
    assert b.getType() == "nest"

    b_sub_vecs = b.getNestSubVecs()
    for i, L_row in enumerate(L):
        assemble_vector(L_row, constraints[i], b=b_sub_vecs[i])

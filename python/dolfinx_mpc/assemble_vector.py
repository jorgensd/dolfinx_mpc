# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.la as _la
import ufl
from dolfinx.common import Timer
from petsc4py import PETSc as _PETSc

import dolfinx_mpc.cpp

from .multipointconstraint import MultiPointConstraint


def apply_lifting(b: _PETSc.Vec, form: List[_fem.Form], bcs: List[List[_fem.DirichletBC]],  # type: ignore
                  constraint: MultiPointConstraint, x0: List[_PETSc.Vec] = [], scale: float = 1.0):  # type: ignore
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
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        _forms = [f._cpp_object for f in form]
        _bcs = [[bc._cpp_object for bc in bcs0] for bcs0 in bcs]
        dolfinx_mpc.cpp.mpc.apply_lifting(b_local.array_w, _forms,
                                          _bcs, x0_r, scale, constraint._cpp_object)
    t.stop()


def assemble_vector(form: ufl.form.Form, constraint: MultiPointConstraint,
                    b: Optional[_PETSc.Vec] = None) -> _PETSc.Vec:  # type: ignore
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
        b = _la.create_petsc_vector(constraint.function_space.dofmap.index_map,
                                    constraint.function_space.dofmap.index_map_bs)
    t = Timer("~MPC: Assemble vector (C++)")
    with b.localForm() as b_local:
        b_local.set(0.0)
        dolfinx_mpc.cpp.mpc.assemble_vector(b_local, form._cpp_object,
                                            constraint._cpp_object)
    t.stop()
    return b


def create_vector_nest(
        L: Sequence[_fem.Form],
        constraints: Sequence[MultiPointConstraint]) -> _PETSc.Vec:  # type: ignore
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

    maps = [(constraint.function_space.dofmap.index_map,
             constraint.function_space.dofmap.index_map_bs)
            for constraint in constraints]
    return _cpp.fem.petsc.create_vector_nest(maps)


def assemble_vector_nest(
        b: _PETSc.Vec,  # type: ignore
        L: Sequence[_fem.Form],
        constraints: Sequence[MultiPointConstraint]):
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

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from typing import List, Optional, Sequence
import dolfinx.fem as _fem
import dolfinx.cpp as _cpp
import ufl
import numpy
import dolfinx_mpc.cpp
from .multipointconstraint import MultiPointConstraint
from petsc4py import PETSc as _PETSc
import contextlib
from dolfinx.common import Timer


def apply_lifting(b: _PETSc.Vec, form: List[_fem.FormMetaClass], bcs: List[List[_fem.DirichletBCMetaClass]],
                  constraint: MultiPointConstraint, x0: Optional[List[_PETSc.Vec]] = [], scale: numpy.float64 = 1.0):
    """
    Apply lifting to vector b, i.e.

    .. math::
        b <- b - scale * K^T (A_j (g_j - x0_j))

    Parameters
    ----------
    b
        PETSc vector to assemble into
    form
        The linear form
    bcs
        List of Dirichlet boundary conditions
    constraint
        The multi point constraint
    x0
        List of vectors
    scale
        Scaling for lifting

    Returns
    -------
    PETSc.Vec
        The assembled linear form
    """
    t = Timer("~MPC: Apply lifting (C++)")
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        dolfinx_mpc.cpp.mpc.apply_lifting(b_local.array_w, form,
                                          bcs, x0_r, scale, constraint._cpp_object)
    t.stop()


def assemble_vector(form: ufl.form.Form, constraint: MultiPointConstraint,
                    b: _PETSc.Vec = None) -> _PETSc.Vec:
    """
    Assemble a linear form into vector b with corresponding multi point constraint

    Parameters
    ----------
    form
        The linear form
    constraint
        The multi point constraint
    b
        PETSc vector to assemble into (optional)

    Returns
    -------
    PETSc.Vec
        The assembled linear form
    """

    if b is None:
        b = _cpp.la.petsc.create_vector(constraint.function_space.dofmap.index_map,
                                        constraint.function_space.dofmap.index_map_bs)
    t = Timer("~MPC: Assemble vector (C++)")
    with b.localForm() as b_local:
        b_local.set(0.0)
        dolfinx_mpc.cpp.mpc.assemble_vector(b_local, form, constraint._cpp_object)
    t.stop()
    return b


def create_vector_nest(
        L: Sequence[_fem.FormMetaClass],
        constraints: Sequence[MultiPointConstraint]):
    assert len(constraints) == len(L)

    maps = [(constraint.function_space.dofmap.index_map,
             constraint.function_space.dofmap.index_map_bs)
             for constraint in constraints]
    return _cpp.fem.petsc.create_vector_nest(maps)


def assemble_vector_nest(
        b: _PETSc.Vec,
        L: Sequence[_fem.FormMetaClass],
        constraints: Sequence[MultiPointConstraint]):
    assert len(constraints) == len(L)

    b_sub_vecs = b.getNestSubVecs()
    for i, L_row in enumerate(L):
        assemble_vector(L_row, constraints[i], b=b_sub_vecs[i])
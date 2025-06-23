# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

import typing
from functools import partial
from typing import Iterable, Sequence

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem as _fem
from dolfinx.la.petsc import create_vector

from dolfinx_mpc.cpp import mpc as _cpp_mpc

from .assemble_matrix import assemble_matrix, assemble_matrix_nest, create_matrix_nest
from .assemble_vector import apply_lifting, assemble_vector, assemble_vector_nest, create_vector_nest
from .multipointconstraint import MultiPointConstraint


def assemble_jacobian_mpc(
    u: typing.Union[Sequence[_fem.Function], _fem.Function],
    jacobian: typing.Union[_fem.Form, typing.Iterable[typing.Iterable[_fem.Form]]],
    preconditioner: typing.Optional[typing.Union[_fem.Form, typing.Iterable[typing.Iterable[_fem.Form]]]],
    bcs: typing.Iterable[_fem.DirichletBC],
    mpc: typing.Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """Assemble the Jacobian matrix and preconditioner.

    A function conforming to the interface expected by SNES.setJacobian can
    be created by fixing the first four arguments:

        functools.partial(assemble_jacobian, u, jacobian, preconditioner,
                          bcs)

    Args:
        u: Function tied to the solution vector within the residual and
            jacobian
        jacobian: Form of the Jacobian
        preconditioner: Form of the preconditioner
        bcs: List of Dirichlet boundary conditions
        mpc: The multi point constraint or a sequence of multi point
        _snes: The solver instance
        x: The vector containing the point to evaluate at
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the preconditioner into
    """
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    try:
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    except PETSc.Error:  # type: ignore
        for x_sub in x.getNestSubVecs():
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    _fem.petsc.assign(x, u)

    # Assemble Jacobian
    J.zeroEntries()
    if J.getType() == "nest":
        assemble_matrix_nest(J, jacobian, mpc, bcs, diagval=1.0)  # type: ignore
    else:
        assemble_matrix(jacobian, mpc, bcs, diagval=1.0, A=J)  # type: ignore
    J.assemble()
    if preconditioner is not None:
        P.zeroEntries()
        if P.getType() == "nest":
            assemble_matrix_nest(P, preconditioner, mpc, bcs, diagval=1.0)  # type: ignore
        else:
            assemble_matrix(mpc, preconditioner, bcs, diagval=1.0, A=P)  # type: ignore

        P.assemble()


def assemble_residual_mpc(
    u: typing.Union[_fem.Function, Sequence[_fem.Function]],
    residual: typing.Union[_fem.Form, typing.Iterable[_fem.Form]],
    jacobian: typing.Union[_fem.Form, typing.Iterable[typing.Iterable[_fem.Form]]],
    bcs: typing.Iterable[_fem.DirichletBC],
    mpc: typing.Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
):
    """Assemble the residual into the vector `F`.

    A function conforming to the interface expected by SNES.setResidual can
    be created by fixing the first four arguments:

        functools.partial(assemble_residual, u, jacobian, preconditioner,
                          bcs)

    Args:
        u: Function(s) tied to the solution vector within the residual and
            Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        F: Vector to assemble the residual into.
    """
    # Update input vector before assigning
    _fem.petsc._ghostUpdate(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore

    # Assign the input vector to the unknowns
    _fem.petsc.assign(x, u)
    if isinstance(u, Sequence):
        assert isinstance(mpc, Sequence)
        for i in range(len(u)):
            mpc[i].homogenize(u[i])
            mpc[i].backsubstitution(u[i])
    else:
        assert isinstance(u, _fem.Function)
        assert isinstance(mpc, MultiPointConstraint)
        mpc.homogenize(u)
        mpc.backsubstitution(u)
    # Assemble the residual
    _fem.petsc._zero_vector(F)
    if x.getType() == "nest":
        assemble_vector_nest(F, residual, mpc)  # type: ignore
    else:
        assert isinstance(residual, _fem.Form)
        assert isinstance(mpc, MultiPointConstraint)
        assemble_vector(residual, mpc, F)

    # Lift vector
    try:
        # Nest and blocked lifting
        bcs1 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(jacobian, 1), bcs)  # type: ignore
        _fem.petsc._assign_block_data(residual, x)  # type: ignore
        apply_lifting(F, jacobian, bcs=bcs1, constraint=mpc, x0=x, scale=-1.0)  # type: ignore
        _fem.petsc._ghostUpdate(F, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
        bcs0 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(residual), bcs)  # type: ignore
        _fem.petsc.set_bc(F, bcs0, x0=x, alpha=-1.0)
    except RuntimeError:
        # Single form lifting
        apply_lifting(F, [jacobian], bcs=[bcs], constraint=mpc, x0=[x], scale=-1.0)  # type: ignore
        _fem.petsc._ghostUpdate(F, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
        _fem.petsc.set_bc(F, bcs, x0=x, alpha=-1.0)
    _fem.petsc._ghostUpdate(F, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):
    def __init__(
        self,
        F: typing.Union[ufl.form.Form, Sequence[ufl.form.Form]],
        u: typing.Union[_fem.Function, Sequence[_fem.Function]],
        mpc: typing.Union[MultiPointConstraint, Sequence[MultiPointConstraint]],
        bcs: typing.Optional[Sequence[_fem.DirichletBC]] = None,
        J: typing.Optional[typing.Union[ufl.form.Form, Iterable[Iterable[ufl.form.Form]]]] = None,
        P: typing.Optional[typing.Union[ufl.form.Form, Iterable[Iterable[ufl.form.Form]]]] = None,
        kind: typing.Optional[typing.Union[str, typing.Iterable[typing.Iterable[str]]]] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        petsc_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[dict[dolfinx.mesh.Mesh, npt.NDArray[np.int32]]] = None,
    ):
        """Class for solving nonlinear problems with SNES.

        Solves problems of the form
        :math:`F_i(u, v) = 0, i=0,...N\\ \\forall v \\in V` where
        :math:`u=(u_0,...,u_N), v=(v_0,...,v_N)` using PETSc SNES as the
        non-linear solver.

        Note: The deprecated version of this class for use with
            NewtonSolver has been renamed NewtonSolverNonlinearProblem.

        Args:
            F: UFL form(s) of residual :math:`F_i`.
            u: Function used to define the residual and Jacobian.
            bcs: Dirichlet boundary conditions.
            J: UFL form(s) representing the Jacobian
                :math:`J_ij = dF_i/du_j`.
            P: UFL form(s) representing the preconditioner.
            kind: The PETSc matrix type(s) for the Jacobian and
                preconditioner (``MatType``).
                See :func:`dolfinx.fem.petsc.create_matrix` for more
                information.
            form_compiler_options: Options used in FFCx compilation of all
                forms. Run ``ffcx --help`` at the command line to see all
                available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See ``python/dolfinx/jit.py`` for all
                available options. Takes priority over all other option
                values.
            petsc_options: Options to pass to the PETSc SNES object.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, ``entity_maps`` must be
                supplied. For each key (a mesh, different to the
                integration domain mesh) a map should be provided relating
                the entities in the integration domain mesh to the entities
                in the key mesh e.g. for a key-value pair ``(msh, emap)``
                in ``entity_maps``, ``emap[i]`` is the entity in ``msh``
                corresponding to entity ``i`` in the integration domain
                mesh.
        """
        # Compile residual and Jacobian forms
        self._F = _fem.form(
            F,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if J is None:
            dus = [ufl.TrialFunction(Fi.arguments()[0].ufl_function_space()) for Fi in F]
            J = _fem.forms.derivative_block(F, u, dus)

        self._J = _fem.form(
            J,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if P is not None:
            self._P = _fem.form(
                P,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        else:
            self._P = None

        self._u = u
        # Set default values if not supplied
        bcs = [] if bcs is None else bcs
        self.mpc = mpc
        # Create PETSc structures for the residual, Jacobian and solution
        # vector
        if kind == "nest" or isinstance(kind, Sequence):
            assert isinstance(mpc, Sequence)
            assert isinstance(self._J, Sequence)
            self._A = create_matrix_nest(self._J, mpc)
        elif kind is None:
            assert isinstance(mpc, MultiPointConstraint)
            self._A = _cpp_mpc.create_matrix(self._J._cpp_object, mpc._cpp_object)
        else:
            raise ValueError("Unsupported kind for matrix: {}".format(kind))

        kind = "nest" if self._A.getType() == "nest" else kind
        if kind == "nest":
            assert isinstance(mpc, Sequence)
            assert isinstance(self._F, Sequence)
            self._b = create_vector_nest(self._F, mpc)
            self._x = create_vector_nest(self._F, mpc)
        else:
            assert isinstance(mpc, MultiPointConstraint)
            self._b = create_vector(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)
            self._x = create_vector(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)

        # Create PETSc structure for preconditioner if provided
        if self._P is not None:
            if kind == "nest":
                assert isinstance(self.P, Sequence)
                assert isinstance(self.mpc, Sequence)
                self._P_mat = create_matrix_nest(self.P, self.mpc)
            else:
                assert isinstance(self._P, _fem.Form)
                self._P_mat = _cpp_mpc.create_matrix(self.P, kind=kind)
        else:
            self._P_mat = None

        # Create the SNES solver and attach the corresponding Jacobian and
        # residual computation functions
        self._snes = PETSc.SNES().create(comm=self.A.comm)  # type: ignore
        self.solver.setJacobian(partial(assemble_jacobian_mpc, u, self.J, self.P, bcs, mpc), self._A, self.P_mat)
        self.solver.setFunction(partial(assemble_residual_mpc, u, self.F, self.J, bcs, mpc), self.b)

        # Set PETSc options
        problem_prefix = f"dolfinx_nonlinearproblem_{id(self)}"
        self.solver.setOptionsPrefix(problem_prefix)
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(problem_prefix)

        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
            self.solver.setFromOptions()

            for k in petsc_options.keys():
                del opts[k]

        opts.prefixPop()

    def solve(self) -> tuple[PETSc.Vec, int, int]:  # type: ignore
        """Solve the problem and update the solution in the problem
        instance.

        Returns:
            The solution, convergence reason and number of iterations.
        """

        # Move current iterate into the work array.
        _fem.petsc.assign(self._u, self.x)

        # Solve problem
        self.solver.solve(None, self.x)

        # Move solution back to function
        dolfinx.fem.petsc.assign(self.x, self._u)
        if isinstance(self.mpc, Sequence):
            for i in range(len(self._u)):
                self.mpc[i].backsubstitution(self._u[i])
                self.mpc[i].backsubstitution(self._u[i])
        else:
            assert isinstance(self._u, _fem.Function)
            self.mpc.homogenize(self._u)
            self.mpc.backsubstitution(self._u)

        converged_reason = self.solver.getConvergedReason()
        return self._u, converged_reason, self.solver.getIterationNumber()


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    """
    Class for solving a linear variational problem with multi point constraints of the form
    a(u, v) = L(v) for all v using PETSc as a linear algebra backend.

    Args:
        a: A bilinear UFL form, the left hand side of the variational problem.
        L: A linear UFL form, the right hand side of the variational problem.
        mpc: The multi point constraint.
        bcs: A list of Dirichlet boundary conditions.
        u: The solution function. It will be created if not provided. The function has
            to be based on the functionspace in the mpc, i.e.

            .. highlight:: python
            .. code-block:: python

                u = dolfinx.fem.Function(mpc.function_space)
        petsc_options: Parameters that is passed to the linear algebra backend PETSc.  #type: ignore
            For available choices for the 'petsc_options' kwarg, see the PETSc-documentation
            https://www.mcs.anl.gov/petsc/documentation/index.html.
        form_compiler_options: Parameters used in FFCx compilation of this form. Run `ffcx --help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by DOLFINx.
        jit_options: Parameters used in CFFI JIT compilation of C code generated by FFCx.
            See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py#L22-L37
            for all available parameters. Takes priority over all other parameter values.
        P: A preconditioner UFL form.
    Examples:
        Example usage:

        .. highlight:: python
        .. code-block:: python

           problem = LinearProblem(a, L, mpc, [bc0, bc1],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    """

    u: typing.Union[_fem.Function, list[_fem.Function]]
    _a: typing.Union[_fem.Form, typing.Sequence[typing.Sequence[_fem.Form]]]
    _L: typing.Union[_fem.Form, typing.Sequence[_fem.Form]]
    _preconditioner: typing.Optional[typing.Union[_fem.Form, typing.Sequence[typing.Sequence[_fem.Form]]]]
    _mpc: typing.Union[MultiPointConstraint, typing.Sequence[MultiPointConstraint]]
    _A: PETSc.Mat  # type: ignore
    _P: typing.Optional[PETSc.Mat]  # type: ignore
    _b: PETSc.Vec  # type: ignore
    _solver: PETSc.KSP  # type: ignore
    _x: PETSc.Vec  # type: ignore
    bcs: typing.List[_fem.DirichletBC]
    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        a: typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]],
        L: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
        mpc: typing.Union[MultiPointConstraint, typing.Sequence[MultiPointConstraint]],
        bcs: typing.Optional[typing.List[_fem.DirichletBC]] = None,
        u: typing.Optional[typing.Union[_fem.Function, typing.Sequence[_fem.Function]]] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        P: typing.Optional[typing.Union[ufl.Form, typing.Iterable[ufl.Form]]] = None,
    ):
        # Compile forms
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options
        self._a = _fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._L = _fem.form(L, jit_options=jit_options, form_compiler_options=form_compiler_options)

        self._mpc = mpc
        # Nest assembly
        if isinstance(mpc, Sequence):
            is_nest = True
            # Sanity check
            for mpc_i in mpc:
                if not mpc_i.finalized:
                    raise RuntimeError("The multi point constraint has to be finalized before calling initializer")
                    # Create function containing solution vector
        else:
            is_nest = False
            if not mpc.finalized:
                raise RuntimeError("The multi point constraint has to be finalized before calling initializer")

        # Create function(s) containing solution vector(s)
        if is_nest:
            if u is None:
                assert isinstance(self._mpc, Sequence)
                self.u = [_fem.Function(self._mpc[i].function_space) for i in range(len(self._mpc))]
            else:
                assert isinstance(self._mpc, Sequence)
                assert isinstance(u, Sequence)
                for i, (mpc_i, u_i) in enumerate(zip(self._mpc, u)):
                    assert isinstance(u_i, _fem.Function)
                    assert isinstance(mpc_i, MultiPointConstraint)
                    if u_i.function_space is not mpc_i.function_space:
                        raise ValueError(
                            "The input function has to be in the function space in the multi-point constraint",
                            "i.e. u = dolfinx.fem.Function(mpc.function_space)",
                        )
                self.u = list(u)
        else:
            if u is None:
                assert isinstance(self._mpc, MultiPointConstraint)
                self.u = _fem.Function(self._mpc.function_space)
            else:
                assert isinstance(u, _fem.Function)
                assert isinstance(self._mpc, MultiPointConstraint)
                if u.function_space is self._mpc.function_space:
                    self.u = u
                else:
                    raise ValueError(
                        "The input function has to be in the function space in the multi-point constraint",
                        "i.e. u = dolfinx.fem.Function(mpc.function_space)",
                    )

        # Create MPC matrix and vector
        self._preconditioner = _fem.form(P, jit_options=jit_options, form_compiler_options=form_compiler_options)

        if is_nest:
            assert isinstance(mpc, Sequence)
            assert isinstance(self._L, Sequence)
            assert isinstance(self._a, Sequence)
            self._A = create_matrix_nest(self._a, mpc)
            self._b = create_vector_nest(self._L, mpc)
            self._x = create_vector_nest(self._L, mpc)
            if self._preconditioner is None:
                self._P = None
            else:
                assert isinstance(self._preconditioner, Sequence)
                self._P = create_matrix_nest(self._preconditioner, mpc)
        else:
            assert isinstance(mpc, MultiPointConstraint)
            assert isinstance(self._L, _fem.Form)
            assert isinstance(self._a, _fem.Form)
            self._A = _cpp_mpc.create_matrix(self._a._cpp_object, mpc._cpp_object)
            self._b = create_vector(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)
            self._x = create_vector(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)
            if self._preconditioner is None:
                self._P = None
            else:
                assert isinstance(self._preconditioner, _fem.Form)
                self._P = _cpp_mpc.create_matrix(self._preconditioner._cpp_object, mpc._cpp_object)

        self.bcs = [] if bcs is None else bcs

        if is_nest:
            assert isinstance(self.u, Sequence)
            comm = self.u[0].function_space.mesh.comm
        else:
            assert isinstance(self.u, _fem.Function)
            comm = self.u.function_space.mesh.comm

        self._solver = PETSc.KSP().create(comm)  # type: ignore
        self._solver.setOperators(self._A, self._P)

        # Give PETSc solver options a unique prefix
        solver_prefix = "dolfinx_mpc_solve_{}".format(id(self))
        self._solver.setOptionsPrefix(solver_prefix)

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(solver_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

    def solve(self) -> typing.Union[list[_fem.Function], _fem.Function]:
        """Solve the problem.

        Returns:
            Function containing the solution"""

        # Assemble lhs
        self._A.zeroEntries()
        if self._A.getType() == "nest":
            assemble_matrix_nest(self._A, self._a, self._mpc, self.bcs, diagval=1.0)  # type: ignore
        else:
            assert isinstance(self._a, _fem.Form)
            assemble_matrix(self._a, self._mpc, bcs=self.bcs, A=self._A)

        self._A.assemble()
        assert self._A.assembled

        # Assemble the preconditioner if provided
        if self._P is not None:
            self._P.zeroEntries()
            if self._P.getType() == "nest":
                assert isinstance(self._preconditioner, typing.Sequence)
                assemble_matrix_nest(self._P, self._preconditioner, self._mpc, self.bcs)  # type: ignore
            else:
                assert isinstance(self._preconditioner, _fem.Form)
                assemble_matrix(self._preconditioner, self._mpc, bcs=self.bcs, A=self._P)
            self._P.assemble()

        # Assemble the residual
        _fem.petsc._zero_vector(self._b)
        if self._x.getType() == "nest":
            assemble_vector_nest(self._b, self._L, self._mpc)  # type: ignore
        else:
            assert isinstance(self._L, _fem.Form)
            assert isinstance(self._mpc, MultiPointConstraint)
            assemble_vector(self._L, self._mpc, self._b)

        # Lift vector
        try:
            # Nest and blocked lifting
            bcs1 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(self._a, 1), self.bcs)  # type: ignore
            apply_lifting(self._b, self._a, bcs=bcs1, constraint=self._mpc)  # type: ignore
            _fem.petsc._ghostUpdate(self._b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
            bcs0 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(self._L), self.bcs)  # type: ignore
            _fem.petsc.set_bc(self._b, bcs0)
        except RuntimeError:
            # Single form lifting
            apply_lifting(self._b, [self._a], bcs=[self.bcs], constraint=self._mpc)  # type: ignore
            _fem.petsc._ghostUpdate(self._b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
            _fem.petsc.set_bc(self._b, self.bcs)
        _fem.petsc._ghostUpdate(self._b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        _fem.petsc._ghostUpdate(self._x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
        _fem.petsc.assign(self._x, self.u)

        if isinstance(self.u, Sequence):
            assert isinstance(self._mpc, Sequence)
            for i in range(len(self.u)):
                self._mpc[i].homogenize(self.u[i])
                self._mpc[i].backsubstitution(self.u[i])
        else:
            assert isinstance(self.u, _fem.Function)
            assert isinstance(self._mpc, MultiPointConstraint)
            self._mpc.homogenize(self.u)
            self._mpc.backsubstitution(self.u)

        return self.u

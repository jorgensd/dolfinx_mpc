# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from typing import cast

from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl
from dolfinx import fem as _fem
from dolfinx.la.petsc import _ghost_update, _zero_vector

from .assemble_matrix import assemble_matrix, create_matrix
from .assemble_vector import (
    apply_lifting,
    apply_mpc_lifting,
    assemble_vector,
    create_vector,
)
from .dirichletbc import BCData
from .multipointconstraint import MultiPointConstraint


def assemble_jacobian_mpc(
    u: Sequence[_fem.Function] | _fem.Function,
    jacobian: _fem.Form | Sequence[Sequence[_fem.Form | None]],
    preconditioner: _fem.Form | Sequence[Sequence[_fem.Form | None]] | None,
    bcs: Iterable[_fem.DirichletBC],
    mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
    bc_data: BCData,
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
    _blocks: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
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
        bc_data: Dof marker and diagonal row cache, built once by the caller so
            that every Newton iteration reuses it. The Jacobian and the
            preconditioner share it: entries are keyed by function space, and
            the two forms are over the same spaces.
        _snes: The solver instance
        x: The vector containing the point to evaluate at
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the preconditioner into
        _blocks: For a monolithic system of several blocks, the offsets of its blocks, see
            :func:`dolfinx_mpc.create_vector`. SNES may pass a vector other than the one the
            problem created, so they are set on `x` here.
    """
    if _blocks is not None:
        x.setAttr("_blocks", _blocks)
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    _ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
    # Assign the input vector to the unknowns
    _fem.petsc.assign(x, u)  # type: ignore
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

    # Assemble Jacobian
    J.zeroEntries()
    assemble_matrix(jacobian, mpc, bcs, diagval=1.0, A=J, bc_data=bc_data)  # type: ignore
    J.assemble()
    if preconditioner is not None:
        P.zeroEntries()
        assemble_matrix(preconditioner, mpc, bcs, diagval=1.0, A=P, bc_data=bc_data)  # type: ignore

        P.assemble()


def assemble_residual_mpc(
    u: _fem.Function | Sequence[_fem.Function],
    residual: _fem.Form | Sequence[_fem.Form],
    jacobian: _fem.Form | Sequence[Sequence[_fem.Form]],
    bcs: Sequence[_fem.DirichletBC],
    mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
    bc_data: BCData,
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
    _blocks: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
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
        mpc: The multi point constraint or a sequence of multi point
            constraints.
        bc_data: Dof marker cache shared with the Jacobian, built once by the
            caller so that every Newton iteration reuses it.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        F: Vector to assemble the residual into.
        _blocks: For a monolithic system of several blocks, the offsets of its blocks, see
            :func:`dolfinx_mpc.create_vector`. A line search evaluates the residual in vectors
            duplicated from the one given to SNES, so they are set on `x` and `F` here.
    """
    if _blocks is not None:
        x.setAttr("_blocks", _blocks)
        F.setAttr("_blocks", _blocks)
    # Update input vector before assigning
    _ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
    # Assign the input vector to the unknowns
    _fem.petsc.assign(x, u)  # type: ignore
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
    _zero_vector(F)
    assemble_vector(residual, mpc, F)  # type: ignore

    # Lift vector. Decide between blocked (nest or monolithic) and single form lifting up front, so
    # that a failure in one of the lifting calls cannot fall through to the other branch
    if isinstance(jacobian, Sequence):
        bcs1 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(jacobian, 1), bcs)  # type: ignore
        apply_lifting(F, jacobian, bcs=bcs1, constraint=mpc, x0=x, scale=-1.0, bc_data=bc_data)  # type: ignore
        _ghost_update(F, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
        bcs0 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(residual), bcs)  # type: ignore
        _fem.petsc.set_bc(F, bcs0, x0=x, alpha=-1.0)
    else:
        apply_lifting(F, [jacobian], bcs=[bcs], constraint=mpc, x0=[x], scale=-1.0, bc_data=bc_data)  # type: ignore
        _ghost_update(F, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
        _fem.petsc.set_bc(F, bcs, x0=x, alpha=-1.0)
    _ghost_update(F, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):
    def __init__(
        self,
        F: ufl.form.Form | Sequence[ufl.form.Form],
        u: _fem.Function | Sequence[_fem.Function],
        mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
        bcs: Sequence[_fem.DirichletBC] | None = None,
        J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        petsc_options_prefix: str = "dolfinx_mpc_nonlinear_problem_",
        petsc_options: dict | None = None,
        entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
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
            kind: The kind of Jacobian and preconditioner matrix, as in
                :func:`dolfinx_mpc.create_matrix`. For a single constraint, a PETSc matrix type
                (``MatType``). For a sequence of constraints, one per block, ``"nest"`` or a nested
                sequence of matrix types gives a ``nest`` system, and ``None``, ``"mpi"`` or a
                matrix type a single monolithic matrix and vector with the blocks one after another.
            form_compiler_options: Options used in FFCx compilation of all
                forms. Run ``ffcx --help`` at the command line to see all
                available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See ``python/dolfinx/jit.py`` for all
                available options. Takes priority over all other option
                values.
            petsc_options_prefix: Options prefix used as the root prefix on
                all internally created PETSc objects (SNES, A, b, x and the
                preconditioner matrix). Typically ends with ``_``. Must be the
                same on all ranks and is usually unique within the
                programme.
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
            if isinstance(F, ufl.form.Form):
                du = ufl.TrialFunction(F.arguments()[0].ufl_function_space())
                J = ufl.derivative(F, u, du)
            else:
                dus = [ufl.TrialFunction(Fi.arguments()[0].ufl_function_space()) for Fi in F]
                J = _fem.forms.derivative_block(F, u, dus)

        self._J = _fem.form(
            J,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if P is not None:
            self._preconditioner = _fem.form(
                P,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        else:
            self._preconditioner = None

        self._u = u
        # Set default values if not supplied
        bcs = [] if bcs is None else bcs
        self.mpc = mpc
        # Create PETSc structures for the residual, Jacobian and solution vector
        if not (kind is None or isinstance(kind, (str, Sequence))):
            raise ValueError(f"Unsupported kind for matrix: {kind!r}")
        if not isinstance(mpc, Sequence) and (kind == "nest" or (kind is not None and not isinstance(kind, str))):
            raise ValueError(f"kind={kind!r} needs a sequence of constraints, one per block")
        self._A = create_matrix(self._J, mpc, kind)

        # The vectors are nested if the matrix is, and monolithic or single otherwise
        vector_kind = "nest" if self._A.getType() == "nest" else None
        self._b = create_vector(self._F, mpc, vector_kind)
        self._x = create_vector(self._F, mpc, vector_kind)

        # Create PETSc structure for preconditioner if provided
        prec = self.preconditioner
        if prec is not None:  # type: ignore
            self._P_mat = create_matrix(prec, mpc, kind)
        else:
            self._P_mat = None  # type: ignore

        # Create the SNES solver and attach the corresponding Jacobian and
        # residual computation functions
        self._snes = PETSc.SNES().create(comm=self.A.comm)  # type: ignore

        # Markers and diagonal rows depend only on the function spaces and the
        # conditions, so one cache covers the Jacobian and the preconditioner
        # and every Newton iteration reuses it.
        bc_data = BCData(bcs)

        # The block layout of a monolithic system is an attribute of the vectors, which SNES may
        # replace by duplicates in the callbacks, so it is handed to them
        blocks = cast("tuple[tuple[int, ...], tuple[int, ...]] | None", self._b.getAttr("_blocks"))
        self.solver.setJacobian(
            partial(assemble_jacobian_mpc, u, self.J, prec, bcs, mpc, bc_data, _blocks=blocks),
            self._A,
            self.P_mat,
        )
        self.solver.setFunction(
            partial(assemble_residual_mpc, u, self.F, self.J, bcs, mpc, bc_data, _blocks=blocks), self.b
        )

        # Set PETSc options
        self._petsc_options_prefix = petsc_options_prefix
        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")
        if self.P_mat is not None:
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")

        # Set options on SNES only
        if petsc_options is not None:
            opts = PETSc.Options()  # type: ignore
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts.setValue(k, v)

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                opts.delValue(k)
            opts.prefixPop()

    def solve(self) -> tuple[_fem.Function | Sequence[_fem.Function], PETSc.ConvergedReason, int]:  # type: ignore
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
        dolfinx.fem.petsc.assign(self.x, self._u)  # type: ignore
        if isinstance(self.mpc, Sequence):
            for i in range(len(self._u)):
                self.mpc[i].homogenize(self._u[i])
                self.mpc[i].backsubstitution(self._u[i])
        else:
            assert isinstance(self._u, _fem.Function)
            self.mpc.homogenize(self._u)
            self.mpc.backsubstitution(self._u)

        converged_reason = self.solver.getConvergedReason()
        return self._u, converged_reason, self.solver.getIterationNumber()  # type: ignore


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
        kind: The kind of matrix, as in :func:`dolfinx.fem.petsc.LinearProblem`. For a single
            constraint, a PETSc matrix type, with ``None`` the default. When ``mpc`` is a sequence,
            one constraint per block, ``"nest"`` or a nested sequence of matrix types assembles a PETSc
            ``nest`` matrix and vector, whose blocks have those types, and any other kind, such as
            ``"mpi"``, a single monolithic matrix and vector with the blocks one after another.
            ``None``, the default, is ``"nest"``, which is what a problem with one constraint per
            block has always used.
    Examples:
        Example usage:

        .. highlight:: python
        .. code-block:: python

           problem = LinearProblem(a, L, mpc, [bc0, bc1],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    """

    _u: _fem.Function | list[_fem.Function]
    _a: _fem.Form | Sequence[Sequence[_fem.Form]]
    _L: _fem.Form | Sequence[_fem.Form]
    _jacobian: _fem.Form | Sequence[Sequence[_fem.Form | None]]
    _preconditioner: _fem.Form | Sequence[Sequence[_fem.Form | None]] | None  # type: ignore
    _mpc: MultiPointConstraint | Sequence[MultiPointConstraint]
    _bc_data: BCData
    _A: PETSc.Mat
    _P: PETSc.Mat | None
    _b: PETSc.Vec
    _solver: PETSc.KSP
    _x: PETSc.Vec
    bcs: list[_fem.DirichletBC]

    def __init__(
        self,
        a: ufl.Form | Sequence[Sequence[ufl.Form]],
        L: ufl.Form | Sequence[ufl.Form],
        mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
        bcs: list[_fem.DirichletBC] | None = None,
        u: _fem.Function | Sequence[_fem.Function] | None = None,
        petsc_options_prefix: str = "dolfinx_mpc_linear_problem_",
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        P: ufl.Form | Sequence[Sequence[ufl.Form]] | None = None,
        entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: str | Sequence[Sequence[str | None]] | None = None,
    ):
        # One constraint per block gives a nest or a monolithic system, one constraint a single matrix
        if not (kind is None or isinstance(kind, (str, Sequence))):
            raise ValueError(
                f"Unsupported kind {kind!r}, expected None, a PETSc matrix type or a nested sequence of them"
            )
        if isinstance(mpc, Sequence):
            # Without a kind a problem with one constraint per block keeps the nest layout it always had
            kind = "nest" if kind is None else kind
        elif kind == "nest" or (kind is not None and not isinstance(kind, str)):
            raise ValueError(f"kind={kind!r} needs a sequence of constraints, one per block")
        # Compile forms
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options
        self._a = _fem.form(
            a,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        self._L = _fem.form(
            L,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )

        self._mpc = mpc
        # Blocked problems
        if isinstance(mpc, Sequence):
            is_blocked = True
            # Sanity check
            for mpc_i in mpc:
                if not mpc_i.finalized:
                    raise RuntimeError("The multi point constraint has to be finalized before calling initializer")
                    # Create function containing solution vector
        else:
            is_blocked = False
            if not mpc.finalized:
                raise RuntimeError("The multi point constraint has to be finalized before calling initializer")

        # Create function(s) containing solution vector(s)
        if is_blocked:
            if u is None:
                assert isinstance(self._mpc, Sequence)
                self._u = [_fem.Function(self._mpc[i].function_space) for i in range(len(self._mpc))]
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
                self._u = list(u)
        else:
            if u is None:
                assert isinstance(self._mpc, MultiPointConstraint)
                self._u = _fem.Function(self._mpc.function_space)
            else:
                assert isinstance(u, _fem.Function)
                assert isinstance(self._mpc, MultiPointConstraint)
                if u.function_space is self._mpc.function_space:
                    self._u = u
                else:
                    raise ValueError(
                        "The input function has to be in the function space in the multi-point constraint",
                        "i.e. u = dolfinx.fem.Function(mpc.function_space)",
                    )

        # Markers and diagonal rows depend only on the function spaces and the
        # conditions, both fixed for this object, so one cache covers the
        # operator and the preconditioner and every solve reuses it.
        self._bc_data = BCData(bcs)

        # Create MPC matrix and vector
        self._preconditioner = _fem.form(  # type: ignore
            P,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )

        if is_blocked:
            assert isinstance(mpc, Sequence)
            assert isinstance(self._L, Sequence)
            assert isinstance(self._a, Sequence)
            self._A = create_matrix(self._a, mpc, kind)
            self._b = create_vector(self._L, mpc, kind)
            self._x = create_vector(self._L, mpc, kind)
            if self._preconditioner is None:
                self._P_mat = None
            else:
                assert isinstance(self._preconditioner, Sequence)
                self._P_mat = create_matrix(self._preconditioner, mpc, kind)
        else:
            assert isinstance(mpc, MultiPointConstraint)
            assert isinstance(self._L, _fem.Form)
            assert isinstance(self._a, _fem.Form)
            self._A = create_matrix(self._a, mpc, kind)
            self._b = create_vector(self._L, mpc)
            self._x = create_vector(self._L, mpc)
            if self._preconditioner is None:
                self._P_mat = None
            else:
                assert isinstance(self._preconditioner, _fem.Form)
                self._P_mat = create_matrix(self._preconditioner, mpc, kind)

        self.bcs = [] if bcs is None else bcs

        if is_blocked:
            assert isinstance(self.u, Sequence)
            comm = self.u[0].function_space.mesh.comm
        else:
            assert isinstance(self.u, _fem.Function)
            comm = self.u.function_space.mesh.comm

        self._solver = PETSc.KSP().create(comm)
        self._solver.setOperators(self._A, self._P_mat)

        self._petsc_options_prefix = petsc_options_prefix
        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")
        if self.P_mat is not None:
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")

        # Set options on KSP only
        if petsc_options is not None:
            opts = PETSc.Options()
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts.setValue(k, v)

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                opts.delValue(k)
            opts.prefixPop()

    def solve(self) -> _fem.Function | Sequence[_fem.Function]:
        """Solve the problem.

        Returns:
            Function containing the solution"""

        # Refresh the constraint offsets, so that a change in the values of the
        # Dirichlet conditions held by the constraint is picked up
        if isinstance(self._mpc, Sequence):
            for mpc_i in self._mpc:
                mpc_i.update_constants()
        else:
            self._mpc.update_constants()

        # Assemble lhs. The layout, single, nest or monolithic, is that of the matrix
        self._A.zeroEntries()
        assemble_matrix(self._a, self._mpc, bcs=self.bcs, diagval=1.0, A=self._A, bc_data=self._bc_data)  # type: ignore

        self._A.assemble()
        assert self._A.assembled

        # Assemble the preconditioner if provided
        if self._P_mat is not None:
            self._P_mat.zeroEntries()
            assemble_matrix(self._preconditioner, self._mpc, bcs=self.bcs, A=self._P_mat, bc_data=self._bc_data)  # type: ignore
            self._P_mat.assemble()

        # Assemble the residual
        _zero_vector(self._b)
        assemble_vector(self._L, self._mpc, self._b)  # type: ignore

        # Lift vector
        # Decide between nest/blocked and single form lifting up front, so that a
        # failure in one of the lifting calls cannot fall through to the other
        # branch and apply the lifting twice
        try:
            bcs1 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(self._a, 1), self.bcs)  # type: ignore
            bcs0 = _fem.bcs.bcs_by_block(_fem.forms.extract_function_spaces(self._L), self.bcs)  # type: ignore
            blocked = True
        except ValueError:
            blocked = False

        if blocked:
            # Nest and blocked lifting
            apply_lifting(self._b, self._a, bcs=bcs1, constraint=self._mpc, bc_data=self._bc_data)  # type: ignore
            apply_mpc_lifting(self._b, self._a, constraint=self._mpc)  # type: ignore
            _ghost_update(self._b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
            _fem.petsc.set_bc(self._b, bcs0)
        else:
            # Single form lifting
            apply_lifting(self._b, [self._a], bcs=[self.bcs], constraint=self._mpc, bc_data=self._bc_data)  # type: ignore
            apply_mpc_lifting(self._b, [self._a], constraint=self._mpc)  # type: ignore
            _ghost_update(self._b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore
            _fem.petsc.set_bc(self._b, self.bcs)
        _ghost_update(self._b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        _ghost_update(self._x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
        _fem.petsc.assign(self._x, self.u)  # type: ignore

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

        return self._u

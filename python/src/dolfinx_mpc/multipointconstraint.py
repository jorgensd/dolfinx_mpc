# Copyright (C) 2020-2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from mpi4py import MPI as _MPI
from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy
import numpy.typing as npt
import ufl

import dolfinx_mpc.cpp

from .container import (
    _UNSET,
    MPCData,
    _cpp_function,
    _deprecated,
    _float_array_types,
    _float_classes,
    _mpc_classes,
    _mpc_data_classes,
    _scalar_type,
    _tolerance,
    _Unset,
)
from .dictcondition import create_dictionary_constraint
from .integralcondition import create_integral_constraint
from .rbe import create_rbe2, create_rbe3


class MultiPointConstraint:
    """
    Hold data for multi point constraint relation ships,
    including new index maps for local assembly of matrices and vectors.

    The constraint is affine, :math:`x = K x_{red} + g`, where :math:`g` is
    supplied through `rhs_coeffs` and through the Dirichlet conditions in
    `bcs`. With neither, :math:`g=0` and the constraint is the usual linear
    one.

    Args:
        V: The function space
        dtype: The scalar type of the coefficients, used by everything built from this
            constraint. Defaults to the default scalar type of DOLFINx, real or complex, at the
            precision of the mesh of `V`. Its precision must be that of the mesh.
        bcs: Dirichlet boundary conditions for the problem. A master degree of
            freedom that is constrained by one of these is removed from the
            equation of its slave, and its contribution folded into the
            constraint offset :math:`g`. As the offset is recomputed from the
            current values of the conditions by :func:`update_constants`, time
            dependent boundary data is supported.
        rhs_coeffs: Function holding an additional inhomogeneity :math:`g_s`
            for the slave degrees of freedom, i.e.
            :math:`u_s = \\sum_j c_j u_{m_j} + g_s`.
    """

    _data: MPCData
    _master_spaces: List[_fem.FunctionSpace]
    _bcs: List[_fem.DirichletBC]
    _rhs_coeffs: Optional[_fem.Function]
    _scale_function: Optional[_fem.Function]
    _rbe2: List[list]
    _rbe3: List[tuple]
    _rbe3_data: Optional[tuple]
    V: _fem.FunctionSpace
    _input_space: _fem.FunctionSpace
    finalized: bool
    _cpp_object: _mpc_classes
    _dtype: npt.DTypeLike
    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        V: _fem.FunctionSpace,
        dtype: npt.DTypeLike | None = None,
        bcs: Optional[List[_fem.DirichletBC]] = None,
        rhs_coeffs: Optional[_fem.Function] = None,
    ):
        dtype = _scalar_type(V.mesh.geometry.x.dtype, dtype)
        # The rows on this process until finalize. Their master_blocks hold a code per master, as
        # the blocks are known only when the constraints are finalized together:
        #   code >= 0       the block itself, its position among the constraints finalized together
        #   code == -1      the block of this constraint: a master in its own space
        #   code == -2 - s  the block of the space self._master_spaces[s]
        # finalize_multipointconstraints resolves the codes to blocks.
        self._data = MPCData.empty(dtype, master_blocks=True)
        # The spaces of masters outside the space of this constraint, as given to add_constraint,
        # add_constraint_from_mpc_data or extend_masters (master_space), each once, in the order
        # first given. Every process appends them in the same order, also without local slaves,
        # so that a code means the same space on every process. A space must be the space of one
        # of the constraints finalized together with this one, or an uncollapsed subspace of it.
        self._master_spaces = []
        self._bcs = [] if bcs is None else list(bcs)
        if rhs_coeffs is not None:
            if not rhs_coeffs.x.array.dtype == dtype:
                raise ValueError("rhs_coeffs must have the same dtype as the MPC")
            if rhs_coeffs.function_space != V:
                raise ValueError("rhs_coeffs must be a Function in the space of the constraint")
        self._rhs_coeffs = rhs_coeffs
        self._scale_function = None
        # Per space on a spider mesh: [W, the tied space, the block of W], the block set by finalize
        self._rbe2 = []
        # Feet of RBE3 constraints on this space, (V, dofs, spiders, weights) per call, built by
        # finalize, which keeps the arrays for update_rbe3
        self._rbe3 = []
        self._rbe3_data = None
        self.V = V
        # Kept after finalize replaces `V` by the extended space, which contains no Dirichlet condition
        self._input_space = V
        self.finalized = False
        self._dtype = dtype

    def add_constraint(
        self,
        V: _fem.FunctionSpace,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        offsets: npt.NDArray[numpy.int32],
        master_space: Optional[_fem.FunctionSpace] = None,
        master_blocks: Optional[npt.NDArray[numpy.int32]] = None,
    ):
        """
        Add new constraint given by numpy arrays.

        Args:
            V: The function space for the constraint
            slaves: List of all slave dofs (using local dof numbering) on this process
            masters: List of all master dofs (using global dof numbering) on this process
            coeffs: The coefficients corresponding to each master.
            owners: The process each master is owned by.
            offsets: Array indicating the location in the masters array for the i-th slave
                in the slaves arrays, i.e.

                .. highlight:: python
                .. code-block:: python

                    masters_of_owned_slave[i] = masters[offsets[i]:offsets[i+1]]

            master_space: The function space all masters belong to, if not `V`. It must be the
                space of another constraint finalized together with this one by
                :func:`finalize_multipointconstraints`, or an uncollapsed subspace of it, and `masters` is in
                the global numbering of that constraint's space.
                The masters of a slave may then be in another block of a blocked problem.
            master_blocks: The block of each master, for masters from several spaces: its position
                in the list of constraints given to :func:`finalize_multipointconstraints`. Each
                master is in the global numbering of its block. Exclusive with `master_space`.

        Note:
            Collective when `master_space` or `master_blocks` is given: every process must call
            it with the same `master_space`, or with `master_blocks` (possibly empty).
        """
        assert V == self.V
        self._raise_if_finalized()
        if master_space is not None and master_blocks is not None:
            raise ValueError("Give either master_space or master_blocks, not both")
        if master_blocks is not None and len(master_blocks) != len(masters):
            raise ValueError("master_blocks must have one entry per master")

        if master_blocks is not None:
            codes = numpy.asarray(master_blocks, dtype=numpy.int32)
        elif master_space is not None:
            codes = numpy.full(len(masters), self._space_code(master_space), dtype=numpy.int32)
        else:
            codes = numpy.full(len(masters), -1, dtype=numpy.int32)
        if len(slaves) > 0:
            self._data = self._data.append(slaves, masters, coeffs, owners, offsets, codes)

    def _space_code(self, space: _fem.FunctionSpace) -> int:
        """The code of the block of the masters in `space`, recorded on every process in the same
        order so that every process resolves it the same way when the constraints are finalized."""
        for s, other in enumerate(self._master_spaces):
            if other is space:
                return -2 - s
        self._master_spaces.append(space)
        return -1 - len(self._master_spaces)

    def extend_masters(
        self,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        master_space: Optional[_fem.FunctionSpace] = None,
    ):
        r"""Add a master to the rows of existing slaves.

        A row is the equation :math:`u_s = \sum_j c_j u_{m_j}` of its slave :math:`u_s`, whichever
        constraint made it. Entry `i` adds a term :math:`k u_d` to its right-hand side, giving
        :math:`u_s = \sum_j c_j u_{m_j} + k u_d`, with :math:`s` = `slaves[i]`,
        :math:`u_d` = `masters[i]` and :math:`k` = `coeffs[i]`. If :math:`u_d` is already a master
        of the row, in the same space, :math:`k` is added to its coefficient. The ghost copies of
        the rows are extended too.

        A call adds one term per slave. Further terms of the same slave are added by further calls,
        each with its own `master_space`, so that a row may have masters in several spaces.

        For instance, a periodic condition :math:`u(x) = u(\mathrm{relation}(x))` extended, by one
        call per entry :math:`\bar H_{bk}` of a macroscopic gradient on a point mesh, with
        coefficient :math:`d_k = (x - \mathrm{relation}(x))_k` for component :math:`b`, ties `u` to
        :math:`\bar H x` plus a periodic field.

        Args:
            slaves: The slave of each entry: owned dofs of the space of the constraint (local,
                unrolled), each already a slave and each at most once.
            masters: The master of each entry, in the global (unrolled) numbering of `master_space`
            coeffs: The coefficient of each entry, of the scalar type of the constraint
            owners: The process owning each master
            master_space: The space of the masters if not that of the constraint. It must be the
                space of another constraint finalized together with this one by
                :func:`finalize_multipointconstraints`, or an uncollapsed subspace of it.

        Raises:
            ValueError: On every process, if a slave is not owned, appears twice or has no row yet,
                or the arrays differ in length.

        Note:
            Collective. Must be called by every process, with the same `master_space`, before the
            constraint is finalized.
        """
        self._raise_if_finalized()
        code = -1 if master_space is None else self._space_code(master_space)
        # Raises ValueError (as the C++ throws std::invalid_argument), identically on every process
        new_data = _cpp_function("extend_mpc_data", self._dtype)(
            self._data._cpp_object,
            -1,
            numpy.asarray(slaves, dtype=numpy.int32),
            numpy.asarray(masters, dtype=numpy.int64),
            numpy.asarray(coeffs, dtype=self._dtype),
            numpy.asarray(owners, dtype=numpy.int32),
            code,
            self.V._cpp_object,
        )
        self._data = MPCData.from_cpp(new_data)

    def add_master_from_point(
        self,
        V: _fem.FunctionSpace,
        slave_marker: Callable[[numpy.ndarray], numpy.ndarray],
        coefficient: Union[Callable[[numpy.ndarray], numpy.ndarray], float, complex],
        point: npt.ArrayLike,
        distance_tol: Optional[float] = None,
    ):
        r"""Add the dofs at a point as a master of the rows of marked slaves.

        Each slave of `V` at a coordinate :math:`x` marked by `slave_marker` gets the term
        :math:`k(x)\, u(p)` added to the right-hand side of its row, by :meth:`extend_masters`,
        with :math:`p` = `point` and :math:`k` = `coefficient`. For a blocked `V`, component
        :math:`b` of a slave gets component :math:`b` of the dofs at :math:`p`; for a single
        component pass the uncollapsed subspace, for instance `V.sub(1)`.

        For instance, the periodic cell of the homogenization demos ties the right edge to the left
        edge plus the corner :math:`B`, :math:`u(L, y) = u(0, y) + u^B`: the periodic constraint
        gives :math:`u(L, y) = u(0, y)`, and this function, with the right edge marked,
        coefficient `1` and `point` :math:`B`, adds :math:`u^B`.

        Args:
            V: The space of this constraint, or an uncollapsed subspace of it
            slave_marker: Marks the coordinates of the slaves, `(3, n)` to `(n,)`. Each marked dof
                must already be a slave, and the dofs at `point` must not be marked.
            coefficient: The coefficient :math:`k` of each slave, from its coordinates, `(3, n)`
                to `(n,)`, or one value for all.
            point: The point of the master, of `gdim` or 3 coordinates. Exactly one block of dofs
                of `V` must be there (see :func:`dofs_at_point`).
            distance_tol: The largest distance between `point` and its dofs. Defaults to `500`
                machine epsilon of the coordinate type of the mesh.

        Note:
            Collective. Must be called by every process, before the constraint is finalized.
        """
        dofs_marked, components, x_slaves, _ = _marked_dofs(V, slave_marker)
        owned = dofs_marked < self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        slaves, components, x_slaves = dofs_marked[owned], components[owned], x_slaves[owned]

        if callable(coefficient):
            coeffs = numpy.asarray(coefficient(x_slaves.T.copy()), dtype=self._dtype).reshape(-1)
        else:
            coeffs = numpy.full(len(slaves), coefficient, dtype=self._dtype)
        dofs, owner = dofs_at_point(V, point, distance_tol)
        self.extend_masters(
            slaves.astype(numpy.int32),
            dofs[components].astype(numpy.int64),
            coeffs,
            numpy.full(len(slaves), owner, dtype=numpy.int32),
        )

    def add_integral_constraint(
        self,
        weight_form,
        value,
        bcs: Optional[List[_fem.DirichletBC]] = None,
        rtol: numpy.floating | float | None = None,
        *,
        coefficient_tol: Optional[float] = None,
    ):
        r"""Constrain a scalar integral of the solution, :math:`L(u) = \gamma`.

        The functional is given as a linear form, and turned into a constraint
        with a single slave by :func:`dolfinx_mpc.create_integral_constraint`;
        see there for the derivation and the cost. The inhomogeneity
        :math:`\gamma/w_s` is written into the ``rhs_coeffs`` function of this
        constraint, which is created here if none was supplied to the
        constructor.

        Args:
            weight_form: A linear form in ``ufl.TestFunction(V)`` defining the
                functional, for instance ``v * ufl.dx``. Its test function must
                be in the function space of this constraint.
            value: The prescribed value :math:`\gamma` of the functional.
            bcs: Dirichlet conditions on the space. A constrained degree of
                freedom is never chosen as the slave. Pass the same conditions
                to the constructor to have a constrained *master* folded into
                the constraint offset. Defaults to the conditions given to the
                constructor.
            rtol: Deprecated, use `coefficient_tol`.
            coefficient_tol: A master whose coefficient is below
                `coefficient_tol` times the largest one is dropped. Defaults to
                `500` machine epsilon of the real type of the constraint.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_finalized()
        if rtol is not None:
            _deprecated("rtol", "`coefficient_tol`")
            coefficient_tol = rtol if coefficient_tol is None else coefficient_tol
        slaves, masters, coeffs, owners, offsets, rhs = create_integral_constraint(
            self.V,
            weight_form,
            value,
            self._bcs if bcs is None else bcs,
            coefficient_tol=_tolerance(coefficient_tol, self._dtype),
        )
        if self._rhs_coeffs is None:
            self._rhs_coeffs = rhs
        else:
            # Slaves of separate constraints are disjoint, so the offsets add
            self._rhs_coeffs.x.array[:] += rhs.x.array
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def add_constraint_from_mpc_data(
        self,
        V: _fem.FunctionSpace,
        mpc_data: Union[_mpc_data_classes, MPCData],
        master_space: Optional[_fem.FunctionSpace] = None,
    ):
        """
        Add new constraint given by an `dolfinc_mpc.cpp.mpc.mpc_data`-object. See
        :meth:`add_constraint` for `master_space`. The `master_blocks` of `mpc_data`, if any,
        are the `master_blocks` of :meth:`add_constraint`, and exclusive with `master_space`.
        """
        self._raise_if_finalized()
        self.add_constraint(
            V,
            mpc_data.slaves,
            mpc_data.masters,
            mpc_data.coeffs,
            mpc_data.owners,
            mpc_data.offsets,
            master_space=master_space,
            master_blocks=mpc_data.master_blocks,
        )

    def finalize(self, filter: Optional[numpy.floating] = None) -> None:
        """
        Finializes the multi point constraint. After this function is called, no new constraints can be added
        to the constraint. This function creates a map from the cells (local to index) to the slave degrees of
        freedom and builds a new index map and function space where unghosted master dofs are added as ghosts.

        Args:
            filter: If given, discard every master whose coefficient satisfies
                :math:`|c_{sj}| < \\mathrm{filter}\\cdot\\max_k|c_{sk}|`, the
                maximum being over the masters of that same slave. A negligible
                coefficient contributes nothing to the constraint, but still
                costs a ghost, a row of the sparsity pattern and an entry in
                every element matrix modification, so removing them can shrink
                :math:`K^HAK` substantially. With `None` (the default) every
                master supplied is kept.

        Note:
            Filtering changes the constraint that is enforced, by exactly the
            terms that are dropped. It is local and adds no communication.

        Note:
            To finalize the constraints of several function spaces, for instance the blocks of a
            :class:`ufl.MixedFunctionSpace`, use :func:`finalize_multipointconstraints`.
        """
        finalize_multipointconstraints([self], filter)

    def update_constants(self) -> None:
        """
        Recompute the constraint offset :math:`g` from the current values of the Dirichlet
        conditions supplied to the constructor.

        Call this whenever the value of one of those conditions changes, for instance between
        time steps, before re-assembling. :class:`LinearProblem` calls it automatically.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_not_finalized()
        if self._rhs_coeffs is not None:
            # Pass the array natively. Zero-copy, zero-allocation.
            num_dofs_local = self.V.dofmap.index_map_bs * (
                self.V.dofmap.index_map.size_local + self.V.dofmap.index_map.num_ghosts
            )
            rhs_coeffs = self._rhs_coeffs.x.array[:num_dofs_local]
            self._cpp_object.set_rhs_coeffs(rhs_coeffs)

        self._cpp_object.update_constants()

    @property
    def constants(self) -> _float_array_types:
        """
        The constraint offset :math:`g` for each degree of freedom local to the process,
        i.e. the affine term in :math:`x = K x_{red} + g`.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.constants

    @property
    def has_inhomogeneity(self) -> bool:
        """
        Whether any process carries a non-zero constraint offset. The value is globally
        reduced, so it is identical on every process.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.has_inhomogeneity

    @property
    def master_blocks(self) -> npt.NDArray[numpy.int32]:
        """
        The block of each master, parallel to ``masters.array``: the position, in the list given
        to :func:`finalize_multipointconstraints`, of the constraint whose space the master is in.
        The local index of a master is in the space of its block.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.master_blocks

    @property
    def dtype(self) -> type:
        """The scalar type of the coefficients, also that of everything built from the constraint."""
        return self._dtype

    @property
    def has_cross_block_masters(self) -> bool:
        """Whether a master on any process is in another block than the slaves."""
        self._raise_if_not_finalized()
        return self._cpp_object.has_cross_block_masters

    def create_periodic_constraint_topological(
        self,
        V: _fem.FunctionSpace,
        meshtag: _mesh.MeshTags,
        tag: int,
        relation: Callable[[numpy.ndarray], numpy.ndarray],
        bcs: List[_fem.DirichletBC],
        scale: Union[_float_classes, float, complex] = 1.0,
        tol: Union[_float_classes, float, None, _Unset] = _UNSET,
        num_threads: Optional[int] = 1,
        *,
        distance_tol: Optional[float] = None,
        coefficient_tol: Optional[float] = None,
    ):
        """
        Create periodic condition for all closure dofs of on all entities in `meshtag` with value `tag`.
        :math:`u(x_i) = scale * u(relation(x_i))` for all of :math:`x_i` on marked entities.

        Args:
            V: The function space to assign the condition to. Should either be the space of the MPC or an uncollapsed
                subspace of it.
               meshtag: MeshTag for entity to apply the periodic condition on
            tag: Tag indicating which entities should be slaves
            relation: Lambda-function describing the geometrical relation
            bcs: Dirichlet boundary conditions for the problem (Periodic constraints will be ignored for these dofs)
            scale: Factor of the masters, of the scalar type of the constraint
            tol: Deprecated, use `distance_tol` and `coefficient_tol`: a value sets both, `None` sets
                `coefficient_tol=0`.
            num_threads: The number of threads to use for certain operations
            distance_tol: The largest distance from a mapped slave point to a master cell for the point to
                be in the cell, and the padding of the bounding boxes of the cells. Defaults to `500`
                machine epsilon of the coordinate type of the mesh.
            coefficient_tol: A master whose coefficient is below `coefficient_tol` times the largest of
                its slave is dropped. `0` keeps every master, so that the coefficients can later be changed
                with :func:`scale_coefficients` or :func:`update_coefficients`. Defaults to `500`
                machine epsilon of the real type of the constraint.
        """
        bcs_ = [bc._cpp_object for bc in bcs]
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        distance_tol, coefficient_tol = self._tolerances(distance_tol, coefficient_tol, tol=tol)
        is_input_space = V is self.V
        if not (is_input_space or self.V.contains(V)):
            raise RuntimeError("The input space has to be an uncollapsed subspace (or the full space) of the MPC")
        mpc_data = _cpp_function("create_periodic_constraint_topological", self._dtype)(
            V._cpp_object,
            meshtag._cpp_object,
            tag,
            relation,
            bcs_,
            scale,
            not is_input_space,
            distance_tol,
            coefficient_tol,
            num_threads,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_periodic_constraint_geometrical(
        self,
        V: _fem.FunctionSpace,
        indicator: Callable[[numpy.ndarray], numpy.ndarray],
        relation: Callable[[numpy.ndarray], numpy.ndarray],
        bcs: List[_fem.DirichletBC],
        scale: Union[_float_classes, float, complex] = 1.0,
        tol: Union[_float_classes, float, None, _Unset] = _UNSET,
        num_threads: Optional[int] = 1,
        *,
        distance_tol: Optional[float] = None,
        coefficient_tol: Optional[float] = None,
    ):
        """
        Create a periodic condition for all degrees of freedom whose physical location satisfies
        :math:`indicator(x_i)==True`, i.e.
        :math:`u(x_i) = scale * u(relation(x_i))` for all :math:`x_i`

        Args:
            V: The function space to assign the condition to. Should either be the space of the MPC or an uncollapsed
                subspace of it.
            indicator: Lambda-function to locate degrees of freedom that should be slaves
            relation: Lambda-function describing the geometrical relation to master dofs
            bcs: Dirichlet boundary conditions for the problem
                 (Periodic constraints will be ignored for these dofs)
            scale: Factor of the masters, of the scalar type of the constraint
            tol: Deprecated, use `distance_tol` and `coefficient_tol`: a value sets both, `None` sets
                `coefficient_tol=0`.
            num_threads: The number of threads to use for certain operations.
            distance_tol: The largest distance from a mapped slave point to a master cell for the point to
                be in the cell, and the padding of the bounding boxes of the cells. Defaults to `500`
                machine epsilon of the coordinate type of the mesh.
            coefficient_tol: A master whose coefficient is below `coefficient_tol` times the largest of
                its slave is dropped. `0` keeps every master, so that the coefficients can later be changed
                with :func:`scale_coefficients` or :func:`update_coefficients`. Defaults to `500`
                machine epsilon of the real type of the constraint.
        """
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        distance_tol, coefficient_tol = self._tolerances(distance_tol, coefficient_tol, tol=tol)
        bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        is_input_space = V is self.V
        if not (is_input_space or self.V.contains(V)):
            raise RuntimeError("The input space has to be an uncollapsed subspace (or the full space) of the MPC")
        mpc_data = _cpp_function("create_periodic_constraint_geometrical", self._dtype)(
            V._cpp_object,
            indicator,
            relation,
            bcs,
            scale,
            not is_input_space,
            distance_tol,
            coefficient_tol,
            num_threads,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_submesh_constraint(
        self,
        V: _fem.FunctionSpace,
        master_space: _fem.FunctionSpace,
        entity_map: _mesh.EntityMap,
        bcs: Optional[List[_fem.DirichletBC]] = None,
        scale: Union[_float_classes, float, complex] = 1.0,
        tol: Union[_float_classes, float, None, _Unset] = _UNSET,
        num_threads: int = 1,
        *,
        coefficient_tol: Optional[float] = None,
    ):
        r"""
        Tie the degrees of freedom of `V` to `master_space` on a related mesh: a submesh and its
        parent, related by `entity_map` as returned by :func:`dolfinx.mesh.create_submesh`.

        Every degree of freedom of `V` in the closure of a cell related to a cell of
        `master_space` becomes a slave, :math:`u(x_i) = \mathrm{scale}\, u_m(x_i)`, with
        :math:`u_m` evaluated in the related cell and component `b` tied to component `b`. With `V`
        on the submesh this is every degree of freedom of `V`, for instance the trace
        :math:`\bar u = u|_\Gamma` on a submesh of facets; with `V` on the parent it is the
        degrees of freedom on the submesh. No search is involved: the related cell is a table
        lookup. For a submesh of facets the parent cell is one attached to the facet, so for a
        discontinuous `master_space` the side is arbitrary.

        Args:
            V: The space of the constraint, or an uncollapsed subspace of it
            master_space: The space of another constraint finalized together with this one by
                :func:`finalize_multipointconstraints`, or an uncollapsed subspace of it. Its mesh and the mesh
                of `V` are the two meshes of `entity_map`, either way round.
            entity_map: Relates the cells of the submesh to entities of the parent, of
                codimension 0 or 1
            bcs: Dirichlet conditions on the space of the constraint. Their degrees of freedom
                are not made slaves.
            scale: Factor of the masters, of the scalar type of the constraint
            tol: Deprecated, use `coefficient_tol`: a value sets it, `None` sets `coefficient_tol=0`.
            num_threads: The number of threads to use
            coefficient_tol: A master whose coefficient is below `coefficient_tol` times the largest of
                its slave is dropped. `0` keeps every basis function of the related cell, so that the
                coefficients can later be changed with :func:`scale_coefficients` or
                :func:`update_coefficients`. Defaults to `500` machine epsilon of the real type of the
                constraint. No distance tolerance is needed, as the related cell is not searched for.

        Raises:
            ValueError: If `entity_map` does not relate the two meshes, relates entities other
                than the cells of the submesh, is of codimension above 1, or the spaces have
                different numbers of components. Raised on every process.

        Note:
            Collective.
        """
        self._raise_if_finalized()
        if not (V is self.V or self.V.contains(V)):
            raise ValueError("V must be the space of the constraint or an uncollapsed subspace of it")
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        if not isinstance(tol, _Unset):
            _deprecated("tol", "`coefficient_tol`")
            if coefficient_tol is None:
                coefficient_tol = 0.0 if tol is None else tol
        bcs_ = [] if bcs is None else [bc._cpp_object for bc in bcs]
        mpc_data = _cpp_function("create_submesh_constraint", self._dtype)(
            V._cpp_object,
            master_space._cpp_object,
            entity_map._cpp_object,
            bcs_,
            scale,
            _tolerance(coefficient_tol, self._dtype),
            num_threads,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data, master_space=master_space)

    def _add_rbe2(self, dofs: list[npt.NDArray[numpy.int32]], W: _fem.FunctionSpace, x=None):
        """Tie `dofs[k]` to spider `k` of `W`, and record `W` for :meth:`update_rbe2`."""
        spiders = [numpy.full(len(d), k, dtype=numpy.int64) for k, d in enumerate(dofs)]
        mpc_data = create_rbe2(
            self.V,
            numpy.concatenate(dofs) if dofs else numpy.zeros(0, dtype=numpy.int32),
            numpy.concatenate(spiders) if spiders else numpy.zeros(0, dtype=numpy.int64),
            W,
            self._dtype,
            x,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data, master_space=W)
        if not any(W is entry[0] for entry in self._rbe2):
            self._rbe2.append([W, self.V, None])

    def add_rbe2_topological(
        self,
        dim: int,
        entities: Union[npt.NDArray[numpy.int32], Sequence[Optional[npt.NDArray[numpy.int32]]]],
        W: _fem.FunctionSpace,
    ):
        r"""
        Tie the dofs on mesh entities rigidly to a point, as the RBE2 element of other codes
        (a rigid "spider").

        Each dof of this constraint's space on the entities is a "foot" of a spider whose "body"
        is a point of the point mesh of `W`. Every component of a foot follows the motion of its
        body,

        .. math::

            u(x) = t + \theta \times (x - x_c),

        where :math:`x_c` is the coordinate of the dofs of `W` at the point, :math:`t` its
        translation and :math:`\theta` its rotation, the dofs of `W` at the point. Without
        rotations, :math:`u(x) = t`. All rotation terms are kept, also where their coefficient is
        zero, so :meth:`update_rbe2` can follow the motion of the meshes.

        Args:
            dim: Topological dimension of the entities
            entities: Entities (local to the process) whose dofs are tied to spider 0, or a
                sequence whose entry `k` holds the entities tied to the spider with input index
                `k` (see :func:`dolfinx_mpc.create_spider_mesh`). An entry may be `None`.
            W: Space on the spider mesh (:func:`dolfinx_mpc.create_spider_mesh`). Its value size
                is the geometric dimension, for translations only, or 6 in 3D and 3 in 2D, for
                translations and rotations. `W` must be the space of another constraint
                finalized together with this one by :func:`finalize_multipointconstraints`.

        Note:
            Collective. Must be called by every process, with the same number of entries in
            `entities`.
        """
        self._raise_if_finalized()
        per_spider = [entities] if isinstance(entities, numpy.ndarray) else list(entities)
        dofs = []
        # The dofs of each spider in turn, collectively, so that a dof on the entities of two
        # spiders is caught as constrained twice
        for e in per_spider:
            e = numpy.zeros(0, dtype=numpy.int32) if e is None else numpy.asarray(e, dtype=numpy.int32)
            dofs.append(_fem.locate_dofs_topological(self.V, dim, e))
        self._add_rbe2(dofs, W)

    def add_rbe2_geometrical(
        self,
        locators: Union[
            Callable[[numpy.ndarray], numpy.ndarray], Sequence[Optional[Callable[[numpy.ndarray], numpy.ndarray]]]
        ],
        W: _fem.FunctionSpace,
    ):
        r"""
        Tie the dofs located geometrically rigidly to a point, as the RBE2 element of other codes
        (a rigid "spider"). See :meth:`add_rbe2_topological` for the relation.

        Args:
            locators: Marks the dofs tied to spider 0, given their coordinates, shape
                `(3, num_points)`, or a sequence whose entry `k` marks the dofs tied to the spider
                with input index `k`. An entry may be `None`.
            W: Space on the spider mesh, see :meth:`add_rbe2_topological`

        Note:
            Collective. Must be called by every process, with the same number of locators.
        """
        self._raise_if_finalized()
        per_spider = [locators] if callable(locators) else list(locators)
        dofs = [
            numpy.zeros(0, dtype=numpy.int32)
            if locator is None
            else numpy.asarray(_fem.locate_dofs_geometrical(self.V, locator), dtype=numpy.int32)
            for locator in per_spider
        ]
        self._add_rbe2(dofs, W, self.V.tabulate_dof_coordinates())

    def update_rbe2(self) -> None:
        """
        Recompute the coefficients of every RBE2 constraint from the current coordinates.

        The feet are at the dof coordinates of the constraint's space, the spiders at those of the
        space on the spider mesh, both read now. Move the meshes, for instance to the deformed
        configuration in an updated Lagrangian analysis, then call this to tie the feet to the
        rigid motion about the new positions. Assemble again afterwards.

        The constraint must be finalized without a `filter`, which could drop a master whose
        coefficient becomes nonzero.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_not_finalized()
        if len(self._rbe2) == 0:
            raise ValueError("The constraint has no RBE2 constraints")
        for W, V, block in self._rbe2:
            dolfinx_mpc.cpp.mpc.update_rbe2(self._cpp_object, V._cpp_object, W._cpp_object, block)

    def _add_rbe3(self, V: _fem.FunctionSpace, dofs: list[npt.NDArray[numpy.int32]], weights, x):
        """Record `dofs[k]` as feet of spider `k`, with weights from `weights`."""
        real = V.mesh.geometry.x.dtype
        for k, d in enumerate(dofs):
            if weights is None:
                w = numpy.ones(len(d), dtype=real)
            elif callable(weights):
                w = numpy.asarray(weights(x[d].T), dtype=real).reshape(-1)
            else:
                w = numpy.full(len(d), weights, dtype=real)
            self._rbe3.append((V, d, numpy.full(len(d), k, dtype=numpy.int64), w))

    def add_rbe3_topological(
        self,
        V: _fem.FunctionSpace,
        dim: int,
        entities: Union[npt.NDArray[numpy.int32], Sequence[Optional[npt.NDArray[numpy.int32]]]],
        weights: Union[None, float, Callable[[numpy.ndarray], numpy.ndarray]] = None,
    ):
        r"""
        Tie the dofs of spiders to the motion of the dofs of `V` on mesh entities, as the RBE3
        element of other codes (a flexible "spider").

        This constraint is on the space of the spider mesh (:func:`dolfinx_mpc.create_spider_mesh`).
        Each spider moves with the rigid motion that best fits its "feet", in the weighted
        least-squares sense,

        .. math::

            \min_{t, \theta} \sum_i w_i |u_i - t - \theta \times (x_i - x_c)|^2,

        where :math:`x_c` is the coordinate of the spider, :math:`t` and :math:`\theta` its
        translation and rotation, and :math:`u_i` the displacement of foot :math:`i` at
        :math:`x_i`. Without rotations, :math:`t` is the weighted mean of the feet. Unlike RBE2, the
        feet keep their stiffness: a load on the spider is spread over them without making them
        rigid.

        The feet may be in several spaces, given by one call each. The constraint is built when
        it is finalized, by :func:`finalize_multipointconstraints` together with the constraints of
        the spaces of the feet.

        Args:
            V: The space of the feet, with one component per dimension
            dim: Topological dimension of the entities
            entities: Entities (local to the process) whose dofs are feet of spider 0, or a
                sequence whose entry `k` holds those of the spider with input index `k`. An entry
                may be `None`.
            weights: The weight of each foot: `None` for one, a number, or a function of the
                coordinates, shape `(3, num_points)`, returning one non-negative weight per foot.
                Evaluated once, here: :meth:`update_rbe3` keeps the weights.

        Note:
            Collective. Must be called by every process, with the same number of entries in
            `entities`.
        """
        self._raise_if_finalized()
        per_spider = [entities] if isinstance(entities, numpy.ndarray) else list(entities)
        dofs = []
        for e in per_spider:
            e = numpy.zeros(0, dtype=numpy.int32) if e is None else numpy.asarray(e, dtype=numpy.int32)
            dofs.append(_fem.locate_dofs_topological(V, dim, e))
        self._add_rbe3(V, dofs, weights, V.tabulate_dof_coordinates() if callable(weights) else None)

    def add_rbe3_geometrical(
        self,
        V: _fem.FunctionSpace,
        locators: Union[
            Callable[[numpy.ndarray], numpy.ndarray], Sequence[Optional[Callable[[numpy.ndarray], numpy.ndarray]]]
        ],
        weights: Union[None, float, Callable[[numpy.ndarray], numpy.ndarray]] = None,
    ):
        """
        Tie the dofs of spiders to the motion of the dofs of `V` located geometrically, as the
        RBE3 element of other codes. See :meth:`add_rbe3_topological` for the relation.

        Args:
            V: The space of the feet, with one component per dimension
            locators: Marks the feet of spider 0, given their coordinates, shape
                `(3, num_points)`, or a sequence whose entry `k` marks the feet of the spider with
                input index `k`. An entry may be `None`.
            weights: See :meth:`add_rbe3_topological`

        Note:
            Collective. Must be called by every process, with the same number of locators.
        """
        self._raise_if_finalized()
        per_spider = [locators] if callable(locators) else list(locators)
        dofs = [
            numpy.zeros(0, dtype=numpy.int32)
            if locator is None
            else numpy.asarray(_fem.locate_dofs_geometrical(V, locator), dtype=numpy.int32)
            for locator in per_spider
        ]
        self._add_rbe3(V, dofs, weights, V.tabulate_dof_coordinates() if callable(weights) else None)

    def _build_rbe3(self, mpcs: Sequence[MultiPointConstraint]) -> None:
        """Build the RBE3 constraint from the feet recorded, with the blocks of their spaces."""
        spaces: list[_fem.FunctionSpace] = []
        for V, *_ in self._rbe3:
            if not any(V is other for other in spaces):
                spaces.append(V)
        blocks = []
        for V in spaces:
            matches = [j for j, other in enumerate(mpcs) if other.V is V]
            if len(matches) != 1:
                raise ValueError(
                    "The feet of an RBE3 constraint must be in the space of exactly one of the "
                    "constraints finalized together with it"
                )
            blocks.append(matches[0])

        def gather(i):
            return [numpy.concatenate([r[i] for r in self._rbe3 if r[0] is V]) for V in spaces]

        data = (self.V, spaces, gather(1), gather(2), gather(3))
        mpc_data = create_rbe3(*data, dtype=self._dtype)
        self.add_constraint(
            self.V,
            mpc_data.slaves,
            mpc_data.masters,
            mpc_data.coeffs,
            mpc_data.owners,
            mpc_data.offsets,
            master_blocks=numpy.asarray(blocks, dtype=numpy.int32)[mpc_data.master_blocks],
        )
        self._rbe3_data = (*data, blocks)

    def update_rbe3(self) -> None:
        """
        Recompute the coefficients of the RBE3 constraint from the current coordinates.

        The feet are at the dof coordinates of their spaces, the spiders at those of the space of
        this constraint, both read now. Move the meshes, then call this. Assemble again
        afterwards.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_not_finalized()
        if self._rbe3_data is None:
            raise ValueError("The constraint has no RBE3 constraint")
        W, spaces, dofs, spiders, weights, blocks = self._rbe3_data
        dolfinx_mpc.cpp.mpc.update_rbe3(
            self._cpp_object,
            W._cpp_object,
            [V._cpp_object for V in spaces],
            blocks,
            [numpy.ascontiguousarray(d, dtype=numpy.int32) for d in dofs],
            [numpy.ascontiguousarray(k, dtype=numpy.int64) for k in spiders],
            [numpy.ascontiguousarray(w) for w in weights],
        )

    def create_slip_constraint(
        self,
        space: _fem.FunctionSpace,
        facet_marker: Tuple[_mesh.MeshTags, int],
        v: _fem.Function,
        bcs: List[_fem.DirichletBC] = [],
    ):
        """
        Create a slip constraint :math:`u \\cdot v=0` over the entities defined in `facet_marker` with the given index.

        Args:
            space: Function space (possibly an uncollapsed subspace) for the current constraint
            facet_marker: Tuple containomg the mesh tag and marker used to locate degrees of freedom
            v: Function containing the directional vector to dot your slip condition (most commonly a normal vector)
            bcs: List of Dirichlet BCs (slip conditions will be ignored on these dofs)

        Examples:
            Create constaint :math:`u\\cdot n=0` of all indices in `mt` marked with `i`

            .. highlight:: python
            .. code-block:: python

                V = dolfinx.fem.functionspace(mesh, ("CG", 1))
                mpc = MultiPointConstraint(V)
                n = dolfinx.fem.Function(V)
                mpc.create_slip_constaint(V, (mt, i), n)

            Create slip constaint for a mixed function space:

            .. highlight:: python
            .. code-block:: python

                cellname = mesh.basix_cell()
                Ve = basix.ufl.element(basix.ElementFamily.P, cellname , 2, shape=(mesh.geometry.dim,))
                Qe = basix.ufl.element(basix.ElementFamily.P, cellname , 1)
                me = basix.ufl.mixed_element([Ve, Qe])
                W = dolfinx.fem.functionspace(mesh, me)
                mpc = MultiPointConstraint(W)
                n_space, _ = W.sub(0).collapse()
                normal = dolfinx.fem.Function(n_space)
                mpc.create_slip_constraint(W.sub(0), (mt, i), normal, bcs=[])

            A slip condition cannot be applied on the same degrees of freedom as a Dirichlet BC, and therefore
            any Dirichlet bc for the space of the multi point constraint should be supplied.

            .. highlight:: python
            .. code-block:: python

                cellname = mesh.basix_cell()
                Ve = basix.ufl.element(basix.ElementFamily.P, cellname , 2, shape=(mesh.geometry.dim,))
                Qe = basix.ufl.element(basix.ElementFamily.P, cellname , 1)
                me = basix.ufl.mixed_element([Ve, Qe])
                W = dolfinx.fem.functionspace(mesh, me)
                mpc = MultiPointConstraint(W)
                n_space, _ = W.sub(0).collapse()
                normal = Function(n_space)
                bc = dolfinx.fem.dirichletbc(inlet_velocity, dofs, W.sub(0))
                mpc.create_slip_constraint(W.sub(0), (mt, i), normal, bcs=[bc])
        """
        bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        if space is self.V:
            sub_space = False
        elif self.V.contains(space):
            sub_space = True
        else:
            raise ValueError("Input space has to be an uncollapsed subspace of the MPC space")
        mpc_data = _cpp_function("create_slip_condition", self._dtype)(
            space._cpp_object,
            facet_marker[0]._cpp_object,
            facet_marker[1],
            v._cpp_object,
            bcs,
            sub_space,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_general_constraint(
        self,
        slave_master_dict: Dict[bytes, Dict[bytes, float]],
        subspace_slave: Optional[int] = None,
        subspace_master: Optional[int] = None,
        *,
        distance_tol: Optional[float] = None,
    ):
        """
        Args:
            V: The function space
            slave_master_dict: Nested dictionary, where the first key is the bit representing the slave dof's
                coordinate in the mesh. The item of this key is a dictionary, where each key of this dictionary
                is the bit representation of the master dof's coordinate, and the item the coefficient for
                the MPC equation.
            subspace_slave: If using mixed or vector space, and only want to use dofs from a sub space
                as slave add index here
            subspace_master: Subspace index for mixed or vector spaces
            distance_tol: The largest distance between the coordinate of a key and that of its dof.
                Defaults to `500` machine epsilon of the coordinate type of the mesh.

        Example:
            If the dof `D` located at `[d0, d1]` should be constrained to the dofs
            `E` and `F` at `[e0, e1]` and `[f0, f1]` as :math:`D = \\alpha E + \\beta F`
            the dictionary should be:

            .. highlight:: python
            .. code-block:: python

                    {numpy.array([d0, d1], dtype=mesh.geometry.x.dtype).tobytes():
                        {numpy.array([e0, e1], dtype=mesh.geometry.x.dtype).tobytes(): alpha,
                        numpy.array([f0, f1], dtype=mesh.geometry.x.dtype).tobytes(): beta}}
        """
        slaves, masters, coeffs, owners, offsets = create_dictionary_constraint(
            self.V,
            slave_master_dict,
            subspace_slave,
            subspace_master,
            dtype=self._dtype,
            distance_tol=_tolerance(distance_tol, self.V.mesh.geometry.x.dtype),
        )
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def _tolerances(
        self,
        distance_tol: Optional[float],
        coefficient_tol: Optional[float],
        tol: Union[_float_classes, float, None, _Unset] = _UNSET,
        eps2: Optional[float] = None,
    ) -> tuple[float, float]:
        """The distance and coefficient tolerance, by default `500` machine epsilon of the coordinate
        type of the mesh and of the real type of the constraint.

        The deprecated `tol` of the periodic constraints sets both, or with `None` keeps every
        master. The deprecated `eps2` of the contact constraints is a squared distance.
        """
        if not isinstance(tol, _Unset):
            _deprecated("tol", "`distance_tol` and `coefficient_tol`", stacklevel=4)
            if tol is None:
                coefficient_tol = 0.0 if coefficient_tol is None else coefficient_tol
            else:
                distance_tol = tol if distance_tol is None else distance_tol
                coefficient_tol = tol if coefficient_tol is None else coefficient_tol
        if eps2 is not None:
            _deprecated("eps2", "`distance_tol`, a distance rather than a squared distance,", stacklevel=4)
            distance_tol = float(numpy.sqrt(eps2)) if distance_tol is None else distance_tol
        return (
            _tolerance(distance_tol, self.V.mesh.geometry.x.dtype),
            _tolerance(coefficient_tol, self._dtype),
        )

    def create_contact_slip_condition(
        self,
        meshtags: _mesh.MeshTags,
        slave_marker: int,
        master_marker: int,
        normal: _fem.Function,
        eps2: Optional[float] = None,
        num_threads: Optional[int] = 1,
        *,
        distance_tol: Optional[float] = None,
        coefficient_tol: Optional[float] = None,
    ):
        """
        Create a slip condition between two sets of facets marker with individual markers.
        The interfaces should be within machine precision of eachother, but the vertices does not need to align.
        The condition created is :math:`u_s \\cdot normal_s = u_m \\cdot normal_m` where `s` is the
        restriction to the slave facets, `m` to the master facets.

        Args:
            meshtags: The meshtags of the set of facets to tie together
            slave_marker: The marker of the slave facets
            master_marker: The marker of the master facets
            normal: The function used in the dot-product of the constraint
            eps2: Deprecated, use `distance_tol`, which is `sqrt(eps2)`.
            num_threads: The number of threads to use for certain operations
            distance_tol: The largest distance from a slave point to a master cell for the point to
                be in the cell, and the padding of the bounding boxes of the cells. Defaults to `500`
                machine epsilon of the coordinate type of the mesh.
            coefficient_tol: A master whose coefficient is below `coefficient_tol` times the largest of
                its slave is dropped. `0` keeps every master. Defaults to `500`
                machine epsilon of the real type of the constraint.
        """
        mpc_data = _cpp_function("create_contact_slip_condition", self._dtype)(
            self.V._cpp_object,
            meshtags._cpp_object,
            slave_marker,
            master_marker,
            normal._cpp_object,
            *self._tolerances(distance_tol, coefficient_tol, eps2=eps2),
            num_threads,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    def create_contact_inelastic_condition(
        self,
        meshtags: _cpp.mesh.MeshTags_int32,
        slave_marker: int,
        master_marker: int,
        eps2: Optional[float] = None,
        allow_missing_masters: bool = False,
        num_threads: Optional[int] = 1,
        *,
        distance_tol: Optional[float] = None,
        coefficient_tol: Optional[float] = None,
    ):
        """
        Create a contact inelastic condition between two sets of facets marker with individual markers.
        The interfaces should be within machine precision of eachother, but the vertices does not need to align.
        The condition created is :math:`u_s = u_m` where `s` is the restriction to the
        slave facets, `m` to the master facets.

        Args:
            meshtags: The meshtags of the set of facets to tie together
            slave_marker: The marker of the slave facets
            master_marker: The marker of the master facets
            eps2: Deprecated, use `distance_tol`, which is `sqrt(eps2)`.
            allow_missing_masters: If true, the function will not throw an error if a degree of freedom
                in the closure of the master entities does not have a corresponding set of slave degree
                of freedom.
            num_threads: The number of threads to use for certain operations
            distance_tol: The largest distance from a slave point to a master cell for the point to
                be in the cell, and the padding of the bounding boxes of the cells. Defaults to `500`
                machine epsilon of the coordinate type of the mesh.
            coefficient_tol: A master whose coefficient is below `coefficient_tol` times the largest of
                its slave is dropped. `0` keeps every master. Defaults to `500`
                machine epsilon of the real type of the constraint.
        """
        mpc_data = _cpp_function("create_contact_inelastic_condition", self._dtype)(
            self.V._cpp_object,
            meshtags._cpp_object,
            slave_marker,
            master_marker,
            *self._tolerances(distance_tol, coefficient_tol, eps2=eps2),
            allow_missing_masters,
            num_threads,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    @property
    def is_slave(self) -> numpy.ndarray:
        """
        Returns a vector of integers where the ith entry indicates if a degree of freedom (local to process) is a slave.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.is_slave

    @property
    def slaves(self):
        """
        Returns the degrees of freedom for all slaves local to process
        """
        self._raise_if_not_finalized()
        return self._cpp_object.slaves

    @property
    def masters(self) -> _cpp.graph.AdjacencyList_int32:
        """
        Returns an adjacency-list whose ith node corresponds to
        a degree of freedom (local to process), and links the corresponding master dofs (local to process).

        Examples:

            .. highlight:: python
            .. code-block:: python

                masters = mpc.masters
                masters_of_dof_i = masters.links(i)
        """
        self._raise_if_not_finalized()
        return self._cpp_object.masters

    def coefficients(self) -> _float_array_types:
        """
        Returns a vector containing the coefficients for the constraint, and the corresponding offsets
        for the ith degree of freedom.

        Examples:

            .. highlight:: python
            .. code-block:: python

                coeffs, offsets = mpc.coefficients()
                coeffs_of_slave_i = coeffs[offsets[i]:offsets[i+1]]
        """
        self._raise_if_not_finalized()
        return self._cpp_object.coefficients()

    def all_coefficients(self) -> Tuple[_float_array_types, npt.NDArray[numpy.int32]]:
        """
        Returns the coefficients of all masters, including those eliminated by a Dirichlet condition,
        in the order supplied before :func:`finalize`, and the offsets for the ith degree of freedom.
        This is the layout taken by :func:`update_coefficients`. The corresponding masters are given
        by :func:`all_masters`.

        Examples:

            .. highlight:: python
            .. code-block:: python

                coeffs, offsets = mpc.all_coefficients()
                coeffs_of_slave_i = coeffs[offsets[i]:offsets[i+1]]
        """
        self._raise_if_not_finalized()
        return self._cpp_object.all_coefficients()

    def all_masters(self) -> npt.NDArray[numpy.int32]:
        """
        Returns the masters (local index in :attr:`function_space`) in the layout of
        :func:`all_coefficients`.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.all_masters()

    def update_coefficients(self, coeffs: _float_array_types) -> None:
        """
        Replace the coefficient of every master, including masters eliminated by a Dirichlet
        condition, and recompute the constraint offset :math:`g`.

        The masters are fixed at creation. A master dropped by `coefficient_tol` or by the `filter`
        of :func:`finalize` cannot be given a coefficient, so create the constraint with
        `coefficient_tol=0` and no filter if the coefficients are to be changed.

        Args:
            coeffs: The new coefficients, in the layout of :func:`all_coefficients`, for all degrees
                of freedom local to the process (owned and ghost).

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_not_finalized()
        self._cpp_object.update_coefficients(numpy.ascontiguousarray(coeffs, dtype=self._dtype))

    def scale_coefficients(
        self,
        scale: Union[_float_classes, float, complex, ufl.core.expr.Expr, _fem.Expression],
    ) -> None:
        """
        Multiply the coefficients of all masters of each slave :math:`s` by a factor
        :math:`f_s`, and recompute the constraint offset :math:`g`. For a periodic constraint
        :math:`u(x_s) = f_s u(relation(x_s))`, which for instance gives a Floquet-Bloch condition
        with :math:`f=e^{i k\\cdot L}`.

        The factors are stored in a function in the space of the constraint, and :math:`f_s` is
        the degree of freedom :math:`s` of that function: the value at the slave for a Lagrange
        space, the corresponding moment for e.g. a Nédélec space.

        Repeated calls compound. Masters eliminated by a Dirichlet condition are scaled as well,
        the user supplied `rhs_coeffs` are not.

        Args:
            scale: A scalar, a :class:`dolfinx.fem.Function` in the constraint's space (copied
                by interpolation), a UFL expression, compiled into a :class:`dolfinx.fem.Expression`
                at the interpolation points of the space, or such a compiled expression. Pass a
                compiled expression to avoid recompilation when the factor is updated through
                :class:`dolfinx.fem.Constant`'s in it.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_not_finalized()
        if self._scale_function is None:
            self._scale_function = _fem.Function(self.V, dtype=self._dtype)
        f = self._scale_function
        if isinstance(scale, (_fem.Expression, _fem.Function)):
            f.interpolate(scale)
        elif isinstance(scale, ufl.core.expr.Expr):
            f.interpolate(_fem.Expression(scale, self.V.element.interpolation_points, dtype=self._dtype))
        else:
            f.x.array[:] = scale
        f.x.scatter_forward()
        # The extended index map appends master ghosts after the ghosts of the input space
        num_dofs_local = len(self._cpp_object.is_slave)
        self._cpp_object.scale_coefficients(f.x.array[:num_dofs_local])

    @property
    def num_local_slaves(self):
        """
        Return the number of slaves owned by the current process.
        """
        self._raise_if_not_finalized()
        return self._cpp_object.num_local_slaves

    @property
    def cell_to_slaves(self):
        """
        Returns an `dolfinx.cpp.graph.AdjacencyList_int32` whose ith node corresponds to
        the ith cell (local to process), and links the corresponding slave degrees of
        freedom in the cell (local to process).

        Examples:

            .. highlight:: python
            .. code-block:: python

                cell_to_slaves = mpc.cell_to_slaves()
                slaves_in_cell_i = cell_to_slaves.links(i)
        """
        self._raise_if_not_finalized()
        return self._cpp_object.cell_to_slaves

    @property
    def function_space(self):
        """
        Return the function space for the multi-point constraint with the updated index map
        """
        self._raise_if_not_finalized()
        return self.V

    @property
    def input_space(self) -> _fem.FunctionSpace:
        """
        The function space the constraint was created with.

        Forms and Dirichlet conditions are stated on this space, while functions holding a
        solution live in :attr:`function_space`, its extension by the masters of the constraint.
        For a system of several blocks, ``[mpc.input_space for mpc in mpcs]`` gives the spaces in
        the order of the blocks, for instance for a :class:`ufl.MixedFunctionSpace`.
        """
        return self._input_space

    def backsubstitution(self, u: Union[_fem.Function, Sequence[_fem.Function], _PETSc.Vec]) -> None:  # type: ignore
        """
        For a Function, impose the multi-point constraint by backsubstiution.
        This function is used after solving the reduced problem to obtain the values
        at the slave degrees of freedom

        .. note::
            It is the users responsibility to destroy the PETSc vector

        Args:
            u: The input function. For a constraint with masters in another block, the function
                of every block, in the order given to :func:`finalize_multipointconstraints`;
                only the function of this constraint's block is changed. The ghosts of the
                functions holding masters must be up to date.
        """
        self._raise_if_not_finalized()
        if isinstance(u, Sequence):
            self._cpp_object.backsubstitution([u_k.x.array for u_k in u])  # type: ignore
            u[self._cpp_object.block].x.scatter_forward()
            return
        try:
            self._cpp_object.backsubstitution(u.x.array)  # type: ignore
            assert isinstance(u, _fem.Function)
            u.x.scatter_forward()
        except AttributeError:
            assert isinstance(u, _PETSc.Vec)
            with u.localForm() as vector_local:
                self._cpp_object.backsubstitution(vector_local.array_w)
            u.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)  # type: ignore

    def homogenize(self, u: _fem.Function) -> None:
        """
        For a vector, homogenize (set to zero) the vector components at the multi-point
        constraint slave DoF indices. This is particularly useful for nonlinear problems.

        Args:
            u: The input vector
        """
        self._cpp_object.homogenize(u.x.array)
        u.x.scatter_forward()

    def _raise_if_finalized(self):
        """
        Raise if the multi point constraint has already been finalized
        """
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")

    def _raise_if_not_finalized(self):
        """
        Raise if the multi point constraint has not yet been finalized
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")


def _marked_dofs(
    V: _fem.FunctionSpace, marker: Callable[[numpy.ndarray], numpy.ndarray]
) -> tuple[npt.NDArray[numpy.int32], npt.NDArray[numpy.int32], npt.NDArray[numpy.floating], int]:
    """The dofs of `V` at the coordinates marked by `marker`, owned and ghosts.

    Returns:
        Their local, unrolled index in the space `V` is an uncollapsed subspace of (or in `V`),
        their component in `V`, their coordinates `(n, 3)`, and the number of components of `V`.
        The component is that of `V`, from its collapsed numbering: the block size of the parent
        need not be that of `V` (a component of a blocked space, or a block of a mixed space).
    """
    if len(V.component()) > 0:
        V_c = V.collapse()[0]
        bs = V_c.dofmap.index_map_bs
        parent_dofs, sub_dofs = _fem.locate_dofs_geometrical((V, V_c), marker)
        x = V_c.tabulate_dof_coordinates()[sub_dofs // bs]
        return parent_dofs, sub_dofs % bs, x, bs
    bs = V.dofmap.index_map_bs
    blocks = _fem.locate_dofs_geometrical(V, marker)
    x = numpy.repeat(V.tabulate_dof_coordinates()[blocks], bs, axis=0)
    dofs = (blocks[:, None] * bs + numpy.arange(bs, dtype=numpy.int32)).ravel()
    return dofs, numpy.tile(numpy.arange(bs, dtype=numpy.int32), len(blocks)), x, bs


def dofs_at_point(
    V: _fem.FunctionSpace, point: npt.ArrayLike, distance_tol: float | None = None
) -> tuple[npt.NDArray[numpy.int64], int]:
    """The global dofs of the block of `V` at a point, and the process owning them.

    The dofs are in the global, unrolled numbering of the space `V` is an uncollapsed subspace of
    (or of `V`), as masters are given to :meth:`MultiPointConstraint.add_constraint` and
    :meth:`MultiPointConstraint.extend_masters`: one per component of `V`. Works for a space on a
    point mesh too.

    Args:
        V: The function space, or an uncollapsed subspace of it
        point: The point, of `gdim` or 3 coordinates
        distance_tol: The largest distance from the point to the dof. Defaults to `500` machine
            epsilon of the coordinate type of the mesh.

    Returns:
        The dofs, one per component, and the owning process. The same on every process.

    Raises:
        ValueError: On every process, unless exactly one block of dofs of `V` is at the point:
            none, or several, as in a discontinuous space or on a point mesh with coinciding
            points.

    Note:
        Collective.
    """
    mesh = V.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim
    if distance_tol is None:
        distance_tol = _tolerance(None, mesh.geometry.x.dtype)
    p = numpy.zeros(3, dtype=numpy.float64)
    given = numpy.asarray(point, dtype=numpy.float64).reshape(-1)
    p[: len(given)] = given

    def at_point(x):
        return numpy.linalg.norm(x[:gdim].T - p[:gdim], axis=1) <= distance_tol

    imap = V.dofmap.index_map
    parent_bs = V.dofmap.index_map_bs
    parent_dofs, components, _, bs = _marked_dofs(V, at_point)
    owned = parent_dofs < imap.size_local * parent_bs

    # The point must hold exactly one block of V, counted on the process owning it, which holds all
    # its components. A discontinuous space, or a point mesh with coinciding points, may have several.
    num_blocks = numpy.array([owned.sum() // bs], dtype=numpy.int64)
    comm.Allreduce(_MPI.IN_PLACE, num_blocks, op=_MPI.SUM)
    if num_blocks[0] != 1:
        raise ValueError(f"{num_blocks[0]} blocks of degrees of freedom of the space at {given}, not one")

    # The dofs and the owner, in one reduction
    found = numpy.full(bs + 1, -1, dtype=numpy.int64)
    found[components[owned]] = imap.local_range[0] * parent_bs + parent_dofs[owned]
    if owned.any():
        found[bs] = comm.rank
    comm.Allreduce(_MPI.IN_PLACE, found, op=_MPI.MAX)
    return found[:bs], int(found[bs])


def finalize_multipointconstraints(
    mpcs: Sequence[MultiPointConstraint], filter: Optional[numpy.floating] = None
) -> None:
    """
    Finalize the multi point constraints of several function spaces together.

    Entry ``k`` of ``mpcs`` constrains its own function space, for instance the ``k``-th block of a
    :class:`ufl.MixedFunctionSpace`. Each is finalized as by :meth:`MultiPointConstraint.finalize`,
    but the checks that need communication are reduced once for all of them, and the meshes of the
    spaces may be distinct, as long as they live on congruent communicators.

    Args:
        mpcs: The constraints to finalize. None may be finalized already, and they must all use the
            same ``dtype``.
        filter: See :meth:`MultiPointConstraint.finalize`. Applied to every constraint.

    Raises:
        ValueError: If the input is inconsistent, or if a dof is both a slave and constrained by a
            Dirichlet condition, a master is also a slave, or the meshes are on communicators
            of different size or rank order. Raised on every process.

    Note:
        Collective. Must be called by every process, with the constraints in the same order.
    """
    mpcs = list(mpcs)
    if len(mpcs) == 0:
        raise ValueError("At least one constraint is required")
    if len({id(mpc) for mpc in mpcs}) != len(mpcs):
        raise ValueError("The same constraint was given more than once")
    for mpc in mpcs:
        mpc._raise_if_finalized()
    dtype = numpy.dtype(mpcs[0]._dtype)
    if any(numpy.dtype(mpc._dtype) != dtype for mpc in mpcs):
        raise ValueError("All constraints must have the same dtype")
    if dtype.type not in (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128):
        raise ValueError(f"Unsupported dtype {dtype} for coefficients")

    # An RBE3 constraint is built now, when the block of each space of its feet is known
    for mpc in mpcs:
        if len(mpc._rbe3) > 0:
            mpc._build_rbe3(mpcs)

    rhs_coeffs = []
    for mpc in mpcs:
        if mpc._rhs_coeffs is None:
            rhs_coeffs.append(numpy.zeros(0, dtype=dtype))
        else:
            num_dofs_local = mpc.V.dofmap.index_map_bs * (
                mpc.V.dofmap.index_map.size_local + mpc.V.dofmap.index_map.num_ghosts
            )
            rhs_coeffs.append(mpc._rhs_coeffs.x.array[:num_dofs_local].astype(dtype))

    def block_of(space: _fem.FunctionSpace) -> int:
        matches = [j for j, other in enumerate(mpcs) if other.V is space]
        if len(matches) == 0:
            matches = [j for j, other in enumerate(mpcs) if other.V.contains(space)]
        if len(matches) != 1:
            raise ValueError(
                "The master space of a constraint must be the function space, or an uncollapsed subspace of "
                "the function space, of exactly one of the constraints finalized together with it"
            )
        return matches[0]

    # The block of each master, from its code (see MultiPointConstraint.__init__). Every process
    # records the same spaces in the same order, so a space that is not one of the blocks raises
    # everywhere.
    master_blocks = []
    for k, mpc in enumerate(mpcs):
        codes = numpy.asarray(mpc._data.master_blocks)
        blocks = codes.copy()
        blocks[codes == -1] = k
        for s, space in enumerate(mpc._master_spaces):
            blocks[codes == -2 - s] = block_of(space)
        if len(mpc._master_spaces) == 0 and (codes == -1).all():
            blocks = numpy.zeros(0, dtype=numpy.int32)
        master_blocks.append(blocks)

    # Raises ValueError (as the C++ throws std::invalid_argument), identically on every process
    cpp_objects = dolfinx_mpc.cpp.mpc.create_multipointconstraints(
        [mpc.V._cpp_object for mpc in mpcs],
        # Copies: the arrays of the rows are read-only views of their C++ object
        [numpy.array(mpc._data.slaves) for mpc in mpcs],
        [numpy.array(mpc._data.masters) for mpc in mpcs],
        [mpc._data.coeffs.astype(dtype) for mpc in mpcs],
        [numpy.array(mpc._data.owners) for mpc in mpcs],
        [numpy.array(mpc._data.offsets) for mpc in mpcs],
        rhs_coeffs,
        [[bc._cpp_object for bc in mpc._bcs] for mpc in mpcs],
        master_blocks,
        filter,
    )

    # The block of each space on a spider mesh, matched before the spaces are replaced
    for mpc in mpcs:
        for entry in mpc._rbe2:
            entry[2] = next(j for j, other in enumerate(mpcs) if other.V is entry[0])

    for mpc, cpp_object in zip(mpcs, cpp_objects):
        mpc._cpp_object = cpp_object
        # Replace function space
        mpc.V = _fem.FunctionSpace(mpc.V.mesh, mpc.V.ufl_element(), cpp_object.function_space)
        mpc.finalized = True
        # Delete variables that are no longer required
        del (mpc._data, mpc._master_spaces)

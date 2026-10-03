# Copyright (C) 2020-2023 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy
import numpy.typing as npt
import ufl
from dolfinx import default_real_type, default_scalar_type

import dolfinx_mpc.cpp
from .container import MPCData, _float_array_types, _mpc_data_classes, _mpc_classes, _float_classes
from .dictcondition import create_dictionary_constraint
from .integralcondition import create_integral_constraint
from .rbe import create_rbe2


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
        dtype: The dtype of the underlying functions
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

    _slaves: npt.NDArray[numpy.int32]
    _masters: npt.NDArray[numpy.int64]
    _coeffs: _float_array_types
    _owners: npt.NDArray[numpy.int32]
    _offsets: npt.NDArray[numpy.int32]
    _master_spaces: List[tuple[npt.NDArray[numpy.int32], Optional[_fem.FunctionSpace]]]
    _bcs: List[_fem.DirichletBC]
    _rhs_coeffs: Optional[_fem.Function]
    _scale_function: Optional[_fem.Function]
    V: _fem.FunctionSpace
    finalized: bool
    _cpp_object: _mpc_classes
    _dtype: npt.DTypeLike
    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        V: _fem.FunctionSpace,
        dtype: npt.DTypeLike = default_scalar_type,
        bcs: Optional[List[_fem.DirichletBC]] = None,
        rhs_coeffs: Optional[_fem.Function] = None,
    ):
        self._slaves = numpy.array([], dtype=numpy.int32)
        self._masters = numpy.array([], dtype=numpy.int64)
        self._coeffs = numpy.array([], dtype=dtype)  # type: ignore
        self._owners = numpy.array([], dtype=numpy.int32)
        self._offsets = numpy.array([0], dtype=numpy.int32)
        self._master_spaces = []
        self._bcs = [] if bcs is None else list(bcs)
        if rhs_coeffs is not None:
            if not rhs_coeffs.x.array.dtype == dtype:
                raise ValueError("rhs_coeffs must have the same dtype as the MPC")
            if rhs_coeffs.function_space != V:
                raise ValueError("rhs_coeffs must be a Function in the space of the constraint")
        self._rhs_coeffs = rhs_coeffs
        self._scale_function = None
        self.V = V
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
                :func:`finalize_multipointconstraints`, and `masters` is in its global numbering.
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

        # Recorded on every process, also without local slaves, so that every process resolves the
        # blocks of the masters the same way when the constraints are finalized
        if master_blocks is not None:
            self._master_spaces.append((numpy.asarray(master_blocks, dtype=numpy.int32), None))
        elif master_space is not None:
            self._master_spaces.append((numpy.full(len(masters), -1, dtype=numpy.int32), master_space))
        else:
            self._master_spaces.append((numpy.full(len(masters), -2, dtype=numpy.int32), None))
        if len(slaves) > 0:
            self._offsets = numpy.append(self._offsets, offsets[1:] + len(self._masters))
            self._slaves = numpy.append(self._slaves, slaves)
            self._masters = numpy.append(self._masters, masters)
            self._coeffs = numpy.array(numpy.append(self._coeffs, coeffs), dtype=self._dtype)
            self._owners = numpy.append(self._owners, owners)

    def add_integral_constraint(
        self,
        weight_form,
        value,
        bcs: Optional[List[_fem.DirichletBC]] = None,
        rtol: numpy.floating | float | None = None,
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
            rtol: Discard a master whose coefficient is below this fraction of
                the largest one. Defaults to
                :func:`dolfinx_mpc.create_integral_constraint`'s own default,
                which scales with the runtime scalar type's precision.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_finalized()
        kwargs = {} if rtol is None else {"rtol": rtol}
        slaves, masters, coeffs, owners, offsets, rhs = create_integral_constraint(
            self.V, weight_form, value, self._bcs if bcs is None else bcs, **kwargs
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
        :meth:`add_constraint` for `master_space`.
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
        scale: _float_classes = default_scalar_type(1.0),  # type: ignore
        tol: Optional[_float_classes] = 500 * numpy.finfo(default_real_type).eps,
        num_threads: Optional[int] = 1,
    ):
        """
        Create periodic condition for all closure dofs of on all entities in `meshtag` with value `tag`.
        :math:`u(x_i) = scale * u(relation(x_i))` for all of :math:`x_i` on marked entities.

        Args:
            V: The function space to assign the condition to. Should either be the space of the MPC or a sub space.
               meshtag: MeshTag for entity to apply the periodic condition on
            tag: Tag indicating which entities should be slaves
            relation: Lambda-function describing the geometrical relation
            bcs: Dirichlet boundary conditions for the problem (Periodic constraints will be ignored for these dofs)
            scale: Float for scaling bc
            tol: Tolerance for adding scaled basis values to MPC. Any contribution that is less than this value
                is ignored. The tolerance is also added as padding for the bounding box trees and corresponding
                collision searches to determine periodic degrees of freedom. With `None`, every basis value is
                kept, so that the coefficients can later be changed with :func:`scale_coefficients` or
                :func:`update_coefficients` without having lost masters. The padding then defaults to
                `500` machine epsilon.
            num_threads: The number of threads to use for certain operations
        """
        bcs_ = [bc._cpp_object for bc in bcs]
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        tol_ = None if tol is None else float(tol)
        if V is self.V:
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_topological(
                self.V._cpp_object,
                meshtag._cpp_object,
                tag,
                relation,
                bcs_,
                scale,
                False,
                tol_,
                num_threads=num_threads,
            )
        elif self.V.contains(V):
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_topological(
                V._cpp_object,
                meshtag._cpp_object,
                tag,
                relation,
                bcs_,
                scale,
                True,
                tol_,
                num_threads=num_threads,
            )
        else:
            raise RuntimeError("The input space has to be a sub space (or the full space) of the MPC")
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_periodic_constraint_geometrical(
        self,
        V: _fem.FunctionSpace,
        indicator: Callable[[numpy.ndarray], numpy.ndarray],
        relation: Callable[[numpy.ndarray], numpy.ndarray],
        bcs: List[_fem.DirichletBC],
        scale: _float_classes = default_scalar_type(1.0),  # type: ignore
        tol: Optional[_float_classes] = 500 * numpy.finfo(default_real_type).eps,
        num_threads: Optional[int] = 1,
    ):
        """
        Create a periodic condition for all degrees of freedom whose physical location satisfies
        :math:`indicator(x_i)==True`, i.e.
        :math:`u(x_i) = scale * u(relation(x_i))` for all :math:`x_i`

        Args:
            V: The function space to assign the condition to. Should either be the space of the MPC or a sub space.
            indicator: Lambda-function to locate degrees of freedom that should be slaves
            relation: Lambda-function describing the geometrical relation to master dofs
            bcs: Dirichlet boundary conditions for the problem
                 (Periodic constraints will be ignored for these dofs)
            scale: Float for scaling bc
            tol: Tolerance for adding scaled basis values to MPC. Any contribution that is less than this value
                is ignored. The tolerance is also added as padding for the bounding box trees and corresponding
                collision searches to determine periodic degrees of freedom. With `None`, every basis value is
                kept, so that the coefficients can later be changed with :func:`scale_coefficients` or
                :func:`update_coefficients` without having lost masters. The padding then defaults to
                `500` machine epsilon.
            num_threads: The number of threads to use for certain operations.
        """
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        tol_ = None if tol is None else float(tol)
        bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        if V is self.V:
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_geometrical(
                self.V._cpp_object, indicator, relation, bcs, scale, False, tol_, num_threads
            )
        elif self.V.contains(V):
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_geometrical(
                V._cpp_object, indicator, relation, bcs, scale, True, tol_, num_threads
            )
        else:
            raise RuntimeError("The input space has to be a sub space (or the full space) of the MPC")
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def add_rbe2_topological(
        self,
        dim: int,
        entities: npt.NDArray[numpy.int32],
        W: _fem.FunctionSpace,
        map: Union[int, npt.NDArray[numpy.integer]] = 0,
    ):
        r"""
        Tie the dofs on mesh entities rigidly to a point, as the RBE2 element of other codes
        (a rigid "spider").

        Each dof of this constraint's space on `entities` is a "foot" of a spider whose "body" is a
        point of the point mesh of `W`. Every component of a foot follows the motion of its body,

        .. math::

            u(x) = t + \theta \times (x - x_c),

        where :math:`x_c` is the coordinate of the point, :math:`t` its translation and
        :math:`\theta` its rotation, the dofs of `W` at the point. Without rotations,
        :math:`u(x) = t`.

        Args:
            dim: Topological dimension of the entities
            entities: Entities (local to the process) whose dofs are tied
            W: Space on the spider mesh (:func:`dolfinx_mpc.create_spider_mesh`). Its value size
                is the geometric dimension, for translations only, or 6 in 3D and 3 in 2D, for
                translations and rotations. `W` must be the space of another constraint
                finalized together with this one by :func:`finalize_multipointconstraints`.
            map: The spider each entity is tied to: the index of its point, see
                :func:`dolfinx_mpc.locate_spider`. One integer for all entities, or one per
                entity.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_finalized()
        entities = numpy.asarray(entities, dtype=numpy.int32)
        points = numpy.broadcast_to(numpy.asarray(map, dtype=numpy.int64), entities.shape)
        dofs, dof_points = [], []
        # The dofs of each spider in turn, so that a dof on entities of two spiders is caught as
        # constrained twice
        for point in numpy.unique(numpy.concatenate(self.V.mesh.comm.allgather(numpy.unique(points)))):
            dofs_p = _fem.locate_dofs_topological(self.V, dim, entities[points == point])
            dofs.append(dofs_p)
            dof_points.append(numpy.full(len(dofs_p), point, dtype=numpy.int64))
        mpc_data = create_rbe2(
            self.V,
            numpy.concatenate(dofs) if dofs else numpy.zeros(0, dtype=numpy.int32),
            numpy.concatenate(dof_points) if dof_points else numpy.zeros(0, dtype=numpy.int64),
            W,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data, master_space=W)

    def add_rbe2_geometrical(
        self,
        locator: Callable[[numpy.ndarray], numpy.ndarray],
        W: _fem.FunctionSpace,
        map: Union[int, Callable[[numpy.ndarray], numpy.ndarray]] = 0,
    ):
        r"""
        Tie the dofs located by `locator` rigidly to a point, as the RBE2 element of other codes
        (a rigid "spider"). See :meth:`add_rbe2_topological` for the relation.

        Args:
            locator: Marks the dofs to tie, given their coordinates, shape `(3, num_points)`
            W: Space on the spider mesh, see :meth:`add_rbe2_topological`
            map: The spider each dof is tied to: the index of its point, see
                :func:`dolfinx_mpc.locate_spider`. One integer for all dofs, or a function of the
                coordinates, shape `(3, num_points)`, returning one index per dof.

        Note:
            Collective. Must be called by every process.
        """
        self._raise_if_finalized()
        dofs = _fem.locate_dofs_geometrical(self.V, locator)
        if callable(map):
            x = self.V.tabulate_dof_coordinates()[dofs].T
            points = numpy.asarray(map(x), dtype=numpy.int64).reshape(-1)
        else:
            points = numpy.full(len(dofs), map, dtype=numpy.int64)
        mpc_data = create_rbe2(self.V, numpy.asarray(dofs, dtype=numpy.int32), points, W)
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data, master_space=W)

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
            space: Function space (possible sub space) for the current constraint
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
            raise ValueError("Input space has to be a sub space of the MPC space")
        mpc_data = dolfinx_mpc.cpp.mpc.create_slip_condition(
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
            self.V, slave_master_dict, subspace_slave, subspace_master
        )
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_contact_slip_condition(
        self,
        meshtags: _mesh.MeshTags,
        slave_marker: int,
        master_marker: int,
        normal: _fem.Function,
        eps2: float = 1e-20,
        num_threads: Optional[int] = 1,
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
            eps2: The tolerance for the squared distance between cells to be considered as a collision
            num_threads: The number of threads to use for certain operations
        """
        if isinstance(eps2, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            eps2 = eps2.item()  # type: ignore
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_slip_condition(
            self.V._cpp_object, meshtags._cpp_object, slave_marker, master_marker, normal._cpp_object, eps2, num_threads
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    def create_contact_inelastic_condition(
        self,
        meshtags: _cpp.mesh.MeshTags_int32,
        slave_marker: int,
        master_marker: int,
        eps2: float = 1e-20,
        allow_missing_masters: bool = False,
        num_threads: Optional[int] = 1,
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
            eps2: The tolerance for the squared distance between cells to be considered as a collision
            allow_missing_masters: If true, the function will not throw an error if a degree of freedom
                in the closure of the master entities does not have a corresponding set of slave degree
                of freedom.
            num_threads: The number of threads to use for certain operations
        """
        if isinstance(eps2, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            eps2 = eps2.item()  # type: ignore
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_inelastic_condition(
            self.V._cpp_object,
            meshtags._cpp_object,
            slave_marker,
            master_marker,
            eps2,
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

        The masters are fixed at creation. A master dropped by `tol` or by the `filter` of
        :func:`finalize` cannot be given a coefficient, so create the constraint with `tol=None`
        and no filter if the coefficients are to be changed.

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

    rhs_coeffs = []
    for mpc in mpcs:
        if mpc._rhs_coeffs is None:
            rhs_coeffs.append(numpy.zeros(0, dtype=dtype))
        else:
            num_dofs_local = mpc.V.dofmap.index_map_bs * (
                mpc.V.dofmap.index_map.size_local + mpc.V.dofmap.index_map.num_ghosts
            )
            rhs_coeffs.append(mpc._rhs_coeffs.x.array[:num_dofs_local].astype(dtype))

    # The block of each master: -2 marks the constraint's own block, -1 the block of the space
    # recorded with it, anything else a block given directly. Every process records the same chunks
    # with the same spaces, so a space that is not one of the blocks raises everywhere.
    master_blocks = []
    for k, mpc in enumerate(mpcs):
        if all(space is None and (blocks == -2).all() for blocks, space in mpc._master_spaces):
            master_blocks.append(numpy.zeros(0, dtype=numpy.int32))
            continue
        resolved = []
        for blocks, space in mpc._master_spaces:
            blocks = blocks.copy()
            blocks[blocks == -2] = k
            if space is not None:
                matches = [j for j, other in enumerate(mpcs) if other.V is space]
                if len(matches) != 1:
                    raise ValueError(
                        "The master space of a constraint must be the function space of exactly one of the "
                        "constraints finalized together with it"
                    )
                blocks[blocks == -1] = matches[0]
            resolved.append(blocks)
        master_blocks.append(numpy.concatenate(resolved) if resolved else numpy.zeros(0, dtype=numpy.int32))

    # Raises ValueError (as the C++ throws std::invalid_argument), identically on every process
    cpp_objects = dolfinx_mpc.cpp.mpc.create_multipointconstraints(
        [mpc.V._cpp_object for mpc in mpcs],
        [mpc._slaves for mpc in mpcs],
        [mpc._masters for mpc in mpcs],
        [mpc._coeffs.astype(dtype) for mpc in mpcs],
        [mpc._owners for mpc in mpcs],
        [mpc._offsets for mpc in mpcs],
        rhs_coeffs,
        [[bc._cpp_object for bc in mpc._bcs] for mpc in mpcs],
        master_blocks,
        filter,
    )

    for mpc, cpp_object in zip(mpcs, cpp_objects):
        mpc._cpp_object = cpp_object
        # Replace function space
        mpc.V = _fem.FunctionSpace(mpc.V.mesh, mpc.V.ufl_element(), cpp_object.function_space)
        mpc.finalized = True
        # Delete variables that are no longer required
        del (mpc._slaves, mpc._masters, mpc._coeffs, mpc._owners, mpc._offsets, mpc._master_spaces)

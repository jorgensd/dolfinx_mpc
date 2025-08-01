# Copyright (C) 2020-2023 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from petsc4py import PETSc as _PETSc

import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy
import numpy.typing as npt
from dolfinx import default_real_type, default_scalar_type

import dolfinx_mpc.cpp

from .dictcondition import create_dictionary_constraint

_mpc_classes = Union[
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_double,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_float,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_double,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_float,
]
_float_classes = Union[numpy.float32, numpy.float64, numpy.complex128, numpy.complex64]
_float_array_types = Union[
    npt.NDArray[numpy.float32],
    npt.NDArray[numpy.float64],
    npt.NDArray[numpy.complex64],
    npt.NDArray[numpy.complex128],
]
_mpc_data_classes = Union[
    dolfinx_mpc.cpp.mpc.mpc_data_double,
    dolfinx_mpc.cpp.mpc.mpc_data_float,
    dolfinx_mpc.cpp.mpc.mpc_data_complex_double,
    dolfinx_mpc.cpp.mpc.mpc_data_complex_float,
]


class MPCData:
    _cpp_object: _mpc_data_classes

    def __init__(
        self,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        offsets: npt.NDArray[numpy.int32],
    ):
        if coeffs.dtype.type == numpy.float32:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_float(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.float64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_double(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.complex64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_float(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.complex128:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_double(slaves, masters, coeffs, owners, offsets)
        else:
            raise ValueError("Unsupported dtype {coeffs.dtype.type} for coefficients")

    @property
    def slaves(self):
        return self._cpp_object.slaves

    @property
    def masters(self):
        return self._cpp_object.masters

    @property
    def coeffs(self):
        return self._cpp_object.coeffs

    @property
    def owners(self):
        return self._cpp_object.owners

    @property
    def offsets(self):
        return self._cpp_object.offsets


class MultiPointConstraint:
    """
    Hold data for multi point constraint relation ships,
    including new index maps for local assembly of matrices and vectors.

    Args:
        V: The function space
        dtype: The dtype of the underlying functions
    """

    _slaves: npt.NDArray[numpy.int32]
    _masters: npt.NDArray[numpy.int64]
    _coeffs: _float_array_types
    _owners: npt.NDArray[numpy.int32]
    _offsets: npt.NDArray[numpy.int32]
    V: _fem.FunctionSpace
    finalized: bool
    _cpp_object: _mpc_classes
    _dtype: numpy.floating
    __slots__ = tuple(__annotations__)

    def __init__(self, V: _fem.FunctionSpace, dtype: numpy.floating = default_scalar_type):
        self._slaves = numpy.array([], dtype=numpy.int32)
        self._masters = numpy.array([], dtype=numpy.int64)
        self._coeffs = numpy.array([], dtype=dtype)  # type: ignore
        self._owners = numpy.array([], dtype=numpy.int32)
        self._offsets = numpy.array([0], dtype=numpy.int32)
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

        """
        assert V == self.V
        self._already_finalized()

        if len(slaves) > 0:
            self._offsets = numpy.append(self._offsets, offsets[1:] + len(self._masters))
            self._slaves = numpy.append(self._slaves, slaves)
            self._masters = numpy.append(self._masters, masters)
            self._coeffs = numpy.append(self._coeffs, coeffs)
            self._owners = numpy.append(self._owners, owners)

    def add_constraint_from_mpc_data(self, V: _fem.FunctionSpace, mpc_data: Union[_mpc_data_classes, MPCData]):
        """
        Add new constraint given by an `dolfinc_mpc.cpp.mpc.mpc_data`-object
        """
        self._already_finalized()
        self.add_constraint(
            V,
            mpc_data.slaves,
            mpc_data.masters,
            mpc_data.coeffs,
            mpc_data.owners,
            mpc_data.offsets,
        )

    def finalize(self) -> None:
        """
        Finializes the multi point constraint. After this function is called, no new constraints can be added
        to the constraint. This function creates a map from the cells (local to index) to the slave degrees of
        freedom and builds a new index map and function space where unghosted master dofs are added as ghosts.
        """
        self._already_finalized()
        self._coeffs.astype(numpy.dtype(self._dtype))
        # Initialize C++ object and create slave->cell maps
        if self._dtype == numpy.float32:
            self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint_float(
                self.V._cpp_object,
                self._slaves,
                self._masters,
                self._coeffs.astype(self._dtype),
                self._owners,
                self._offsets,
            )
        elif self._dtype == numpy.float64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint_double(
                self.V._cpp_object,
                self._slaves,
                self._masters,
                self._coeffs.astype(self._dtype),
                self._owners,
                self._offsets,
            )
        elif self._dtype == numpy.complex64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_float(
                self.V._cpp_object,
                self._slaves,
                self._masters,
                self._coeffs.astype(self._dtype),
                self._owners,
                self._offsets,
            )

        elif self._dtype == numpy.complex128:
            self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_double(
                self.V._cpp_object,
                self._slaves,
                self._masters,
                self._coeffs.astype(self._dtype),
                self._owners,
                self._offsets,
            )
        else:
            raise ValueError("Unsupported dtype {coeffs.dtype.type} for coefficients")

        # Replace function space
        self.V = _fem.FunctionSpace(self.V.mesh, self.V.ufl_element(), self._cpp_object.function_space)

        self.finalized = True
        # Delete variables that are no longer required
        del (self._slaves, self._masters, self._coeffs, self._owners, self._offsets)

    def create_periodic_constraint_topological(
        self,
        V: _fem.FunctionSpace,
        meshtag: _mesh.MeshTags,
        tag: int,
        relation: Callable[[numpy.ndarray], numpy.ndarray],
        bcs: List[_fem.DirichletBC],
        scale: _float_classes = default_scalar_type(1.0),
        tol: _float_classes = 500 * numpy.finfo(default_real_type).eps,
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
                collision searches to determine periodic degrees of freedom.
        """
        bcs_ = [bc._cpp_object for bc in bcs]
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        if V is self.V:
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_topological(
                self.V._cpp_object, meshtag._cpp_object, tag, relation, bcs_, scale, False, float(tol)
            )
        elif self.V.contains(V):
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_topological(
                V._cpp_object, meshtag._cpp_object, tag, relation, bcs_, scale, True, float(tol)
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
        scale: _float_classes = default_scalar_type(1.0),
        tol: _float_classes = 500 * numpy.finfo(default_real_type).eps,
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
                collision searches to determine periodic degrees of freedom.
        """
        if isinstance(scale, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            scale = scale.item()  # type: ignore
        bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        if V is self.V:
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_geometrical(
                self.V._cpp_object, indicator, relation, bcs, scale, False, float(tol)
            )
        elif self.V.contains(V):
            mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint_geometrical(
                V._cpp_object, indicator, relation, bcs, scale, True, float(tol)
            )
        else:
            raise RuntimeError("The input space has to be a sub space (or the full space) of the MPC")
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

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

                cellname = mesh.ufl_cell().cellname()
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

                cellname = mesh.ufl_cell().cellname()
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
        """
        if isinstance(eps2, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            eps2 = eps2.item()  # type: ignore
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_slip_condition(
            self.V._cpp_object,
            meshtags._cpp_object,
            slave_marker,
            master_marker,
            normal._cpp_object,
            eps2,
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    def create_contact_inelastic_condition(
        self,
        meshtags: _cpp.mesh.MeshTags_int32,
        slave_marker: int,
        master_marker: int,
        eps2: float = 1e-20,
        allow_missing_masters: bool = False,
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
        """
        if isinstance(eps2, numpy.generic):  # nanobind conversion of numpy dtypes to general Python types
            eps2 = eps2.item()  # type: ignore
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_inelastic_condition(
            self.V._cpp_object, meshtags._cpp_object, slave_marker, master_marker, eps2, allow_missing_masters
        )
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    @property
    def is_slave(self) -> numpy.ndarray:
        """
        Returns a vector of integers where the ith entry indicates if a degree of freedom (local to process) is a slave.
        """
        self._not_finalized()
        return self._cpp_object.is_slave

    @property
    def slaves(self):
        """
        Returns the degrees of freedom for all slaves local to process
        """
        self._not_finalized()
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
        self._not_finalized()
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
        self._not_finalized()
        return self._cpp_object.coefficients()

    @property
    def num_local_slaves(self):
        """
        Return the number of slaves owned by the current process.
        """
        self._not_finalized()
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
        self._not_finalized()
        return self._cpp_object.cell_to_slaves

    @property
    def function_space(self):
        """
        Return the function space for the multi-point constraint with the updated index map
        """
        self._not_finalized()
        return self.V

    def backsubstitution(self, u: Union[_fem.Function, _PETSc.Vec]) -> None:  # type: ignore
        """
        For a Function, impose the multi-point constraint by backsubstiution.
        This function is used after solving the reduced problem to obtain the values
        at the slave degrees of freedom

        .. note::
            It is the users responsibility to destroy the PETSc vector

        Args:
            u: The input function
        """
        try:
            self._cpp_object.backsubstitution(u.x.array)
            u.x.scatter_forward()
        except AttributeError:
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

    def _already_finalized(self):
        """
        Check if we have already finalized the multi point constraint
        """
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")

    def _not_finalized(self):
        """
        Check if we have finalized the multi point constraint
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")

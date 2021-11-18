# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import Callable, List, Union, Dict

import dolfinx
import numpy
from petsc4py import PETSc

import dolfinx_mpc.cpp
from dolfinx.cpp.fem import Form_complex128 as Form_C, Form_float64 as Form_R
from .dictcondition import create_dictionary_constraint
from .periodic_condition import create_periodic_condition_topological, create_periodic_condition_geometrical


def cpp_dirichletbc(bc):
    """Unwrap Dirichlet BC objects as cpp objects"""
    if isinstance(bc, dolfinx.DirichletBC):
        return bc._cpp_object
    elif isinstance(bc, (tuple, list)):
        return list(map(lambda sub_bc: cpp_dirichletbc(sub_bc), bc))
    return bc


class MultiPointConstraint():
    """
    Multi-point constraint class.
    This class will hold data for multi point constraint relation ships,
    including new index maps for local assembly of matrices and vectors.
    """

    def __init__(self, V: dolfinx.fem.FunctionSpace):
        """
        Initialize the multi point constraint for a given function space.
        """
        self._slaves = numpy.array([], dtype=numpy.int32)
        self._masters = numpy.array([], dtype=numpy.int64)
        self._coeffs = numpy.array([], dtype=PETSc.ScalarType)
        self._owners = numpy.array([], dtype=numpy.int32)
        self._offsets = numpy.array([0], dtype=numpy.int32)
        self.V = V
        self.finalized = False

    def add_constraint(self, V: dolfinx.FunctionSpace, slaves_: "numpy.ndarray[numpy.int32]",
                       masters_: "numpy.ndarray[numpy.int64]", coeffs_: "numpy.ndarray[PETSc.ScalarType]",
                       owners_: "numpy.ndarray[numpy.int32]", offsets_: "numpy.ndarray[numpy.int32]"):
        """
        Add new constraint given by numpy arrays.
        Parameters
        ----------
            V
                The function space for the constraint
            slaves_
                List of all slave dofs (using local dof numbering) on this process
            masters_
                List of all master dofs (using global dof numbering) on this process
            coeffs_
                The coefficients corresponding to each master.
            owners_
                The process each master is owned by.
            offsets_
                Array indicating the location in the masters array for the i-th slave
                in the slaves arrays. I.e.
                masters_of_owned_slave[i] = masters[offsets[i]:offsets[i+1]]

        """
        assert(V == self.V)
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")

        if len(slaves_) > 0:
            self._offsets = numpy.append(self._offsets, offsets_[1:] + len(self._masters))
            self._slaves = numpy.append(self._slaves, slaves_)
            self._masters = numpy.append(self._masters, masters_)
            self._coeffs = numpy.append(self._coeffs, coeffs_)
            self._owners = numpy.append(self._owners, owners_)

    def add_constraint_from_mpc_data(self, V: dolfinx.FunctionSpace, mpc_data: dolfinx_mpc.cpp.mpc.mpc_data):
        """
        Add new constraint given by an `dolfinc_mpc.cpp.mpc.mpc_data`-object
        """
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")
        self.add_constraint(V, mpc_data.slaves, mpc_data.masters, mpc_data.coeffs, mpc_data.owners, mpc_data.offsets)

    def finalize(self) -> None:
        """
        Finializes the multi point constraint. After this function is called, no new constraints can be added
        to the constraint. This function creates a map from the cells (local to index) to the slave degrees of
        freedom and builds a new index map and function space where unghosted master dofs are added as ghosts.
        """
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")

        # Initialize C++ object and create slave->cell maps
        self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint(
            self.V._cpp_object, self._slaves, self._masters, self._coeffs, self._owners, self._offsets)
        # Replace function space
        self.V = dolfinx.FunctionSpace(None, self.V.ufl_element(), self._cpp_object.function_space)

        self.finalized = True
        # Delete variables that are no longer required
        del (self._slaves, self._masters, self._coeffs, self._owners, self._offsets)

    def create_periodic_constraint_topological(self, meshtag: dolfinx.MeshTags, tag: int,
                                               relation: Callable[[numpy.ndarray], numpy.ndarray],
                                               bcs: list([dolfinx.DirichletBC]), scale: PETSc.ScalarType = 1):
        """
        Create periodic condition for all dofs in MeshTag with given marker:
        u(x_i) = scale * u(relation(x_i))
        for all x_i on marked entities.

        Parameters
        ----------
        meshtag
            MeshTag for entity to apply the periodic condition on
        tag
            Tag indicating which entities should be slaves
        relation
            Lambda-function describing the geometrical relation
        bcs
            Dirichlet boundary conditions for the problem
            (Periodic constraints will be ignored for these dofs)
        scale
            Float for scaling bc
        """
        slaves, masters, coeffs, owners, offsets = create_periodic_condition_topological(
            self.V, meshtag, tag, relation, bcs, scale)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_periodic_constraint_geometrical(self, V: dolfinx.FunctionSpace,
                                               indicator: Callable[[numpy.ndarray], numpy.ndarray],
                                               relation: Callable[[numpy.ndarray], numpy.ndarray],
                                               bcs: List[dolfinx.DirichletBC], scale: PETSc.ScalarType = 1):
        """
        Create a periodic condition for all degrees of freedom whose physical location satisfies indicator(x)
        u(x_i) = scale * u(relation(x_i)) for all x_i where indicator(x_i) == True

        Parameters
        ----------
        indicator
            Lambda-function to locate degrees of freedom that should be slaves
        relation
            Lambda-function describing the geometrical relation to master dofs
        bcs
            Dirichlet boundary conditions for the problem
            (Periodic constraints will be ignored for these dofs)
        scale
            Float for scaling bc
        """
        slaves, masters, coeffs, owners, offsets = create_periodic_condition_geometrical(
            self.V, indicator, relation, bcs, scale)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_slip_constraint(self, facet_marker: tuple([dolfinx.MeshTags, int]), v: dolfinx.Function,
                               sub_space: dolfinx.FunctionSpace = None, sub_map: numpy.ndarray = numpy.array([]),
                               bcs: list([dolfinx.DirichletBC]) = []):
        """
        Create a slip constraint dot(u, v)=0 over the entities defined in a dolfinx.Meshtags
        marked with index i. normal is the normal vector defined as a vector function.

        Parameters
        ----------
        facet_marker
            Tuple containg the mesh tag and marker used to locate degrees of freedom that should be constrained
        v
            Dolfin function containing the directional vector to dot your slip condition (most commonly a normal vector)
        sub_space
           If the vector v is in a sub space of the multi point function space, supply the sub space
        sub_map
           Map from sub-space to parent space
        bcs
           List of Dirichlet BCs (slip conditions will be ignored on these dofs)

        Example
        -------
        Create constaint dot(u, n)=0 of all indices in mt marked with i
             create_slip_constaint((mt,i), n)

        Create slip constaint for u when in a sub space:
             me = MixedElement(VectorElement("CG", triangle, 2), FiniteElement("CG", triangle 1))
             W = FunctionSpace(mesh, me)
             V, V_to_W = W.sub(0).collapse(True)
             n = Function(V)
             create_slip_constraint((mt, i), normal, V, V_to_W, bcs=[])

        A slip condition cannot be applied on the same degrees of freedom as a Dirichlet BC, and therefore
        any Dirichlet bc for the space of the multi point constraint should be supplied.
             me = MixedElement(VectorElement("CG", triangle, 2), FiniteElement("CG", triangle 1))
             W = FunctionSpace(mesh, me)
             W0 = W.sub(0)
             V, V_to_W = W0.collapse(True)
             n = Function(V)
             bc = dolfinx.DirichletBC(inlet_velocity, dofs, W0)
             create_slip_constraint((mt, i), normal, V, V_to_W, bcs=[bc])
        """
        if sub_space is None:
            W = [self.V._cpp_object]
        else:
            W = [self.V._cpp_object, sub_space._cpp_object]
        mesh_tag, marker = facet_marker
        mpc_data = dolfinx_mpc.cpp.mpc.create_slip_condition(W, mesh_tag, marker, v._cpp_object,
                                                             numpy.asarray(sub_map, dtype=numpy.int32),
                                                             cpp_dirichletbc(bcs))
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_general_constraint(self, slave_master_dict: Dict[bytes, Dict[bytes, float]],
                                  subspace_slave: int = None, subspace_master: int = None):
        """
        Parameters
        ----------
        V
            The function space
        slave_master_dict
            Nested dictionary, where the first key is the bit representing the slave dof's coordinate in the mesh.
            The item of this key is a dictionary, where each key of this dictionary is the bit representation
            of the master dof's coordinate, and the item the coefficient for the MPC equation.
        subspace_slave
            If using mixed or vector space, and only want to use dofs from a sub space as slave add index here
        subspace_master
            Subspace index for mixed or vector spaces

        Example
        -------
        If the dof D located at [d0,d1] should be constrained to the dofs
        E and F at [e0,e1] and [f0,f1] as
        D = alpha E + beta F
        the dictionary should be:
            {numpy.array([d0, d1], dtype=numpy.float64).tobytes():
                {numpy.array([e0, e1], dtype=numpy.float64).tobytes(): alpha,
                numpy.array([f0, f1], dtype=numpy.float64).tobytes(): beta}}
        """
        slaves, masters, coeffs, owners, offsets = create_dictionary_constraint(
            self.V, slave_master_dict, subspace_slave, subspace_master)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_contact_slip_condition(self, meshtags: dolfinx.MeshTags, slave_marker: int, master_marker: int,
                                      normal: dolfinx.Function):
        """
        Create a slip condition between two sets of facets marker with individual markers.
        The interfaces should be within machine precision of eachother, but the vertices does not need to align.
        The condition created is dot(u_s, normal_s) = dot(u_m, normal_m) where s stands for the restriction to the
        slave facets, m to the master facets.

        Parameters
        ----------
        meshtags
            The meshtags of the set of facets to tie together
        slave_marker
            The marker of the slave facets
        master_marker
            The marker of the master facets
        normal
            The function used in the dot-product of the constraint
        """
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_slip_condition(
            self.V._cpp_object, meshtags, slave_marker, master_marker, normal._cpp_object)
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    def create_contact_inelastic_condition(self, meshtags: dolfinx.MeshTags, slave_marker: int, master_marker: int):
        """
        Create a contact inelastic condition between two sets of facets marker with individual markers.
        The interfaces should be within machine precision of eachother, but the vertices does not need to align.
        The condition created is u_s = u_m where s stands for the restriction to the
        slave facets, m to the master facets.

        Parameters
        ----------
        meshtags
            The meshtags of the set of facets to tie together
        slave_marker
            The marker of the slave facets
        master_marker
            The marker of the master facets
        """
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_inelastic_condition(
            self.V._cpp_object, meshtags, slave_marker, master_marker)
        self.add_constraint_from_mpc_data(self.V, mpc_data)

    @property
    def is_slave(self) -> numpy.ndarray:
        """
        Returns a vector of integers where the ith entry indicates if a degree of freedom (local to process) is a slave.
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.is_slave

    @property
    def slaves(self):
        """
        Returns the degrees of freedom for all slaves local to process
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.slaves

    @property
    def masters(self):
        """
        Returns an `dolfinx.cpp.graph.AdjacencyList_int32` whose ith node corresponds to
        a degree of freedom (local to process), and links the corresponding master dofs (local to process).

        Example
        -------
        masters = mpc.masters
        masters_of_dof_i = masters.links(i)
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.masters

    def coefficients(self):
        """
        Returns a vector containing the coefficients for the constraint, and the corresponding offsets
        for the ith degree of freedom.

        Example
        -------
        coeffs, offsets = mpc.coefficients()
        coeffs_of_slave_i = coeffs[offsets[i]:offsets[i+1]]
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.coefficients()

    @property
    def num_local_slaves(self):
        """
        Return the number of slaves owned by the current process.
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        if self.finalized:
            return self._cpp_object.num_local_slaves

    @property
    def cell_to_slaves(self):
        """
        Returns an `dolfinx.cpp.graph.AdjacencyList_int32` whose ith node corresponds to
        the ith cell (local to process), and links the corresponding slave degrees of
        freedom in the cell (local to process).

        Example
        -------
        cell_to_slaves = mpc.cell_to_slaves()
        slaves_in_cell_i = cell_to_slaves.links(i)
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.cell_to_slaves

    def create_sparsity_pattern(self, cpp_form: Union[Form_C, Form_R]):
        """
        Create sparsity-pattern for MPC given a compiled DOLFINx form

        Parameters
        ----------
        cpp_form
            The form
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return dolfinx_mpc.cpp.mpc.create_sparsity_pattern(cpp_form, self._cpp_object)

    @property
    def function_space(self):
        """
        Return the function space for the multi-point constraint with the updated index map
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        else:
            return self.V

    def backsubstitution(self, vector: PETSc.Vec) -> None:
        """
        For a vector, impose the multi-point constraint by backsubstiution.
        This function is used after solving the reduced problem to obtain the values
        at the slave degrees of freedom

        Parameters
        ----------
        vector
            The input vector
        """
        # Unravel data from constraint
        with vector.localForm() as vector_local:
            self._cpp_object.backsubstitution(vector_local.array_w)
        vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

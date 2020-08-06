import dolfinx_mpc.cpp
import warnings
import dolfinx

from .contactcondition import create_contact_condition
from .slipcondition import create_slip_condition
from .periodic_condition import create_periodic_condition
from .dictcondition import create_dictionary_constraint


class MultiPointConstraint():
    def __init__(self, V):
        self.local_slaves = []
        self.ghost_slaves = []
        self.local_masters = []
        self.ghost_masters = []
        self.local_coeffs = []
        self.ghost_coeffs = []
        self.local_owners = []
        self.ghost_owners = []
        self.offsets = [0]
        self.ghost_offsets = [0]
        self.V = V
        self.finalized = False

    def add_constraint(self, V, slaves,
                       masters, coeffs, owners, offsets):
        """
        Add a constraint to your multipoint-constraint.
        Takes in tuples of slaves, masters, coeffs, owners, and offsets.
        Each tuple consist of one list, where the first list are related
        to slaves owned by the processor, the second list is related
        to the ghost slaves.
        """
        assert(V == self.V)
        if self.finalized:
            raise RuntimeError(
                "MultiPointConstraint has already been finalized")
        local_slaves, ghost_slaves = slaves
        local_masters, ghost_masters = masters
        local_coeffs, ghost_coeffs = coeffs
        local_owners, ghost_owners = owners
        local_offsets, ghost_offsets = offsets
        if len(local_slaves) > 0:
            self.offsets.extend(offset+len(self.local_masters)
                                for offset in local_offsets[1:])
            self.local_slaves.extend(local_slaves)
            self.local_masters.extend(local_masters)
            self.local_coeffs.extend(local_coeffs)
            self.local_owners.extend(local_owners)
        if len(ghost_slaves) > 0:
            self.ghost_offsets.extend(offset+len(self.ghost_masters)
                                      for offset in ghost_offsets[1:])
            self.ghost_slaves.extend(ghost_slaves)
            self.ghost_masters.extend(ghost_masters)
            self.ghost_coeffs.extend(ghost_coeffs)
            self.ghost_owners.extend(ghost_owners)

    def finalize(self):
        """
        Finalizes a multi point constraint by adding all constraints together,
        locating slave cells and building a new index map with
        additional ghosts
        """
        if self.finalized:
            raise RuntimeError(
                "MultiPointConstraint has already been finalized")
        num_local_slaves = len(self.local_slaves)
        num_local_masters = len(self.local_masters)
        slaves = self.local_slaves
        masters = self.local_masters
        coeffs = self.local_coeffs
        owners = self.local_owners
        offsets = self.offsets
        # Merge local and ghost arrays
        if len(self.ghost_slaves) > 0:
            ghost_offsets = [g_offset + num_local_masters
                             for g_offset in self.ghost_offsets[1:]]
            slaves.extend(self.ghost_slaves)
            masters.extend(self.ghost_masters)
            coeffs.extend(self.ghost_coeffs)
            owners.extend(self.ghost_owners)
            offsets.extend(ghost_offsets)
        # Initialize C++ object and create slave->cell maps
        self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint(
            self.V._cpp_object, slaves, num_local_slaves)
        # Add masters and compute new index maps
        self._cpp_object.add_masters(masters, coeffs, owners, offsets)
        # Replace function space
        V_cpp = dolfinx.cpp.function.FunctionSpace(self.V.mesh,
                                                   self.V.element,
                                                   self._cpp_object.dofmap())
        self.V_mpc = dolfinx.FunctionSpace(None, self.V.ufl_element(), V_cpp)
        self.finalized = True
        # Delete variables that are no longer required
        del (self.local_slaves, self.local_masters,
             self.local_coeffs, self.local_owners, self.offsets,
             self.ghost_coeffs, self.ghost_masters, self.ghost_offsets,
             self.ghost_owners, self.ghost_slaves)

    def create_contact_constraint(self, meshtag, slave_marker, master_marker):
        """
        Create a contact constraint between two mesh entities,
        defined through markers.
        """
        slaves, masters, coeffs, owners, offsets = create_contact_condition(
            self.V, meshtag, slave_marker, master_marker)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_periodic_constraint(self, meshtag, tag, relation, bcs, scale=1):
        """
        Create a periodic boundary condition (possibly scaled).
        Input:
            meshtag: MeshTag for entity to apply the periodic condition on
            tag: Tag indicating which entities should be slaves
            relation: Lambda function describing the geometrical relation
            bcs: Dirichlet boundary conditions for the problem
               (Periodic constraints will be ignored for these dofs)
            scale: Float for scaling bc
        """
        slaves, masters, coeffs, owners, offsets = create_periodic_condition(
            self.V, meshtag, tag, relation, bcs, scale)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_slip_constraint(self, sub_space, normal, sub_map, entity_data,
                               bcs=[]):
        """
        Create a slip constraint dot(u,normal)=0 where u is either in the
        constrained
        space or a sub-space. Need to supply the map between the space of the
        normal function (from the collapsed sub space) and the constraint
        space,
        as well as which mesh entities this constraint shall be applied on
        """
        if sub_space is None:
            W = self.V
        else:
            W = (self.V, sub_space)
        slaves, masters, coeffs, owners, offsets = create_slip_condition(
            W, normal, sub_map, entity_data, bcs)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_general_constraint(self, slave_master_dict, subspace_slave=None,
                                  subspace_master=None):
        """
        Input:
            V - The function space
            slave_master_dict - The dictionary.
            subspace_slave - If using mixed or vector space,
                            and only want to use dofs from
                            a sub space as slave add index here
            subspace_master - Subspace index for mixed or vector spaces
        Example:
            If the dof D located at [d0,d1] should be constrained to the dofs
             E and F at [e0,e1] and [f0,f1] as
            D = alpha E + beta F
            the dictionary should be:
                {np.array([d0, d1], dtype=np.float64).tobytes():
                    {np.array([e0, e1], dtype=np.float64).tobytes(): alpha,
                    np.array([f0, f1], dtype=np.float64).tobytes(): beta}}
        """
        slaves, masters, coeffs, owners, offsets\
            = create_dictionary_constraint(
                self.V, slave_master_dict, subspace_slave, subspace_master)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def slaves(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.slaves()

    def masters_local(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.masters_local()

    def coefficients(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.coefficients()

    def num_local_slaves(self):
        if self.finalized:
            return self._cpp_object.num_local_slaves()
        else:
            return len(self.local_slaves)

    def index_map(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.index_map()

    def slave_cells(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.slave_cells()

    def cell_to_slaves(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.cell_to_slaves()

    def create_sparsity_pattern(self, cpp_form):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.create_sparsity_pattern(cpp_form)

    def function_space(self):
        if not self.finalized:
            warnings.warn(
                "Returning original function space for MultiPointConstraint")
            return self.V
        else:
            return self.V_mpc

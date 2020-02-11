from dolfinx import function
from dolfinx_mpc import cpp
import typing
import numba
from numba.typed import List
from petsc4py import PETSc

class MultiPointConstraint(cpp.mpc.MultiPointConstraint):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace],
            master_slave_map: typing.Dict[int, typing.Dict[int, float]]):

        """Representation of MultiPointConstraint which is imposed on
        a linear system.

        """
        slaves = []
        masters = []
        coefficients = []
        offsets = []

        for slave in master_slave_map.keys():
            offsets.append(len(masters))
            slaves.append(slave)
            for master in master_slave_map[slave]:
                masters.append(master)
                coefficients.append(master_slave_map[slave][master])
        # Add additional element to indicate end of list
        # (to determine number of master dofs later on)
        offsets.append(len(masters))
        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V
        super().__init__(_V, slaves, masters, coefficients, offsets)




    def backsubstitution(self,vector, dofmap):
        slaves = self.slaves()
        masters, coefficients = self.masters_and_coefficients()
        offsets = self.master_offsets()
        index_map = self.index_map()
        slave_cells = self.slave_cells()
        cell_to_slave, cell_to_slave_offset = self.cell_to_slave_mapping()
        if len(slave_cells)==0:
            sc_nb = List.empty_list(numba.types.int64)
        else:
            sc_nb = List()
        [sc_nb.append(sc) for sc in slave_cells]

        # Can be empty list locally, so has to be wrapped to be used with numba
        if len(cell_to_slave)==0:
            c2s_nb = List.empty_list(numba.types.int64)
        else:
            c2s_nb = List()
        [c2s_nb.append(c2s) for c2s in cell_to_slave]
        if len(cell_to_slave_offset)==0:
            c2so_nb = List.empty_list(numba.types.int64)
        else:
            c2so_nb = List()
        [c2so_nb.append(c2so) for c2so in cell_to_slave_offset]

        ghost_info = (index_map.local_range, index_map.ghosts, index_map.indices(True))
        mpc = (slaves, sc_nb, c2s_nb, c2so_nb, masters, coefficients, offsets)

        backsubstitution_numba(vector, dofmap.dof_array, mpc, ghost_info)
        vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return vector


@numba.njit
def backsubstitution_numba(b, dofmap, mpc, ghost_info):
    """
        Insert mpc values into vector bc
        """
    (slaves, slave_cells, cell_to_slave, cell_to_slave_offset, masters, coefficients, offsets) = mpc
    (local_range, ghosts,global_indices) = ghost_info
    slaves_visited = List.empty_list(numba.types.int64)
    # Loop through slave cells
    for (index, cell_index) in enumerate(slave_cells):
        cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                    cell_to_slave_offset[index+1]]
        local_dofs = dofmap[3 * cell_index:3 * cell_index + 3]

        # Find the global index of the slaves on the cell in the slaves-array
        global_slaves_index = []
        for gi in range(len(slaves)):
            if slaves[gi] in cell_slaves:
                global_slaves_index.append(gi)

        for slave_index in global_slaves_index:
            slave = slaves[slave_index]
            k = -1
            # Find local position of slave dof
            for local_dof in local_dofs:
                if global_indices[local_dof] == slave:
                    k = local_dof
            assert k != -1
            # Check if we have already inserted for this slave
            if not slave in slaves_visited:
                slaves_visited.append(slave)
                slaves_masters = masters[offsets[slave_index]:
                                         offsets[slave_index+1]]
                slaves_coeffs = coefficients[offsets[slave_index]:
                                             offsets[slave_index+1]]
                for (master, coeff) in zip(slaves_masters, slaves_coeffs):
                    # Find local index for master
                    local_master_index = -1
                    if master < local_range[1] and master >= local_range[0]:
                        local_master_index = master-local_range[0]
                    else:
                        for q, ghost in enumerate(ghosts):
                            if master == ghost:
                                local_master_index = q + local_range[1] - local_range[0]
                                break
                    assert q!= -1
                    b[k] += coeff*b[local_master_index]

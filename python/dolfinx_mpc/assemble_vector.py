# Copyright (C) 2020 Jørgen S. DOkken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numba
import numpy
from numba.typed import List

import dolfinx

from .numba_setup import PETSc, ffi


def assemble_vector(form, multipointconstraint, bcs=[numpy.array([]), numpy.array([])]):
    bc_dofs, bc_values = bcs
    V = form.arguments()[0].ufl_function_space()

    # Unpack mesh and dofmap data
    c = V.mesh.topology.connectivity(2, 0).array()
    pos = V.mesh.topology.connectivity(2, 0).offsets()
    geom = V.mesh.geometry.points
    dofs = V.dofmap.dof_array

    # Data from multipointconstraint
    masters, coefficients = multipointconstraint.masters_and_coefficients()
    cell_to_slave, cell_to_slave_offset = multipointconstraint.cell_to_slave_mapping()
    slaves = multipointconstraint.slaves()
    offsets = numpy.array(multipointconstraint.master_offsets())
    slave_cells = numpy.array(multipointconstraint.slave_cells())
    masters = numpy.array(masters)
    coefficients = numpy.array(coefficients)

    # Get index map and ghost info
    index_map = multipointconstraint.index_map()
    ghost_info = (index_map.local_range,
                  index_map.indices(True), index_map.ghosts)

    # Wrapping for numba to be able to do "if i in slave_cells"
    if len(slave_cells) == 0:
        sc_nb = List.empty_list(numba.types.int64)
    else:
        sc_nb = List()
    [sc_nb.append(sc) for sc in slave_cells]

    # Can be empty list locally, so has to be wrapped to be used with numba
    if len(cell_to_slave) == 0:
        c2s_nb = List.empty_list(numba.types.int64)
    else:
        c2s_nb = List()
    [c2s_nb.append(c2s) for c2s in cell_to_slave]
    if len(cell_to_slave_offset) == 0:
        c2so_nb = List.empty_list(numba.types.int64)
    else:
        c2so_nb = List()
    [c2so_nb.append(c2so) for c2so in cell_to_slave_offset]

    vector = dolfinx.cpp.la.create_vector(index_map)

    ufc_form = dolfinx.jit.ffcx_jit(form)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    gdim = V.mesh.geometry.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    with vector.localForm() as b:
        b.set(0.0)
        assemble_vector_numba(numpy.asarray(b), kernel, (c, pos), geom, gdim, dofs,num_dofs_per_element,
                              (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info,
                              (bc_dofs, bc_values))
    return vector


@numba.njit
def assemble_vector_numba(b, kernel, mesh, x, gdim, dofmap, num_dofs_per_element, mpc, ghost_info, bcs):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    (bcs, values) = bcs
    (slaves, masters, coefficients, offsets,
     slave_cells, cell_to_slave, cell_to_slave_offset) = mpc
    local_range, global_indices, ghosts = ghost_info

    connections, pos = mesh
    orientation = numpy.array([0], dtype=numpy.int32)
    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.ones(1, dtype=PETSc.ScalarType)
    b_local = numpy.zeros(num_dofs_per_element, dtype=PETSc.ScalarType)
    index = 0
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(orientation),
               ffi.from_buffer(orientation))
        # if len(bcs) > 1:
        #     for k in range(3):
        #         if bcs[dofmap[i * 3 + k]]:
        #             b_local[k] = 0
        if i in slave_cells:
            b_local_copy = b_local.copy()
            # Determine which slaves are in this cell,
            # and which global index they have in 1D arrays
            cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                        cell_to_slave_offset[index+1]]
            index += 1
            glob = dofmap[ num_dofs_per_element * i: num_dofs_per_element * i +  num_dofs_per_element]
            # Find which slaves belongs to each cell
            global_slaves = []
            for gi, slave in enumerate(slaves):
                if slaves[gi] in cell_slaves:
                    global_slaves.append(gi)
            # Loop over the slaves
            for s_0 in range(len(global_slaves)):
                slave_index = global_slaves[s_0]
                cell_masters = masters[offsets[slave_index]:
                                       offsets[slave_index+1]]
                cell_coeffs = coefficients[offsets[slave_index]:
                                           offsets[slave_index+1]]
                # Variable for local position of slave dof
                # slave_local = 0
                # for k in range(len(glob)):
                #     if global_indices[glob[k]] == slaves[slave_index]:
                #         slave_local = k

                # Loop through each master dof to take individual contributions
                for m_0 in range(len(cell_masters)):
                    if slaves[slave_index] == cell_masters[m_0]:
                        print("No slaves (since slave is same as master dof)")
                        continue

                    # Find local dof and add contribution to another place
                    for k in range(len(glob)):
                        if global_indices[glob[k]] == slaves[slave_index]:
                            c0 = cell_coeffs[m_0]
                            # Map to local index
                            local_index = -1
                            if cell_masters[m_0] < local_range[1] and cell_masters[m_0] >= local_range[0]:
                                local_index = cell_masters[m_0]-local_range[0]
                            else:
                                # Inverse mapping from ghost info
                                for q, ghost in enumerate(ghosts):
                                    if cell_masters[m_0] == ghost:
                                        local_index = q + \
                                            local_range[1]-local_range[0]
                                        # break
                                        pass
                            assert local_index != -1
                            b[local_index] += c0*b_local_copy[k]
                            b_local[k] = 0
        for j in range(num_dofs_per_element):
            b[dofmap[i * num_dofs_per_element + j]] += b_local[j]

    # for k in range(len(bcs)):
    #     if bcs[k]:

    #         b[k] = values[k]

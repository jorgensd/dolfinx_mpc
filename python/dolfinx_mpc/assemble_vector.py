# Copyright (C) 2020 Jørgen S. DOkken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numba
import numpy

import dolfinx

from .numba_setup import PETSc, ffi
from .assemble_matrix import in_numpy_array


def assemble_vector(form, multipointconstraint,
                    bcs=[numpy.array([]), numpy.array([])]):
    bc_dofs, bc_values = bcs
    V = form.arguments()[0].ufl_function_space()

    # Unpack mesh and dofmap data
    c = V.mesh.topology.connectivity(2, 0).array()
    pos = V.mesh.topology.connectivity(2, 0).offsets()
    geom = V.mesh.geometry.points
    dofs = V.dofmap.dof_array

    # Get cell orientation data
    edge_reflections = V.mesh.topology.get_edge_reflections()
    face_reflections = V.mesh.topology.get_face_reflections()
    face_rotations = V.mesh.topology.get_face_rotations()
    # FIXME: Need to get all of this data indep of gdim
    facet_permutations = numpy.array([], dtype=numpy.uint8)
    # FIXME: Numba does not support edge reflections
    edge_reflections = numpy.array([], dtype=numpy.bool)
    permutation_data = (edge_reflections, face_reflections,
                        face_rotations, facet_permutations)
    # FIXME: should be local facet index
    facet_index = numpy.array([], dtype=numpy.int32)

    # Data from multipointconstraint
    masters, coefficients = multipointconstraint.masters_and_coefficients()
    cell_to_slave, c2s_offset = multipointconstraint.cell_to_slave_mapping()
    slaves = multipointconstraint.slaves()
    offsets = multipointconstraint.master_offsets()
    slave_cells = multipointconstraint.slave_cells()
    mpc_data = (slaves, masters, coefficients,
                offsets, slave_cells, cell_to_slave, c2s_offset)

    # Get index map and ghost info
    index_map = multipointconstraint.index_map()

    ghost_info = (index_map.local_range, index_map.indices(True),
                  index_map.block_size, index_map.ghosts)

    vector = dolfinx.cpp.la.create_vector(index_map)

    ufc_form = dolfinx.jit.ffcx_jit(form)

    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    gdim = V.mesh.geometry.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    with vector.localForm() as b:
        b.set(0.0)
        assemble_vector_numba(numpy.asarray(b), kernel, (c, pos), geom, gdim,
                              facet_index, permutation_data,
                              dofs, num_dofs_per_element, mpc_data,
                              ghost_info, (bc_dofs, bc_values))
    return vector


@numba.njit
def assemble_vector_numba(b, kernel, mesh, x, gdim, facet_index,
                          permutation_data, dofmap, num_dofs_per_element,
                          mpc, ghost_info, bcs):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs
    (slaves, masters, coefficients, offsets,
     slave_cells, cell_to_slave, cell_to_slave_offset) = mpc
    (edge_reflections, face_reflections,
     face_rotations, facet_permutations) = permutation_data

    local_range, global_indices, block_size, ghosts = ghost_info
    local_size = local_range[1] - local_range[0]
    connections, pos = mesh
    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
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
        # Cell orientation data
        face_reflection = face_reflections[i, :]
        # FIXME: Numba does not support edge reflections
        edge_reflection = edge_reflections  # edge_reflections[i,:]
        face_rotation = face_rotations[i, :]

        kernel(ffi_fb(b_local), ffi_fb(coeffs),
               ffi_fb(constants),
               ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_permutations), ffi_fb(face_reflection),
               ffi_fb(edge_reflection), ffi_fb(face_rotation))

        # FIXME: Add support for bcs on master dof
        # if len(bcs) > 0:
        #     for k in range(3):
        #         if bcs[dofmap[i * 3 + k]]:
        #             b_local[k] = 0
        if in_numpy_array(slave_cells, i):
            b_local_copy = b_local.copy()
            # Determine which slaves are in this cell,
            # and which global index they have in 1D arrays
            cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                        cell_to_slave_offset[index+1]]
            index += 1
            glob = dofmap[num_dofs_per_element * i:
                          num_dofs_per_element * i + num_dofs_per_element]
            # Find which slaves belongs to each cell
            global_slaves = []
            for gi, slave in enumerate(slaves):
                if in_numpy_array(cell_slaves, slaves[gi]):
                    global_slaves.append(gi)
            # Loop over the slaves
            for s_0 in range(len(global_slaves)):
                slave_index = global_slaves[s_0]
                cell_masters = masters[offsets[slave_index]:
                                       offsets[slave_index+1]]
                cell_coeffs = coefficients[offsets[slave_index]:
                                           offsets[slave_index+1]]

                # Loop through each master dof to take individual contributions
                for m_0 in range(len(cell_masters)):

                    # Find local dof and add contribution to another place
                    for k in range(len(glob)):
                        if global_indices[glob[k]] == slaves[slave_index]:
                            c0 = cell_coeffs[m_0]
                            # Map to local index
                            local_index = -1
                            in_range = (cell_masters[m_0] <
                                        block_size * local_range[1] and
                                        cell_masters[m_0] >=
                                        block_size * local_range[0])
                            if in_range:
                                local_index = (cell_masters[m_0] -
                                               block_size * local_range[0])
                            else:
                                # Inverse mapping from ghost info
                                for q, ghost in enumerate(ghosts):
                                    if local_index != -1:
                                        break
                                    for comp in range(block_size):
                                        cm0 = cell_masters[m_0]
                                        if cm0 == block_size * ghost + comp:
                                            local_index = ((q + local_size)
                                                           * block_size + comp)
                                            break

                            assert local_index != -1
                            b[local_index] += c0*b_local_copy[k]
                            b_local[k] = 0
        for j in range(num_dofs_per_element):
            b[dofmap[i * num_dofs_per_element + j]] += b_local[j]

    # FIXME: Add support for bcs on master dofs
    # for k in range(len(bcs)):
    #     if bcs[k]:

    #         b[k] = values[k]


@numba.njit
def block_numba(b, kernel, mesh, x, gdim, facet_index,
                permutation_data, dofmap, num_dofs_per_element,
                mpc, ghost_info, bcs):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs
    (slaves, masters, coefficients, offsets,
     slave_cells, cell_to_slave, cell_to_slave_offset) = mpc
    (edge_reflections, face_reflections,
     face_rotations, facet_permutations) = permutation_data

    local_range, global_indices, block_size, ghosts = ghost_info
    connections, pos = mesh
    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    coeffs = numpy.zeros(2, dtype=PETSc.ScalarType)
    constants = numpy.zeros(2, dtype=PETSc.ScalarType)
    constants = numpy.ones(2, dtype=PETSc.ScalarType)
    b_local = numpy.zeros(num_dofs_per_element, dtype=PETSc.ScalarType)

    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        # Cell orientation data
        face_reflection = face_reflections[i, :]
        # FIXME: Numba does not support edge reflections
        edge_reflection = edge_reflections  # edge_reflections[i,:]
        face_rotation = face_rotations[i, :]

        kernel(ffi_fb(b_local), ffi_fb(coeffs),
               ffi_fb(constants),
               ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_permutations), ffi_fb(face_reflection),
               ffi_fb(edge_reflection), ffi_fb(face_rotation))

        for j in range(num_dofs_per_element):
            b[dofmap[i * num_dofs_per_element + j]] += b_local[j]

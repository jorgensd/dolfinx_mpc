# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
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
    pos = V.mesh.geometry.dofmap().offsets()
    x_dofs = V.mesh.geometry.dofmap().array()
    x = V.mesh.geometry.x
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
    masters_local = multipointconstraint.masters_local()
    slave_cells = multipointconstraint.slave_cells()
    mpc_data = (slaves, masters_local, coefficients,
                offsets, slave_cells, cell_to_slave, c2s_offset)

    # Get index map and ghost info
    index_map = multipointconstraint.index_map()

    ghost_info = (index_map.local_range, index_map.indices(True),
                  index_map.block_size, index_map.ghosts)

    vector = dolfinx.cpp.la.create_vector(index_map)
    ufc_form = dolfinx.jit.ffcx_jit(form)

    # Pack constants and coefficients
    cpp_form = dolfinx.Form(form)._cpp_object
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    gdim = V.mesh.geometry.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Assemble vector with all entries
    dolfinx.cpp.fem.assemble_vector(vector, cpp_form)

    # Assemble over slave cells and add extra contributions elsewhere
    with vector.localForm() as b:
        assemble_vector_numba(numpy.asarray(b), kernel, (pos, x_dofs, x), gdim,
                              form_coeffs, form_consts,
                              facet_index, permutation_data,
                              dofs, num_dofs_per_element, mpc_data,
                              ghost_info, (bc_dofs, bc_values))
    return vector


@numba.njit
def assemble_vector_numba(b, kernel, mesh, gdim,
                          coeffs, constants, facet_index,
                          permutation_data, dofmap, num_dofs_per_element,
                          mpc, ghost_info, bcs):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs

    (edge_reflections, face_reflections,
     face_rotations, facet_permutations) = permutation_data

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    b_local = numpy.zeros(num_dofs_per_element, dtype=PETSc.ScalarType)
    slave_cell_index = 0
    slave_cells = mpc[4]
    for cell_index, cell in enumerate(pos[:-1]):
        if not in_numpy_array(slave_cells, cell_index):
            continue

        num_vertices = pos[cell_index + 1] - pos[cell_index]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)

        # FIXME: Numba does not support edge reflections
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_permutations),
               ffi_fb(face_reflections[cell_index, :]),
               ffi_fb(edge_reflections), ffi_fb(face_rotations[cell_index, :]))

        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, slave_cell_index, b_local,
                                 b_local_copy, mpc, dofmap,
                                 num_dofs_per_element, ghost_info)
        slave_cell_index += 1
        for j in range(num_dofs_per_element):
            position = dofmap[cell_index * num_dofs_per_element + j]
            b[position] += (b_local[j] - b_local_copy[j])


@numba.njit(cache=True)
def modify_mpc_contributions(b, cell_index,
                             slave_cell_index, b_local, b_copy,  mpc, dofmap,
                             num_dofs_per_element, ghost_info):
    """
    Modify local entries of b_local with MPC info and add modified
    entries to global vector b.
    """

    # Unwrap MPC data
    (slaves, masters_local, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offset) = mpc
    # Unwrap ghost data
    local_range, global_indices, block_size, ghosts = ghost_info

    # Determine which slaves are in this cell,
    # and which global index they have in 1D arrays
    cell_slaves = cell_to_slave[cell_to_slave_offset[slave_cell_index]:
                                cell_to_slave_offset[slave_cell_index+1]]

    glob = dofmap[num_dofs_per_element * cell_index:
                  num_dofs_per_element * cell_index + num_dofs_per_element]
    # Find which slaves belongs to each cell
    global_slaves = []
    for gi, slave in enumerate(slaves):
        if in_numpy_array(cell_slaves, slaves[gi]):
            global_slaves.append(gi)
    # Loop over the slaves
    for s_0 in range(len(global_slaves)):
        slave_index = global_slaves[s_0]
        cell_masters = masters_local[offsets[slave_index]:
                                     offsets[slave_index+1]]
        cell_coeffs = coefficients[offsets[slave_index]:
                                   offsets[slave_index+1]]

        # Loop through each master dof to take individual contributions
        for m_0 in range(len(cell_masters)):
            # Find local dof and add contribution to another place
            for k in range(len(glob)):
                if global_indices[glob[k]] == slaves[slave_index]:
                    c0 = cell_coeffs[m_0]
                    b[cell_masters[m_0]] += c0*b_copy[k]
                    b_local[k] = 0

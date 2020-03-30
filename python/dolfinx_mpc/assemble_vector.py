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
    dofs = V.dofmap.list.array()

    # Get cell orientation data
    permutation_info = V.mesh.topology.get_cell_permutation_info()
    facet_permutation_info = V.mesh.topology.get_facet_permutations()

    # Data from multipointconstraint
    coefficients = multipointconstraint.coefficients()
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
    # FIXME: Here we most likely need to loop through integral
    # id's if we are using restricted domains.
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    gdim = V.mesh.geometry.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Assemble vector with all entries
    dolfinx.cpp.fem.assemble_vector(vector, cpp_form)

    # Assemble over slave cells and add extra contributions elsewhere
    with vector.localForm() as b:
        assemble_cells(numpy.asarray(b), kernel, (pos, x_dofs, x), gdim,
                       form_coeffs, form_consts,
                       permutation_info,
                       dofs, num_dofs_per_element, mpc_data,
                       ghost_info, (bc_dofs, bc_values))

    exterior_integrals = form.integrals_by_type("exterior_facet")
    if len(exterior_integrals) > 0:
        # Assemble exterior facet integrals
        integrals = cpp_form.integrals()
        for i in range(len(exterior_integrals)):
            facet_info = pack_facet_info(V.mesh, integrals, i)
            subdomain_id = exterior_integrals[i].subdomain_id()
            if subdomain_id == "everywhere":
                subdomain_id = -1
            facet_kernel = ufc_form.create_exterior_facet_integral(
                subdomain_id).tabulate_tensor
            with vector.localForm() as b:
                assemble_exterior_facets(numpy.asarray(b), facet_kernel,
                                         facet_info,
                                         (pos, x_dofs, x),
                                         gdim, form_coeffs, form_consts,
                                         (permutation_info,
                                          facet_permutation_info),
                                         dofs,
                                         num_dofs_per_element,
                                         mpc_data, ghost_info,
                                         (bc_dofs, bc_values))

    return vector


def pack_facet_info(mesh, integrals, i):
    """
    Given the mesh, FormIntgrals and the index of the i-th exterior
    facet integral, for each active facet, find the cell index and
    local facet index and pack them in a numpy nd array
    """
    # FIXME: Should be moved to dolfinx C++ layer
    # Set up data required for exterior facet assembly
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim-1
    # This connectivities has been computed by normal assembly
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)

    active_facets = integrals.integral_domains(
        dolfinx.fem.FormIntegrals.Type.exterior_facet, i)
    facet_info = numpy.zeros((len(active_facets), 2),
                             dtype=numpy.int64)
    for j, facet in enumerate(active_facets):
        cells = f_to_c.links(facet)
        assert(len(cells) == 1)
        local_facets = c_to_f.links(cells[0])
        # Should be wrapped in convenience numba function
        local_index = numpy.flatnonzero(facet == local_facets)[0]
        facet_info[j, :] = [cells[0], local_index]
    return facet_info


@numba.njit
def assemble_cells(b, kernel, mesh, gdim,
                   coeffs, constants,
                   permutation_info, dofmap, num_dofs_per_element,
                   mpc, ghost_info, bcs):
    """Assemble additional MPC contributions for cell integrals"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

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

        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_perm),
               permutation_info[cell_index])

        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, slave_cell_index, b_local,
                                 b_local_copy, mpc, dofmap,
                                 num_dofs_per_element, ghost_info)
        slave_cell_index += 1

        for j in range(num_dofs_per_element):
            position = dofmap[cell_index * num_dofs_per_element + j]
            b[position] += (b_local[j] - b_local_copy[j])


@numba.njit
def assemble_exterior_facets(b, kernel, facet_info, mesh, gdim,
                             coeffs, constants,
                             permutation_info, dofmap,
                             num_dofs_per_element,
                             mpc, ghost_info, bcs):
    """Assemble additional MPC contributions for facets"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs

    cell_perms, facet_perms = permutation_info

    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    b_local = numpy.zeros(num_dofs_per_element, dtype=PETSc.ScalarType)
    slave_cells = mpc[4]
    for i in range(facet_info.shape[0]):
        cell_index, local_facet = facet_info[i]
        cell = pos[cell_index]
        facet_index[0] = local_facet
        if not in_numpy_array(slave_cells, cell_index):
            continue
        slave_cell_index = numpy.flatnonzero(slave_cells == cell_index)[0]
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        facet_perm[0] = facet_perms[local_facet, cell_index]
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_perm),
               cell_perms[cell_index])

        b_local_copy = b_local.copy()

        modify_mpc_contributions(b, cell_index, slave_cell_index, b_local,
                                 b_local_copy, mpc, dofmap,
                                 num_dofs_per_element, ghost_info)
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

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numba
import numpy

import dolfinx
import dolfinx.common
import dolfinx.log

from .assemble_matrix import in_numpy_array, pack_facet_info
from .numba_setup import PETSc, ffi

Timer = dolfinx.common.Timer


def assemble_vector(form, constraint,
                    bcs=[numpy.array([]), numpy.array([])]):
    dolfinx.log.log(dolfinx.log.LogLevel.INFO, "Assemble MPC vector")
    timer_vector = Timer("~MPC: Assemble vector")
    bc_dofs, bc_values = bcs
    V = form.arguments()[0].ufl_function_space()
    # Unpack mesh and dofmap data
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x
    dofs = V.dofmap.list.array
    block_size = V.dofmap.index_map_bs
    # Data from multipointconstraint
    slave_cells = constraint.slave_cells()
    coefficients = numpy.array(constraint.coefficients(), dtype=PETSc.ScalarType)
    masters = constraint.masters_local()
    slave_cell_to_dofs = constraint.cell_to_slaves()
    cell_to_slave = slave_cell_to_dofs.array
    c_to_s_off = slave_cell_to_dofs.offsets
    slaves_local = constraint.slaves()
    masters_local = masters.array
    offsets = masters.offsets

    mpc_data = (slaves_local, masters_local, coefficients, offsets,
                slave_cells, cell_to_slave, c_to_s_off)
    # Get index map and ghost info
    index_map = constraint.index_map()
    vector = dolfinx.cpp.la.create_vector(index_map, block_size)
    ufc_form = dolfinx.jit.ffcx_jit(V.mesh.mpi_comm(), form)

    # Pack constants and coefficients
    cpp_form = dolfinx.Form(form)._cpp_object
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = numpy.array(dolfinx.cpp.fem.pack_constants(cpp_form), dtype=PETSc.ScalarType)

    gdim = V.mesh.geometry.dim
    tdim = V.mesh.topology.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Assemble vector with all entries
    dolfinx.cpp.fem.assemble_vector(vector.array_w, cpp_form)
    # Assemble over cells
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)
    if num_cell_integrals > 0:
        V.mesh.topology.create_entity_permutations()
        permutation_info = numpy.array(V.mesh.topology.get_cell_permutation_info(), dtype=numpy.uint32)
        timer = Timer("~MPC: Assemble matrix (cells)")
        for subdomain_id in subdomain_ids:
            cell_kernel = ufc_form.create_cell_integral(subdomain_id).tabulate_tensor
            active_cells = numpy.array(cpp_form.domains(dolfinx.fem.IntegralType.cell, subdomain_id), dtype=numpy.int64)
            slave_cell_indices = numpy.flatnonzero(numpy.isin(active_cells, slave_cells))
            with vector.localForm() as b:
                assemble_cells(numpy.asarray(b), cell_kernel, active_cells[slave_cell_indices],
                               (pos, x_dofs, x), gdim, form_coeffs, form_consts,
                               permutation_info, dofs, block_size, num_dofs_per_element, mpc_data, (bc_dofs, bc_values))
        timer.stop()

    # Assemble exterior facet integrals
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)
    if num_exterior_integrals > 0:
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        permutation_info = numpy.array(V.mesh.topology.get_cell_permutation_info(), dtype=numpy.uint32)
        facet_permutation_info = V.mesh.topology.get_facet_permutations()
        timer = Timer("MPC Assemble vector (exterior facets)")
        for subdomain_id in subdomain_ids:
            active_facets = numpy.array(cpp_form.domains(
                dolfinx.fem.IntegralType.exterior_facet, subdomain_id), dtype=numpy.int64)
            facet_info = pack_facet_info(V.mesh, active_facets)
            facet_kernel = ufc_form.create_exterior_facet_integral(subdomain_id).tabulate_tensor
            with vector.localForm() as b:
                assemble_exterior_facets(numpy.asarray(b), facet_kernel, facet_info, (pos, x_dofs, x), gdim,
                                         form_coeffs, form_consts, (permutation_info, facet_permutation_info),
                                         dofs, block_size, num_dofs_per_element, mpc_data, (bc_dofs, bc_values))
        timer.stop()
    timer_vector.stop()
    return vector


@numba.njit
def assemble_cells(b, kernel, active_cells, mesh, gdim, coeffs, constants, permutation_info,
                   dofmap, block_size, num_dofs_per_element, mpc, bcs):
    """Assemble additional MPC contributions for cell integrals"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    geometry = numpy.zeros((pos[1] - pos[0], gdim))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=PETSc.ScalarType)

    for slave_cell_index, cell_index in enumerate(active_cells):
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        # FIXME: This assumes a particular geometry dof layout
        cell = pos[cell_index]
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
        modify_mpc_contributions_local(b, cell_index, slave_cell_index, b_local,
                                       b_local_copy, mpc, dofmap, block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[num_dofs_per_element * cell_index + j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit
def assemble_exterior_facets(b, kernel, facet_info, mesh, gdim, coeffs, constants, permutation_info,
                             dofmap, block_size, num_dofs_per_element, mpc, bcs):
    """Assemble additional MPC contributions for facets"""
    ffi_fb = ffi.from_buffer
    (bcs, values) = bcs

    cell_perms, facet_perms = permutation_info

    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    geometry = numpy.zeros((pos[1] - pos[0], gdim))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=PETSc.ScalarType)
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
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]), ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm), cell_perms[cell_index])

        b_local_copy = b_local.copy()

        modify_mpc_contributions_local(b, cell_index, slave_cell_index, b_local, b_local_copy,
                                       mpc, dofmap, block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[num_dofs_per_element * cell_index + j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit(cache=True)
def modify_mpc_contributions_local(b, cell_index, slave_cell_index, b_local, b_copy, mpc, dofmap,
                                   block_size, num_dofs_per_element):
    """
    Modify local entries of b_local with MPC info and add modified
    entries to global vector b.
    """

    # Unwrap MPC data
    (slaves, masters_local, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offset) = mpc

    # Determine which slaves are in this cell,
    # and which global index they have in 1D arrays
    cell_slaves = cell_to_slave[cell_to_slave_offset[slave_cell_index]:
                                cell_to_slave_offset[slave_cell_index + 1]]

    cell_blocks = dofmap[num_dofs_per_element * cell_index:
                         num_dofs_per_element * cell_index + num_dofs_per_element]

    # Loop over the slaves
    for slave_index in cell_slaves:
        cell_masters = masters_local[offsets[slave_index]:
                                     offsets[slave_index + 1]]
        cell_coeffs = coefficients[offsets[slave_index]:
                                   offsets[slave_index + 1]]

        # Loop through each master dof to take individual contributions
        for m0, c0 in zip(cell_masters, cell_coeffs):
            # Find local dof and add contribution to another place
            for i in range(num_dofs_per_element):
                for j in range(block_size):
                    if cell_blocks[i] * block_size + j == slaves[slave_index]:
                        b[m0] += c0 * b_copy[i * block_size + j]
                        b_local[i * block_size + j] = 0

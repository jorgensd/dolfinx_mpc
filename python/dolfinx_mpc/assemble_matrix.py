# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numba
import numpy

import dolfinx

from .numba_setup import (PETSc, ffi, insert, mode, set_values,
                          set_values_local, sink)


def assemble_matrix(form, multipointconstraint, bcs=None):
    """
    Assembles a ufl form given a multi point constraint and possible
    Dirichlet boundary conditions.
    NOTE: Dirichlet conditions cant be on master dofs.
    """
    bc_array = numpy.array([])
    if bcs is not None:
        for bc in bcs:
            # Extract local index of possible sub space
            bc_array = numpy.append(bc_array, bc.dof_indices[:, 0])
    # Get data from function space
    assert(form.arguments()[0].ufl_function_space() ==
           form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()
    dofmap = V.dofmap
    dofs = dofmap.dof_array
    indexmap = dofmap.index_map
    ghost_info = (indexmap.local_range, indexmap.block_size,
                  indexmap.indices(True))

    # Get data from mesh
    pos = V.mesh.geometry.dofmap().offsets()
    x_dofs = V.mesh.geometry.dofmap().array()
    x = V.mesh.geometry.x

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

    # Generate ufc_form
    ufc_form = dolfinx.jit.ffcx_jit(form)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object

    # Pack constants and coefficients
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    pattern = multipointconstraint.create_sparsity_pattern(cpp_form)
    pattern.assemble()

    A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
    A.zeroEntries()
    # Unravel data from MPC
    slave_cells = multipointconstraint.slave_cells()
    masters, coefficients = multipointconstraint.masters_and_coefficients()
    masters, coefficients = masters, coefficients
    cell_to_slave, c_to_s_off = multipointconstraint.cell_to_slave_mapping()
    slaves = multipointconstraint.slaves()
    offsets = multipointconstraint.master_offsets()
    mpc_data = (slaves, masters, coefficients, offsets,
                slave_cells, cell_to_slave, c_to_s_off)

    # General assembly data
    num_dofs_per_element = dofmap.dof_layout.num_dofs
    gdim = V.mesh.geometry.dim

    assemble_matrix_numba(A.handle, kernel, (pos, x_dofs, x), gdim,
                          form_coeffs, form_consts, facet_index,
                          permutation_data,
                          dofs, num_dofs_per_element, mpc_data,
                          ghost_info, bc_array)
    A.assemble()

    # Add one on diagonal for diriclet bc and slave dofs
    add_diagonal(A.handle, slaves)
    A.assemble()
    if bcs is not None:
        if cpp_form.function_space(0).id == cpp_form.function_space(1).id:
            dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_space(0),
                                         bcs, 1.0)
    A.assemble()
    return A


@numba.njit
def add_diagonal(A, dofs):
    """
    Insert 1 on the diagonal for each dof
    """
    ffi_fb = ffi.from_buffer
    A_diag = numpy.ones((1, 1), dtype=PETSc.ScalarType)
    for i, dof in enumerate(dofs):
        dof_pos = numpy.array([dof], dtype=numpy.int32)
        # This is done on every processors, that's why we use
        # PETSC.InsertMode.INSERT
        ierr_diag = set_values(A, 1, ffi_fb(dof_pos),
                               1, ffi_fb(dof_pos),
                               ffi_fb(A_diag), insert)
        assert(ierr_diag == 0)
    sink(A_diag, dof_pos)


@numba.njit
def in_numpy_array(array, value):
    """
    Convenience function replacing "value in array" for numpy arrays in numba
    """
    in_array = False
    for item in array:
        if item == value:
            in_array = True
            break
    return in_array


@numba.njit
def assemble_matrix_numba(A, kernel, mesh, gdim, coeffs, constants,
                          facet_index, permutation_data, dofmap,
                          num_dofs_per_element, mpc, ghost_info, bcs):
    """
    Numba assembler for cell integrals
    """
    ffi_fb = ffi.from_buffer
    slave_cells = mpc[4]  # Note: packing order for MPC really important

    # Get mesh and geometry data
    pos, x_dofmap, x = mesh
    (edge_reflections, face_reflections,
     face_rotations, facet_permutations) = permutation_data

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1]-pos[0], gdim))

    A_local = numpy.zeros((num_dofs_per_element, num_dofs_per_element),
                          dtype=PETSc.ScalarType)

    # Loop over all cells
    slave_cell_index = 0
    for cell_index, cell in enumerate(pos[:-1]):
        num_vertices = pos[cell_index + 1] - pos[cell_index]

        # Compute vertices of cell from mesh data
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]

        A_local.fill(0.0)
        # FIXME: Numba does not support edge reflections
        kernel(ffi_fb(A_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants),
               ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_permutations),
               ffi_fb(face_reflections[cell_index, :]),
               ffi_fb(edge_reflections),
               ffi_fb(face_rotations[cell_index, :]))

        # Local dof position
        local_pos = dofmap[num_dofs_per_element * cell_index:
                           num_dofs_per_element * cell_index
                           + num_dofs_per_element]
        # Remove all contributions for dofs that are in the Dirichlet bcs
        if len(bcs) > 0:
            for k in range(len(local_pos)):
                is_bc_dof = in_numpy_array(bcs, local_pos[k])
                if is_bc_dof:
                    A_local[k, :] = 0
                    A_local[:, k] = 0

        # If this slave contains a slave dof, modify local contribution
        if in_numpy_array(slave_cells, cell_index):
            modify_mpc_cell(A, slave_cell_index, A_local, local_pos,
                            mpc, ghost_info, num_dofs_per_element)
            slave_cell_index += 1

        # Insert local contribution
        ierr_loc = set_values_local(A, num_dofs_per_element, ffi_fb(local_pos),
                                    num_dofs_per_element, ffi_fb(local_pos),
                                    ffi_fb(A_local), mode)
        assert(ierr_loc == 0)

    sink(A_local, local_pos)


@numba.njit
def modify_mpc_cell(A, slave_cell_index, A_local, local_pos,
                    mpc, ghost_info, num_dofs_per_element):
    """
    Modifies A_local as it contains slave degrees of freedom.
    Adds contributions to corresponding master degrees of freedom in A.
    """
    ffi_fb = ffi.from_buffer

    # Unpack MPC data
    (slaves, masters, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offset) = mpc
    # Unpack ghost data
    (local_range, block_size, global_indices) = ghost_info
    local_size = local_range[1]-local_range[0]

    # Rows taken over by master
    A_row = numpy.zeros((num_dofs_per_element, 1), dtype=PETSc.ScalarType)
    # Columns taken over by master
    A_col = numpy.zeros((1, num_dofs_per_element), dtype=PETSc.ScalarType)
    # Extra insertions at master diagonal
    A_master = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Cross-master coefficients, m0, m1 for same constraint
    A_c0 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    A_c1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Cross-master coefficients, m0, m1, master to multiple slave
    A_m0m1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    A_m1m0 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Index arrays used in set_values and set_values_local
    m0_index = numpy.zeros(1, dtype=numpy.int32)
    m1_index = numpy.zeros(1, dtype=numpy.int32)

    A_local_copy = A_local.copy()
    cell_slaves = cell_to_slave[cell_to_slave_offset[slave_cell_index]:
                                cell_to_slave_offset[slave_cell_index+1]]

    # Find which slaves belongs to each cell
    global_slaves = []
    for gi, slave in enumerate(slaves):
        if in_numpy_array(cell_slaves, slaves[gi]):
            global_slaves.append(gi)
    for s_0 in range(len(global_slaves)):
        slave_index = global_slaves[s_0]
        cell_masters = masters[offsets[slave_index]:
                               offsets[slave_index+1]]
        cell_coeffs = coefficients[offsets[slave_index]:
                                   offsets[slave_index+1]]
        # Variable for local position of slave dof
        slave_local = -1
        for k in range(len(local_pos)):
            g_pos = local_pos[k] + block_size*local_range[0]
            if g_pos == slaves[slave_index]:
                slave_local = k
        assert slave_local != -1

        # Loop through each master dof to take individual contributions
        for m_0 in range(len(cell_masters)):
            ce = cell_coeffs[m_0]

            # Reset local contribution matrices
            A_row.fill(0.0)
            A_col.fill(0.0)

            # Move local contributions with correct coefficients
            A_row[:, 0] = ce*A_local_copy[:, slave_local]
            A_col[0, :] = ce*A_local_copy[slave_local, :]
            A_master[0, 0] = ce*A_row[slave_local, 0]

            # Remove row contribution going to central addition
            A_col[0, slave_local] = 0
            A_row[slave_local, 0] = 0

            # Remove local contributions moved to master
            A_local[:, slave_local] = 0
            A_local[slave_local, :] = 0

            # If one of the other local indices are a slave,
            # move them to the corresponding master dof
            # and multiply by the corresponding coefficient
            for o_slave_index in range(len(global_slaves)):
                other_slave = global_slaves[o_slave_index]
                # If not another slave, continue
                if other_slave == slave_index:
                    continue
                # Find local index of the other slave
                o_slave_local = -1
                for k in range(len(local_pos)):
                    l0 = block_size * local_range[0]
                    if local_pos[k] + l0 == slaves[other_slave]:
                        o_slave_local = k
                        break
                assert(o_slave_local != -1)
                other_cell_masters = masters[offsets[other_slave]:
                                             offsets[other_slave+1]]
                o_coeffs = coefficients[offsets[other_slave]:
                                        offsets[other_slave+1]]
                # Find local index of other masters
                for m_1 in range(len(other_cell_masters)):
                    A_m0m1.fill(0)
                    A_m1m0.fill(0)
                    o_c = o_coeffs[m_1]
                    A_m0m1[0, 0] = o_c*A_row[o_slave_local, 0]
                    A_m1m0[0, 0] = o_c*A_col[0, o_slave_local]
                    A_row[o_slave_local, 0] = 0
                    A_col[0, o_slave_local] = 0
                    m0_index[0] = cell_masters[m_0]
                    m1_index[0] = other_cell_masters[m_1]

                    # Only insert once per pair,
                    # but remove local values for all slaves
                    if o_slave_index > s_0:
                        ierr_m0m1 = set_values(A, 1, ffi_fb(m0_index),
                                               1, ffi_fb(m1_index),
                                               ffi_fb(A_m0m1), mode)
                        assert(ierr_m0m1 == 0)
                        ierr_m1m0 = set_values(A, 1, ffi_fb(m1_index),
                                               1, ffi_fb(m0_index),
                                               ffi_fb(A_m1m0), mode)
                        assert(ierr_m1m0 == 0)

            # Add slave rows to master column
            global_pos = numpy.zeros(local_pos.size, dtype=numpy.int32)
            # Map ghosts to global index and slave to master
            for (h, g) in enumerate(local_pos):
                if g >= block_size * local_size:
                    global_pos[h] = global_indices[g]
                else:
                    global_pos[h] = g + block_size*local_range[0]
            global_pos[slave_local] = cell_masters[m_0]
            m0_index[0] = cell_masters[m_0]

            ierr_row = set_values(A, num_dofs_per_element, ffi_fb(global_pos),
                                  1, ffi_fb(m0_index), ffi_fb(A_row), mode)
            assert(ierr_row == 0)

            # Add slave columns to master row
            ierr_col = set_values(A, 1, ffi_fb(m0_index), num_dofs_per_element,
                                  ffi_fb(global_pos), ffi_fb(A_col), mode)
            assert(ierr_col == 0)
            # Add slave contributions to A_(master, master)
            ierr_m0m0 = set_values(A, 1, ffi_fb(m0_index), 1, ffi_fb(m0_index),
                                   ffi_fb(A_master), mode)
            assert(ierr_m0m0 == 0)

        # Add contributions for different masters on the same cell
        for m_0 in range(len(cell_masters)):
            for m_1 in range(m_0+1, len(cell_masters)):

                A_c0.fill(0.0)
                c0, c1 = cell_coeffs[m_0], cell_coeffs[m_1]
                A_c0[0, 0] += c0*c1*A_local_copy[slave_local, slave_local]
                m0_index[0] = cell_masters[m_0]
                m1_index[0] = cell_masters[m_1]
                ierr_c0 = set_values(A, 1, ffi_fb(m0_index), 1,
                                     ffi_fb(m1_index), ffi_fb(A_c0), mode)
                assert(ierr_c0 == 0)
                A_c1.fill(0.0)
                A_c1[0, 0] += c0*c1*A_local_copy[slave_local, slave_local]
                ierr_c1 = set_values(A, 1, ffi_fb(m1_index), 1,
                                     ffi_fb(m0_index), ffi_fb(A_c1), mode)
                assert(ierr_c1 == 0)

    # FIXME: add handling of Dirichlet condition on master nodes
    # if len(bcs)>0:
    #     bc_value = numpy.array([[1]], dtype=PETSc.ScalarType)
    #     for i in range(len(bcs)):
    #         if bcs[i]:
    #             bc_row = numpy.array([i],dtype=numpy.int32)
    #             ierr_bc = set_values(A, 1, ffi_fb(bc_row), 1,
    #                                  ffi_fb(bc_row),
    #                                  ffi_fb(bc_value), mode)
    #             assert(ierr_bc == 0)

    sink(A_m0m1, A_m1m0, m1_index, A_row, A_col, m0_index, global_pos,
         A_master, A_c0, A_c1)
    return A_local

# Copyright (C) 2020 Jørgen S. DOkken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from .numba_setup import mode, set_values_local, set_values, ffi, PETSc
import numba
import dolfinx
import numpy
from numba.typed import List
# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass


def assemble_matrix(form, multipointconstraint, bcs=numpy.array([])):
    # Get data from function space
    assert(form.arguments()[0].ufl_function_space() ==
           form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()
    dofmap = V.dofmap
    dofs = dofmap.dof_array
    indexmap = dofmap.index_map
    ghost_info = (indexmap.local_range, indexmap.indices(True))

    # Get data from mesh
    c2v = V.mesh.topology.connectivity(V.mesh.topology.dim, 0)
    c = c2v.array()
    pos = c2v.offsets()
    geom = V.mesh.geometry.points

    # Generate ufc_form
    ufc_form = dolfinx.jit.ffcx_jit(form)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object
    A = multipointconstraint.generate_petsc_matrix(cpp_form)
    A.zeroEntries()

    # Unravel data from MPC
    slave_cells = numpy.array(multipointconstraint.slave_cells())
    masters, coefficients = multipointconstraint.masters_and_coefficients()
    cell_to_slave, c_to_s_off = multipointconstraint.cell_to_slave_mapping()
    slaves = multipointconstraint.slaves()
    offsets = multipointconstraint.master_offsets()
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
    if len(c_to_s_off) == 0:
        c2so_nb = List.empty_list(numba.types.int64)
    else:
        c2so_nb = List()
    [c2so_nb.append(c2so) for c2so in c_to_s_off]

    mpc_data = (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb)
    assemble_matrix_numba(A.handle, kernel, (c, pos), geom,
                          dofs, mpc_data, ghost_info, bcs)
    A.assemble()
    return A


@numba.njit
def assemble_matrix_numba(A, kernel, mesh, x, dofmap, mpc, ghost_info, bcs):
    ffi_fb = ffi.from_buffer

    (slaves, masters, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offset) = mpc
    (local_range, global_indices) = ghost_info
    local_size = local_range[1]-local_range[0]
    connections, pos = mesh
    orientation = numpy.array([0], dtype=numpy.int32)
    # FIXME: Need to determine geometry, coeffs and constants from input data
    geometry = numpy.zeros((3, 2))
    coeffs = numpy.zeros(0, dtype=PETSc.ScalarType)
    constants = numpy.zeros(0, dtype=PETSc.ScalarType)
    A_local = numpy.zeros((3, 3), dtype=PETSc.ScalarType)
    # Rows taken over by master
    A_row = numpy.zeros((3, 1), dtype=PETSc.ScalarType)
    # Columns taken over by master
    A_col = numpy.zeros((1, 3), dtype=PETSc.ScalarType)
    # Extra insertions at master diagonal
    A_master = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Setting 0 Dirichlet in original matrix for slave diagonal
    # A_slave = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Cross-master coefficients, m0, m1 for same constraint
    A_c0 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    A_c1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Cross-master coefficients, m0, m1, master to multiple slave
    A_m0m1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    A_m1m0 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    # Index arrays used in set_values and set_values_local
    m0_index = numpy.zeros(1, dtype=numpy.int32)
    m1_index = numpy.zeros(1, dtype=numpy.int32)

    # Loop over slave cells
    index = 0
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]

        A_local.fill(0.0)
        kernel(ffi_fb(A_local), ffi_fb(coeffs),
               ffi_fb(constants),
               ffi_fb(geometry), ffi_fb(orientation),
               ffi_fb(orientation))

        local_pos = dofmap[3 * i:3 * i + 3]
        if len(bcs) > 1:
            for k in range(len(local_pos)):
                if bcs[local_range[0] + local_pos[k]]:
                    A_local[k, :] = 0

        if i in slave_cells:
            A_local_copy = A_local.copy()
            cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                        cell_to_slave_offset[index+1]]
            index += 1

            # Find which slaves belongs to each cell
            global_slaves = []
            for gi, slave in enumerate(slaves):
                if slaves[gi] in cell_slaves:
                    global_slaves.append(gi)
            for s_0 in range(len(global_slaves)):
                slave_index = global_slaves[s_0]
                cell_masters = masters[offsets[slave_index]:
                                       offsets[slave_index+1]]
                cell_coeffs = coefficients[offsets[slave_index]:
                                           offsets[slave_index+1]]
                # Variable for local position of slave dof
                slave_local = 0
                for k in range(len(local_pos)):
                    if local_pos[k] + local_range[0] == slaves[slave_index]:
                        slave_local = k

                # Loop through each master dof to take individual contributions
                for m_0 in range(len(cell_masters)):
                    ce = cell_coeffs[m_0]

                    # Special case if master is the same as slave,
                    # aka nothing should be done
                    if slaves[slave_index] == cell_masters[m_0]:
                        print("No slaves (since slave is same as master dof)")
                        continue

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
                            if local_pos[k] + local_range[0] == slaves[other_slave]:
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

                            # Only insert once per pair, but remove local values for all slaves
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
                        if g >= local_size:
                            global_pos[h] = global_indices[g]
                        else:
                            global_pos[h] = g + local_range[0]
                    global_pos[slave_local] = cell_masters[m_0]
                    m0_index[0] = cell_masters[m_0]
                    ierr_row = set_values(A, 3, ffi_fb(global_pos),
                                          1, ffi_fb(m0_index),
                                          ffi_fb(A_row), mode)
                    assert(ierr_row == 0)

                    # Add slave columns to master row
                    ierr_col = set_values(A, 1, ffi_fb(m0_index),
                                          3, ffi_fb(global_pos),
                                          ffi_fb(A_col), mode)
                    assert(ierr_col == 0)
                    # Add slave contributions to A_(master, master)
                    ierr_m0m0 = set_values(A, 1, ffi_fb(m0_index),
                                           1, ffi_fb(m0_index),
                                           ffi_fb(A_master), mode)
                    assert(ierr_m0m0 == 0)

                # Add contributions for different masters on the same cell

                for m_0 in range(len(cell_masters)):
                    for m_1 in range(m_0+1, len(cell_masters)):

                        A_c0.fill(0.0)
                        c0, c1 = cell_coeffs[m_0], cell_coeffs[m_1]
                        A_c0[0, 0] += c0*c1*A_local_copy[slave_local,
                                                         slave_local]
                        m0_index[0] = cell_masters[m_0]
                        m1_index[0] = cell_masters[m_1]
                        ierr_c0 = set_values(A, 1, ffi_fb(m0_index),
                                             1, ffi_fb(m1_index),
                                             ffi_fb(A_c0), mode)
                        assert(ierr_c0 == 0)
                        A_c1.fill(0.0)
                        A_c1[0, 0] += c0*c1*A_local_copy[slave_local,
                                                         slave_local]
                        ierr_c1 = set_values(A, 1, ffi_fb(m1_index),
                                             1, ffi_fb(m0_index),
                                             ffi_fb(A_c1), mode)
                        assert(ierr_c1 == 0)

        # Insert local contribution
        ierr_loc = set_values_local(A, 3, ffi_fb(local_pos),
                                    3, ffi_fb(local_pos),
                                    ffi_fb(A_local), mode)
        assert(ierr_loc == 0)

    # Insert actual Dirichlet condition
    # if len(bcs)>1:
    #     bc_value = numpy.array([[1]], dtype=PETSc.ScalarType)
    #     for i in range(len(bcs)):
    #         if bcs[i]:
    #             bc_row = numpy.array([i],dtype=numpy.int32)
    #             ierr_bc = set_values(A, 1, ffi_fb(bc_row), 1,
    #                                  ffi_fb(bc_row),
    #                                  ffi_fb(bc_value), mode)
    #             assert(ierr_bc == 0)

    # insert zero dirchlet to freeze slave dofs for back substitution
    # for i, slave in enumerate(slaves):
    #     # Do not add to matrix if slave is equal to master
    #     # (equivalent of empty condition)
    #     cell_masters = masters[offsets[i]:offsets[i+1]]
    #     if slave in cell_masters:
    #         break
    #     A_slave[0,0] = 1
    #     slave_pos = numpy.array([slave],dtype=numpy.int32)
    #     ierr_slave = set_values(A, 1,ffi_fb(slave_pos),
    #                             1, ffi_fb(slave_pos),
    #                             ffi_fb(A_slave), mode)
    #     assert(ierr_slave == 0)
    sink(A_m0m1, A_m1m0, m0_index, m1_index, A_row, global_pos,
         A_col, A_master, A_c0, A_c1, A_local, local_pos)

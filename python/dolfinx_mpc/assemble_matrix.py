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


from .numba_setup import (PETSc, ffi, insert, mode, set_values,
                          set_values_local, sink)


def assemble_matrix(form, multipointconstraint, bcs=[]):
    """
    Assembles a ufl form given a multi point constraint and possible
    Dirichlet boundary conditions.
    NOTE: Dirichlet conditions cant be on master dofs.
    """
    dolfinx.log.log(dolfinx.log.LogLevel.INFO, "Assemble MPC matrix")
    bc_array = numpy.array([])
    if len(bcs) > 0:
        for bc in bcs:
            # Extract local index of possible sub space
            bc_array = numpy.append(bc_array, bc.dof_indices[:, 0])
    # Get data from function space
    assert(form.arguments()[0].ufl_function_space() ==
           form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()
    dofmap = V.dofmap
    dofs = dofmap.list.array()
    indexmap = dofmap.index_map
    ghost_info = (indexmap.local_range, indexmap.block_size,
                  indexmap.global_indices(False), indexmap.ghosts)

    # Get data from mesh
    pos = V.mesh.geometry.dofmap.offsets()
    x_dofs = V.mesh.geometry.dofmap.array()
    x = V.mesh.geometry.x

    # Generate ufc_form
    ufc_form = dolfinx.jit.ffcx_jit(form)

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object

    # Pack constants and coefficients
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Create sparsity pattern
    # trad_pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_form)
    # trad_pattern.assemble()
    tt = dolfinx.common.Timer("MPC: Assemble matrix (sparsitypattern total)")
    pattern = multipointconstraint.create_sparsity_pattern(cpp_form)
    pattern.assemble()
    tt.stop()

    # # extra_nonz_loc = pattern.num_nonzeros()-trad_pattern.num_nonzeros()
    # # if extra_nonz_loc < 0:
    # #     raise ValueError("Should not be less nonzeros locally")
    # # comm = V.mesh.mpi_comm()
    # # extra_nonz_glob = sum(comm.allgather(extra_nonz_loc))
    # # glob_nonz = sum(comm.allgather(trad_pattern.num_nonzeros()))
    # if comm.rank == 0:
    #     print("Num extra Nonzeros: {0:d} (Total {2:d}, #dofs {1:d})"
    #             .format(extra_nonz_glob, V.dim, glob_nonz))

    A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    with dolfinx.common.Timer("MPC: Assemble (classicial components)") as t:
        dolfinx.cpp.fem.assemble_matrix(A, cpp_form, bcs)
    # Unravel data from MPC
    slave_cells = multipointconstraint.slave_cells()
    coefficients = multipointconstraint.coefficients()
    masters = multipointconstraint.masters_local()
    slave_cell_to_dofs = multipointconstraint.slave_cell_to_dofs()
    cell_to_slave = slave_cell_to_dofs.array()
    c_to_s_off = slave_cell_to_dofs.offsets()
    slaves = multipointconstraint.slaves()
    slaves_local = multipointconstraint.slaves_local()
    masters_local = masters.array()
    offsets = masters.offsets()
    mpc_data = (slaves, masters_local, coefficients, offsets,
                slave_cells, cell_to_slave, c_to_s_off, slaves_local)

    # General assembly data
    num_dofs_per_element = dofmap.dof_layout.num_dofs
    gdim = V.mesh.geometry.dim
    tdim = V.mesh.topology.dim

    # Get integrals as FormInegral
    formintegral = cpp_form.integrals()

    # Assemble over cells
    subdomain_ids = formintegral.integral_ids(
        dolfinx.cpp.fem.FormIntegrals.Type.cell)
    num_cell_integrals = len(subdomain_ids)

    if num_cell_integrals > 0:
        V.mesh.topology.create_entity_permutations()
        permutation_info = V.mesh.topology.get_cell_permutation_info()

    for i in range(num_cell_integrals):
        subdomain_id = subdomain_ids[i]
        with dolfinx.common.Timer("MPC: Assemble matrix (cell kernel)") as t:
            cell_kernel = ufc_form.create_cell_integral(
                subdomain_id).tabulate_tensor
        active_cells = numpy.array(formintegral.integral_domains(
            dolfinx.cpp.fem.FormIntegrals.Type.cell, i), dtype=numpy.int64)
        slave_cell_indices = numpy.flatnonzero(
            numpy.isin(active_cells, slave_cells))
        with dolfinx.common.Timer("MPC: Assemble matrix (numba cells)") as t:
            assemble_cells(A.handle, cell_kernel,
                           active_cells[slave_cell_indices],
                           (pos, x_dofs, x),
                           gdim, form_coeffs, form_consts,
                           permutation_info,
                           dofs, num_dofs_per_element, mpc_data,
                           ghost_info, bc_array)

    # Assemble over exterior facets
    subdomain_ids = formintegral.integral_ids(
        dolfinx.cpp.fem.FormIntegrals.Type.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)

    # Get cell orientation data
    if num_exterior_integrals > 0:
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        permutation_info = V.mesh.topology.get_cell_permutation_info()
        facet_permutation_info = V.mesh.topology.get_facet_permutations()
        perm = (permutation_info, facet_permutation_info)
    for j in range(num_exterior_integrals):
        facet_info = pack_facet_info(V.mesh, formintegral, j)

        subdomain_id = subdomain_ids[j]
        with dolfinx.common.Timer("MPC: Assemble matrix (ext. facet kernel)"
                                  ) as t:
            facet_kernel = ufc_form.create_exterior_facet_integral(
                subdomain_id).tabulate_tensor
        with dolfinx.common.Timer("MPC: Assemble matrix (numba ext. facet)"
                                  ) as t:
            assemble_exterior_facets(A.handle, facet_kernel,
                                     (pos, x_dofs, x), gdim,
                                     form_coeffs, form_consts,
                                     perm, dofs, num_dofs_per_element,
                                     facet_info, mpc_data, ghost_info,
                                     bc_array)

    with dolfinx.common.Timer("MPC: Assemble matrix (diagonal handling)") as t:
        A.assemble()

        # Add one on diagonal for diriclet bc and slave dofs
        add_diagonal(A.handle, slaves)
        A.assemble()
        if bcs is not None:
            if cpp_form.function_space(0).id == cpp_form.function_space(1).id:
                dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_space(0),
                                             bcs, 1.0)
        A.assemble()
        t.elapsed()
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


@numba.njit(cache=True)
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
    facet_info = pack_facet_info_numba(active_facets,
                                       (c_to_f.array(), c_to_f.offsets()),
                                       (f_to_c.array(), f_to_c.offsets()))
    return facet_info


@numba.njit(fastmath=True, cache=True)
def pack_facet_info_numba(active_facets, c_to_f, f_to_c):
    facet_info = numpy.zeros((len(active_facets), 2),
                             dtype=numpy.int64)
    c_to_f_pos, c_to_f_offs = c_to_f
    f_to_c_pos, f_to_c_offs = f_to_c

    for j, facet in enumerate(active_facets):
        cells = f_to_c_pos[f_to_c_offs[facet]:f_to_c_offs[facet+1]]
        assert(len(cells) == 1)
        local_facets = c_to_f_pos[c_to_f_offs[cells[0]]:
                                  c_to_f_offs[cells[0]+1]]
        # Should be wrapped in convenience numba function
        local_index = numpy.flatnonzero(facet == local_facets)[0]
        facet_info[j, :] = [cells[0], local_index]
    return facet_info


@numba.njit
def assemble_cells(A, kernel, active_cells, mesh, gdim, coeffs, constants,
                   permutation_info, dofmap,
                   num_dofs_per_element, mpc, ghost_info, bcs):
    """
    Assemble MPC contributions for cell integrals
    """
    ffi_fb = ffi.from_buffer

    # Get mesh and geometry data
    pos, x_dofmap, x = mesh

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1]-pos[0], gdim))

    A_local = numpy.zeros((num_dofs_per_element, num_dofs_per_element),
                          dtype=PETSc.ScalarType)

    # Loop over all cells
    slave_cell_index = 0
    for cell_index in active_cells:
        num_vertices = pos[cell_index + 1] - pos[cell_index]

        # Compute vertices of cell from mesh data
        # FIXME: This assumes a particular geometry dof layout
        cell = pos[cell_index]
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]

        A_local.fill(0.0)
        # FIXME: Numba does not support edge reflections
        kernel(ffi_fb(A_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm),
               permutation_info[cell_index])

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

        A_local_copy = A_local.copy()
        # If this slave contains a slave dof, modify local contribution
        modify_mpc_cell(A, slave_cell_index, A_local, A_local_copy, local_pos,
                        mpc, ghost_info, num_dofs_per_element)
        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy
        slave_cell_index += 1

        # Insert local contribution
        ierr_loc = set_values_local(A, num_dofs_per_element, ffi_fb(local_pos),
                                    num_dofs_per_element, ffi_fb(local_pos),
                                    ffi_fb(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_pos)


@numba.njit
def assemble_exterior_facets(A, kernel, mesh, gdim, coeffs, consts, perm,
                             dofmap, num_dofs_per_element, facet_info,
                             mpc, ghost_info, bcs):
    """Assemble MPC contributions over exterior facet integrals"""

    slave_cells = mpc[4]  # Note: packing order for MPC really important

    # Mesh data
    pos, x_dofmap, x = mesh

    # Empty arrays for facet information
    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1] - pos[0], gdim))

    A_local = numpy.zeros((num_dofs_per_element, num_dofs_per_element),
                          dtype=PETSc.ScalarType)

    # Permutation info
    cell_perms, facet_perms = perm

    # Loop over all external facets that are active
    for i in range(facet_info.shape[0]):
        cell_index, local_facet = facet_info[i]
        if not in_numpy_array(slave_cells, cell_index):
            continue
        slave_cell_index = numpy.flatnonzero(slave_cells == cell_index)[0]

        cell = pos[cell_index]
        facet_index[0] = local_facet
        num_vertices = pos[cell_index + 1] - pos[cell_index]

        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]

        A_local.fill(0.0)
        facet_perm[0] = facet_perms[local_facet, cell_index]
        kernel(ffi.from_buffer(A_local),
               ffi.from_buffer(coeffs[cell_index, :]),
               ffi.from_buffer(consts),
               ffi.from_buffer(geometry),
               ffi.from_buffer(facet_index),
               ffi.from_buffer(facet_perm),
               cell_perms[cell_index])

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

        A_local_copy = A_local.copy()
        # If this slave contains a slave dof, modify local contribution
        modify_mpc_cell(A, slave_cell_index, A_local, A_local_copy, local_pos,
                        mpc, ghost_info, num_dofs_per_element)

        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy

        # Insert local contribution
        ierr_loc = set_values_local(A,
                                    num_dofs_per_element, ffi.from_buffer(
                                        local_pos),
                                    num_dofs_per_element, ffi.from_buffer(
                                        local_pos),
                                    ffi.from_buffer(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_pos)


@numba.njit
def modify_mpc_cell(A, slave_cell_index, A_local, A_local_copy, local_pos,
                    mpc, ghost_info, num_dofs_per_element):
    """
    Modifies A_local as it contains slave degrees of freedom.
    Adds contributions to corresponding master degrees of freedom in A.
    """
    ffi_fb = ffi.from_buffer

    # Unpack MPC data
    (slaves, masters_local, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offset, slaves_local) = mpc

    # Unpack ghost data
    local_range, block_size, global_indices, ghosts = ghost_info

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

    cell_slaves = cell_to_slave[cell_to_slave_offset[slave_cell_index]:
                                cell_to_slave_offset[slave_cell_index+1]]

    # Find which slaves belongs to each cell
    global_slaves = []
    for gi, slave in enumerate(slaves):
        if in_numpy_array(cell_slaves, slaves[gi]):
            global_slaves.append(gi)
    for s_0 in range(len(global_slaves)):
        slave_index = global_slaves[s_0]
        cell_masters = masters_local[offsets[slave_index]:
                                     offsets[slave_index+1]]
        cell_coeffs = coefficients[offsets[slave_index]:
                                   offsets[slave_index+1]]
        # Variable for local position of slave dof
        slave_local = numpy.flatnonzero(
            slaves_local[slave_index] == local_pos)[0]
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
                o_slave_local = numpy.flatnonzero(
                    slaves_local[other_slave] == local_pos)[0]

                other_cell_masters = masters_local[offsets[other_slave]:
                                                   offsets[other_slave+1]]
                o_coeffs = coefficients[offsets[other_slave]:
                                        offsets[other_slave+1]]
                # Find local index of other masters
                for m_1 in range(len(other_cell_masters)):
                    A_m0m1.fill(0)
                    A_m1m0.fill(0)
                    o_c = o_coeffs[m_1]
                    A_m0m1[0, 0] = ce*o_c*A_local_copy[slave_local,
                                                       o_slave_local]
                    A_m1m0[0, 0] = ce*o_c*A_local_copy[o_slave_local,
                                                       slave_local]
                    A_row[o_slave_local, 0] = 0
                    A_col[0, o_slave_local] = 0
                    m0_index[0] = cell_masters[m_0]
                    m1_index[0] = other_cell_masters[m_1]
                    # Insert only once per slave pair on each cell
                    if o_slave_index > s_0:
                        ierr_m0m1 = set_values_local(A, 1, ffi_fb(m0_index),
                                                     1, ffi_fb(m1_index),
                                                     ffi_fb(A_m0m1), mode)
                        assert(ierr_m0m1 == 0)
                        ierr_m1m0 = set_values_local(A, 1, ffi_fb(m1_index),
                                                     1, ffi_fb(m0_index),
                                                     ffi_fb(A_m1m0), mode)
                        assert(ierr_m1m0 == 0)

            # Add slave column to master column
            mpc_pos = local_pos.copy()
            mpc_pos[slave_local] = cell_masters[m_0]
            m0_index[0] = cell_masters[m_0]

            ierr_row = set_values_local(A,
                                        num_dofs_per_element, ffi_fb(mpc_pos),
                                        1, ffi_fb(m0_index),
                                        ffi_fb(A_row), mode)
            assert(ierr_row == 0)

            # Add slave row to master row
            ierr_col = set_values_local(A,
                                        1, ffi_fb(m0_index),
                                        num_dofs_per_element, ffi_fb(mpc_pos),
                                        ffi_fb(A_col), mode)

            assert(ierr_col == 0)
            # Add slave contributions to A_(master, master)
            ierr_m0m0 = set_values_local(A,
                                         1, ffi_fb(m0_index),
                                         1, ffi_fb(m0_index),
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
                ierr_c0 = set_values_local(A,
                                           1, ffi_fb(m0_index),
                                           1, ffi_fb(m1_index),
                                           ffi_fb(A_c0), mode)
                assert(ierr_c0 == 0)
                A_c1.fill(0.0)
                A_c1[0, 0] += c0*c1*A_local_copy[slave_local, slave_local]
                ierr_c1 = set_values_local(A,
                                           1, ffi_fb(m1_index),
                                           1, ffi_fb(m0_index),
                                           ffi_fb(A_c1), mode)
                assert(ierr_c1 == 0)

    sink(A_m0m1, A_m1m0, m1_index, A_row, A_col, m0_index, mpc_pos,
         A_master, A_c0, A_c1)

    return A_local

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

from .numba_setup import PETSc, ffi, mode, set_values_local, sink
from dolfinx_mpc import cpp
Timer = dolfinx.common.Timer


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


def pack_facet_info(mesh, active_facets):
    """
    Given the mesh, FormIntgrals and the index of the i-th exterior
    facet integral, for each active facet, find the cell index and
    local facet index and pack them in a numpy nd array
    """
    # FIXME: Should be moved to dolfinx C++ layer
    # Set up data required for exterior facet assembly
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1
    # This connectivities has been computed by normal assembly
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    facet_info = pack_facet_info_numba(active_facets,
                                       (c_to_f.array, c_to_f.offsets),
                                       (f_to_c.array, f_to_c.offsets))
    return facet_info


@numba.njit(fastmath=True, cache=True)
def pack_facet_info_numba(active_facets, c_to_f, f_to_c):
    facet_info = numpy.zeros((len(active_facets), 2), dtype=numpy.int64)
    c_to_f_pos, c_to_f_offs = c_to_f
    f_to_c_pos, f_to_c_offs = f_to_c

    for j, facet in enumerate(active_facets):
        cells = f_to_c_pos[f_to_c_offs[facet]:f_to_c_offs[facet + 1]]
        assert(len(cells) == 1)
        local_facets = c_to_f_pos[c_to_f_offs[cells[0]]: c_to_f_offs[cells[0] + 1]]
        # Should be wrapped in convenience numba function
        local_index = numpy.flatnonzero(facet == local_facets)[0]
        facet_info[j, :] = [cells[0], local_index]
    return facet_info


def assemble_matrix_cpp(form, constraint, bcs=[]):
    assert(form.arguments()[0].ufl_function_space() == form.arguments()[1].ufl_function_space())

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object
    A = cpp.mpc.create_matrix(cpp_form, constraint._cpp_object)
    A.zeroEntries()
    cpp.mpc.assemble_matrix(A, cpp_form, constraint._cpp_object, bcs)

    # Add one on diagonal for Dirichlet boundary conditions
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0], bcs, 1)
    with dolfinx.common.Timer("~MPC: A.assemble()"):
        A.assemble()
    return A


def assemble_matrix(form, constraint, bcs=[], A=None):
    """
    Assembles a ufl form given a multi point constraint and possible
    Dirichlet boundary conditions.
    NOTE: Dirichlet conditions cant be on master dofs.
    """

    timer_matrix = Timer("~MPC: Assemble matrix")

    # Get data from function space
    assert(form.arguments()[0].ufl_function_space() == form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()
    dofmap = V.dofmap
    dofs = dofmap.list.array

    # Unravel data from MPC
    slave_cells = constraint.slave_cells()
    coefficients = numpy.array(constraint.coefficients(), dtype=PETSc.ScalarType)
    masters = constraint.masters_local()
    slave_cell_to_dofs = constraint.cell_to_slaves()
    cell_to_slave = slave_cell_to_dofs.array
    c_to_s_off = slave_cell_to_dofs.offsets
    slaves_local = constraint.slaves()
    num_local_slaves = constraint.num_local_slaves()
    masters_local = masters.array
    offsets = masters.offsets
    mpc_data = (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off)

    # Gather BC data
    bc_array = numpy.array([])
    if len(bcs) > 0:
        for bc in bcs:
            bc_indices, _ = bc.dof_indices()
            bc_array = numpy.append(bc_array, bc_indices)

    # Get data from mesh
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x

    # Generate ufc_form

    ufc_form = dolfinx.jit.ffcx_jit(V.mesh.mpi_comm(), form)

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object

    # Pack constants and coefficients
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = numpy.array(dolfinx.cpp.fem.pack_constants(cpp_form), dtype=PETSc.ScalarType)

    # Create sparsity pattern
    if A is None:
        pattern = constraint.create_sparsity_pattern(cpp_form)
        with Timer("~MPC: Assemble sparsity pattern"):
            pattern.assemble()
        with Timer("~MPC: Assemble matrix (Create matrix)"):
            A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    with Timer("~MPC: Assemble unconstrained matrix"):
        dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, bcs)

    # General assembly data
    block_size = dofmap.dof_layout.block_size()
    num_dofs_per_element = dofmap.dof_layout.num_dofs

    gdim = V.mesh.geometry.dim
    tdim = V.mesh.topology.dim

    # Assemble over cells
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)

    if num_cell_integrals > 0:
        timer = Timer("~MPC: Assemble matrix (cells)")
        V.mesh.topology.create_entity_permutations()
        permutation_info = numpy.array(V.mesh.topology.get_cell_permutation_info(), dtype=numpy.uint32)
        for subdomain_id in subdomain_ids:
            cell_kernel = ufc_form.create_cell_integral(subdomain_id).tabulate_tensor
            active_cells = numpy.array(cpp_form.domains(dolfinx.fem.IntegralType.cell, subdomain_id), dtype=numpy.int64)
            slave_cell_indices = numpy.flatnonzero(numpy.isin(active_cells, slave_cells))
            assemble_cells(A.handle, cell_kernel, active_cells[slave_cell_indices], (pos, x_dofs, x), gdim, form_coeffs,
                           form_consts, permutation_info, dofs, block_size, num_dofs_per_element, mpc_data, bc_array)
        timer.stop()

    # Assemble over exterior facets
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)

    # Get cell orientation data
    if num_exterior_integrals > 0:
        timer = Timer("~MPC: Assemble matrix (ext. facet kernel)")
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        permutation_info = numpy.array(V.mesh.topology.get_cell_permutation_info(), dtype=numpy.uint32)
        facet_permutation_info = V.mesh.topology.get_facet_permutations()
        perm = (permutation_info, facet_permutation_info)
        for subdomain_id in subdomain_ids:
            active_facets = numpy.array(cpp_form.domains(
                dolfinx.fem.IntegralType.exterior_facet, subdomain_id), dtype=numpy.int64)
            facet_info = pack_facet_info(V.mesh, active_facets)
            facet_kernel = ufc_form.create_exterior_facet_integral(subdomain_id).tabulate_tensor
            assemble_exterior_facets(A.handle, facet_kernel, (pos, x_dofs, x), gdim, form_coeffs, form_consts,
                                     perm, dofs, block_size, num_dofs_per_element, facet_info, mpc_data, bc_array)
        timer.stop()

    with Timer("~MPC: Assemble matrix (diagonal handling)"):
        # Add one on diagonal for diriclet bc and slave dofs
        # NOTE: In the future one could use a constant in the DirichletBC
        if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
            dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0],
                                         bcs, 1.0)

    # Add mpc entries on diagonal
    add_diagonal(A.handle, slaves_local[:num_local_slaves])

    with Timer("~MPC: Assemble matrix (Finalize matrix)"):
        A.assemble()
    timer_matrix.stop()
    return A


@numba.jit
def add_diagonal(A, dofs):
    """
    Insert one on diagonal of matrix for given dofs.
    """
    ffi_fb = ffi.from_buffer
    dof_list = numpy.zeros(1, dtype=numpy.int32)
    dof_value = numpy.ones(1, dtype=PETSc.ScalarType)
    for dof in dofs:
        dof_list[0] = dof
        ierr_loc = set_values_local(A, 1, ffi_fb(dof_list), 1, ffi_fb(dof_list), ffi_fb(dof_value), mode)
        assert(ierr_loc == 0)
    sink(dof_list, dof_value)


@numba.njit
def assemble_cells(A, kernel, active_cells, mesh, gdim, coeffs, constants,
                   permutation_info, dofmap, block_size, num_dofs_per_element, mpc, bcs):
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
    geometry = numpy.zeros((pos[1] - pos[0], gdim))

    A_local = numpy.zeros((block_size * num_dofs_per_element, block_size
                           * num_dofs_per_element), dtype=PETSc.ScalarType)

    # Loop over all cells
    for slave_cell_index, cell_index in enumerate(active_cells):
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
        kernel(ffi_fb(A_local), ffi_fb(coeffs[cell_index, :]), ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm), permutation_info[cell_index])

        # Local dof position
        local_blocks = dofmap[num_dofs_per_element
                              * cell_index: num_dofs_per_element * cell_index + num_dofs_per_element]
        # Remove all contributions for dofs that are in the Dirichlet bcs
        if len(bcs) > 0:
            for j in range(num_dofs_per_element):
                for k in range(block_size):
                    is_bc_dof = in_numpy_array(bcs, local_blocks[j] * block_size + k)
                    if is_bc_dof:
                        A_local[j * block_size + k, :] = 0
                        A_local[:, j * block_size + k] = 0

        A_local_copy = A_local.copy()

        # Find local position of slaves
        (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off) = mpc
        slave_indices = cell_to_slave[c_to_s_off[slave_cell_index]: c_to_s_off[slave_cell_index + 1]]
        num_slaves = len(slave_indices)
        local_indices = numpy.full(num_slaves, -1, dtype=numpy.int32)
        is_slave = numpy.full(block_size * num_dofs_per_element, False, dtype=numpy.bool_)
        for i in range(num_slaves):
            for j in range(num_dofs_per_element):
                for k in range(block_size):
                    if local_blocks[j] * block_size + k == slaves_local[slave_indices[i]]:
                        local_indices[i] = j * block_size + k
                        is_slave[j * block_size + k] = True
        mpc_cell = (slave_indices, masters_local, coefficients, offsets, local_indices, is_slave)
        modify_mpc_cell_new(A, num_dofs_per_element, block_size, A_local, local_blocks, mpc_cell)
        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy
        # Expand local blocks to dofs
        local_dofs = numpy.zeros(block_size * num_dofs_per_element, dtype=numpy.int32)
        for i in range(num_dofs_per_element):
            for j in range(block_size):
                local_dofs[i * block_size + j] = local_blocks[i] * block_size + j
        # Insert local contribution
        ierr_loc = set_values_local(A, block_size * num_dofs_per_element, ffi_fb(local_dofs),
                                    block_size * num_dofs_per_element, ffi_fb(local_dofs), ffi_fb(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_dofs)


@numba.njit
def modify_mpc_cell_new(A, num_dofs, block_size, Ae, local_blocks, mpc_cell):

    (slave_indices, masters, coeffs, offsets, local_indices, is_slave) = mpc_cell
    # Flatten slave->master structure to 1D arrays
    flattened_masters = []
    flattened_slaves = []
    slaves_loc = []
    flattened_coeffs = []
    for i in range(len(slave_indices)):
        local_masters = masters[offsets[slave_indices[i]]: offsets[slave_indices[i] + 1]]
        local_coeffs = coeffs[offsets[slave_indices[i]]: offsets[slave_indices[i] + 1]]
        slaves_loc.extend([i, ] * len(local_masters))
        flattened_slaves.extend([local_indices[i], ] * len(local_masters))
        flattened_masters.extend(local_masters)
        flattened_coeffs.extend(local_coeffs)
    # Create element matrix stripped of slave entries
    Ae_original = numpy.copy(Ae)
    Ae_stripped = numpy.zeros((block_size * num_dofs, block_size * num_dofs), dtype=PETSc.ScalarType)
    for i in range(block_size * num_dofs):
        for j in range(block_size * num_dofs):
            if is_slave[i] and is_slave[j]:
                Ae_stripped[i, j] = 0
            else:
                Ae_stripped[i, j] = Ae_original[i, j]

    m0 = numpy.zeros(1, dtype=numpy.int32)
    m1 = numpy.zeros(1, dtype=numpy.int32)
    Am0m1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    Arow = numpy.zeros((block_size * num_dofs, 1), dtype=PETSc.ScalarType)
    Acol = numpy.zeros((1, block_size * num_dofs), dtype=PETSc.ScalarType)

    num_masters = len(flattened_masters)
    ffi_fb = ffi.from_buffer
    for i in range(num_masters):
        local_index = flattened_slaves[i]
        master = flattened_masters[i]
        coeff = flattened_coeffs[i]
        Ae[:, local_index] = 0
        Ae[local_index, :] = 0
        m0[0] = master
        Arow[:, 0] = coeff * Ae_stripped[:, local_index]
        Acol[0, :] = coeff * Ae_stripped[local_index, :]
        Am0m1[0, 0] = coeff**2 * Ae_original[local_index, local_index]
        mpc_dofs = numpy.zeros(block_size * num_dofs, dtype=numpy.int32)
        for j in range(num_dofs):
            for k in range(block_size):
                mpc_dofs[j * block_size + k] = local_blocks[j] * block_size + k
        mpc_dofs[local_index] = master
        ierr_row = set_values_local(A, block_size * num_dofs, ffi_fb(mpc_dofs), 1, ffi_fb(m0), ffi_fb(Arow), mode)
        assert(ierr_row == 0)

        # Add slave row to master row
        ierr_col = set_values_local(A, 1, ffi_fb(m0), block_size * num_dofs, ffi_fb(mpc_dofs), ffi_fb(Acol), mode)
        assert(ierr_col == 0)

        ierr_master = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m0), ffi_fb(Am0m1), mode)
        assert(ierr_master == 0)

        # Create[0, ...., N]\[i]
        other_masters = numpy.arange(0, num_masters - 1, dtype=numpy.int32)
        if i < num_masters - 1:
            other_masters[i] = num_masters - 1
        for j in other_masters:
            other_local_index = flattened_slaves[j]
            other_master = flattened_masters[j]
            other_coeff = flattened_coeffs[j]
            m1[0] = other_master
            if slaves_loc[i] != slaves_loc[j]:
                Am0m1[0, 0] = coeff * other_coeff * Ae_original[local_index, other_local_index]
                ierr_other_slave = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m1), ffi_fb(Am0m1), mode)
                assert(ierr_other_slave == 0)
            else:
                Am0m1[0, 0] = coeff * other_coeff * Ae_original[local_index, local_index]
                ierr_same_slave = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m1), ffi_fb(Am0m1), mode)
                assert(ierr_same_slave == 0)
    sink(Arow, Acol, Am0m1, m0, m1, mpc_dofs)


@numba.njit
def assemble_exterior_facets(A, kernel, mesh, gdim, coeffs, consts, perm,
                             dofmap, block_size, num_dofs_per_element, facet_info, mpc, bcs):
    """Assemble MPC contributions over exterior facet integrals"""

    slave_cells = mpc[4]  # Note: packing order for MPC really important

    # Mesh data
    pos, x_dofmap, x = mesh

    # Empty arrays for facet information
    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1] - pos[0], gdim))

    A_local = numpy.zeros((num_dofs_per_element * block_size,
                           num_dofs_per_element * block_size), dtype=PETSc.ScalarType)

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
        kernel(ffi.from_buffer(A_local), ffi.from_buffer(coeffs[cell_index, :]), ffi.from_buffer(consts),
               ffi.from_buffer(geometry), ffi.from_buffer(facet_index), ffi.from_buffer(facet_perm),
               cell_perms[cell_index])

        local_blocks = dofmap[num_dofs_per_element
                              * cell_index: num_dofs_per_element * cell_index + num_dofs_per_element]

        # Remove all contributions for dofs that are in the Dirichlet bcs
        if len(bcs) > 0:
            for k in range(len(local_blocks)):
                is_bc_dof = in_numpy_array(bcs, local_blocks[k])
                if is_bc_dof:
                    A_local[k, :] = 0
                    A_local[:, k] = 0

        A_local_copy = A_local.copy()
        #  If this slave contains a slave dof, modify local contribution
        # modify_mpc_cell_local(A, slave_cell_index, A_local, A_local_copy, local_pos, mpc, num_dofs_per_element)
        # Find local position of slaves
        (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off) = mpc
        slave_indices = cell_to_slave[c_to_s_off[slave_cell_index]: c_to_s_off[slave_cell_index + 1]]
        num_slaves = len(slave_indices)
        local_indices = numpy.full(num_slaves, -1, dtype=numpy.int32)
        is_slave = numpy.full(block_size * num_dofs_per_element, False, dtype=numpy.bool_)
        for i in range(num_slaves):
            for j in range(num_dofs_per_element):
                for k in range(block_size):
                    if local_blocks[j] * block_size + k == slaves_local[slave_indices[i]]:
                        local_indices[i] = j * block_size + k
                        is_slave[j * block_size + k] = True
        mpc_cell = (slave_indices, masters_local, coefficients, offsets, local_indices, is_slave)
        modify_mpc_cell_new(A, num_dofs_per_element, block_size, A_local, local_blocks, mpc_cell)

        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy
        # Expand local blocks to dofs
        local_dofs = numpy.zeros(block_size * num_dofs_per_element, dtype=numpy.int32)
        for i in range(num_dofs_per_element):
            for j in range(block_size):
                local_dofs[i * block_size + j] = local_blocks[i] * block_size + j
        # Insert local contribution
        ierr_loc = set_values_local(A, block_size * num_dofs_per_element, ffi.from_buffer(local_dofs),
                                    block_size * num_dofs_per_element, ffi.from_buffer(local_dofs),
                                    ffi.from_buffer(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_dofs)


@numba.njit
def modify_mpc_cell_local(A, slave_cell_index, A_local, A_local_copy, local_pos, mpc, num_dofs_per_element):
    """
    Modifies A_local as it contains slave degrees of freedom.
    Adds contributions to corresponding master degrees of freedom in A.
    """
    ffi_fb = ffi.from_buffer

    # Unpack MPC data
    (slaves, masters_local, coefficients, offsets, slave_cells, cell_to_slave, cell_to_slave_offsets) = mpc

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

    cell_slaves = cell_to_slave[cell_to_slave_offsets[slave_cell_index]: cell_to_slave_offsets[slave_cell_index + 1]]
    # Find local indices for each slave
    slave_indices = numpy.zeros(len(cell_slaves), dtype=numpy.int32)
    for i, slave_index in enumerate(cell_slaves):
        for j, dof in enumerate(local_pos):
            if dof == slaves[slave_index]:
                slave_indices[i] = j
                break

    for i, slave_index in enumerate(cell_slaves):
        cell_masters = masters_local[offsets[slave_index]: offsets[slave_index + 1]]
        cell_coeffs = coefficients[offsets[slave_index]: offsets[slave_index + 1]]
        local_idx = slave_indices[i]
        # Loop through each master dof to take individual contributions
        for i_0, (master, coeff) in enumerate(zip(cell_masters, cell_coeffs)):
            # Reset local contribution matrices
            A_row.fill(0.0)
            A_col.fill(0.0)
            # Move local contributions with correct coefficients
            A_row[:, 0] = coeff * A_local_copy[:, local_idx]
            A_col[0, :] = coeff * A_local_copy[local_idx, :]
            A_master[0, 0] = coeff * A_row[local_idx, 0]

            # Remove row contribution going to central addition
            A_col[0, local_idx] = 0
            A_row[local_idx, 0] = 0

            # Remove local contributions moved to master
            A_local[:, local_idx] = 0
            A_local[local_idx, :] = 0

            # If one of the other local indices are a slave,
            # move them to the corresponding master dof
            # and multiply by the corresponding coefficient
            for j, other_slave in enumerate(cell_slaves):
                o_local_idx = slave_indices[j]
                # Zero out for other slave
                A_row[o_local_idx, 0] = 0
                A_col[0, o_local_idx] = 0

                # Masters for other cell
                other_cell_masters = masters_local[offsets[other_slave]: offsets[other_slave + 1]]
                other_coeffs = coefficients[offsets[other_slave]: offsets[other_slave + 1]]
                # Add cross terms for masters
                for (other_master, other_coeff) in zip(other_cell_masters, other_coeffs):
                    A_m0m1.fill(0)
                    A_m1m0.fill(0)
                    A_m0m1[0, 0] = coeff * other_coeff * A_local_copy[local_idx, o_local_idx]
                    A_m1m0[0, 0] = coeff * other_coeff * A_local_copy[o_local_idx, local_idx]
                    m0_index[0] = master
                    m1_index[0] = other_master
                    # Insert only once per slave pair on each cell
                    if j > i:
                        ierr_m0m1 = set_values_local(A, 1, ffi_fb(m0_index), 1, ffi_fb(m1_index), ffi_fb(A_m0m1), mode)
                        assert(ierr_m0m1 == 0)
                        ierr_m1m0 = set_values_local(A, 1, ffi_fb(m1_index), 1, ffi_fb(m0_index), ffi_fb(A_m1m0), mode)
                        assert(ierr_m1m0 == 0)

            # Add slave column to master column
            mpc_pos = local_pos.copy()
            mpc_pos[local_idx] = master
            m0_index[0] = master

            ierr_row = set_values_local(A, num_dofs_per_element, ffi_fb(mpc_pos),
                                        1, ffi_fb(m0_index), ffi_fb(A_row), mode)
            assert(ierr_row == 0)

            # Add slave row to master row
            ierr_col = set_values_local(A, 1, ffi_fb(m0_index), num_dofs_per_element,
                                        ffi_fb(mpc_pos), ffi_fb(A_col), mode)
            assert(ierr_col == 0)
            # Add slave contributions to A_(master, master)
            ierr_m0m0 = set_values_local(A, 1, ffi_fb(m0_index), 1, ffi_fb(m0_index), ffi_fb(A_master), mode)

            assert(ierr_m0m0 == 0)

            # Add contributions for different masters on the same cell
            for i_1 in range(i_0 + 1, len(cell_masters)):
                A_c0.fill(0.0)
                c1 = cell_coeffs[i_1]
                A_c0[0, 0] += coeff * c1 * A_local_copy[local_idx, local_idx]
                m0_index[0] = master
                m1_index[0] = cell_masters[i_1]
                ierr_c0 = set_values_local(A, 1, ffi_fb(m0_index), 1, ffi_fb(m1_index), ffi_fb(A_c0), mode)
                assert(ierr_c0 == 0)
                A_c1.fill(0.0)
                A_c1[0, 0] += coeff * c1 * A_local_copy[local_idx, local_idx]
                ierr_c1 = set_values_local(A, 1, ffi_fb(m1_index), 1, ffi_fb(m0_index), ffi_fb(A_c1), mode)
                assert(ierr_c1 == 0)
    sink(A_m0m1, A_m1m0, m1_index, A_row, A_col, m0_index, mpc_pos, A_master, A_c0, A_c1)

    return A_local

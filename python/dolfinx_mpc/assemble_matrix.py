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
        dolfinx.fem.IntegralType.exterior_facet, i)
    facet_info = pack_facet_info_numba(active_facets,
                                       (c_to_f.array, c_to_f.offsets),
                                       (f_to_c.array, f_to_c.offsets))
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


def assemble_matrix_cpp(form, constraint, bcs=[]):
    timer_matrix = Timer("~MPC: Assemble matrix (C++)")

    assert(form.arguments()[0].ufl_function_space() ==
           form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object
    pattern = constraint.create_sparsity_pattern(cpp_form)
    pattern.assemble()
    with Timer("~MPC: Assemble matrix (Create matrix)"):
        A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
        A.zeroEntries()
    cpp.mpc.assemble_matrix(A, cpp_form, constraint._cpp_object, bcs)

    # Add Dirichlet boundary conditions (including slave dofs)
    slaves_local = constraint.slaves()
    local_slave_dofs_trimmed = slaves_local[:constraint.num_local_slaves()]
    V = form.arguments()[0].ufl_function_space()
    bc_mpc = [dolfinx.DirichletBC(
        dolfinx.Function(V), local_slave_dofs_trimmed)]
    if len(bcs) > 0:
        bc_mpc.extend(bcs)
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0], bc_mpc, 1)
    A.assemble()
    timer_matrix.stop()
    return A


def assemble_matrix(form, constraint, bcs=[]):
    """
    Assembles a ufl form given a multi point constraint and possible
    Dirichlet boundary conditions.
    NOTE: Dirichlet conditions cant be on master dofs.
    """
    timer_matrix = Timer("~MPC: Assemble matrix")

    # Get data from function space
    assert(form.arguments()[0].ufl_function_space() ==
           form.arguments()[1].ufl_function_space())
    V = form.arguments()[0].ufl_function_space()
    dofmap = V.dofmap
    dofs = dofmap.list.array

    # Unravel data from MPC
    slave_cells = constraint.slave_cells()
    coefficients = constraint.coefficients()
    masters = constraint.masters_local()
    slave_cell_to_dofs = constraint.cell_to_slaves()
    cell_to_slave = slave_cell_to_dofs.array
    c_to_s_off = slave_cell_to_dofs.offsets
    slaves_local = constraint.slaves()
    num_local_slaves = constraint.num_local_slaves()
    masters_local = masters.array
    offsets = masters.offsets
    mpc_data = (slaves_local, masters_local, coefficients, offsets,
                slave_cells, cell_to_slave, c_to_s_off)

    # Gather BC data
    bc_array = numpy.array([])
    local_slave_dofs_trimmed = slaves_local[:num_local_slaves]
    bc_mpc = [dolfinx.DirichletBC(
        dolfinx.Function(V), local_slave_dofs_trimmed)]

    if len(bcs) > 0:
        bc_mpc.extend(bcs)
        for bc in bcs:
            # Extract local index of possible sub space
            bc_array = numpy.append(bc_array, bc.dof_indices[:, 0])

    # Get data from mesh
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x

    # Generate ufc_form
    ufc_form = dolfinx.jit.ffcx_jit(form)

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object

    # Pack constants and coefficients
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Create sparsity pattern
    pattern = constraint.create_sparsity_pattern(cpp_form)
    pattern.assemble()
    with Timer("~MPC: Assemble matrix (Create matrix)"):
        A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
        A.zeroEntries()

    # Assemble the matrix with all entries
    with Timer("~MPC: Assemble matrix (classical components)"):
        dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, bcs)

    # General assembly data
    num_dofs_per_element = (dofmap.dof_layout.num_dofs *
                            dofmap.dof_layout.block_size())

    gdim = V.mesh.geometry.dim
    tdim = V.mesh.topology.dim

    # Get integrals as FormIntegral
    formintegral = cpp_form.integrals

    # Assemble over cells
    subdomain_ids = formintegral.integral_ids(
        dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)

    if num_cell_integrals > 0:
        V.mesh.topology.create_entity_permutations()
        permutation_info = V.mesh.topology.get_cell_permutation_info()

    for i in range(num_cell_integrals):
        subdomain_id = subdomain_ids[i]
        with Timer("~MPC: Assemble matrix (cell kernel)"):
            cell_kernel = ufc_form.create_cell_integral(
                subdomain_id).tabulate_tensor
        active_cells = numpy.array(formintegral.integral_domains(
            dolfinx.fem.IntegralType.cell, i), dtype=numpy.int64)
        slave_cell_indices = numpy.flatnonzero(
            numpy.isin(active_cells, slave_cells))
        with Timer("~MPC: Assemble matrix (numba cells)"):
            assemble_cells(A.handle, cell_kernel,
                           active_cells[slave_cell_indices],
                           (pos, x_dofs, x),
                           gdim, form_coeffs, form_consts,
                           permutation_info,
                           dofs, num_dofs_per_element, mpc_data,
                           bc_array)

    # Assemble over exterior facets
    subdomain_ids = formintegral.integral_ids(
        dolfinx.fem.IntegralType.exterior_facet)
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
        with Timer("~MPC: Assemble matrix (ext. facet kernel)"):
            facet_kernel = ufc_form.create_exterior_facet_integral(
                subdomain_id).tabulate_tensor
        with Timer("~MPC: Assemble matrix (numba ext. facet)"):
            assemble_exterior_facets(A.handle, facet_kernel,
                                     (pos, x_dofs, x), gdim,
                                     form_coeffs, form_consts,
                                     perm, dofs, num_dofs_per_element,
                                     facet_info, mpc_data,
                                     bc_array)

    with Timer("~MPC: Assemble matrix (diagonal handling)"):
        # Add one on diagonal for diriclet bc and slave dofs
        # NOTE: In the future one could use a constant in the DirichletBC
        if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
            dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0],
                                         bc_mpc, 1.0)
    with Timer("~MPC: Assemble matrix (Finalize matrix)"):
        A.assemble()

    timer_matrix.stop()
    return A


@numba.njit
def assemble_cells(A, kernel, active_cells, mesh, gdim, coeffs, constants,
                   permutation_info, dofmap,
                   num_dofs_per_element, mpc, bcs):
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
        modify_mpc_cell_local(A, slave_cell_index, A_local, A_local_copy,
                              local_pos, mpc, num_dofs_per_element)
        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy

        # Insert local contribution
        ierr_loc = set_values_local(A, num_dofs_per_element, ffi_fb(local_pos),
                                    num_dofs_per_element, ffi_fb(local_pos),
                                    ffi_fb(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_pos)


@numba.njit
def assemble_exterior_facets(A, kernel, mesh, gdim, coeffs, consts, perm,
                             dofmap, num_dofs_per_element, facet_info,
                             mpc, bcs):
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
        modify_mpc_cell_local(A, slave_cell_index, A_local, A_local_copy,
                              local_pos, mpc, num_dofs_per_element)

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
def modify_mpc_cell_local(A, slave_cell_index, A_local, A_local_copy,
                          local_pos, mpc, num_dofs_per_element):
    """
    Modifies A_local as it contains slave degrees of freedom.
    Adds contributions to corresponding master degrees of freedom in A.
    """
    ffi_fb = ffi.from_buffer

    # Unpack MPC data
    (slaves, masters_local, coefficients, offsets, slave_cells,
     cell_to_slave, cell_to_slave_offsets) = mpc

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

    cell_slaves = cell_to_slave[cell_to_slave_offsets[slave_cell_index]:
                                cell_to_slave_offsets[slave_cell_index+1]]
    # Find local indices for each slave
    slave_indices = numpy.zeros(len(cell_slaves), dtype=numpy.int32)
    for i, slave_index in enumerate(cell_slaves):
        for j, dof in enumerate(local_pos):
            if dof == slaves[slave_index]:
                slave_indices[i] = j
                break

    for i, slave_index in enumerate(cell_slaves):
        cell_masters = masters_local[offsets[slave_index]:
                                     offsets[slave_index+1]]
        cell_coeffs = coefficients[offsets[slave_index]:
                                   offsets[slave_index+1]]
        local_idx = slave_indices[i]
        # Loop through each master dof to take individual contributions
        for master, coeff in zip(cell_masters, cell_coeffs):
            # Reset local contribution matrices
            A_row.fill(0.0)
            A_col.fill(0.0)
            # Move local contributions with correct coefficients
            A_row[:, 0] = coeff*A_local_copy[:, local_idx]
            A_col[0, :] = coeff*A_local_copy[local_idx, :]
            A_master[0, 0] = coeff*A_row[local_idx, 0]

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
                other_cell_masters = masters_local[offsets[other_slave]:
                                                   offsets[other_slave+1]]
                other_coeffs = coefficients[offsets[other_slave]:
                                            offsets[other_slave+1]]
                # Add cross terms for masters
                for (other_master, other_coeff) in zip(other_cell_masters,
                                                       other_coeffs):
                    A_m0m1.fill(0)
                    A_m1m0.fill(0)
                    A_m0m1[0, 0] = coeff*other_coeff*A_local_copy[local_idx,
                                                                  o_local_idx]
                    A_m1m0[0, 0] = coeff*other_coeff*A_local_copy[o_local_idx,
                                                                  local_idx]
                    m0_index[0] = master
                    m1_index[0] = other_master
                    # Insert only once per slave pair on each cell
                    if j > i:
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
            mpc_pos[local_idx] = master
            m0_index[0] = master

            ierr_row = set_values_local(A, num_dofs_per_element,
                                        ffi_fb(mpc_pos),
                                        1, ffi_fb(m0_index),
                                        ffi_fb(A_row), mode)
            assert(ierr_row == 0)

            # Add slave row to master row
            ierr_col = set_values_local(A,
                                        1, ffi_fb(m0_index),
                                        num_dofs_per_element,
                                        ffi_fb(mpc_pos),
                                        ffi_fb(A_col), mode)
            assert(ierr_col == 0)
            # Add slave contributions to A_(master, master)
            ierr_m0m0 = set_values_local(A,
                                         1, ffi_fb(m0_index),
                                         1, ffi_fb(m0_index),
                                         ffi_fb(A_master), mode)

            assert(ierr_m0m0 == 0)

        # Add contributions for different masters on the same cell
        for i_0, (m0, c0) in enumerate(zip(cell_masters, cell_coeffs)):
            for i_1 in range(i_0+1, len(cell_masters)):
                A_c0.fill(0.0)
                c1 = cell_coeffs[i_1]
                A_c0[0, 0] += c0*c1 * \
                    A_local_copy[local_idx, local_idx]
                m0_index[0] = m0
                m1_index[0] = cell_masters[i_1]
                ierr_c0 = set_values_local(A,
                                           1, ffi_fb(m0_index),
                                           1, ffi_fb(m1_index),
                                           ffi_fb(A_c0), mode)
                assert(ierr_c0 == 0)
                A_c1.fill(0.0)
                A_c1[0, 0] += c0*c1 * \
                    A_local_copy[local_idx, local_idx]
                ierr_c1 = set_values_local(A,
                                           1, ffi_fb(m1_index),
                                           1, ffi_fb(m0_index),
                                           ffi_fb(A_c1), mode)
                assert(ierr_c1 == 0)
    sink(A_m0m1, A_m1m0, m1_index, A_row, A_col, m0_index, mpc_pos,
         A_master, A_c0, A_c1)

    return A_local

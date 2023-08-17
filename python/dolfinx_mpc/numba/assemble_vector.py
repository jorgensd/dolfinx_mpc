# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from typing import Optional, Tuple

import cffi
import dolfinx.cpp as _cpp
import dolfinx.fem as _fem
import dolfinx.la as _la
import dolfinx.log as _log
import numba
import numpy
import numpy.typing as npt
from dolfinx.common import Timer
from petsc4py import PETSc as _PETSc

from dolfinx_mpc.multipointconstraint import MultiPointConstraint

from .helpers import _forms, extract_slave_cells, pack_slave_facet_info
from .numba_setup import initialize_petsc

ffi, _ = initialize_petsc()


def assemble_vector(form: _forms, constraint: MultiPointConstraint, b: Optional[_PETSc.Vec] = None) -> _PETSc.Vec:
    """
    Assemble a compiled DOLFINx form into vector b.

    Args:
        form: The complied linear form
        constraint: The multi point constraint
        b: PETSc vector to assemble into (optional)
    """

    _log.log(_log.LogLevel.INFO, "Assemble MPC vector")
    timer_vector = Timer("~MPC: Assemble vector (numba)")

    # Unpack Function space data
    V = form.function_spaces[0]
    x_dofs = V.mesh.geometry.dofmap
    x = V.mesh.geometry.x
    dofs = V.dofmap.map()
    block_size = V.dofmap.index_map_bs

    # Data from multipointconstraint
    coefficients = constraint.coefficients()[0]
    masters_adj = constraint.masters
    c_to_s_adj = constraint.cell_to_slaves
    cell_to_slave = c_to_s_adj.array
    c_to_s_off = c_to_s_adj.offsets
    is_slave = constraint.is_slave
    mpc_data = (masters_adj.array, coefficients, masters_adj.offsets, cell_to_slave, c_to_s_off, is_slave)
    slave_cells = extract_slave_cells(c_to_s_off)

    # Get index map and ghost info
    if b is None:
        index_map = constraint.function_space.dofmap.index_map
        vector = _la.create_petsc_vector(index_map, block_size)
    else:
        vector = b

    # Pack constants and coefficients
    form_coeffs = _cpp.fem.pack_coefficients(form._cpp_object)
    form_consts = _cpp.fem.pack_constants(form._cpp_object)

    tdim = V.mesh.topology.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Assemble vector with all entries
    with vector.localForm() as b_local:
        _cpp.fem.assemble_vector(b_local.array_w, form._cpp_object,
                                 form_consts, form_coeffs)

    # Check if we need facet permutations
    # FIXME: access apply_dof_transformations here
    e0 = form.function_spaces[0].element
    needs_transformation_data = e0.needs_dof_transformations or form._cpp_object.needs_facet_permutations
    cell_perms = numpy.array([], dtype=numpy.uint32)
    if needs_transformation_data:
        V.mesh.topology.create_entity_permutations()
        cell_perms = V.mesh.topology.get_cell_permutation_info()
    if e0.needs_dof_transformations:
        raise NotImplementedError("Dof transformations not implemented")
    # Assemble over cells
    subdomain_ids = form._cpp_object.integral_ids(_fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)

    is_complex = numpy.issubdtype(_PETSc.ScalarType, numpy.complexfloating)
    nptype = "complex128" if is_complex else "float64"
    ufcx_form = form.ufcx_form
    if num_cell_integrals > 0:
        V.mesh.topology.create_entity_permutations()

        # NOTE: This depends on enum ordering in ufcx.h
        cell_form_pos = ufcx_form.form_integral_offsets[0]
        for i, id in enumerate(subdomain_ids):
            cell_kernel = getattr(ufcx_form.form_integrals[cell_form_pos + i], f"tabulate_tensor_{nptype}")
            active_cells = form._cpp_object.domains(_fem.IntegralType.cell, id)
            coeffs_i = form_coeffs[(_fem.IntegralType.cell, id)]
            with vector.localForm() as b:
                assemble_cells(numpy.asarray(b), cell_kernel, active_cells[numpy.isin(active_cells, slave_cells)],
                               (x_dofs, x), coeffs_i, form_consts,
                               cell_perms, dofs, block_size, num_dofs_per_element, mpc_data)

    # Assemble exterior facet integrals
    subdomain_ids = form._cpp_object.integral_ids(_fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)
    if num_exterior_integrals > 0:
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        # Get facet permutations if required
        facet_perms = numpy.array([], dtype=numpy.uint8)
        if form._cpp_object.needs_facet_permutations:
            facet_perms = V.mesh.topology.get_facet_permutations()
        perm = (cell_perms, form._cpp_object.needs_facet_permutations, facet_perms)
        # NOTE: This depends on enum ordering in ufcx.h
        ext_facet_pos = ufcx_form.form_integral_offsets[1]
        for i, id in enumerate(subdomain_ids):
            facet_kernel = getattr(ufcx_form.form_integrals[ext_facet_pos + i],
                                   f"tabulate_tensor_{nptype}")
            coeffs_i = form_coeffs[(_fem.IntegralType.exterior_facet, id)]
            facets = form._cpp_object.domains(_fem.IntegralType.exterior_facet, id)
            facet_info = pack_slave_facet_info(facets, slave_cells)
            num_facets_per_cell = len(V.mesh.topology.connectivity(tdim, tdim - 1).links(0))
            with vector.localForm() as b:
                assemble_exterior_slave_facets(numpy.asarray(b), facet_kernel, facet_info, (x_dofs, x),
                                               coeffs_i, form_consts, perm,
                                               dofs, block_size, num_dofs_per_element, mpc_data, num_facets_per_cell)
    timer_vector.stop()
    return vector


@numba.njit
def assemble_cells(b: npt.NDArray[_PETSc.ScalarType],
                   kernel: cffi.FFI, active_cells: npt.NDArray[numpy.int32],
                   mesh: Tuple[npt.NDArray[numpy.int32],
                               npt.NDArray[numpy.float64]],
                   coeffs: npt.NDArray[_PETSc.ScalarType],
                   constants: npt.NDArray[_PETSc.ScalarType],
                   permutation_info: npt.NDArray[numpy.uint32],
                   dofmap: npt.NDArray[numpy.int32],
                   block_size: int,
                   num_dofs_per_element: int,
                   mpc: Tuple[npt.NDArray[numpy.int32], npt.NDArray[_PETSc.ScalarType],
                              npt.NDArray[numpy.int32], npt.NDArray[numpy.int32],
                              npt.NDArray[numpy.int32], npt.NDArray[numpy.int32]]):
    """Assemble additional MPC contributions for cell integrals"""
    ffi_fb = ffi.from_buffer

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # Unpack mesh data
    x_dofmap, x = mesh

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((x_dofmap.shape[1], 3))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=_PETSc.ScalarType)

    for cell_index in active_cells:

        # Compute mesh geometry for cell
        geometry[:, :] = x[x_dofmap[cell_index]]

        # Assemble local element vector
        b_local.fill(0.0)
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm))
        # NOTE: Here we need to add the apply_dof_transformation function

        # Modify global vector and local cell contributions
        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, b_local, b_local_copy, mpc, dofmap,
                                 block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[cell_index, j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit
def assemble_exterior_slave_facets(b: npt.NDArray[_PETSc.ScalarType],
                                   kernel: cffi.FFI,
                                   facet_info: npt.NDArray[numpy.int32],
                                   mesh: Tuple[npt.NDArray[numpy.int32],
                                               npt.NDArray[numpy.float64]],
                                   coeffs: npt.NDArray[_PETSc.ScalarType],
                                   constants: npt.NDArray[_PETSc.ScalarType],
                                   permutation_info: npt.NDArray[numpy.uint32],
                                   dofmap: npt.NDArray[numpy.int32],
                                   block_size: int,
                                   num_dofs_per_element: int,
                                   mpc: Tuple[npt.NDArray[numpy.int32], npt.NDArray[_PETSc.ScalarType],
                                              npt.NDArray[numpy.int32], npt.NDArray[numpy.int32],
                                              npt.NDArray[numpy.int32], npt.NDArray[numpy.int32]],
                                   num_facets_per_cell: int):
    """Assemble additional MPC contributions for facets"""
    ffi_fb = ffi.from_buffer

    # Unpack facet permutation info
    cell_perms, needs_facet_perm, facet_perms = permutation_info
    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # Unpack mesh data
    x_dofmap, x = mesh

    geometry = numpy.zeros((x_dofmap.shape[1], 3))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=_PETSc.ScalarType)
    for i in range(facet_info.shape[0]):
        # Extract cell index (local to process) and facet index (local to cell) for kernel
        cell_index, local_facet = facet_info[i]
        facet_index[0] = local_facet

        # Extract cell geometry
        geometry[:, :] = x[x_dofmap[cell_index]]

        # Compute local facet kernel
        if needs_facet_perm:
            facet_perm[0] = facet_perms[cell_index * num_facets_per_cell + local_facet]
        b_local.fill(0.0)
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm))
        # NOTE: Here we need to add the apply_dof_transformation

        # Modify local contributions and add global MPC contributions
        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, b_local, b_local_copy, mpc, dofmap,
                                 block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[cell_index, j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit(cache=True)
def modify_mpc_contributions(b: npt.NDArray[_PETSc.ScalarType], cell_index: int,
                             b_local: npt.NDArray[_PETSc.ScalarType],
                             b_copy: npt.NDArray[_PETSc.ScalarType],
                             mpc: Tuple[npt.NDArray[numpy.int32], npt.NDArray[_PETSc.ScalarType],
                                        npt.NDArray[numpy.int32], npt.NDArray[numpy.int32],
                                        npt.NDArray[numpy.int32], npt.NDArray[numpy.int32]],
                             dofmap: npt.NDArray[numpy.int32],
                             block_size: int,
                             num_dofs_per_element: int):
    """
    Modify local entries of b_local with MPC info and add modified
    entries to global vector b.
    """

    # Unwrap MPC data
    masters, coefficients, offsets, cell_to_slave, cell_to_slave_offset, is_slave = mpc

    # Determine which slaves are in this cell,
    # and which global index they have in 1D arrays
    cell_slaves = cell_to_slave[cell_to_slave_offset[cell_index]:
                                cell_to_slave_offset[cell_index + 1]]

    # Get local index of slaves in cell
    cell_blocks = dofmap[cell_index]
    local_index = numpy.empty(len(cell_slaves), dtype=numpy.int32)
    for i in range(num_dofs_per_element):
        for j in range(block_size):
            dof = cell_blocks[i] * block_size + j
            if is_slave[dof]:
                location = numpy.flatnonzero(cell_slaves == dof)[0]
                local_index[location] = i * block_size + j

    # Move contribution from each slave to the corresponding master dof
    # and zero out local b
    for local, slave in zip(local_index, cell_slaves):
        cell_masters = masters[offsets[slave]: offsets[slave + 1]]
        cell_coeffs = coefficients[offsets[slave]: offsets[slave + 1]]
        for m0, c0 in zip(cell_masters, cell_coeffs):
            b[m0] += c0 * b_copy[local]
            b_local[local] = 0

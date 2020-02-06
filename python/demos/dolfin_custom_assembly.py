# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""
from IPython import embed

from numba.typed import List
import ctypes
import ctypes.util
import importlib
import math
import os
import time

import cffi
import numba
import numba.cffi_support
import numpy as np
import pytest
from petsc4py import PETSc, get_config as PETSc_get_config

import dolfinx
import dolfinx_mpc
import ufl

petsc_dir = PETSc_get_config()['PETSC_DIR']

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RecursionError("Unknown PETSc index type.")

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        "Cannot translate PETSc scalar type to a C type, complex: {} size: {}.".format(complex, scalar_size))


# Load PETSc library via ctypes
petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_ctypes = ctypes.CDLL(petsc_lib_name)
else:
    try:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

# Get the PETSc MatSetValuesLocal function via ctypes
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValues
MatSetValuesLocal_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValuesLocal_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
MatSetValues_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
MatSetValuesLocal_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
del petsc_lib_ctypes


ADD_VALUES = PETSc.InsertMode.ADD_VALUES


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                 numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                 numba.types.complex64)


# Get MatSetValues from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValues(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);

""".format(c_int_t, c_scalar_t))
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))

if petsc_lib_name is not None:
    petsc_lib_cffi = ffi.dlopen(petsc_lib_name)
else:
    try:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise
MatSetValues_abi = petsc_lib_cffi.MatSetValues
MatSetValuesLocal_abi = petsc_lib_cffi.MatSetValuesLocal

# Make MatSetValuesLocal from PETSc available via cffi in API mode
worker = os.getenv('PYTEST_XDIST_WORKER', None)
module_name = "_petsc_cffi_{}".format(worker)
if dolfinx.MPI.comm_world.Get_rank() == 0:
    os.environ["CC"] = "mpicc"
    ffibuilder = cffi.FFI()
    ffibuilder.cdef("""
        typedef int... PetscInt;
        typedef ... PetscScalar;
        typedef int... InsertMode;
        int MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
        int MatSetValues(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
    """)
    ffibuilder.set_source(module_name, """
        # include "petscmat.h"
    """,
                          libraries=['petsc'],
                          include_dirs=[os.path.join(petsc_dir, 'include')],
                          library_dirs=[os.path.join(petsc_dir, 'lib')],
                          extra_compile_args=[])
    ffibuilder.compile(verbose=False)

dolfinx.MPI.comm_world.barrier()

spec = importlib.util.find_spec(module_name)
if spec is None:
    raise ImportError("Failed to find CFFI generated module")
module = importlib.util.module_from_spec(spec)

numba.cffi_support.register_module(module)
MatSetValues_api = module.lib.MatSetValues
MatSetValuesLocal_api = module.lib.MatSetValuesLocal

numba.cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba_scalar_t)
set_values = MatSetValues_api
set_values_local = MatSetValuesLocal_api
mode = PETSc.InsertMode.ADD_VALUES

# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass

@numba.njit
def backsubstitution(b, mpc):
    (slaves, masters, coefficients, offsets) = mpc
    for i, slave in enumerate(slaves):
        masters_i = masters[offsets[i]:offsets[i+1]]
        coeffs_i = coefficients[offsets[i]:offsets[i+1]]
        for j, (master,coeff) in enumerate(zip(masters_i, coeffs_i)):
            if slave == master:
                print("No slaves (since slave is same as master dof)")
                continue

            b[slave] += coeff*b[master]



@numba.njit
def assemble_vector_ufc(b, kernel, mesh, x, dofmap, mpc, ghost_info, bcs):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    (bcs, values) = bcs
    (slaves, masters, coefficients, offsets, slave_cells,cell_to_slave, cell_to_slave_offset)  = mpc
    local_range, global_indices = ghost_info

    connections, pos = mesh
    orientation = np.array([0], dtype=np.int32)
    geometry = np.zeros((3, 2))
    coeffs = np.zeros(1, dtype=PETSc.ScalarType)
    constants = np.zeros(1, dtype=PETSc.ScalarType)

    b_local = np.zeros(3, dtype=PETSc.ScalarType)
    index = 0
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(orientation),
               ffi.from_buffer(orientation))
        if len(bcs)>1:
            for k in range(3):
                if bcs[dofmap[i * 3 + k]]:
                    b_local[k] = 0

        if i not in slave_cells:
            for j in range(3):
                b[dofmap[i * 3 + j]] += b_local[j]
        else:
            b_local_copy = b_local.copy()
            # Determine which slaves are in this cell, and which global index they have in 1D arrays
            cell_slaves = cell_to_slave[cell_to_slave_offset[index]: cell_to_slave_offset[index +1]]
            index += 1
            glob = dofmap[3 * i:3 * i + 3]
            # Find which slaves belongs to each cell
            global_slaves = []
            for gi, slave in enumerate(slaves):
                if slaves[gi] in cell_slaves:
                    global_slaves.append(gi)
            # Loop over the slaves
            for s_0 in range(len(global_slaves)):
                slave_index = global_slaves[s_0]
                cell_masters = masters[offsets[slave_index]:offsets[slave_index+1]]
                cell_coeffs = coefficients[offsets[slave_index]:offsets[slave_index+1]]
                # Variable for local position of slave dof
                slave_local = 0
                for k in range(len(glob)):
                    print("HERE", global_indices[glob[k]], slaves[slave_index])
                    if global_indices[glob[k]] == slaves[slave_index]:
                        slave_local = k

                # Loop through each master dof to take individual contributions
                for m_0 in range(len(cell_masters)):
                    if slaves[slave_index] == cell_masters[m_0]:
                        print("No slaves (since slave is same as master dof)")
                        continue

                    # Find local dof and add contribution to another place
                    for k in range(len(glob)):
                        if global_indices[glob[k]] == slaves[slave_index]:
                            #b[cell_masters[m_0]] += cell_coeffs[m_0]*b_local_copy[k]
                            b_local[k] = 0
            for k in range(len(glob)):
                print(glob[k], k)
                b[glob[k]] += b_local[k]

    for k in range(len(bcs)):
        if bcs[k]:

            b[k] = values[k]

@numba.njit
def assemble_matrix_ufc(A, kernel, mesh, x, dofmap, mpc, ghost_info, bcs):
    # FIXME: Need to add structures to reverse engineer number of dofs in local matrix
    (slaves, masters, coefficients, offsets, slave_cells,cell_to_slave, cell_to_slave_offset)  = mpc
    (local_range, global_indices) = ghost_info
    local_size = local_range[1]-local_range[0]
    connections, pos = mesh
    orientation = np.array([0], dtype=np.int32)
    # FIXME: Need to determine geometry, coeffs and constants from input data
    geometry = np.zeros((3, 2))
    coeffs = np.zeros(0, dtype=PETSc.ScalarType)
    constants = np.zeros(0, dtype=PETSc.ScalarType)
    A_local = np.zeros((3,3), dtype=PETSc.ScalarType)

    A_mpc_row = np.zeros((3,1), dtype=PETSc.ScalarType) # Rows taken over by master
    A_mpc_col = np.zeros((1,3), dtype=PETSc.ScalarType) # Columns taken over by master
    A_mpc_master = np.zeros((1,1), dtype=PETSc.ScalarType) # Extra insertions at master diagonal
    A_mpc_slave =  np.zeros((1,1), dtype=PETSc.ScalarType) # Setting 0 Dirichlet in original matrix for slave diagonal
    A_mpc_mixed0 =  np.zeros((1,1), dtype=PETSc.ScalarType) # Cross-master coefficients, m0, m1 for same constraint
    A_mpc_mixed1 =  np.zeros((1,1), dtype=PETSc.ScalarType) # Cross-master coefficients, m1, m0 for same constraint
    A_mpc_m0m1 =  np.zeros((1,1), dtype=PETSc.ScalarType) # Cross-master coefficients, m0, m1, master to multiple slave
    A_mpc_m1m0 =  np.zeros((1,1), dtype=PETSc.ScalarType) # Cross-master coefficients, m0, m1, master to multiple slaves
    m0_index = np.zeros(1, dtype=np.int32)
    m1_index = np.zeros(1, dtype=np.int32)



    # Loop over slave cells
    index = 0
    for i,cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]

        A_local.fill(0.0)
        kernel(ffi.from_buffer(A_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(orientation),
               ffi.from_buffer(orientation))

        local_pos = dofmap[3 * i:3 * i + 3]
        if len(bcs)>1:
            for k in range(len(local_pos)):
                if bcs[local_range[0] + local_pos[k]]:
                    A_local[k,:] = 0

        if i in slave_cells:
            A_local_copy = A_local.copy()
            cell_slaves = cell_to_slave[cell_to_slave_offset[index]: cell_to_slave_offset[index +1]]
            index += 1

            # Find which slaves belongs to each cell
            global_slaves = []
            for gi, slave in enumerate(slaves):
                if slaves[gi] in cell_slaves:
                    global_slaves.append(gi)
            for s_0 in range(len(global_slaves)):
                slave_index = global_slaves[s_0]
                cell_masters = masters[offsets[slave_index]:offsets[slave_index+1]]
                cell_coeffs = coefficients[offsets[slave_index]:offsets[slave_index+1]]
                # Variable for local position of slave dof
                slave_local = 0
                for k in range(len(local_pos)):
                    if local_pos[k] + local_range[0] == slaves[slave_index]:
                        slave_local = k

                # Loop through each master dof to take individual contributions
                for m_0 in range(len(cell_masters)):
                    ce = cell_coeffs[m_0]
                    # Special case if master is the same as slave, aka nothing should be done
                    if slaves[slave_index] == cell_masters[m_0]:
                        print("No slaves (since slave is same as master dof)")
                        continue

                    # Reset local contribution matrices
                    A_mpc_row.fill(0.0)
                    A_mpc_col.fill(0.0)

                    # Move local contributions with correct coefficients
                    A_mpc_row[:,0] = ce*A_local_copy[:,slave_local]
                    A_mpc_col[0,:] = ce*A_local_copy[slave_local,:]
                    A_mpc_master[0,0] = ce*A_mpc_row[slave_local,0]

                    # Remove row contribution going to central addition
                    A_mpc_col[0, slave_local] = 0
                    A_mpc_row[slave_local,0] = 0

                    # Remove local contributions moved to master
                    A_local[:,slave_local] = 0
                    A_local[slave_local,:] = 0

                    # If one of the other local indices are a slave, move them to the corresponding
                    # master dof and multiply by the corresponding coefficient
                    for other_slave in global_slaves:
                        # If not another slave, continue
                        if other_slave == slave_index:
                            continue
                        #Find local index of the other slave
                        other_slave_local = 0
                        for k in range(len(local_pos)):
                            if local_pos[k] == slaves[other_slave]:
                                other_slave_local = k
                        other_cell_masters = masters[offsets[other_slave]:offsets[other_slave+1]]
                        other_coefficients = coefficients[offsets[other_slave]:offsets[other_slave+1]]
                        # Find local index of other masters
                        for m_1 in range(len(other_cell_masters)):
                            A_mpc_m0m1.fill(0)
                            A_mpc_m1m0.fill(0)
                            other_coeff = other_coefficients[m_1]
                            A_mpc_m0m1[0,0] = other_coeff*A_mpc_row[other_slave_local,0]
                            A_mpc_m1m0[0,0] = other_coeff*A_mpc_col[0,other_slave_local]
                            A_mpc_row[other_slave_local,0] = 0
                            A_mpc_col[0,other_slave_local] = 0
                            ierr_m0m1 = set_values(A, 1,ffi.from_buffer(m0_index),  1, ffi.from_buffer(m1_index),
                                                   ffi.from_buffer(A_mpc_m0m1), mode)
                            assert(ierr_m0m1 == 0)
                            # Do not set twice if set on diagonal
                            if cell_masters[m_0] != other_cell_masters[m_1]:
                                ierr_m1m0 = set_values(A, 1,ffi.from_buffer(m1_index),  1, ffi.from_buffer(m0_index),
                                                       ffi.from_buffer(A_mpc_m1m0), mode)
                                assert(ierr_m1m0 == 0)


                    # --------------------------------Add slave rows to master column-------------------------------
                    row_local_pos = np.zeros(local_pos.size, dtype=np.int32)
                    for (h,g) in enumerate(local_pos):
                        # Map ghosts to global index and slave to master
                        if g >= local_size:
                            row_local_pos[h] = global_indices[g]
                        else:
                            row_local_pos[h] = g + local_range[0]
                    row_local_pos[slave_local] = cell_masters[m_0]
                    master_row_index = np.array([cell_masters[m_0]],dtype=np.int32)
                    ierr_row = set_values(A, 3,ffi.from_buffer(row_local_pos),  1, ffi.from_buffer(master_row_index),
                                          ffi.from_buffer(A_mpc_row), mode)
                    assert(ierr_row == 0)

                    # --------------------------------Add slave columns to master row-------------------------------
                    col_local_pos = np.zeros(local_pos.size, dtype=np.int32)
                    for (h,g) in enumerate(local_pos):
                        if g >= local_size:
                            col_local_pos[h] = global_indices[g]
                        else:
                            col_local_pos[h] = g + local_range[0]
                    col_local_pos[slave_local] = cell_masters[m_0]

                    master_col_index = np.array([cell_masters[m_0]],dtype=np.int32)
                    ierr_col = set_values(A, 1,ffi.from_buffer(master_col_index), 3, ffi.from_buffer(col_local_pos),
                                          ffi.from_buffer(A_mpc_col), mode)
                    assert(ierr_col == 0)


                    # --------------------------------Add slave contributions to A_(master,master)-----------------
                    master_index = np.array([cell_masters[m_0]],dtype=np.int32)
                    ierr_m0m0 = set_values(A, 1,ffi.from_buffer(master_index),  1, ffi.from_buffer(master_index),
                                          ffi.from_buffer(A_mpc_master), mode)
                    assert(ierr_m0m0 == 0)


                # Add contributions for different masters on the same cell
                for m_0 in range(len(cell_masters)):
                    for m_1 in range(m_0+1, len(cell_masters)):
                        A_mpc_mixed0.fill(0.0)
                        A_mpc_mixed0[0,0] += cell_coeffs[m_0]*cell_coeffs[m_1]*A_local_copy[slave_local,slave_local]


                        mixed0_index = np.array([cell_masters[m_0]],dtype=np.int32)
                        mixed1_index = np.array([cell_masters[m_1]],dtype=np.int32)
                        ierr_mixed0 = set_values(A, 1,ffi.from_buffer(mixed0_index),  1, ffi.from_buffer(mixed1_index),
                                               ffi.from_buffer(A_mpc_mixed0), mode)
                        assert(ierr_mixed0 == 0)
                        A_mpc_mixed1.fill(0.0)
                        mixed2_index = np.array([cell_masters[m_0]],dtype=np.int32)
                        mixed3_index = np.array([cell_masters[m_1]],dtype=np.int32)
                        A_mpc_mixed1[0,0] += cell_coeffs[m_0]*cell_coeffs[m_1]*A_local_copy[slave_local,slave_local]

                        ierr_mixed1 = set_values(A, 1,ffi.from_buffer(mixed3_index),  1, ffi.from_buffer(mixed2_index),
                                             ffi.from_buffer(A_mpc_mixed1), mode)
                        assert(ierr_mixed1 == 0)


        # Insert local contribution
        ierr_loc = set_values_local(A, 3, ffi.from_buffer(local_pos), 3, ffi.from_buffer(local_pos), ffi.from_buffer(A_local), mode)
        assert(ierr_loc == 0)


    # Insert actual Dirichlet condition
    # if len(bcs)>1:
    #     bc_value = np.array([[1]], dtype=PETSc.ScalarType)
    #     for i in range(len(bcs)):
    #         if bcs[i]:
    #             bc_row = np.array([i],dtype=np.int32)
    #             ierr_bc = set_values(A, 1, ffi.from_buffer(bc_row), 1, ffi.from_buffer(bc_row), ffi.from_buffer(bc_value), mode)
    #             assert(ierr_bc == 0)

    # insert zero dirchlet to freeze slave dofs for back substitution
    # for i, slave in enumerate(slaves):
    #     # Do not add to matrix if slave is equal to master (equivalent of empty condition)
    #     cell_masters = masters[offsets[i]:offsets[i+1]]
    #     if slave in cell_masters:
    #         break
    #     A_mpc_slave[0,0] = 1
    #     slave_pos = np.array([slave],dtype=np.int32)
    #     ierr_slave = set_values(A, 1,ffi.from_buffer(slave_pos),  1, ffi.from_buffer(slave_pos),
    #                             ffi.from_buffer(A_mpc_slave), mode)
    #     assert(ierr_slave == 0)
    sink(A_mpc_m0m1, A_mpc_m1m0, m0_index, m1_index,
         A_mpc_row, row_local_pos, master_row_index,
         A_mpc_col, master_col_index, col_local_pos,
         A_mpc_master, master_index,
         A_mpc_mixed0, mixed0_index, mixed1_index,
         A_mpc_mixed1, mixed2_index, mixed3_index,
         A_local, local_pos
    )


def master_dofs(x):
    logical = np.logical_and(np.logical_or(np.isclose(x[:,0],0),np.isclose(x[:,0],1)),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0]
    except IndexError:
        return []

def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []
def bc0(x):
    return np.logical_and(np.isclose(x[:,1], 0),np.isclose(x[:,1], 0))

def bc_value(bcs, x):
    vals = np.zeros(len(bcs))
    for i in range(len(bcs)):
        if bcs[i]:
            vals[i] = 0
    return vals


def test_mpc_assembly():

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 3)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap.dof_array

    # Find master and slave dofs from geometrical search
    dof_coords = V.tabulate_dof_coordinates()
    masters = master_dofs(dof_coords)
    indexmap = V.dofmap.index_map
    ghost_info = (indexmap.local_range, indexmap.indices(True))

    # Build slaves and masters in parallel with global dof numbering scheme
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(np.array(masters, dtype=np.int32))
    masters = np.hstack(masters)
    slaves = slave_dofs(dof_coords)
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+V.dofmap.index_map.local_range[0]
    slaves = dolfinx.MPI.comm_world.allgather(np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    coeffs = [0.11,0.43]
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters,coeffs):
            slave_master_dict[slave][master] = coeff

    # Define bcs
    bcs = np.array([])#bc0(dof_coords)
    values = np.array([]) # values = bc_value(bcs, dof_coords)
    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    l = ufl.inner(f, v)*ufl.dx

    # Timing assembly of matrix with C++
    start = time.time()
    A1 = dolfinx.fem.assemble_matrix(a)
    # end = time.time()
    A1.assemble()

    # print()
    # print("--"*10)
    # print("Matrix Time (C++, pass 1):", end - start)
    # start = time.time()
    # L1 = dolfinx.fem.assemble_vector(l)
    # print(L1.array)
    # end = time.time()
    # print("Vector Time (C++, pass 1):", end - start)
    # print("--"*10)
    # second pass in C++
    # A1.zeroEntries()
    # start = time.time()
    # dolfinx.fem.assemble_matrix(A1, a)
    # end = time.time()
    # A1.assemble()
    # print("Matrix Time (C++, pass 2):", end - start)
    # start = time.time()
    L1 = dolfinx.fem.assemble_vector(l)
    end = time.time()
    # print("Vector Time (C++, pass 1):", end - start)


    # slave_master_dict = {3:{0:0.3,2:0.5}}
    # slave_master_dict = {0:{0:1}}
    # slave_master_dict = {1:{0:0.1},3:{0:0.3,2:0.73}}
    # slave_master_dict = {1:{7:0.1}}


    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)
    slave_cells = np.array(mpc.slave_cells())
    # pattern = dolfinx.cpp.fem.create_sparsity_pattern(dolfinx.Form(a)._cpp_object)
    #pattern.info_statistics()
    #print("-----"*10)
    # dolfinx.cpp.fem.create_matrix(dolfinx.Form(a)._cpp_object)
    # pattern.info_statistics()
    #print("-"*25)
    # Add mpc non-zeros to sparisty pattern#
    print(slave_master_dict)

    A2 = mpc.generate_petsc_matrix(dolfinx.Form(a)._cpp_object)
    A2.zeroEntries()
    A1.assemble()

    # Data from mpc
    masters, coefficients = mpc.masters_and_coefficients()
    cell_to_slave, cell_to_slave_offset = mpc.cell_to_slave_mapping()
    slaves = mpc.slaves()
    offsets = mpc.master_offsets()


    # Wrapping for numba to be able to do "if i in slave_cells"
    if len(slave_cells)==0:
        sc_nb = List.empty_list(numba.types.int64)
    else:
        sc_nb = List()
    [sc_nb.append(sc) for sc in slave_cells]

    # Can be empty list locally, so has to be wrapped to be used with numba
    if len(cell_to_slave)==0:
        c2s_nb = List.empty_list(numba.types.int64)
    else:
        c2s_nb = List()
    [c2s_nb.append(c2s) for c2s in cell_to_slave]
    if len(cell_to_slave_offset)==0:
        c2so_nb = List.empty_list(numba.types.int64)
    else:
        c2so_nb = List()
    [c2so_nb.append(c2so) for c2so in cell_to_slave_offset]

    # Assemble using generated tabulate_tensor kernel and Numba assembler
    ufc_form = dolfinx.jit.ffcx_jit(a)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        A2.zeroEntries()
        start = time.time()
        assemble_matrix_ufc(A2.handle, kernel, (c, pos), geom, dofs,
                            (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info,bcs)
        end = time.time()
        print("Matrix Time (numba/cffi, pass {}): {}".format(i, end - start))

        A2.assemble()


    # A2.view()
    assert(A2.norm() == 11.34150823303497)
    return
    print(A2.norm())
    # K = np.array([[1,0,0,0,0,0,0,0],
    #                [0,1,0,0,0,0,0,0],
    #                [0,0,1,0,0,0,0,0],
    #                [0,0,0,1,0,0,0,0],
    #                [0,0,0,0,1,0,0,0],
    #                [0,0,0,0,0,1,0,0],
    #                [0,0,0,0,0,beta,0,0],
    #                [0,0,0,0,0,0,1,0],
    #                [0,0,0,0,0,0,0,1]])
    # global_A = np.matmul(np.matmul(K.T,A1[:,:]),K)
    # print(global_A)
    # print(A2[:,:])
    # for local, glob in zip([0,1,2,3,4,5,7,8],[0,1,2,3,4,5,6,7]):
    #     for local2, glob2 in zip([0,1,2,3,4,5,7,8],[0,1,2,3,4,5,6,7]):
    #         assert(A2[local,local2] == global_A[glob, glob2])
    # # print(reduced)
    # assert np.max(d-reduced) <= 1e-6
    # print(A1.view())
    # print(A2.view())
    b_func = dolfinx.Function(V)
    ufc_form = dolfinx.jit.ffcx_jit(l)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        with b_func.vector.localForm() as b:
            b.set(0.0)
            start = time.time()
            assemble_vector_ufc(np.asarray(b), kernel, (c, pos), geom, dofs,
                                (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info, (bcs, values))
            end = time.time()
            print("Vector Time (numba/cffi, pass {}): {}".format(i, end - start))

    print(L1.array)
    b_func.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    global_L = dolfinx.MPI.comm_world.allgather(b_func.vector.array)
    from dolfinx.io import XDMFFile
    XDMFFile(dolfinx.MPI.comm_world, "lhs.xdmf").write(b_func)

    print("----")

    return
    # Solve problem
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A2)

    # Solve
    uh = dolfinx.Function(V)
    solver.solve(b_func.vector, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    print("solution")
    print(uh.vector.array)
    with uh.vector.localForm() as b:
        backsubstitution(b, (slaves, masters, coefficients, offsets))
    print(uh.vector.array)
    for slave in slave_master_dict.keys():
        u_slave = 0
        for master in slave_master_dict[slave].keys():
            u_slave += slave_master_dict[slave][master]*uh.vector.array[master]
        print(uh.vector.array[slave], u_slave)
    from dolfinx.io import VTKFile
    VTKFile("uh.pvd").write(uh)
    # assert((A1 - A2).norm() == pytest.approx(0.0))


if __name__ == "__main__":
    test_mpc_assembly()

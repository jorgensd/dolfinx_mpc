# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ctypes
import ctypes.util
import importlib
import os
import pathlib

import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np

import cffi
import petsc4py.lib
from mpi4py import MPI
from petsc4py import get_config as PETSc_get_config
from petsc4py import PETSc

# Get details of PETSc install
petsc_dir = PETSc_get_config()['PETSC_DIR']
petsc_arch = petsc4py.lib.getPathArchPETSc()[1]

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
    raise RuntimeError(
        "Cannot translate PETSc index size into a C type, index_size: {}."
        .format(index_size))

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
        "Cannot translate PETSc scalar type to a C type, complex: {} size: {}."
        .format(complex, scalar_size))


# Load PETSc library via ctypes
petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_ctypes = ctypes.CDLL(petsc_lib_name)
else:
    try:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(
            petsc_dir, petsc_arch, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(
            petsc_dir, petsc_arch, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

# Get the PETSc MatSetValuesLocal function via ctypes
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index),
    ctypes.c_void_p, ctypes.c_int)
del petsc_lib_ctypes


ADD_VALUES = PETSc.InsertMode.ADD_VALUES


# CFFI - register complex types
ffi = cffi.FFI()
cffi_support.register_type(ffi.typeof(
    'double _Complex'), numba.types.complex128)
cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)

# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValues(void* mat, {0} nrow, const {0}* irow,
                             {0} ncol, const {0}* icol,
                             const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
            {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))

if petsc_lib_name is not None:
    petsc_lib_cffi = ffi.dlopen(petsc_lib_name)
else:
    try:
        petsc_lib_cffi = ffi.dlopen(os.path.join(
            petsc_dir, petsc_arch, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_cffi = ffi.dlopen(os.path.join(
            petsc_dir, petsc_arch, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise
MatSetValues_abi = petsc_lib_cffi.MatSetValues
MatSetValuesLocal_abi = petsc_lib_cffi.MatSetValuesLocal

# Make MatSetValuesLocal from PETSc available via cffi in API mode
worker = os.getenv('ASSEMBLE_XDIST_WORKER', None)
module_name = "_petsc_cffi_{}".format(worker)
if MPI.COMM_WORLD.Get_rank() == 0:
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
                          include_dirs=[os.path.join(petsc_dir, petsc_arch,
                                                     'include'),
                                        os.path.join(petsc_dir, 'include')],
                          library_dirs=[os.path.join(
                              petsc_dir, petsc_arch, 'lib')],
                          extra_compile_args=[])

    # Build module in same directory as test file
    path = pathlib.Path(__file__).parent.absolute()
    # ffibuilder.compile(tmpdir=path, verbose=False)
    ffibuilder.compile(verbose=False)

MPI.COMM_WORLD.Barrier()
module = importlib.import_module("dolfinx_mpc." + module_name)

cffi_support.register_module(module)
MatSetValues_api = module.lib.MatSetValues
MatSetValuesLocal_api = module.lib.MatSetValuesLocal

cffi_support.register_type(module.ffi.typeof("PetscScalar"),
                           numba_scalar_t)
set_values = MatSetValues_api
set_values_local = MatSetValuesLocal_api
mode = PETSc.InsertMode.ADD_VALUES
insert = PETSc.InsertMode.INSERT_VALUES


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass

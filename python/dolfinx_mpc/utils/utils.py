# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import dolfinx
import dolfinx_mpc
import ufl
import time

from mpi4py import MPI


def create_transformation_matrix(dim, slaves, masters, coeffs, offsets):
    """
    Creates the transformation matrix K (dim x dim-len(slaves)) f
    or a given set of slaves, masters and coefficients.
    All input is given as 1D arrays, where offsets[j] indicates where
    the first master and corresponding coefficient of slaves[j] is located.

    Example:

    For dim=3, where:
      u_1 = alpha u_0 + beta u_2

    Input:
      slaves = [1]
      masters = [0, 2]
      coeffs = [alpha, beta]
      offsets = [0, 1]

    Output:
      K = [[1,0], [alpha beta], [0,1]]
    """
    K = np.zeros((dim, dim - len(slaves)))
    for i in range(K.shape[0]):
        if i in slaves:
            index = np.argwhere(slaves == i)[0, 0]
            masters_index = masters[offsets[index]: offsets[index+1]]
            coeffs_index = coeffs[offsets[index]: offsets[index+1]]
            for master, coeff in zip(masters_index, coeffs_index):
                count = sum(master > np.array(slaves))
                K[i, master - count] = coeff
        else:
            count = sum(i > slaves)
            K[i, i-count] = 1
    return K


def PETScVector_to_global_numpy(vector):
    """
    Gather a PETScVector from different processors on
    all processors as a numpy array
    """
    numpy_vec = np.zeros(vector.size, dtype=vector.array.dtype)
    l_min = vector.owner_range[0]
    l_max = vector.owner_range[1]
    numpy_vec[l_min:l_max] += vector.array
    numpy_vec = sum(MPI.COMM_WORLD.allgather(numpy_vec))
    return numpy_vec


def PETScMatrix_to_global_numpy(A):
    """
    Gather a PETScMatrix from different processors on
    all processors as a numpy nd array.
    """

    B = A.convert("dense")
    B_np = B.getDenseArray()
    A_numpy = np.zeros((A.size[0], A.size[1]), dtype=B_np.dtype)
    o_range = A.getOwnershipRange()
    A_numpy[o_range[0]:o_range[1], :] = B_np
    A_numpy = sum(MPI.COMM_WORLD.allgather(A_numpy))
    return A_numpy


def compare_vectors(reduced_vec, vec, global_zeros):
    """
    Compare two numpy vectors of different lengths,
    where global_zeros are the indices of vec that are not in
    reduced vec.
    """
    count = 0
    for i in range(len(vec)):
        if i in global_zeros:
            count += 1
            assert np.isclose(vec[i], 0)
        else:
            assert(np.isclose(reduced_vec[i-count], vec[i]))


def compare_matrices(reduced_A, A, global_ones):
    """
    Compare two numpy matrices of different sizes, where global_ones
    corresponds to places where matrix has 1 on the diagonal,
    and the reduced_matrix does not have entries.
    Example:
    Input:
      A = [[0.1, 0 0.3], [0, 1, 0], [0.2, 0, 0.4]]
      reduced_A = [[0.1, 0.3], [0.2, 0.4]]
      global_ones = [1]
    """

    A_numpy_padded = np.zeros(A.shape, dtype=reduced_A.dtype)
    count = 0
    for i in range(A.shape[0]):
        if i in global_ones:
            A_numpy_padded[i, i] = 1
            count += 1
            continue
        m = 0
        for j in range(A.shape[1]):
            if j in global_ones:
                m += 1
                continue
            else:
                A_numpy_padded[i, j] = reduced_A[i-count, j-m]
    D = np.abs(A - A_numpy_padded)

    max_index = np.unravel_index(np.argmax(D, axis=None), D.shape)
    if D[max_index] > 1e-6:
        print("Unequal ({0:.2e}) at ".format(D[max_index]), max_index)
        print(A_numpy_padded[max_index], A[max_index])
        print("Unequal at:", np.argwhere(D != 0))

    # Check that all entities are close
    assert np.allclose(A, A_numpy_padded)


def cache_numba(matrix=False, vector=False, backsubstitution=False):
    """
    Build a minimal numba cache for all operations
    """
    print("Building Numba cache...")
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD,
                                  MPI.COMM_WORLD.size,
                                  MPI.COMM_WORLD.size)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(dolfinx.Constant(mesh, 1), v) * ufl.dx
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object,
                                                   np.array([],
                                                            dtype=np.int64),
                                                   np.array([],
                                                            dtype=np.int64),
                                                   np.array([],
                                                            dtype=np.float64),
                                                   np.array([],
                                                            dtype=np.int64))
    if matrix:
        start = time.time()
        A = dolfinx_mpc.assemble_matrix(a, mpc, [])
        end = time.time()
        print("CACHE Matrix assembly time: {0:.2e} ".format(end-start))
        A.assemble()
    if vector:
        start = time.time()
        b = dolfinx_mpc.assemble_vector(L, mpc)
        end = time.time()
        print("CACHE Vector assembly time: {0:.2e} ".format(end-start))

        if backsubstitution:
            c = b.copy()
            start = time.time()
            dolfinx_mpc.backsubstitution(mpc, c, V.dofmap)
            end = time.time()
            print("CACHE backsubstitution: {0:.2e} ".format(end-start))

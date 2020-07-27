# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from contextlib import ExitStack

import dolfinx_mpc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.common
import dolfinx.log
import dolfinx.la


def create_transformation_matrix(V, constraint):
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
    slaves_local = constraint.slaves()[:constraint.num_local_slaves()]
    # Should gather slaves here
    slaves = slaves_local
    masters = constraint.masters_local().array()
    coeffs = constraint.coefficients()
    offsets = constraint.masters_local().offsets()
    K = np.zeros((V.dim, V.dim - len(slaves)), dtype=PETSc.ScalarType)

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
    dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                    "Building Numba cache...")
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
                                                   np.array([0],
                                                            dtype=np.int64),
                                                   np.array([],
                                                            dtype=np.int32))
    if matrix:
        with dolfinx.common.Timer("MPC: Cache Matrix"):
            A = dolfinx_mpc.assemble_matrix(a, mpc, [])
            A.assemble()
    if vector:
        with dolfinx.common.Timer("MPC: Cache Vector"):
            b = dolfinx_mpc.assemble_vector(L, mpc)

        if backsubstitution:
            c = b.copy()
            with dolfinx.common.Timer("MPC: Cache Backsubstitution"):
                dolfinx_mpc.backsubstitution(mpc, c, V.dofmap)


def log_info(message):
    """
    Wrapper for logging a simple string on the zeroth communicator
    Reverting the log level
    """
    old_level = dolfinx.log.get_log_level()
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        message)
        dolfinx.log.set_log_level(old_level)


def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [dolfinx.cpp.la.create_vector(
        V.dofmap.index_map) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm())
                     for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        x = V.tabulate_dof_coordinates()
        dofs = [V.sub(i).dofmap.list.array() for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][V.sub(i).dofmap.list.array()] = 1.0

        # Build rotational null space basis
        if gdim == 2:
            basis[2][dofs[0]] = -x[dofs[0], 1]
            basis[2][dofs[1]] = x[dofs[1], 0]
        elif gdim == 3:
            basis[3][dofs[0]] = -x[dofs[0], 1]
            basis[3][dofs[1]] = x[dofs[1], 0]

            basis[4][dofs[0]] = x[dofs[0], 2]
            basis[4][dofs[2]] = -x[dofs[2], 0]
            basis[5][dofs[2]] = x[dofs[2], 1]
            basis[5][dofs[1]] = -x[dofs[1], 2]

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(dim)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp

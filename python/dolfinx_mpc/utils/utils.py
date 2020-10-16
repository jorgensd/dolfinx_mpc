# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from contextlib import ExitStack

import dolfinx_mpc.cpp
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.common
import dolfinx.la
import dolfinx.log


def rotation_matrix(axis, angle):
    # See https://en.wikipedia.org/wiki/Rotation_matrix,
    # Subsection: Rotation_matrix_from_axis_and_angle.
    if np.isclose(np.inner(axis, axis), 1):
        n_axis = axis
    else:
        # Normalize axis
        n_axis = axis / np.sqrt(np.inner(axis, axis))

    # Define cross product matrix of axis
    axis_x = np.array([[0, -n_axis[2], n_axis[1]],
                       [n_axis[2], 0, -n_axis[0]],
                       [-n_axis[1], n_axis[0], 0]])
    id = np.cos(angle) * np.eye(3)
    outer = (1 - np.cos(angle)) * np.outer(n_axis, n_axis)
    return np.sin(angle) * axis_x + id + outer


def facet_normal_approximation(V, mt, mt_id, tangent=False):
    timer = dolfinx.common.Timer("~MPC: Facet normal projection")
    comm = V.mesh.mpi_comm()
    n = ufl.FacetNormal(V.mesh)
    nh = dolfinx.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    if tangent:
        if V.mesh.geometry.dim == 1:
            raise ValueError("Tangent not defined for 1D problem")
        elif V.mesh.geometry.dim == 2:
            a = ufl.inner(u, v) * ds
            L = ufl.inner(ufl.as_vector([-n[1], n[0]]), v) * ds
        else:
            def tangential_proj(u, n):
                """
                See for instance:
                https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
                """
                return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u
            c = dolfinx.Constant(V.mesh, [1, 1, 1])
            a = ufl.inner(u, v) * ds
            L = ufl.inner(tangential_proj(c, n), v) * ds
    else:
        a = (ufl.inner(u, v) * ufl.ds)
        L = ufl.inner(n, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_dofs = np.array(
        range(imap.size_local * imap.block_size), dtype=np.int32)
    top_facets = mt.indices[np.flatnonzero(mt.values == mt_id)]
    top_dofs = dolfinx.fem.locate_dofs_topological(
        V, V.mesh.topology.dim - 1, top_facets)[:, 0]
    deac_dofs = all_dofs[np.isin(all_dofs, top_dofs, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    cpp_form = dolfinx.Form(a)._cpp_object
    pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_form)
    block_size = V.dofmap.index_map.block_size
    blocks = np.unique(deac_dofs // block_size)
    dolfinx_mpc.cpp.mpc.add_pattern_diagonal(
        pattern, blocks, block_size)
    pattern.assemble()

    u_0 = dolfinx.Function(V)
    u_0.vector.set(0)

    bc_deac = dolfinx.fem.DirichletBC(u_0, deac_dofs)
    A = dolfinx.cpp.la.create_matrix(comm, pattern)
    A.zeroEntries()
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0],
                                     [bc_deac], 1.0)
    # Assemble the matrix with all entries
    dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, [bc_deac])
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)

    dolfinx.fem.apply_lifting(b, [a], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc_deac])

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    timer.stop()
    return nh


def gather_slaves_global(constraint):
    """
    Given a multi point constraint,
    return slaves for all processors with global dof numbering
    """
    loc_to_glob = np.array(
        constraint.index_map().global_indices(False), dtype=np.int64)
    if constraint.num_local_slaves() > 0:
        glob_slaves = np.array(loc_to_glob[
            constraint.slaves()[:constraint.num_local_slaves()]],
            dtype=np.int64)
    else:
        glob_slaves = np.array([], dtype=np.int64)

    slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    return slaves


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
    # Gather slaves from all procs
    loc_to_glob = np.array(
        constraint.index_map().global_indices(False), dtype=np.int64)
    if constraint.num_local_slaves() > 0:
        local_slaves = constraint.slaves()[:constraint.num_local_slaves()]
        glob_slaves = np.array(loc_to_glob[local_slaves], dtype=np.int64)
    else:
        local_slaves = np.array([], dtype=np.int32)
        glob_slaves = np.array([], dtype=np.int64)

    global_slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    masters = constraint.masters_local().array
    coeffs = constraint.coefficients()
    offsets = constraint.masters_local().offsets
    K = np.zeros((V.dim, V.dim - len(global_slaves)), dtype=PETSc.ScalarType)
    # Add entries to
    for i in range(K.shape[0]):
        local_index = np.flatnonzero(i == loc_to_glob[local_slaves])
        if len(local_index) > 0:
            # If local master add coeffs
            local_index = local_index[0]
            masters_index = loc_to_glob[masters[offsets[local_index]:
                                                offsets[local_index + 1]]]
            coeffs_index = coeffs[offsets[local_index]: offsets[local_index + 1]]
            for master, coeff in zip(masters_index, coeffs_index):
                count = sum(master > global_slaves)
                K[i, master - count] = coeff
        elif i in global_slaves:
            # Do not add anything if there is a master
            pass
        else:
            # For all other nodes add one over number of procs to sum to 1
            count = sum(i > global_slaves)
            K[i, i - count] = 1 / MPI.COMM_WORLD.size
    # Gather K
    K = np.sum(MPI.COMM_WORLD.allgather(K), axis=0)
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


def compare_vectors(reduced_vec, vec, constraint):
    """
    Compare two numpy vectors of different lengths,
    where the constraints slaves are not in the reduced vector
    """
    global_zeros = gather_slaves_global(constraint)
    count = 0
    for i in range(len(vec)):
        if i in global_zeros:
            count += 1
            assert np.isclose(vec[i], 0)
        else:
            assert(np.isclose(reduced_vec[i - count], vec[i]))


def compare_matrices(reduced_A, A, constraint):
    """
    Compare a reduced matrix (stemming from numpy global matrix product),
    with a matrix stemming from MPC computations (having identity
    rows for slaves) given a multi-point constraint
    """
    global_ones = gather_slaves_global(constraint)
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
                A_numpy_padded[i, j] = reduced_A[i - count, j - m]
    D = np.abs(A - A_numpy_padded)

    max_index = np.unravel_index(np.argmax(D, axis=None), D.shape)
    if D[max_index] > 1e-6:
        print("Unequal ({0:.2e}) at ".format(D[max_index]), max_index)
        print(A_numpy_padded[max_index], A[max_index])
        print("Unequal at:", np.argwhere(D != 0))

    # Check that all entities are close
    assert np.allclose(A, A_numpy_padded)


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


def rigid_motions_nullspace(V):
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
        dofs = [V.sub(i).dofmap.list.array for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][V.sub(i).dofmap.list.array] = 1.0

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

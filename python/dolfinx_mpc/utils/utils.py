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
import dolfinx.geometry
import dolfinx.cpp


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


def determine_closest_dofs(V, point):
    """
    Determine the closest dofs (in a single block) to a point and the distance
    """
    tdim = V.mesh.topology.dim
    bb_tree = dolfinx.geometry.BoundingBoxTree(V.mesh, tdim)
    midpoint_tree = bb_tree.create_midpoint_tree(V.mesh)
    # Find facet closest
    closest_cell_data = dolfinx.geometry.compute_closest_entity(bb_tree, midpoint_tree, V.mesh, point)
    closest_cell, min_distance = closest_cell_data[0][0], closest_cell_data[1][0]
    cell_imap = V.mesh.topology.index_map(tdim)
    # Set distance high if cell is not owned
    if cell_imap.size_local * cell_imap.block_size <= closest_cell:
        min_distance = 1e5
    # Find processor with cell closest to point
    global_distances = MPI.COMM_WORLD.allgather(min_distance)
    owning_processor = np.argmin(global_distances)

    dofmap = V.dofmap
    imap = dofmap.index_map
    ghost_owner = imap.ghost_owner_rank()
    block_size = imap.block_size
    local_max = imap.size_local * block_size
    # Determine which block of dofs is closest
    min_distance = max(min_distance, 1e5)
    minimal_distance_block = None
    min_dof_owner = owning_processor
    if MPI.COMM_WORLD.rank == owning_processor:
        x = V.tabulate_dof_coordinates()
        cell_dofs = dofmap.cell_dofs(closest_cell)
        for dof in cell_dofs:
            # Only do distance computation for one node per block
            if dof % block_size == 0:
                block = dof // block_size
                distance = np.linalg.norm(dolfinx.cpp.geometry.compute_distance_gjk(point, x[dof]))
                if distance < min_distance:
                    # If cell owned by processor, but not the closest dof
                    if dof < local_max:
                        minimal_distance_block = block
                        min_dof_owner = MPI.COMM_WORLD.rank
                    else:
                        min_dof_owner = ghost_owner[block - imap.size_local]
                        minimal_distance_block = block
                    min_distance = distance
    min_dof_owner = MPI.COMM_WORLD.bcast(min_dof_owner, root=owning_processor)
    # If dofs not owned by cell
    if owning_processor != min_dof_owner:
        owning_processor = min_dof_owner

    if MPI.COMM_WORLD.rank == min_dof_owner:
        # Re-search using the closest cell
        x = V.tabulate_dof_coordinates()
        cell_dofs = dofmap.cell_dofs(closest_cell)
        for dof in cell_dofs:
            # Only do distance computation for one node per block
            if dof % block_size == 0:
                block = dof // block_size
                distance = np.linalg.norm(dolfinx.cpp.geometry.compute_distance_gjk(point, x[dof]))
                if distance < min_distance:
                    # If cell owned by processor, but not the closest dof
                    if dof < local_max:
                        minimal_distance_block = block
                        min_dof_owner = MPI.COMM_WORLD.rank
                    else:
                        min_dof_owner = ghost_owner[block - imap.size_local]
                        minimal_distance_block = block
                    min_distance = distance
        assert(min_dof_owner == owning_processor)
        return owning_processor, [minimal_distance_block * block_size + i for i in range(block_size)]
    else:
        return owning_processor, None


def create_point_to_point_constraint(V, slave_point, master_point, vector=None):
    # Determine which processor owns the dof closest to the slave and master point
    slave_proc, slave_dofs = determine_closest_dofs(V, slave_point)
    master_proc, master_dofs = determine_closest_dofs(V, master_point)

    # Create local to global mapping and map masters
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False), dtype=np.int64)
    # Output structures
    local_slaves, ghost_slaves = [], []
    local_masters, ghost_masters = [], []
    local_coeffs, ghost_coeffs = [], []
    local_owners, ghost_owners = [], []
    local_offsets, ghost_offsets = [], []
    # Information required to handle vector as input
    zero_indices, slave_index = None, None
    if vector is not None:
        zero_indices = np.argwhere(np.isclose(vector, 0)).T[0]
        slave_index = np.argmax(np.abs(vector))
    if MPI.COMM_WORLD.rank == slave_proc:
        if vector is None:
            local_slaves = slave_dofs
        else:
            assert(len(vector) == len(slave_dofs))
            # Check for input vector (Should be of same length as number of slaves)
            # All entries should not be zero
            assert(not np.isin(slave_index, zero_indices))
            # Check vector for zero contributions
            local_slaves = [slave_dofs[slave_index]]
            for i, slave in enumerate(slave_dofs):
                if i != slave_index and not np.isin(i, zero_indices):
                    local_masters.append(loc_to_glob[slave])
                    local_owners.append(slave_proc)
                    local_coeffs.append(-vector[i] / vector[slave_index])

    if MPI.COMM_WORLD.rank == slave_proc and slave_proc == master_proc:
        # If slaves and masters are on the same processor finalize local work
        if vector is None:
            local_masters = loc_to_glob[master_dofs]
            local_owners = np.full(len(local_masters), master_proc, dtype=np.int32)
            local_coeffs = np.ones(len(local_masters), dtype=PETSc.ScalarType)
            local_offsets = np.arange(0, len(local_masters) + 1, dtype=np.int32)
        else:
            for i, master in enumerate(master_dofs):
                if not np.isin(i, zero_indices):
                    local_masters.append(loc_to_glob[master])
                    local_owners.append(master_proc)
                    local_coeffs.append(vector[i] / vector[slave_index])
            local_offsets = [0, len(local_masters)]
    else:
        # Send/Recv masters from other processor
        if MPI.COMM_WORLD.rank == master_proc:
            MPI.COMM_WORLD.send(loc_to_glob[master_dofs], dest=slave_proc, tag=10)

        if MPI.COMM_WORLD.rank == slave_proc:
            global_masters = MPI.COMM_WORLD.recv(source=master_proc, tag=10)

    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    imap = V.dofmap.index_map
    ghost_processors = []
    if MPI.COMM_WORLD.rank == slave_proc and slave_proc != master_proc:
        for i, master in enumerate(global_masters):
            if not np.isin(i, zero_indices):
                local_masters.append(master)
                local_owners.append(master_proc)
                if vector is None:
                    local_coeffs.append(1)
                else:
                    local_coeffs.append(vector[i] / vector[slave_index])
        if vector is None:
            local_offsets = np.arange(0, len(local_slaves) + 1, dtype=np.int32)
        else:
            local_offsets = np.array([0, len(local_masters)], dtype=np.int32)
        if slave_dofs[0] / imap.block_size in shared_indices.keys():
            ghost_processors = list(shared_indices[slave_dofs[0] / imap.block_size])

    # Broadcast processors containg slave
    ghost_processors = MPI.COMM_WORLD.bcast(ghost_processors, root=slave_proc)

    if MPI.COMM_WORLD.rank == slave_proc:
        for proc in ghost_processors:
            MPI.COMM_WORLD.send(loc_to_glob[local_slaves], dest=proc, tag=20 + proc)
            MPI.COMM_WORLD.send(local_coeffs, dest=proc, tag=30 + proc)
            MPI.COMM_WORLD.send(local_owners, dest=proc, tag=40 + proc)
            MPI.COMM_WORLD.send(local_masters, dest=proc, tag=50 + proc)
            MPI.COMM_WORLD.send(local_offsets, dest=proc, tag=60 + proc)

    # Receive data for ghost slaves
    if np.isin(MPI.COMM_WORLD.rank, ghost_processors):
        # Convert recieved slaves to the corresponding ghost index
        recv_slaves = MPI.COMM_WORLD.recv(source=slave_proc, tag=20 + MPI.COMM_WORLD.rank)
        ghost_coeffs = MPI.COMM_WORLD.recv(source=slave_proc, tag=30 + MPI.COMM_WORLD.rank)
        ghost_owners = MPI.COMM_WORLD.recv(source=slave_proc, tag=40 + MPI.COMM_WORLD.rank)
        ghost_masters = MPI.COMM_WORLD.recv(source=slave_proc, tag=50 + MPI.COMM_WORLD.rank)
        ghost_offsets = MPI.COMM_WORLD.recv(source=slave_proc, tag=60 + MPI.COMM_WORLD.rank)

        ghosts = imap.indices(True)[imap.size_local * imap.block_size:]
        ghost_slaves = np.zeros(len(recv_slaves), dtype=np.int32)
        for i, slave in enumerate(recv_slaves):
            idx = np.argwhere(ghosts == slave)[0, 0]

            ghost_slaves[i] = imap.size_local * imap.block_size + idx

    return (local_slaves, ghost_slaves), (local_masters, ghost_masters), (local_coeffs, ghost_coeffs),\
        (local_owners, ghost_owners), (local_offsets, ghost_offsets)


def create_normal_approximation(V, facets):
    """
    Creates a normal approximation for the dofs in the closure of the attached facets.
    Where a dof is attached to multiple facets, an average is computed
    """
    n = dolfinx.Function(V)
    with n.vector.localForm() as vector:
        vector.set(0)
        dolfinx_mpc.cpp.mpc.create_normal_approximation(V._cpp_object, facets, vector.array_w)
    n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return n

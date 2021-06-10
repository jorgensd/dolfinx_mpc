# Copyright (C) 2020-2021 Jørgen S. Dokken
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
import scipy.sparse

import dolfinx
import dolfinx.common
import dolfinx.la
import dolfinx.log
import dolfinx.geometry
import dolfinx.cpp


__all__ = ["rotation_matrix", "facet_normal_approximation", "gather_PETScVector",
           "gather_PETScMatrix", "compare_MPC_LHS", "log_info", "rigid_motions_nullspace",
           "determine_closest_block", "compare_MPC_RHS", "create_normal_approximation",
           "gather_transformation_matrix", "compare_CSR", "create_point_to_point_constraint"]


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
        a = (ufl.inner(u, v) * ds)
        L = ufl.inner(n, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_blocks = np.arange(imap.size_local, dtype=np.int32)
    top_facets = mt.indices[np.flatnonzero(mt.values == mt_id)]
    top_blocks = dolfinx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, top_facets)
    deac_blocks = all_blocks[np.isin(all_blocks, top_blocks, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    cpp_form = dolfinx.Form(a)._cpp_object
    pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()
    u_0 = dolfinx.Function(V)
    u_0.vector.set(0)

    bc_deac = dolfinx.fem.DirichletBC(u_0, deac_blocks)
    A = dolfinx.cpp.la.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, [bc_deac])
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, cpp_form.function_spaces[0],
                                        [bc_deac], 1.0)
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)

    dolfinx.fem.apply_lifting(b, [a], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc_deac])

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    timer.stop()
    return nh


def gather_slaves_global(constraint):
    """
    Given a multi point constraint,
    return slaves for all processors with global dof numbering
    """
    imap = constraint.index_map()
    num_local_slaves = constraint.num_local_slaves()
    block_size = constraint.function_space().dofmap.index_map_bs
    if num_local_slaves > 0:
        slave_blocks = constraint.slaves()[:num_local_slaves] // block_size
        slave_rems = constraint.slaves()[:num_local_slaves] % block_size
        glob_slaves = imap.local_to_global(slave_blocks) * block_size + slave_rems
    else:
        glob_slaves = np.array([], dtype=np.int64)

    slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    return slaves


def gather_transformation_matrix(constraint, root=0):
    """
    Creates the transformation matrix K (dim x dim-len(slaves)) for a given MPC
    and gathers it as a scipy CSR matrix on process 'root'.

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
    V = constraint.V
    imap = constraint.index_map()
    block_size = V.dofmap.index_map_bs
    num_local_slaves = constraint.num_local_slaves()
    # Gather all global_slaves
    if num_local_slaves > 0:
        local_blocks = constraint.slaves()[: num_local_slaves] // block_size
        local_rems = constraint.slaves()[: num_local_slaves] % block_size
        glob_slaves = imap.local_to_global(local_blocks) * block_size + local_rems
    else:
        glob_slaves = np.array([], dtype=np.int64)

    global_slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    masters = constraint.masters_local().array
    master_blocks = masters // block_size
    master_rems = masters % block_size
    coeffs = constraint.coefficients()
    offsets = constraint.masters_local().offsets

    # Create sparse K matrix
    K_val, rows, cols = [], [], []
    # Add local contributions to K from local slaves
    for local_index, slave in enumerate(glob_slaves):
        masters_index = (imap.local_to_global(master_blocks[offsets[local_index]: offsets[local_index + 1]])
                         * block_size + master_rems[offsets[local_index]: offsets[local_index + 1]])
        coeffs_index = coeffs[offsets[local_index]: offsets[local_index + 1]]
        for master, coeff in zip(masters_index, coeffs_index):
            count = sum(master > global_slaves)
            K_val.append(coeff)
            rows.append(slave)
            cols.append(master - count)

    # Add identity for all dofs on diagonal
    l_range = V.dofmap.index_map.local_range
    global_dofs = np.arange(l_range[0] * block_size, l_range[1] * block_size)
    is_slave = np.isin(global_dofs, glob_slaves)
    for i, dof in enumerate(global_dofs):
        if not is_slave[i]:
            K_val.append(1)
            rows.append(dof)
            cols.append(dof - sum(dof > global_slaves))

    # Gather K to root
    K_vals = MPI.COMM_WORLD.gather(np.asarray(K_val, dtype=PETSc.ScalarType), root=root)
    rows_g = MPI.COMM_WORLD.gather(np.asarray(rows, dtype=np.int64), root=root)
    cols_g = MPI.COMM_WORLD.gather(np.asarray(cols, dtype=np.int64), root=root)

    if MPI.COMM_WORLD.rank == root:
        K_sparse = scipy.sparse.coo_matrix((np.hstack(K_vals), (np.hstack(rows_g), np.hstack(cols_g)))).tocsr()
        return K_sparse


def petsc_to_local_CSR(A: PETSc.Mat, mpc: dolfinx_mpc.MultiPointConstraint):
    """
    Convert a PETSc matrix to a local CSR matrix (scipy) including ghost entries
    """
    global_indices = np.asarray(mpc.function_space().dofmap.index_map.global_indices(), dtype=PETSc.IntType)
    sort_index = np.argsort(global_indices)
    is_A = PETSc.IS().createGeneral(global_indices[sort_index])
    A_loc = A.createSubMatrices(is_A)[0]
    ai, aj, av = A_loc.getValuesCSR()
    A_csr = scipy.sparse.csr_matrix((av, aj, ai))
    return A_csr[global_indices[:, None], global_indices]


def gather_PETScMatrix(A: PETSc.Mat, root=0):
    """
    Given a distributed PETSc matrix, gather in on process 'root' in
    a scipy CSR matrix
    """
    ai, aj, av = A.getValuesCSR()
    aj_all = MPI.COMM_WORLD.gather(aj, root=root)
    av_all = MPI.COMM_WORLD.gather(av, root=root)
    ai_all = MPI.COMM_WORLD.gather(ai, root=root)
    if MPI.COMM_WORLD.rank == root:
        ai_cum = [0]
        for ai in ai_all:
            offsets = ai[1:] + ai_cum[-1]
            ai_cum.extend(offsets)
        return scipy.sparse.csr_matrix((np.hstack(av_all), np.hstack(aj_all), ai_cum))


def gather_PETScVector(vector, root=0):
    """
    Gather a PETScVector from different processors on
    process 'root' as an numpy array
    """
    numpy_vec = np.zeros(vector.size, dtype=vector.array.dtype)
    l_min = vector.owner_range[0]
    l_max = vector.owner_range[1]
    numpy_vec[l_min: l_max] += vector.array
    numpy_vec = sum(MPI.COMM_WORLD.allgather(numpy_vec))
    return numpy_vec


def compare_CSR(A, B, atol=1e-10):
    """ Compuare CSR matrices A and B """
    diff = np.abs(A - B)
    assert(diff.max() < atol)


def compare_MPC_LHS(A_org: PETSc.Mat, A_mpc: PETSc.Mat,
                    mpc: dolfinx_mpc.MultiPointConstraint, root: int = 0):
    """
    Compare an unmodified matrix for the problem with the one assembled with a
    multi point constraint.

    The unmodified matrix is multiplied with K^T A K, where K is the global transformation matrix.
    """
    timer = dolfinx.common.Timer("~MPC: Compare matrices")
    comm = mpc.V.mesh.mpi_comm()
    V = mpc.V
    assert(root < comm.size)

    K = gather_transformation_matrix(mpc, root=root)
    A_csr = gather_PETScMatrix(A_org, root=root)

    # Get global slaves
    glob_slaves = gather_slaves_global(mpc)
    A_mpc_csr = gather_PETScMatrix(A_mpc, root=root)

    if MPI.COMM_WORLD.rank == root:
        KTAK = K.T * A_csr * K

        # Remove identity rows of MPC matrix
        all_cols = np.arange(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
        cols_except_slaves = np.flatnonzero(np.isin(all_cols, glob_slaves, invert=True).astype(np.int32))
        mpc_without_slaves = A_mpc_csr[cols_except_slaves[:, None], cols_except_slaves]

        # Compute difference
        compare_CSR(KTAK, mpc_without_slaves)

    timer.stop()


def compare_MPC_RHS(b_org: PETSc.Vec, b: PETSc.Vec, constraint: dolfinx_mpc.MultiPointConstraint, root: int = 0):
    """
    Compare an unconstrained RHS with an MPC rhs.
    """
    glob_slaves = gather_slaves_global(constraint)
    b_org_np = dolfinx_mpc.utils.gather_PETScVector(b_org, root=root)
    b_np = dolfinx_mpc.utils.gather_PETScVector(b, root=root)
    K = gather_transformation_matrix(constraint, root=root)

    comm = constraint.V.mesh.mpi_comm()
    if comm.rank == root:
        reduced_b = K.T @ b_org_np

        all_cols = np.arange(constraint.V.dofmap.index_map.size_global * constraint.V.dofmap.index_map_bs)
        cols_except_slaves = np.flatnonzero(np.isin(all_cols, glob_slaves, invert=True).astype(np.int32))
        assert np.allclose(b_np[glob_slaves], 0)
        assert np.allclose(b_np[cols_except_slaves], reduced_b)


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
    nullspace_basis = [dolfinx.cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.array for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][dofs[i]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        if gdim == 2:
            basis[2][dofs[0]] = -x1
            basis[2][dofs[1]] = x0
        elif gdim == 3:
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0

            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(dim)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


def determine_closest_block(V, point):
    """
    Determine the closest dofs (in a single block) to a point and the distance
    """
    # Create boundingboxtree of cells connected to boundary facets
    tdim = V.mesh.topology.dim
    boundary_facets = np.flatnonzero(dolfinx.cpp.mesh.compute_boundary_facets(V.mesh.topology))
    V.mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = V.mesh.topology.connectivity(tdim - 1, tdim)
    boundary_cells = []
    for facet in boundary_facets:
        boundary_cells.extend(f_to_c.links(facet))
    cell_imap = V.mesh.topology.index_map(tdim)
    boundary_cells = np.array(np.unique(boundary_cells), dtype=np.int32)
    boundary_cells = boundary_cells[boundary_cells < cell_imap.size_local]
    bb_tree = dolfinx.geometry.BoundingBoxTree(V.mesh, tdim, boundary_cells)
    midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(V.mesh, tdim, boundary_cells)

    # Find facet closest
    closest_midpoint_facet, R_init = dolfinx.geometry.compute_closest_entity(midpoint_tree, point, V.mesh)
    closest_cell, R = dolfinx.geometry.compute_closest_entity(bb_tree, point, V.mesh, R_init)

    # Set distance high if cell is not owned
    if cell_imap.size_local <= closest_cell:
        R = 1e5
    # Find processor with cell closest to point
    global_distances = MPI.COMM_WORLD.allgather(R)
    owning_processor = np.argmin(global_distances)

    dofmap = V.dofmap
    imap = dofmap.index_map
    ghost_owner = imap.ghost_owner_rank()
    local_max = imap.size_local
    # Determine which block of dofs is closest
    min_distance = max(R, 1e5)
    minimal_distance_block = None
    min_dof_owner = owning_processor
    if MPI.COMM_WORLD.rank == owning_processor:
        x = V.tabulate_dof_coordinates()
        cell_blocks = dofmap.cell_dofs(closest_cell)
        for block in cell_blocks:
            distance = np.linalg.norm(dolfinx.cpp.geometry.compute_distance_gjk(point, x[block]))
            if distance < min_distance:
                # If cell owned by processor, but not the closest dof
                if block < local_max:
                    min_dof_owner = MPI.COMM_WORLD.rank
                else:
                    min_dof_owner = ghost_owner[block - local_max]
                minimal_distance_block = block
                min_distance = distance
    min_dof_owner = MPI.COMM_WORLD.bcast(min_dof_owner, root=owning_processor)
    # If dofs not owned by cell
    if owning_processor != min_dof_owner:
        owning_processor = min_dof_owner

    if MPI.COMM_WORLD.rank == min_dof_owner:
        # Re-search using the closest cell
        x = V.tabulate_dof_coordinates()
        cell_blocks = dofmap.cell_dofs(closest_cell)
        for block in cell_blocks:
            distance = np.linalg.norm(dolfinx.cpp.geometry.compute_distance_gjk(point, x[block]))
            if distance < min_distance:
                # If cell owned by processor, but not the closest dof
                if block < local_max:
                    min_dof_owner = MPI.COMM_WORLD.rank
                else:
                    min_dof_owner = ghost_owner[block - local_max]
                minimal_distance_block = block
                min_distance = distance
        assert(min_dof_owner == owning_processor)
        return owning_processor, [minimal_distance_block]
    else:
        return owning_processor, []


def create_point_to_point_constraint(V, slave_point, master_point, vector=None):
    # Determine which processor owns the dof closest to the slave and master point
    slave_proc, slave_block = determine_closest_block(V, slave_point)
    master_proc, master_block = determine_closest_block(V, master_point)
    is_master_proc = MPI.COMM_WORLD.rank == master_proc
    is_slave_proc = MPI.COMM_WORLD.rank == slave_proc

    block_size = V.dofmap.index_map_bs
    imap = V.dofmap.index_map
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
    if is_slave_proc:
        assert(len(slave_block) == 1)
        slave_block_g = imap.local_to_global(slave_block)[0]
        if vector is None:
            local_slaves = np.arange(slave_block[0] * block_size, slave_block[0]
                                     * block_size + block_size, dtype=np.int32)
        else:
            assert(len(vector) == block_size)
            # Check for input vector (Should be of same length as number of slaves)
            # All entries should not be zero
            assert(not np.isin(slave_index, zero_indices))
            # Check vector for zero contributions
            local_slaves = np.array([slave_block[0] * block_size + slave_index], dtype=np.int32)
            for i in range(block_size):
                if i != slave_index and not np.isin(i, zero_indices):
                    local_masters.append(slave_block_g * block_size + i)
                    local_owners.append(slave_proc)
                    local_coeffs.append(-vector[i] / vector[slave_index])

    global_masters = None
    if is_master_proc:
        assert(len(master_block) == 1)
        master_block_g = imap.local_to_global(master_block)[0]
        masters_as_glob = np.arange(master_block_g * block_size,
                                    master_block_g * block_size + block_size, dtype=np.int64)
    else:
        masters_as_glob = np.array([], dtype=np.int64)

    ghost_processors = []
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)

    if is_master_proc and is_slave_proc:
        # If slaves and masters are on the same processor finalize local work
        if vector is None:
            local_masters = masters_as_glob
            local_owners = np.full(len(local_masters), master_proc, dtype=np.int32)
            local_coeffs = np.ones(len(local_masters), dtype=PETSc.ScalarType)
            local_offsets = np.arange(0, len(local_masters) + 1, dtype=np.int32)
        else:
            for i in range(len(masters_as_glob)):
                if not np.isin(i, zero_indices):
                    local_masters.append(masters_as_glob[i])
                    local_owners.append(master_proc)
                    local_coeffs.append(vector[i] / vector[slave_index])
            local_offsets = [0, len(local_masters)]
    else:
        # Send/Recv masters from other processor
        if is_master_proc:
            MPI.COMM_WORLD.send(masters_as_glob, dest=slave_proc, tag=10)
        if is_slave_proc:
            global_masters = MPI.COMM_WORLD.recv(source=master_proc, tag=10)
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
            if slave_block[0] in shared_indices.keys():
                ghost_processors = list(shared_indices[slave_block[0]])

    # Broadcast processors containg slave
    ghost_processors = MPI.COMM_WORLD.bcast(ghost_processors, root=slave_proc)
    if is_slave_proc:
        for proc in ghost_processors:
            MPI.COMM_WORLD.send(slave_block_g * block_size + local_slaves %
                                block_size, dest=proc, tag=20 + proc)
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

        # Unroll ghost blocks
        ghosts = imap.ghosts
        ghost_dofs = [g * block_size + i for g in ghosts for i in range(block_size)]

        ghost_slaves = np.zeros(len(recv_slaves), dtype=np.int32)
        local_size = imap.size_local
        for i, slave in enumerate(recv_slaves):
            idx = np.argwhere(ghost_dofs == slave)[0, 0]
            ghost_slaves[i] = local_size * block_size + idx
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

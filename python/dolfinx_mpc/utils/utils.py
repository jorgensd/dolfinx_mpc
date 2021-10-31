# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from contextlib import ExitStack

from typing import List, Tuple
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


__all__ = ["rotation_matrix", "facet_normal_approximation", "log_info", "rigid_motions_nullspace",
           "determine_closest_block", "create_normal_approximation", "create_point_to_point_constraint"]


def rotation_matrix(axis, angle):
    """
    Create a rotation matrix for rotation around an axis with a given angle
    See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Parameters
    ----------
    axis
       Three dimensional vector describing the rotation axis
    angle
       The rotation angle
    """
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


def facet_normal_approximation(V: dolfinx.FunctionSpace, mt: dolfinx.MeshTags,
                               mt_id: int, tangent: bool = False):
    """
    Approximate the facet normal by projecting it into the function space for a set of facets

    Parameters
    ----------
    V
        The function space to project into
    mt
        The `dolfinx.Meshtags` containing facet markers
    mt_id
        The id for the facets in `mt` we want to represent the normal at
    tangent
        To approximate the tangent to the facet set this flag to `True`.
    """
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
    pattern = dolfinx.fem.create_sparsity_pattern(cpp_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()
    u_0 = dolfinx.Function(V)
    u_0.vector.set(0)

    bc_deac = dolfinx.fem.DirichletBC(u_0, deac_blocks)
    A = dolfinx.cpp.la.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)
    dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, form_consts, form_coeffs, [bc_deac._cpp_object])
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, cpp_form.function_spaces[0],
                                        [bc_deac._cpp_object], 1.0)
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


def log_info(message: str):
    """
    Wrapper for logging a simple string on the zeroth communicator
    and resetting log level after sending message

    Parameters
    ----------
    message
        The message to print
    """
    old_level = dolfinx.log.get_log_level()
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO, message)
        dolfinx.log.set_log_level(old_level)


def rigid_motions_nullspace(V: dolfinx.FunctionSpace) -> PETSc.NullSpace:
    """
    Build nullspace for 2D/3D elasticity. The nullspace are
    the translations and rotations.

    Parameters
    ----------
    V
        The function space
    Returns
    -------
    PETSc.NullSpace
        The PETSc nullspace
    """

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


def determine_closest_block(V, point) -> Tuple[int, List[np.int32]]:
    """
    Determine the closest dofs (in a single block) to a point.

    Parameters
    ----------
    V
        The function space
    point
        The point

    Returns
    -------
    Tuple[int, List[np.int32]]
        A tuple whose first entry is which rank owns the degree of freedom,
        and the second entry is the degree of freedom (not expanded by block size).
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


def create_point_to_point_constraint(V: dolfinx.FunctionSpace, slave_point, master_point,
                                     vector=None):
    """
    Creates a constraint between to points `p1` and `p2`, `dof_at_p1 = dof_at_p2`

    Parameters
    ----------
    V
        The function space
    slave_point
        The first point p1
    master_point
        The second point p2
    vector
        If vector is supplied the constraint is `dot(dofs_at_p1, vector)=dot(dofs_at_p2, vector)`


    """
    # Determine which processor owns the dof closest to the slave and master point
    slave_proc, slave_block = determine_closest_block(V, slave_point)
    master_proc, master_block = determine_closest_block(V, master_point)
    is_master_proc = MPI.COMM_WORLD.rank == master_proc
    is_slave_proc = MPI.COMM_WORLD.rank == slave_proc

    block_size = V.dofmap.index_map_bs
    imap = V.dofmap.index_map
    # Output structures
    slaves, masters, coeffs, owners, offsets = [], [], [], [], []
    # Information required to handle vector as input
    zero_indices, slave_index = None, None
    if vector is not None:
        zero_indices = np.argwhere(np.isclose(vector, 0)).T[0]
        slave_index = np.argmax(np.abs(vector))
    if is_slave_proc:
        assert(len(slave_block) == 1)
        slave_block_g = imap.local_to_global(slave_block)[0]
        if vector is None:
            slaves = np.arange(slave_block[0] * block_size, slave_block[0]
                               * block_size + block_size, dtype=np.int32)
        else:
            assert(len(vector) == block_size)
            # Check for input vector (Should be of same length as number of slaves)
            # All entries should not be zero
            assert(not np.isin(slave_index, zero_indices))
            # Check vector for zero contributions
            slaves = np.array([slave_block[0] * block_size + slave_index], dtype=np.int32)
            for i in range(block_size):
                if i != slave_index and not np.isin(i, zero_indices):
                    masters.append(slave_block_g * block_size + i)
                    owners.append(slave_proc)
                    coeffs.append(-vector[i] / vector[slave_index])

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
            masters = masters_as_glob
            owners = np.full(len(masters), master_proc, dtype=np.int32)
            coeffs = np.ones(len(masters), dtype=PETSc.ScalarType)
            offsets = np.arange(0, len(masters) + 1, dtype=np.int32)
        else:
            for i in range(len(masters_as_glob)):
                if not np.isin(i, zero_indices):
                    masters.append(masters_as_glob[i])
                    owners.append(master_proc)
                    coeffs.append(vector[i] / vector[slave_index])
            offsets = [0, len(masters)]
    else:
        # Send/Recv masters from other processor
        if is_master_proc:
            MPI.COMM_WORLD.send(masters_as_glob, dest=slave_proc, tag=10)
        if is_slave_proc:
            global_masters = MPI.COMM_WORLD.recv(source=master_proc, tag=10)
            for i, master in enumerate(global_masters):
                if not np.isin(i, zero_indices):
                    masters.append(master)
                    owners.append(master_proc)
                    if vector is None:
                        coeffs.append(1)
                    else:
                        coeffs.append(vector[i] / vector[slave_index])
            if vector is None:
                offsets = np.arange(0, len(slaves) + 1, dtype=np.int32)
            else:
                offsets = np.array([0, len(masters)], dtype=np.int32)
            if slave_block[0] in shared_indices.keys():
                ghost_processors = list(shared_indices[slave_block[0]])

    # Broadcast processors containg slave
    ghost_processors = MPI.COMM_WORLD.bcast(ghost_processors, root=slave_proc)
    if is_slave_proc:
        for proc in ghost_processors:
            MPI.COMM_WORLD.send(slave_block_g * block_size + slaves %
                                block_size, dest=proc, tag=20 + proc)
            MPI.COMM_WORLD.send(coeffs, dest=proc, tag=30 + proc)
            MPI.COMM_WORLD.send(owners, dest=proc, tag=40 + proc)
            MPI.COMM_WORLD.send(masters, dest=proc, tag=50 + proc)
            MPI.COMM_WORLD.send(offsets, dest=proc, tag=60 + proc)

    # Receive data for ghost slaves
    ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners, ghost_offsets = [], [], [], [], []
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
    slaves = np.asarray(np.append(slaves, ghost_slaves), dtype=np.int32)
    masters = np.asarray(np.append(masters, ghost_masters), dtype=np.int64)
    coeffs = np.asarray(np.append(coeffs, ghost_coeffs), dtype=np.int32)
    owners = np.asarray(np.append(owners, ghost_owners), dtype=np.int32)
    offsets = np.asarray(np.append(offsets, ghost_offsets), dtype=np.int32)
    return slaves, masters, coeffs, owners, offsets


def create_normal_approximation(V:dolfinx.FunctionSpace, facets):
    """
    Creates a normal approximation for the dofs in the closure of the attached facets.
    Where a dof is attached to multiple facets, an average is computed.

    Parameters
    ----------
    V
        The function space
    facets
        List of facets (indices local to process)
    """
    n = dolfinx.Function(V)
    with n.vector.localForm() as vector:
        vector.set(0)
        dolfinx_mpc.cpp.mpc.create_normal_approximation(V._cpp_object, facets, vector.array_w)
    n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return n

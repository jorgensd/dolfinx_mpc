# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Helper for finding master slave relationship for contact problems
import dolfinx
import dolfinx.geometry as geometry
import dolfinx.fem as fem
import dolfinx_mpc
import numpy as np


get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def find_master_slave_relationship(V, interface_info, cell_info):
    """
    Function return the master, slaves, coefficients and offsets
    for coupling two interfaces.
    Input:
        V - The function space that should be constrained.
        interface_info - Tuple containing a mesh function
                         (for facets) and two ints, one specifying the
                          slave interface, the other the master interface.
        cell_info - Tuple containing a mesh function (for cells)
                    and an int indicating which cells should be
                    master cells.
    """
    # Extract mesh info
    mesh = V.mesh
    tdim = V.mesh.topology.dim
    fdim = tdim - 1

    # Collapse sub spaces
    Vs = [V.sub(i).collapse() for i in range(tdim)]

    # Get dofmap info
    comm = V.mesh.mpi_comm()
    dofs = V.dofmap.list.array()
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    indexmap = V.dofmap.index_map
    bs = indexmap.block_size
    global_indices = np.array(indexmap.global_indices(False),
                              dtype=np.int64)
    lmin = bs*indexmap.local_range[0]
    lmax = bs*indexmap.local_range[1]
    ghost_owners = indexmap.ghost_owner_rank()

    # Extract interface_info
    (mt, slave_i_value, master_i_value) = interface_info
    (ct, master_value) = cell_info

    # Find all dofs on slave interface
    slave_facets = mt.indices[np.flatnonzero(mt.values == slave_i_value)]
    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, slave_facets)[:, 0]
        global_dofs = global_indices[i_dofs]
        owned = np.logical_and(lmin <= global_dofs, global_dofs < lmax)
        interface_dofs.append(i_dofs[owned])

    # Extract all top cell values from MeshTags
    tree = geometry.BoundingBoxTree(mesh, tdim)
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mt, slave_i_value)
    n_vec = nh.vector.getArray()

    # Create local dictionaries for the master, slaves and coeffs
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    local_slaves = []
    local_coordinates = np.zeros((len(interface_dofs[tdim-1]), 3),
                                 dtype=np.float64)
    local_normals = np.zeros((len(interface_dofs[tdim-1]), tdim),
                             dtype=np.float64)

    # Gather all slaves fromm each processor,
    # with its corresponding normal vector and coordinate
    for i in range(len(interface_dofs[tdim-1])):
        local_slave = interface_dofs[tdim-1][i]
        global_slave = global_indices[local_slave]
        if lmin <= global_slave < lmax:
            local_slaves.append(global_slave)
            local_coordinates[i] = dof_coords[local_slave]
            for j in range(tdim):
                local_normals[i, j] = n_vec[interface_dofs[j][i]]

    # Gather info on all processors
    slaves = np.hstack(comm.allgather(local_slaves))
    slave_coordinates = np.concatenate(comm.allgather(local_coordinates))
    slave_normals = np.concatenate(comm.allgather(local_normals))

    # Local range of cells
    cellmap = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cellmap.local_range
    ltog_cells = np.array(cellmap.global_indices(False), dtype=np.int64)

    # Local dictionaries which will hold data for each slave
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    for i, slave in enumerate(slaves):
        master_dict[i] = []
        master_owner_dict[i] = []
        coeffs_dict[i] = []

        # Find all masters from same coordinate as the slave (same proc)
        if lmin <= slave < lmax:
            local_index = np.flatnonzero(slave == local_slaves)[0]
            for j in range(tdim-1):
                coeff = -slave_normals[i][j]/slave_normals[i][tdim-1]
                if not np.isclose(coeff, 0):
                    master_dict[i].append(
                        global_indices[interface_dofs[j][local_index]])
                    master_owner_dict[i].append(comm.rank)
                    coeffs_dict[i].append(coeff)

        # Find bb cells for masters on the other interface (no ghosts)
        possible_cells = np.array(
            dolfinx.geometry.compute_collisions_point(tree,
                                                      slave_coordinates[i]))
        is_top = np.isin(possible_cells, ct_top_cells)
        is_owned = np.zeros(len(possible_cells), dtype=bool)
        if len(possible_cells) > 0:
            is_owned = np.logical_and(cmin <= ltog_cells[possible_cells],
                                      ltog_cells[possible_cells] < cmax)
        top_cells = possible_cells[np.logical_and(is_owned, is_top)]
        # Check if actual cell in mesh is within a distance of 1e-14
        close_cell = dolfinx.cpp.geometry.select_colliding_cells(
            mesh, list(top_cells), slave_coordinates[i], 1)
        if len(close_cell) > 0:
            master_cell = close_cell[0]
            master_dofs = dofs[num_dofs_per_element * master_cell:
                               num_dofs_per_element * master_cell
                               + num_dofs_per_element]
            global_masters = global_indices[master_dofs]

            # Get basis values, and add coeffs of sub space
            # to masters with corresponding coeff
            basis_values = get_basis(V._cpp_object,
                                     slave_coordinates[i], master_cell)
            for k in range(tdim):
                for local_idx, dof in enumerate(global_masters):
                    l_coeff = basis_values[local_idx, k]
                    l_coeff *= (slave_normals[i][k] /
                                slave_normals[i][tdim-1])
                    if not np.isclose(l_coeff, 0):
                        # If master i ghost, find its owner
                        if lmin <= dof < lmax:
                            master_owner_dict[i].append(comm.rank)
                        else:
                            master_owner_dict[i].append(ghost_owners[
                                dof//bs-indexmap.size_local])
                        master_dict[i].append(dof)
                        coeffs_dict[i].append(l_coeff)

    # Gather entries on all processors
    masters = np.array([], dtype=np.int64)
    coeffs = np.array([], dtype=np.int64)
    owner_ranks = np.array([], dtype=np.int64)
    offsets = np.zeros(len(slaves)+1, dtype=np.int64)
    for key in master_dict.keys():
        master_glob = np.array(np.concatenate(comm.allgather(
            master_dict[key])), dtype=np.int64)
        coeff_glob = np.array(np.concatenate(comm.allgather(
            coeffs_dict[key])), dtype=np.float64)
        owner_glob = np.array(np.concatenate(comm.allgather(
            master_owner_dict[key])), dtype=np.int64)
        masters = np.hstack([masters, master_glob])
        coeffs = np.hstack([coeffs, coeff_glob])
        owner_ranks = np.hstack([owner_ranks, owner_glob])
        offsets[key+1] = len(masters)

    return (slaves, masters, coeffs, offsets, owner_ranks)

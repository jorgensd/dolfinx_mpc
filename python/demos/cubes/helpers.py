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

    # Extract interface_info
    (mt, slave_i_value, master_i_value) = interface_info
    (ct, master_value) = cell_info

    # Locate dofs on both interfaces
    indexmap = V.dofmap.index_map
    global_indices = np.array(
        indexmap.global_indices(False), dtype=np.int64)
    lmin = indexmap.block_size*indexmap.local_range[0]
    lmax = indexmap.block_size*indexmap.local_range[1]

    slave_facets = mt.indices[np.flatnonzero(mt.values == slave_i_value)]
    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, slave_facets)[:, 0]
        global_dofs = global_indices[i_dofs]
        owned = np.logical_and(lmin <= global_dofs, global_dofs < lmax)
        interface_dofs.append(i_dofs[owned])

    # Get dof info
    dofs = V.dofmap.list.array()
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    slaves, masters, coeffs, offsets, owner_ranks = [], [], [], [0], []

    # Extract all top cell values from MeshTags
    tree = geometry.BoundingBoxTree(mesh, tdim)
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mt, slave_i_value)
    n_vec = nh.vector.getArray()

    # New computation below

    comm = V.mesh.mpi_comm()
    slave_index = 0
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    local_slaves = []
    local_coordinates = np.zeros((len(interface_dofs[tdim-1]), 3),
                                 dtype=np.float64)
    local_normals = np.zeros((len(interface_dofs[tdim-1]), tdim),
                             dtype=np.float64)

    # Gather all slaves form each processor, with its corresponding normal vector
    # and coordinate
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
    [cmin, cmax] = V.mesh.topology.index_map(tdim).local_range

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
                master_dict[i].append(
                    global_indices[interface_dofs[j][local_index]])
                master_owner_dict[i].append(comm.rank)
                coeffs_dict[i].append(-slave_normals[i]
                                      [j]/slave_normals[i][tdim-1])

        # Find bb cells for masters on the other interface (no ghosts)
        possible_cells = np.array(dolfinx.geometry.
                                  compute_collisions_point(tree,
                                                           slave_coordinates[i]))
        is_owned = cmin + possible_cells < cmax
        is_top = np.isin(possible_cells, ct_top_cells)
        top_cells = possible_cells[np.logical_and(is_owned, is_top)]

        # Check if actual cell in mesh is within a distance of 1e-14
        close_cell = dolfinx.cpp.geometry.select_colliding_cells(
            mesh, list(top_cells), slave_coordinates[i], 1)
        if len(close_cell) > 0:
            master_cell = close_cell[0]
            master_dofs = dofs[num_dofs_per_element * master_cell:
                               num_dofs_per_element * master_cell
                               + num_dofs_per_element]

            # Get basis values, and add coeffs of sub space
            # to masters with corresponding coeff
            basis_values = get_basis(V._cpp_object,
                                     slave_coordinates[i], master_cell)
            for k in range(tdim):
                for local_idx, dof in enumerate(master_dofs):
                    l_coeff = basis_values[local_idx, k]
                    l_coeff *= (slave_normals[i][k] /
                                slave_normals[i][tdim-1])
                    if not np.isclose(l_coeff, 0):
                        master_owner_dict[i].append(comm.rank)
                        master_dict[i].append(global_indices[dof])
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
    m2 = masters
    c2 = coeffs
    o2 = offsets
    or2 = owner_ranks
    s2 = slaves
    if comm.rank == 0:
        print(masters)
        print(coeffs)
        print(offsets)
        print(owner_ranks)
        print("---"*25)

    return (slaves, masters, coeffs, offsets, owner_ranks)

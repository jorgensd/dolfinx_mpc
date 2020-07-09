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

    # Create facet-to-cell-connectivity
    # mesh.topology.create_connectivity(fdim, tdim)
    # mesh.topology.create_connectivity(tdim, fdim)
    # tree = geometry.BoundingBoxTree(mesh, tdim)
    # facet_to_cell = mesh.topology.connectivity(fdim, tdim)

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
    print(comm.rank, cmin, cmax)
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

        # Find cells for masters on the other interface (no ghosts)
        possible_cells = np.array(dolfinx.geometry.
                                  compute_collisions_point(tree,
                                                           slave_coordinates[i]
                                                           ))
        is_owned = cmin + possible_cells < cmax
        is_top = np.isin(possible_cells, ct_top_cells)
        top_cells = possible_cells[np.logical_and(is_owned, is_top)]

        # Find cell a top cell within 1e-14 distance
        close_cell = dolfinx.cpp.geometry.select_colliding_cells(
            mesh, list(top_cells), slave_coordinates[i], 1)

        print(i, comm.rank, close_cell, possible_cells)

    print("Slaves", comm.rank, slaves)
    print("Masters", comm.rank, master_dict)
    print("Owner", comm.rank, master_owner_dict)
    print("Coeffs", comm.rank, coeffs_dict)
    print(len(slaves))
    return (np.array(slaves, dtype=np.int64),
            np.array(masters, dtype=np.int64),
            np.array(coeffs, dtype=np.float64),
            np.array(offsets, dtype=np.int64),
            np.array(owner_ranks, dtype=np.int32))

    for facet in slave_facets:
        # Find cells connected to facets (Should only be one)
        cell_indices = facet_to_cell.links(facet)
        assert(len(cell_indices) == 1)
        cell_index = cell_indices[0]
        cell_dofs = dofs[num_dofs_per_element * cell_index:
                         num_dofs_per_element * cell_index
                         + num_dofs_per_element]
        local_indices = []
        # Find local indices for dofs on interface for each dimension
        for i in range(tdim):
            dofs_on_interface = np.isin(cell_dofs, interface_dofs[i])
            local_indices.append(np.flatnonzero(dofs_on_interface))

        # First find all slaves, and the corresponding normal vector for each dof,
        # and gather them globally

        for dof_index in local_indices[tdim-1]:
            slave_l = cell_dofs[dof_index]
            slave_coords = dof_coords[slave_l]

            global_slave = global_indices[slave_l]

        # Cache for slave dof location, used for computing coefficients
        # of masters on other cell
        # Use the highest dim as the slave dof
        for dof_index in local_indices[tdim-1]:
            slave_l = cell_dofs[dof_index]
            slave_coords = dof_coords[slave_l]

            global_slave = global_indices[slave_l]
            slave_indices = []
            slave_normal = np.zeros(tdim, dtype=np.float64)

            if global_slave not in slaves:
                slaves.append(global_slave)
                master_dict[slave_index] = []
                master_owner_dict[slave_index] = []
                coeffs_dict[slave_index] = []
                # Find the other dofs in same dof coordinate and
                # add as master
                for i in range(tdim - 1):
                    for index in local_indices[i]:
                        dof_i = cell_dofs[index]
                        if np.allclose(dof_coords[dof_i], slave_coords):
                            global_master = global_indices[dof_i]
                            coeff = - n_vec[dof_i]/n_vec[slave_l]
                            if not np.isclose(coeff, 0):
                                owner_ranks.append(comm.rank)
                                masters.append(global_master)
                                coeffs.append(coeff)

                            slave_indices.append(dof_i)
                            break
                slave_indices.append(slave_l)
                assert(len(slave_indices) == tdim)
                possible_cells = np.array(dolfinx.geometry.
                                          compute_collisions_point(tree,
                                                                   slave_coords
                                                                   ))

                # Check if multiple possible cells is in top cube
                top_cells = possible_cells[np.isin(possible_cells,
                                                   ct_top_cells)]
                # Find cell a top cell within 1e-14 distance
                close_cell = dolfinx.cpp.geometry.select_colliding_cells(
                    mesh, list(top_cells), slave_coords, 1)
                # global_close_cells = np.hstack(
                #     V.mesh.mpi_comm().allgather(close_cell))
                # # Check if we have found any cells in master domain
                # if len(global_close_cells) == 0:
                #     raise RuntimeError("No top cells found for slave at"
                #                        + "({0:.2f}, {1:.2f}, {2:.2f})".format(
                #                            slave_coords[0], slave_coords[1],
                #                            slave_coords[2]))
                if len(close_cell) == 1:
                    cell = close_cell[0]
                    other_dofs = dofs[num_dofs_per_element * cell:
                                      num_dofs_per_element * cell
                                      + num_dofs_per_element]

                    # Get basis values, and add coeffs of sub space
                    # to masters with corresponding coeff
                    basis_values = get_basis(V._cpp_object,
                                             slave_coords, cell)
                    for k in range(tdim):
                        for local_idx, dof in enumerate(other_dofs):
                            l_coeff = basis_values[local_idx,
                                                   k]
                            l_coeff *= (n_vec[slave_indices[k]] /
                                        n_vec[slave_indices[tdim
                                                            - 1]])
                            if not np.isclose(l_coeff, 0):
                                owner_ranks.append(comm.rank)
                                masters.append(global_indices[dof])
                                coeffs.append(l_coeff)

                    offsets.append(len(masters))
                # assert(len(masters) == len(owner_ranks))
                slave_index += 1
            # if len(slaves) != len(offsets)-1:
            #     raise RuntimeError(
            #         "Something went wrong in master slave construction")
    print(V.mesh.mpi_comm().rank, slaves)

    assert(not np.all(np.isin(slaves, masters)))
    return (np.array(slaves, dtype=np.int64),
            np.array(masters, dtype=np.int64),
            np.array(coeffs, dtype=np.float64),
            np.array(offsets, dtype=np.int64),
            np.array(owner_ranks, dtype=np.int32))

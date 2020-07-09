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
    slave_facets = mt.indices[np.flatnonzero(mt.values == slave_i_value)]
    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, slave_facets)[:, 0]
        interface_dofs.append(i_dofs)

    # Get dof info
    global_indices = V.dofmap.index_map.global_indices(False)
    dofs = V.dofmap.list.array()
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Create facet-to-cell-connectivity
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    tree = geometry.BoundingBoxTree(mesh, tdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)

    # Extract all top cell values from MeshTags
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mt, slave_i_value)
    n_vec = nh.vector.getArray()
    slaves, masters, coeffs, offsets, owner_ranks = [], [], [], [0], []
    comm = V.mesh.mpi_comm()
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

        # Cache for slave dof location, used for computing coefficients
        # of masters on other cell
        # Use the highest dim as the slave dof
        for dof_index in local_indices[tdim-1]:
            slave_l = cell_dofs[dof_index]
            slave_coords = dof_coords[slave_l]

            global_slave = global_indices[slave_l]
            slave_indices = []
            if global_slave not in slaves:
                slaves.append(global_slave)
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

                # Check if we have found any cells in master domain
                if len(close_cell) == 0:
                    raise RuntimeError("No top cells found for slave at"
                                       + "({0:.2f}, {1:.2f}, {2:.2f})".format(
                                           slave_coords[0], slave_coords[1],
                                           slave_coords[2]))
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
            assert(len(masters) == len(owner_ranks))
            if len(slaves) != len(offsets)-1:
                raise RuntimeError(
                    "Something went wrong in master slave construction")
    assert(not np.all(np.isin(slaves, masters)))
    return (np.array(slaves, dtype=np.int64),
            np.array(masters, dtype=np.int64),
            np.array(coeffs, dtype=np.float64),
            np.array(offsets, dtype=np.int64),
            np.array(owner_ranks, dtype=np.int32))

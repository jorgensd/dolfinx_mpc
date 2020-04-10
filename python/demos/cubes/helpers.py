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
                         (for facets) and an int indicating
                         the marked interface.
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
    (mt, interface_value) = interface_info
    (ct, master_value) = cell_info

    # Locate dofs on both interfaces
    i_facets = mt.indices[np.flatnonzero(mt.values == interface_value)]

    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, i_facets)[:, 0]
        interface_dofs.append(i_dofs)

    # Get dof info
    global_indices = V.dofmap.index_map.global_indices(False)
    dofs = V.dofmap.list.array()
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Create facet-to-cell-connectivity
    mesh.create_connectivity(fdim, tdim)
    mesh.create_connectivity(tdim, fdim)
    tree = geometry.BoundingBoxTree(mesh, tdim)
    cell_to_facet = mesh.topology.connectivity(tdim, fdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)

    # Extract all top cell values from MeshTags
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mt, interface_value)
    n_vec = nh.vector.getArray()
    slaves, masters, coeffs, offsets = [], [], [], [0]

    for facet in i_facets:
        # Find cells connected to facets (Should only be one)
        cell_indices = facet_to_cell.links(facet)
        assert(len(cell_indices) == 1)
        cell_index = cell_indices[0]
        local_celltag_index = np.flatnonzero(ct.indices == cell_index)
        cell_dofs = dofs[num_dofs_per_element * cell_index:
                         num_dofs_per_element * cell_index
                         + num_dofs_per_element]
        local_indices = []
        # Find local indices for dofs on interface for each dimension
        for i in range(tdim):
            dofs_on_interface = np.isin(cell_dofs, interface_dofs[i])
            local_indices.append(np.flatnonzero(dofs_on_interface))

        if len(local_celltag_index) > 0:
            assert(len(local_celltag_index) == 1)
            is_slave_cell = not (
                ct.values[local_celltag_index[0]] == master_value)
        else:
            is_slave_cell = True

        # Cache for slave dof location, used for computing coefficients
        # of masters on other cell
        # Use the highest dim as the slave dof
        for dof_index in local_indices[tdim-1]:
            slave_l = cell_dofs[dof_index]
            slave_coords = dof_coords[slave_l]

            global_slave = global_indices[slave_l]
            slave_indices = []
            if is_slave_cell and global_slave not in slaves:
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
                                masters.append(global_master)
                                coeffs.append(coeff)
                            slave_indices.append(dof_i)
                            break
                slave_indices.append(slave_l)
                assert(len(slave_indices) == tdim)
                possible_cells = dolfinx.geometry.\
                    compute_collisions_point(tree, slave_coords)[0]
                # Check if multiple possible cells is in top cube
                top_cells = possible_cells[np.isin(possible_cells,
                                                   ct_top_cells)]
                # Check if we have found any cells in master domain
                # and handle special case of multiple cells
                if len(top_cells) == 0:
                    raise RuntimeError("No top cells found for slave at"
                                       + "({0:.2f}, {0:.2f}, {0:.2f})".format(
                                           slave_coords[0], slave_coords[1],
                                           slave_coords[2]))
                elif len(top_cells) == 1:
                    cells = top_cells
                else:
                    # Check if it is actually colliding with a cell
                    actual_collisions = dolfinx.geometry.\
                        compute_entity_collisions_mesh(
                            tree, mesh, slave_coords)[0]
                    mesh_top_cells = actual_collisions[np.isin(
                        actual_collisions, top_cells)]
                    # Fallback if it is not colliding with any cell
                    # due to difference in FEM cell and geometry cell
                    if len(mesh_top_cells) == 0:
                        fem_cells = dolfinx_mpc.cpp.mpc.\
                            check_cell_point_collision(
                                top_cells, V._cpp_object, slave_coords)
                        fem_top_cells = top_cells[fem_cells]
                        cells = fem_top_cells
                    else:
                        cells = mesh_top_cells
                    if len(cells) == 0:
                        # If not uniquely determined at this point
                        # Choose first that has facets on interface
                        cells = top_cells
                assert(len(cells) > 0)
                for cell in cells:
                    facets = cell_to_facet.links(cell)
                    facet_on_interface = np.any(np.isin(facets, i_facets))

                    if facet_on_interface:
                        other_dofs = dofs[num_dofs_per_element * cell:
                                          num_dofs_per_element * cell
                                          + num_dofs_per_element]

                        # Get basis values, and add coeffs of sub space
                        # to masters with corresponding coeff
                        basis_values = get_basis(V._cpp_object,
                                                 slave_coords, cell)
                        for k in range(tdim):
                            for local_idx, dof in enumerate(other_dofs):
                                l_coeff = 0
                                if dof in interface_dofs[k]:
                                    l_coeff = basis_values[local_idx,
                                                           k]
                                    l_coeff *= (n_vec[slave_indices[k]] /
                                                n_vec[slave_indices[tdim
                                                                    - 1]])
                                if not np.isclose(l_coeff, 0):
                                    masters.append(global_indices[dof])
                                    coeffs.append(l_coeff)
                        offsets.append(len(masters))
                        break
                if len(slaves) != len(offsets)-1:
                    raise RuntimeError(
                        "Something went wrong in master slave construction")
    assert(not np.all(np.isin(slaves, masters)))
    return (np.array(slaves, dtype=np.int64),
            np.array(masters, dtype=np.int64),
            np.array(coeffs, dtype=np.float64),
            np.array(offsets, dtype=np.int64))

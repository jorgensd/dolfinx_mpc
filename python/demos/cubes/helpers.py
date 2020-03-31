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


comp_col_pts = dolfinx.geometry.compute_collisions_point
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
    facet_tree = geometry.BoundingBoxTree(mesh, fdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)

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
                # Find facets that possibly contain the slave dof
                # FIXME: Does not work for quads, possibly due to
                # https://github.com/FEniCS/dolfinx/issues/767
                possible_facets = comp_col_pts(
                    facet_tree, slave_coords)[0]
                for o_facet in possible_facets:
                    facet_on_interface = o_facet in i_facets
                    c_cells = facet_to_cell.links(o_facet)
                    in_top_cube = False
                    if facet_on_interface:
                        assert(len(c_cells) == 1)
                        local_celltag_index2 = np.flatnonzero(
                            ct.indices == c_cells[0])

                        if len(local_celltag_index2 > 0):
                            assert(len(local_celltag_index2 == 1))
                            in_top_cube = ct.values[
                                local_celltag_index2[0]] == master_value
                        else:
                            in_top_cube = False
                    if in_top_cube:
                        other_index = c_cells[0]
                        other_dofs = dofs[num_dofs_per_element
                                          * other_index:
                                          num_dofs_per_element
                                          * other_index
                                          + num_dofs_per_element]

                        # Get basis values, and add coeffs of sub space
                        # to masters with corresponding coeff
                        basis_values = get_basis(V._cpp_object,
                                                 slave_coords,
                                                 other_index)
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

    return (np.array(slaves, dtype=np.int64),
            np.array(masters, dtype=np.int64),
            np.array(coeffs, dtype=np.float64),
            np.array(offsets, dtype=np.int64))

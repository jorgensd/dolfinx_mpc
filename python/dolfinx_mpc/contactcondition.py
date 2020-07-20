from .multipointconstraint import facet_normal_approximation
import dolfinx.fem as fem
import dolfinx.geometry as geometry
import dolfinx_mpc.cpp as cpp
import dolfinx.cpp as dcpp
import numpy as np


def create_contact_condition(V, meshtag, slave_marker, master_marker):
    # Extract information from functionspace and mesh
    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    bs = V.dofmap.index_map.block_size
    x = V.tabulate_dof_coordinates()
    local_size = V.dofmap.index_map.size_local*bs
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                           dtype=np.int64)
    comm = V.mesh.mpi_comm()

    # Compute approximate facet normal used in contact condition
    # over slave facets
    nh = facet_normal_approximation(V, meshtag, slave_marker)
    n_vec = nh.vector.getArray()

    # Locate facets with slaves (both owned and ghosted slaves)
    slave_facets = meshtag.indices[meshtag.values == slave_marker]

    # Find masters from the same block as the local slave (no ghosts)
    masters_in_slave_block = []
    for i in range(tdim-1):
        loc_master_i_dofs = fem.locate_dofs_topological(
            (V.sub(i), V.sub(i).collapse()), fdim, slave_facets)[:, 0]
        # Remove ghosts
        loc_master = loc_master_i_dofs[loc_master_i_dofs < local_size]
        masters_in_slave_block.append(loc_master)
        del loc_master_i_dofs, loc_master

    # Locate local slaves (owned and ghosted)
    slaves = fem.locate_dofs_topological(
        (V.sub(tdim-1), V.sub(tdim-1).collapse()), fdim, slave_facets)[:, 0]
    slave_coordinates = x[slaves]
    num_owned_slaves = len(slaves[slaves < local_size])

    # Find masters and coefficients in same block as slave
    masters_block = {}
    for i, slave in enumerate(slaves[:num_owned_slaves]):
        for j in range(tdim-1):
            master = masters_in_slave_block[j][i]
            coeff = -n_vec[master]/n_vec[slave]
            if not np.isclose(coeff, 0):
                if slave in masters_block.keys():
                    masters_block[slave]["masters"].append(master)
                    masters_block[slave]["coeffs"].append(coeff)
                    masters_block[slave]["owners"].append(comm.rank)
                else:
                    masters_block[slave] = {"masters": [master],
                                            "coeffs": [coeff],
                                            "owners": [comm.rank]}
            del coeff, master

    # Find all local cells that can contain masters
    facet_to_cell = V.mesh.topology.connectivity(fdim, tdim)
    if facet_to_cell is None:
        V.mesh.topology.create_connectivity(fdim, tdim)
        facet_to_cell = V.mesh.topology.connectivity(fdim, tdim)
    master_facets = meshtag.indices[meshtag.values == master_marker]
    local_master_cells = []
    for facet in master_facets:
        local_master_cells.extend(facet_to_cell.links(facet))
        del facet

    local_master_cells = np.unique(local_master_cells)

    # Find masters that is owned by this processor
    tree = geometry.BoundingBoxTree(V.mesh, tdim)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    loc2glob_cell = np.array(cell_map.global_indices(False),
                             dtype=np.int64)
    del cell_map

    # Loop through all local slaves, check if there is any
    # colliding master cell that is local
    slave_to_master_cell = {}
    for i, (dof, coord) in enumerate(zip(slaves[:num_owned_slaves],
                                         slave_coordinates
                                         [:num_owned_slaves])):
        possible_cells = np.array(
            geometry.compute_collisions_point(tree, coord.T),
            dtype=np.int32)
        if len(possible_cells) > 0:
            is_master_cell = np.isin(possible_cells, local_master_cells)
            glob_cells = loc2glob_cell[possible_cells]
            is_owned = np.logical_and(cmin <= glob_cells, glob_cells < cmax)
            candidates = possible_cells[np.logical_and(
                is_master_cell, is_owned)]
            verified_candidate = dcpp.geometry.select_colliding_cells(
                V.mesh, list(candidates), coord.T, 1)
            if len(verified_candidate) > 0:
                slave_to_master_cell[dof] = (verified_candidate[0], i)
            del (is_master_cell, glob_cells, is_owned,
                 candidates, verified_candidate)
        del possible_cells
    del cmin, cmax, loc2glob_cell

    # Loop through the slave dofs with masters on other interface and
    # compute corresponding coefficients
    masters_local = {}
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    for slave in slave_to_master_cell.keys():
        # Find normal vector for local slave (through blocks)
        b_off = slave % bs
        block_dofs = np.array([slave-b_off+i for i in range(bs)])
        n = list(n_vec[block_dofs])
        # Compute basis values for master at slave coordinate
        cell = slave_to_master_cell[slave][0]
        coord = slave_coordinates[slave_to_master_cell[slave][1]]
        basis_values = cpp.mpc.get_basis_functions(
            V._cpp_object, coord, cell)
        cell_dofs = V.dofmap.cell_dofs(cell)
        masters, owners, coeffs = [], [], []

        # Loop through all dofs in cell and append those
        # that yield a non-zero coefficient
        for k in range(tdim):
            for local_idx, dof in enumerate(cell_dofs):
                coeff = (basis_values[local_idx, k] *
                         n[k] / n[tdim-1])
                if not np.isclose(coeff, 0):
                    if dof < local_size:
                        owners.append(comm.rank)
                    else:
                        owners.append(ghost_owners[dof//bs-local_size//bs])
                    masters.append(dof)
                    coeffs.append(coeff)
                del coeff
        if len(masters) > 0:
            masters_local[slave] = {"masters": masters,
                                    "coeffs": coeffs, "owners": owners}
        del (b_off, block_dofs, n, cell, coord, basis_values, cell_dofs,
             masters, owners, coeffs)

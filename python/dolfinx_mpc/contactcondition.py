import dolfinx_mpc.utils
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
    comm = V.mesh.mpi_comm()

    # Compute approximate facet normal used in contact condition
    # over slave facets

    nh = dolfinx_mpc.utils.facet_normal_approximation(V, meshtag, slave_marker)
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
    if comm.size == 1:
        # Gather all masters and create constraint
        masters, coeffs, owners, offsets = [], [], [], [0]
        for slave in slaves[:num_owned_slaves]:
            if slave in masters_block.keys():
                masters.extend(masters_block[slave]["masters"])
                coeffs.extend(masters_block[slave]["coeffs"])
                owners.extend(masters_block[slave]["owners"])
            if slave in masters_local.keys():
                masters.extend(masters_local[slave]["masters"])
                coeffs.extend(masters_local[slave]["coeffs"])
                owners.extend(masters_local[slave]["owners"])
            offsets.append(len(masters))
        num_owned_slaves = len(slaves)
        return ((slaves, []), (masters, []),
                (coeffs, []), (owners, []), (offsets, [0]))

    # Find all processors which a slave collides with
    # (according to the bounding boxes). Send the slaves global dof number,
    # physical coordinate and corresponding normal vector to those procs.
    slaves_to_send = {}
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                           dtype=np.int64)
    for (dof, coord) in zip(slaves[:num_owned_slaves],
                            slave_coordinates[:num_owned_slaves]):
        # Find normal vector for local slave (through blocks)
        b_off = slave % bs
        block_dofs = np.array([slave-b_off+i for i in range(bs)])
        n = list(n_vec[block_dofs])

        # Get processors where boundingbox collides with point
        procs = np.array(cpp.mpc.compute_process_collisions(
            tree._cpp_object, coord.T), dtype=np.int32)
        procs = procs[procs != comm.rank]
        for proc in procs:
            if proc in slaves_to_send.keys():
                slaves_to_send[proc]["slaves"].append(loc_to_glob[dof])
                slaves_to_send[proc]["coords"].append(coord.tolist())
                slaves_to_send[proc]["normals"].append(n)
            else:
                slaves_to_send[proc] = {"slaves": [loc_to_glob[dof]],
                                        "coords": [coord.tolist()],
                                        "normals": [n]}
        del b_off, block_dofs, n, procs
    # Send information about collisions between slave and bounding
    #  box to the other processors
    bb_collision_recv = list(slaves_to_send.keys())
    for proc in range(comm.size):
        if proc in slaves_to_send.keys():
            comm.send(slaves_to_send[proc], dest=proc, tag=1)
        elif proc != comm.rank:
            comm.send(None, dest=proc, tag=1)
    del slaves_to_send

    # Receive information about slaves from other processors
    recv_slaves, recv_owners, recv_coords, recv_normals = [], [], [], []
    other_procs = np.arange(comm.size)
    other_procs = other_procs[other_procs != comm.rank]
    for proc in other_procs:
        slaves_to_recv = comm.recv(source=proc, tag=1)
        if slaves_to_recv is not None:
            recv_slaves.extend(slaves_to_recv["slaves"])
            recv_owners.extend([proc]*len(slaves_to_recv["slaves"]))
            recv_coords.extend(slaves_to_recv["coords"])
            recv_normals.extend(slaves_to_recv["normals"])
        del slaves_to_recv

    # If received slaves, find possible masters
    masters_for_recv_slaves = {}
    if len(recv_slaves) > 0:
        # NOTE: Repeat of previously used function
        # This should probably be captured as a separate function
        recv_slave_to_master_cell = {}
        for i, (dof, coord) in enumerate(zip(recv_slaves, recv_coords)):
            possible_cells = np.array(
                geometry.compute_collisions_point(tree, coord),
                dtype=np.int32)
            if len(possible_cells) > 0:
                is_master_cell = np.isin(possible_cells, local_master_cells)
                glob_cells = loc2glob_cell[possible_cells]
                is_owned = np.logical_and(
                    cmin <= glob_cells, glob_cells < cmax)
                candidates = possible_cells[np.logical_and(
                    is_master_cell, is_owned)]
                verified_candidate = dcpp.geometry.select_colliding_cells(
                    V.mesh, list(candidates), coord, 1)
                if len(verified_candidate) > 0:
                    recv_slave_to_master_cell[dof] = (verified_candidate[0], i)
                del (is_master_cell, glob_cells, is_owned,
                     candidates, verified_candidate)
            del possible_cells

        # Loop through received slaves and compute possible local contributions
        for slave in recv_slave_to_master_cell.keys():
            cell = recv_slave_to_master_cell[slave][0]
            index = recv_slave_to_master_cell[slave][1]
            coord = recv_coords[index]
            basis_values = cpp.mpc.get_basis_functions(
                V._cpp_object, coord, cell)
            cell_dofs = V.dofmap.cell_dofs(cell)
            masters, owners, coeffs = [], [], []

            # Loop through all dofs in cell and append those
            # that yield a non-zero coefficient
            for k in range(tdim):
                for local_idx, dof in enumerate(cell_dofs):
                    coeff = (basis_values[local_idx, k] *
                             recv_normals[index][k]
                             / recv_normals[index][tdim-1])
                    if not np.isclose(coeff, 0):
                        if dof < local_size:
                            owners.append(comm.rank)
                        else:
                            owners.append(ghost_owners[dof//bs-local_size//bs])
                        masters.append(loc_to_glob[dof])
                        coeffs.append(coeff)
                    del coeff
            if len(masters) > 0:
                # Sort masters by the slave ownership
                if recv_owners[index] in masters_for_recv_slaves.keys():
                    masters_for_recv_slaves[recv_owners[index]][slave] = {
                        "masters": masters, "coeffs": coeffs, "owners": owners}
                else:
                    masters_for_recv_slaves[recv_owners[index]] = {
                        slave: {"masters": masters, "coeffs": coeffs,
                                "owners": owners}}
            del (cell, basis_values, cell_dofs, index,
                 masters, owners, coeffs)
    # Send masters back to owning processor
    # master_send_procs = list(masters_for_recv_slaves.keys())
    for proc in set(recv_owners):
        if proc in masters_for_recv_slaves.keys():
            comm.send(masters_for_recv_slaves[proc], dest=proc, tag=2)
        else:
            comm.send(None, dest=proc, tag=2)
    # Receive masters from other processors and sort them by slave.
    # If duplicates, add both of them. One will be randomly selected later
    recv_masters = {}
    recv_masters_procs = []
    for proc in bb_collision_recv:
        recv_master_from_proc = comm.recv(source=proc, tag=2)
        if recv_master_from_proc is not None:
            recv_masters_procs.append(proc)
            for global_slave in recv_master_from_proc.keys():
                # NOTE: Could be done faster if the global_to_local
                # function in indexmap was exposed to the python
                # layer
                idx = np.where(global_slave ==
                               loc_to_glob[slaves[:num_owned_slaves]])[0][0]
                if slaves[idx] in recv_masters.keys():
                    recv_masters[slaves[idx]][proc] = (
                        recv_master_from_proc[global_slave])
                else:
                    recv_masters[slaves[idx]] = {
                        proc: recv_master_from_proc[global_slave]}
        del recv_master_from_proc

    # Remove duplicates by random selection
    recv_masters_unique = {}
    if len(recv_masters.keys()) > 0:
        for slave in recv_masters.keys():
            if len(recv_masters[slave].keys()) == 1:
                key = next(iter(recv_masters[slave]))
                recv_masters_unique[slave] = recv_masters[slave][key]
            else:
                selected_proc = np.random.choice(np.fromiter(
                    recv_masters[slave].keys(), dtype=np.int32))
                recv_masters_unique[slave] = recv_masters[slave][selected_proc]
    del recv_masters

    # Gather all masters for local slave in a
    # 1D list with corresponding offsets
    masters, coeffs, owners, offsets = [], [], [], [0]
    for slave in slaves[:num_owned_slaves]:
        if slave in masters_block.keys():
            masters.extend(loc_to_glob[masters_block[slave]["masters"]])
            coeffs.extend(masters_block[slave]["coeffs"])
            owners.extend(masters_block[slave]["owners"])
        # Always select local masters over received ones for the same slave
        if slave in masters_local.keys():
            masters.extend(loc_to_glob[masters_local[slave]["masters"]])
            coeffs.extend(masters_local[slave]["coeffs"])
            owners.extend(masters_local[slave]["owners"])
        elif slave in recv_masters_unique.keys():
            masters.extend(recv_masters_unique[slave]["masters"])
            coeffs.extend(recv_masters_unique[slave]["coeffs"])
            owners.extend(recv_masters_unique[slave]["owners"])
        offsets.append(len(masters))
    del recv_masters_unique, masters_block, masters_local

    # Get the shared indices for every block owned by the processors
    shared_indices = cpp.mpc.compute_shared_indices(V._cpp_object)
    send_ghost_masters = {}

    for (i, slave) in enumerate(slaves[:num_owned_slaves]):
        block = slave // bs
        if block in shared_indices.keys():
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[slave]] = {
                        "masters": masters[offsets[i]:offsets[i+1]],
                        "coeffs": coeffs[offsets[i]:offsets[i+1]],
                        "owners": owners[offsets[i]:offsets[i+1]]
                    }
                else:
                    send_ghost_masters[proc] = {loc_to_glob[slave]: {
                        "masters": masters[offsets[i]:offsets[i+1]],
                        "coeffs": coeffs[offsets[i]:offsets[i+1]],
                        "owners": owners[offsets[i]:offsets[i+1]]}}
        del block

    # Send masters for ghosted slaves to respective procs
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=3)
    del send_ghost_masters

    # For ghosted slaves, find their respective owner
    local_blocks_size = V.dofmap.index_map.size_local
    ghost_recv = []
    for slave in slaves[num_owned_slaves:]:
        block = slave//bs - local_blocks_size
        owner = ghost_owners[block]
        ghost_recv.append(owner)
        del block, owner
    ghost_recv = set(ghost_recv)

    # Receive masters for ghosted slaves
    recv_ghost_masters = {}
    for owner in ghost_recv:
        proc_masters = comm.recv(source=owner, tag=3)
        recv_ghost_masters.update(proc_masters)

    # Add ghost masters to array
    ghost_masters, ghost_coeffs, ghost_owners, ghost_offsets\
        = [], [], [], [0]
    for slave in loc_to_glob[slaves[num_owned_slaves:]]:
        ghost_masters.extend(recv_ghost_masters[slave]["masters"])
        ghost_coeffs.extend(recv_ghost_masters[slave]["coeffs"])
        ghost_owners.extend(recv_ghost_masters[slave]["owners"])
        ghost_offsets.append(len(ghost_masters))
    del recv_ghost_masters

    return ((slaves[:num_owned_slaves], slaves[num_owned_slaves:]),
            (masters, ghost_masters), (coeffs, ghost_coeffs),
            (owners, ghost_owners), (offsets, ghost_offsets))

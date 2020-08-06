import dolfinx.fem as fem
import numpy as np
import dolfinx_mpc.cpp as cpp


def create_slip_condition(V, normal, n_to_W, meshtag_info,
                          bcs=[]):
    if isinstance(V, tuple):
        W, Wsub = V
    else:
        W, Wsub = V, None
    mesh = W.mesh
    comm = mesh.mpi_comm()

    meshtag, marker = meshtag_info
    n_vec = normal.vector.getArray()

    # Info from parent space
    global_indices = np.array(W.dofmap.index_map.global_indices(False))
    ghost_owners = W.dofmap.index_map.ghost_owner_rank()
    bs = W.dofmap.index_map.block_size
    local_size = W.dofmap.index_map.size_local
    pairs = []
    x = W.tabulate_dof_coordinates()

    # Output arrays
    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]

    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(bc.dof_indices[:, 0])

    marked_entities = meshtag.indices[meshtag.values == marker]
    # Find dofs for each sub-space, and determine which are in Dirichlet BCS
    # and are local
    if Wsub is None:
        entity_dofs = fem.locate_dofs_topological(
            W, meshtag.dim, marked_entities)
        local_dofs = entity_dofs[entity_dofs[:, 0] < local_size * bs]
        # Determine which dofs are located at the same coordinate
        # and create a mapping for the local slaves
        for i in range(len(local_dofs[:, 0])):
            pairs.append(np.flatnonzero(np.isclose(np.linalg.norm(
                x[local_dofs[:, 0]]-x[local_dofs[i, 0]], axis=1), 0)))
        if len(pairs) > 0:
            unique_pairs = np.unique(pairs, axis=0)
            for pair in unique_pairs:
                # Determine slave by largest normal component
                normal_dofs = entity_dofs[pair].T[0]
                slave_index = np.argmax(np.abs(n_vec[normal_dofs]))
                pair_masters, pair_coeffs, pair_owners = [], [], []
                if not normal_dofs[slave_index] in bc_dofs:
                    master_indices = np.delete(
                        np.arange(W.num_sub_spaces()), slave_index)
                    for index in master_indices:
                        coeff = -(n_vec[normal_dofs[index]] /
                                  n_vec[normal_dofs[slave_index]])
                        if not np.isclose(coeff, 0):
                            pair_masters.append(
                                global_indices[normal_dofs[index]])
                            pair_coeffs.append(coeff)
                            if normal_dofs[index] < local_size*bs:
                                pair_owners.append(comm.rank)
                            else:
                                pair_owners.append(
                                    ghost_owners[normal_dofs[index]//bs
                                                 - local_size])
                    # If all coeffs are 0 (normal aligned with axis),
                    # append one to enforce Dirichlet BC 0
                    if len(pair_masters) == 0:
                        local_master = normal_dofs[master_indices[0]]
                        pair_masters = [global_indices[local_master]]
                        pair_coeffs = [-n_vec[local_master] /
                                       n_vec[normal_dofs[slave_index]]]
                        if local_master < local_size*bs:
                            pair_owners.append(comm.rank)
                        else:
                            pair_owners.append(
                                ghost_owners[local_master
                                             // bs - local_size])

                    slaves.append(normal_dofs[slave_index])
                    masters.extend(pair_masters)
                    coeffs.extend(pair_coeffs)
                    offsets.append(len(masters))
                    owners.extend(pair_owners)
    else:
        # If the constraint is on a subspace, which normal
        # is in the collapsed subspace
        n_imap = normal.function_space.dofmap.index_map
        entity_dofs = fem.locate_dofs_topological(
            (Wsub, normal.function_space), meshtag.dim, marked_entities)
        local_dofs = entity_dofs[entity_dofs[:, 1] < n_imap.size_local *
                                 n_imap.block_size]

        # Determine which dofs are located at the same coordinate
        # and create a mapping for the local slaves
        for i in range(len(local_dofs[:, 0])):
            pairs.append(np.flatnonzero(np.isclose(np.linalg.norm(
                x[local_dofs[:, 0]]-x[local_dofs[i, 0]], axis=1), 0)))

        if len(pairs) > 0:
            unique_pairs = np.unique(pairs, axis=0)

            for pair in unique_pairs:
                normal_dofs = entity_dofs[pair, 1]
                # Determine slave by largest normal component
                slave_index = np.argmax(np.abs(n_vec[normal_dofs]))
                pair_masters, pair_coeffs, pair_owners = [], [], []
                if not n_to_W[normal_dofs[slave_index]] in bc_dofs:
                    master_indices = np.delete(
                        np.arange(normal.function_space.num_sub_spaces()),
                        slave_index)
                    for index in master_indices:
                        coeff = -(n_vec[normal_dofs[index]] /
                                  n_vec[normal_dofs[slave_index]])
                        if not np.isclose(coeff, 0):
                            pair_masters.append(
                                global_indices[n_to_W[normal_dofs[index]]])
                            pair_coeffs.append(coeff)
                            if n_to_W[normal_dofs[index]] < local_size*bs:
                                pair_owners.append(comm.rank)
                            else:
                                pair_owners.append(
                                    ghost_owners[n_to_W[normal_dofs[index]]//bs
                                                 - local_size])
                    # If all coeffs are 0 (normal aligned with axis),
                    # append one to enforce Dirichlet BC 0
                    if len(pair_masters) == 0:
                        local_master = normal_dofs[master_indices[0]]
                        pair_masters = [n_to_W[global_indices[local_master]]]
                        pair_coeffs = [-n_vec[local_master] /
                                       n_vec[normal_dofs[slave_index]]]
                        if local_master < local_size*bs:
                            pair_owners.append(comm.rank)
                        else:
                            pair_owners.append(
                                ghost_owners[local_master//bs
                                             - local_size])

                    slaves.append(n_to_W[normal_dofs[slave_index]])
                    masters.extend(pair_masters)
                    coeffs.extend(pair_coeffs)
                    offsets.append(len(masters))
                    owners.extend(pair_owners)

    # Send ghost slaves to other processors
    shared_indices = cpp.mpc.compute_shared_indices(
        W._cpp_object)
    blocks = np.array(slaves)//bs
    ghost_indices = np.flatnonzero(
        np.isin(blocks, np.array(list(shared_indices.keys()))))
    slaves_to_send = {}
    for slave_index in ghost_indices:
        for proc in shared_indices[slaves[slave_index]//bs]:
            if (proc in slaves_to_send.keys()):
                slaves_to_send[proc][global_indices[
                    slaves[slave_index]]] = {
                    "masters": masters[offsets[slave_index]:
                                       offsets[slave_index+1]],
                    "coeffs": coeffs[offsets[slave_index]:
                                     offsets[slave_index+1]],
                    "owners": owners[offsets[slave_index]:
                                     offsets[slave_index+1]]
                }

            else:
                slaves_to_send[proc] = {global_indices[
                    slaves[slave_index]]: {
                    "masters": masters[offsets[slave_index]:
                                       offsets[slave_index+1]],
                    "coeffs": coeffs[offsets[slave_index]:
                                     offsets[slave_index+1]],
                    "owners": owners[offsets[slave_index]:
                                     offsets[slave_index+1]]}}

    for proc in slaves_to_send.keys():
        comm.send(slaves_to_send[proc], dest=proc, tag=1)

    # Receive ghost slaves
    if Wsub is None:
        ghost_dofs = np.array(entity_dofs[local_size * bs
                                          <= entity_dofs[:, 0]])
    else:
        ghost_dofs = np.array(entity_dofs[n_imap.size_local *
                                          n_imap.block_size
                                          <= entity_dofs[:, 1]])

    ghost_blocks = ghost_dofs[:, 0]//bs - local_size
    receiving_procs = []
    for block in ghost_blocks:
        receiving_procs.append(ghost_owners[block])
    unique_procs = set(receiving_procs)
    recv_slaves = {}

    for proc in unique_procs:
        recv_data = comm.recv(source=proc, tag=1)
        recv_slaves.update(recv_data)
    ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners,\
        ghost_offsets = [], [], [], [], [0]
    for i, slave in enumerate(global_indices[ghost_dofs[:, 0]]):
        if slave in recv_slaves.keys():
            ghost_slaves.append(ghost_dofs[i, 0])
            ghost_masters.extend(recv_slaves[slave]["masters"])
            ghost_coeffs.extend(recv_slaves[slave]["coeffs"])
            ghost_owners.extend(recv_slaves[slave]["owners"])
            ghost_offsets.append(len(ghost_masters))

    return ((slaves, ghost_slaves),
            (masters, ghost_masters), (coeffs, ghost_coeffs),
            (owners, ghost_owners), (offsets, ghost_offsets))

import typing

import numpy as np
import dolfinx.fem as fem
import dolfinx.function as function


def close_to(point):
    """
    Convenience function for locating a point [x,y,z]
    within an array x [[x0,...,xN],[y0,...,yN], [z0,...,zN]].
    If the point is not 3D it is padded with zeros
    """
    return lambda x: np.isclose(x, point).all(axis=0)


def create_dictionary_constraint(V: function.FunctionSpace, slave_master_dict:
                                 typing.Dict[bytes, typing.Dict[bytes, float]],
                                 subspace_slave=None,
                                 subspace_master=None
                                 ):
    """
    Returns a multi point constraint for a given function space
    and dictionary constraint.
    Input:
        V - The function space
        slave_master_dict - The dictionary.
        subspace_slave - If using mixed or vector space,
                         and only want to use dofs from
                         a sub space as slave add index here
        subspace_master - Subspace index for mixed or vector spaces
    Example:
        If the dof D located at [d0,d1] should be constrained to the dofs E and
        F at [e0,e1] and [f0,f1] as
        D = alpha E + beta F
        the dictionary should be:
            {np.array([d0, d1], dtype=np.float64).tobytes():
                 {np.array([e0, e1], dtype=np.float64).tobytes(): alpha,
                  np.array([f0, f1], dtype=np.float64).tobytes(): beta}}
    """
    dfloat = np.float64
    comm = V.mesh.mpi_comm()
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local*bs
    loc_to_glob = V.dofmap.index_map.global_indices(False)
    owned_entities = {}
    ghosted_entities = {}
    non_local_entities = {}
    slaves_local = {}
    slaves_ghost = {}
    for i, slave_point in enumerate(slave_master_dict.keys()):
        num_masters = len(list(slave_master_dict[slave_point].keys()))
        # Status for current slave, -1 if not on proc, 0 if ghost, 1 if owned
        slave_status = -1
        # Wrap slave point as numpy array
        slave_point_nd = np.zeros((3, 1), dtype=dfloat)
        for j, coord in enumerate(np.frombuffer(slave_point,
                                                dtype=dfloat)):
            slave_point_nd[j] = coord
        if subspace_slave is None:
            slave_dofs = np.array(fem.locate_dofs_geometrical(
                V, close_to(slave_point_nd))[:, 0])
        else:
            Vsub = V.sub(subspace_slave).collapse()
            slave_dofs = np.array(fem.locate_dofs_geometrical(
                (V.sub(subspace_slave), Vsub), close_to(slave_point_nd))[:, 0])
        if len(slave_dofs) == 1:
            # Decide if slave is ghost or not
            if slave_dofs[0] < local_size:
                slaves_local[i] = slave_dofs[0]
                owned_entities[i] = {
                    "masters": np.full(num_masters, -1, dtype=np.int64),
                    "coeffs": np.full(num_masters, -1, dtype=np.float64),
                    "owners": np.full(num_masters, -1, dtype=np.int32),
                    "master_count": 0,
                    "local_index": []}
                slave_status = 1
            else:
                slaves_ghost[i] = slave_dofs[0]
                ghosted_entities[i] = {
                    "masters": np.full(num_masters, -1, dtype=np.int64),
                    "coeffs": np.full(num_masters, -1, dtype=np.float64),
                    "owners": np.full(num_masters, -1, dtype=np.int32),
                    "master_count": 0,
                    "local_index": []}
                slave_status = 0
        elif len(slave_dofs) > 1:
            raise RuntimeError("Multiple slaves found at same point. " +
                               "You should use sub-space locators.")
        # Wrap as list to ensure order later
        master_points = list(slave_master_dict[slave_point].keys())
        master_points_nd = np.zeros((
            3, len(master_points)), dtype=dfloat)
        for (j, master_point) in enumerate(master_points):
            # Wrap bytes as numpy array
            for k, coord in enumerate(np.frombuffer(master_point,
                                                    dtype=dfloat)):
                master_points_nd[k, j] = coord
            if subspace_master is None:
                master_dofs = fem.locate_dofs_geometrical(
                    V, close_to(master_points_nd[:, j:j+1]))[:, 0]
            else:
                Vsub = V.sub(subspace_master).collapse()
                master_dofs = np.array(fem.locate_dofs_geometrical(
                    (V.sub(subspace_master), Vsub),
                    close_to(master_points_nd[:, j:j+1]))[:, 0])

            # Only add masters owned by this processor
            master_dofs = master_dofs[master_dofs < local_size]
            if len(master_dofs) == 1:
                if slave_status == -1:
                    if i in non_local_entities.keys():
                        non_local_entities[i]["masters"].append(
                            loc_to_glob[master_dofs[0]])
                        non_local_entities[i]["coeffs"].append(
                            slave_master_dict[slave_point][master_point])
                        non_local_entities[i]["owners"].append(comm.rank),
                        non_local_entities[i]["local_index"].append(j)
                    else:
                        non_local_entities[i] = {
                            "masters": [loc_to_glob[master_dofs[0]]],
                            "coeffs":
                            [slave_master_dict[slave_point][master_point]],
                            "owners": [comm.rank],
                            "local_index": [j]}
                elif slave_status == 0:
                    ghosted_entities[i]["masters"][j] = \
                        loc_to_glob[master_dofs[0]]
                    ghosted_entities[i]["owners"][j] = \
                        comm.rank
                    ghosted_entities[i]["coeffs"][j] = \
                        slave_master_dict[slave_point][master_point]
                    ghosted_entities[i]["local_index"].append(j)
                elif slave_status == 1:
                    owned_entities[i]["masters"][j] = \
                        loc_to_glob[master_dofs[0]]
                    owned_entities[i]["owners"][j] = \
                        comm.rank
                    owned_entities[i]["coeffs"][j] = \
                        slave_master_dict[slave_point][master_point]
                    owned_entities[i]["local_index"].append(j)

                else:
                    raise RuntimeError(
                        "Invalid slave status: {0:d}".format(slave_status)
                        + "(-1,0,1 are valid options)")
            elif len(master_dofs) > 1:
                raise RuntimeError("Multiple masters found at same point. "
                                   + "You should use sub-space locators.")

    # Send the ghost and owned entities to processor 0 to gather them
    data_to_send = [owned_entities, ghosted_entities, non_local_entities]
    if comm.rank != 0:
        comm.send(data_to_send, dest=0, tag=1)
    del owned_entities, ghosted_entities, non_local_entities
    # Gather all info on proc 0 and sort data
    if comm.rank == 0:
        recv = {0: data_to_send}
        for proc in range(1, comm.size):
            recv[proc] = comm.recv(source=proc, tag=1)

        for proc in range(comm.size):
            # Loop through all masters
            other_procs = np.arange(comm.size)
            other_procs = other_procs[other_procs != proc]
            # Loop through all owned slaves and ghosts, and update
            # the master entries
            for pair in [[0, 1], [1, 0]]:
                i, j = pair
                for slave in recv[proc][i].keys():
                    for o_proc in other_procs:
                        # If slave is ghost on other proc add local masters
                        if slave in recv[o_proc][j].keys():
                            # Update master with possible entries from ghost
                            o_masters = recv[o_proc][j][slave]["local_index"]
                            for o_master in o_masters:
                                recv[proc][i][slave]["masters"][o_master] = \
                                    recv[o_proc][j][slave]["masters"][o_master]
                                recv[proc][i][slave]["coeffs"][o_master] = \
                                    recv[o_proc][j][slave]["coeffs"][o_master]
                                recv[proc][i][slave]["owners"][o_master] = \
                                    recv[o_proc][j][slave]["owners"][o_master]
                        # If proc only has master, but not the slave
                        if slave in recv[o_proc][2].keys():
                            o_masters = recv[o_proc][2][slave]["local_index"]
                            # As non owned indices only store non-zero entries
                            for k, o_master in enumerate(o_masters):
                                recv[proc][i][slave]["masters"][o_master] = \
                                    recv[o_proc][2][slave]["masters"][k]
                                recv[proc][i][slave]["coeffs"][o_master] = \
                                    recv[o_proc][2][slave]["coeffs"][k]
                                recv[proc][i][slave]["owners"][o_master] = \
                                    recv[o_proc][2][slave]["owners"][k]
            if proc == comm.rank:
                owned_slaves = recv[proc][0]
                ghosted_slaves = recv[proc][1]
            else:
                # If no owned masters, do not send masters
                if len(recv[proc][0].keys()) > 0:
                    comm.send(recv[proc][0], dest=proc, tag=55)
                if len(recv[proc][1].keys()) > 0:
                    comm.send(recv[proc][1], dest=proc, tag=66)
    else:
        if len(slaves_local.keys()) > 0:
            owned_slaves = comm.recv(source=0, tag=55)
        if len(slaves_ghost.keys()) > 0:
            ghosted_slaves = comm.recv(source=0, tag=66)

    # Flatten slaves (local)
    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    for slave_index in slaves_local.keys():
        slaves.append(slaves_local[slave_index])
        masters.extend(owned_slaves[slave_index]["masters"])
        owners.extend(owned_slaves[slave_index]["owners"])
        coeffs.extend(owned_slaves[slave_index]["coeffs"])
        offsets.append(len(masters))

    # Flatten slaves (ghosts)
    ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners,\
        ghost_offsets = [], [], [], [], [0]

    for slave_index in slaves_ghost.keys():
        ghost_slaves.append(slaves_ghost[slave_index])
        ghost_masters.extend(ghosted_slaves[slave_index]["masters"])
        ghost_owners.extend(ghosted_slaves[slave_index]["owners"])
        ghost_coeffs.extend(ghosted_slaves[slave_index]["coeffs"])
        ghost_offsets.append(len(ghost_masters))
    return ((slaves, ghost_slaves), (masters, ghost_masters),
            (coeffs, ghost_coeffs), (owners, ghost_owners),
            (offsets, ghost_offsets))

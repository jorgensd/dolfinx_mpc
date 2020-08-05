import numpy as np
import dolfinx.fem
import dolfinx.geometry
import dolfinx_mpc.cpp


def create_periodic_condition(V, mt, tag, relation, bcs, scale=1):
    tdim = V.mesh.topology.dim
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local * bs
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                           dtype=np.int64)

    x = V.tabulate_dof_coordinates()
    comm = V.mesh.mpi_comm()
    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(bc.dof_indices[:, 0])

    slave_entities = mt.indices[mt.values == tag]
    slave_dofs = dolfinx.fem.locate_dofs_topological(
        V, mt.dim, slave_entities)[:, 0]
    # Filter out Dirichlet BC dofs
    slave_dofs = slave_dofs[np.isin(
        slave_dofs, bc_dofs, invert=True)]
    num_local_dofs = len(slave_dofs[slave_dofs
                                    < local_size])
    # Compute coordinates where each slave has to evaluate its masters
    tree = dolfinx.geometry.BoundingBoxTree(V.mesh, dim=tdim)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    loc2glob_cell = np.array(cell_map.global_indices(False),
                             dtype=np.int64)
    del cell_map
    master_coordinates = relation(x[slave_dofs].T).T

    # If data
    constraint_data = {slave_dof: {}
                       for slave_dof in slave_dofs[:num_local_dofs]}
    remaining_slaves = list(slave_dofs[:num_local_dofs])
    num_remaining = len(remaining_slaves)
    search_data = {}
    for (slave_dof, master_coordinate) in zip(
            slave_dofs[:num_local_dofs], master_coordinates):
        block_idx = slave_dof % bs
        procs = [0]
        if comm.size > 1:
            procs = np.array(
                dolfinx_mpc.cpp.mpc.compute_process_collisions(
                    tree._cpp_object, master_coordinate), dtype=np.int32)
        # Check if masters can be local
        search_globally = True
        masters_, coeffs_, owners_ = [], [], []
        if comm.rank in procs:
            possible_cells = \
                np.array(
                    dolfinx.geometry.compute_collisions_point(
                        tree, master_coordinate), dtype=np.int32)
            if len(possible_cells) > 0:
                glob_cells = loc2glob_cell[possible_cells]
                is_owned = np.logical_and(
                    cmin <= glob_cells, glob_cells < cmax)
                verified_candidate = \
                    dolfinx.cpp.geometry.select_colliding_cells(
                        V.mesh, list(possible_cells[is_owned]),
                        master_coordinate, 1)
                if len(verified_candidate) > 0:
                    basis_values = \
                        dolfinx_mpc.cpp.mpc.get_basis_functions(
                            V._cpp_object, master_coordinate,
                            verified_candidate[0])
                    cell_dofs = V.dofmap.cell_dofs(verified_candidate[0])
                    for local_idx, dof in enumerate(cell_dofs):
                        if not np.isclose(
                                basis_values[local_idx, block_idx], 0):
                            if dof < local_size:
                                owners_.append(comm.rank)
                            else:
                                owners_.append(
                                    ghost_owners[dof//bs-local_size//bs])
                            masters_.append(loc_to_glob[dof])
                            coeffs_.append(
                                scale*basis_values[local_idx, block_idx])
            if len(masters_) > 0:
                constraint_data[slave_dof]["masters"] = masters_
                constraint_data[slave_dof]["coeffs"] = coeffs_
                constraint_data[slave_dof]["owners"] = owners_
                remaining_slaves.remove(slave_dof)
                num_remaining -= 1
                search_globally = False
        # If masters not found on this processor
        if search_globally:
            other_procs = procs[procs != comm.rank]
            for proc in other_procs:
                if proc in search_data.keys():
                    search_data[proc][slave_dof] = master_coordinate
                else:
                    search_data[proc] = {slave_dof: master_coordinate}

    # Send search data
    recv_procs = []
    for proc in range(comm.size):
        if proc != comm.rank:
            if proc in search_data.keys():
                comm.send(search_data[proc], dest=proc, tag=11)
                recv_procs.append(proc)
            else:
                comm.send(None, dest=proc, tag=11)
    # Receive search data
    for proc in range(comm.size):
        if proc != comm.rank:
            in_data = comm.recv(source=proc, tag=11)
            out_data = {}
            if in_data is not None:
                for slave_dof in in_data.keys():
                    block_idx = slave_dof % bs
                    coordinate = in_data[slave_dof]
                    m_, c_, o_ = [], [], []
                    possible_cells = \
                        np.array(
                            dolfinx.geometry.compute_collisions_point(
                                tree, coordinate), dtype=np.int32)
                    if len(possible_cells) > 0:
                        glob_cells = loc2glob_cell[possible_cells]
                        is_owned = np.logical_and(
                            cmin <= glob_cells, glob_cells < cmax)
                        verified_candidate = \
                            dolfinx.cpp.geometry.select_colliding_cells(
                                V.mesh, list(possible_cells[is_owned]),
                                coordinate, 1)
                        if len(verified_candidate) > 0:
                            basis_values = \
                                dolfinx_mpc.cpp.mpc.get_basis_functions(
                                    V._cpp_object, coordinate,
                                    verified_candidate[0])
                            cell_dofs = V.dofmap.cell_dofs(
                                verified_candidate[0])
                            for idx, dof in enumerate(cell_dofs):
                                if not np.isclose(
                                        basis_values[idx, block_idx], 0):
                                    if dof < local_size:
                                        o_.append(comm.rank)
                                    else:
                                        o_.append(ghost_owners[
                                            dof//bs-local_size//bs])
                                    m_.append(loc_to_glob[dof])
                                    c_.append(
                                        scale * basis_values[idx, block_idx])
                    if len(m_) > 0:
                        out_data[slave_dof] = {"masters": m_,
                                               "coeffs": c_, "owners": o_}
                comm.send(out_data, dest=proc, tag=22)
    for proc in recv_procs:
        master_info = comm.recv(source=proc, tag=22)
        for dof in master_info.keys():
            # First come first serve principle
            if dof in remaining_slaves:
                constraint_data[dof]["masters"] = \
                    master_info[dof]["masters"]
                constraint_data[dof]["coeffs"] = master_info[dof]["coeffs"]
                constraint_data[dof]["owners"] = master_info[dof]["owners"]
                remaining_slaves.remove(dof)
                num_remaining -= 1
    assert(num_remaining == 0)

    # Get the shared indices per block and send all dofs
    send_ghost_masters = {}
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    shared_blocks = np.array(list(shared_indices.keys()))
    local_shared_blocks = shared_blocks[shared_blocks <
                                        V.dofmap.index_map.size_local]
    del shared_blocks
    for (i, dof) in enumerate(slave_dofs[:num_local_dofs]):
        block = dof//bs
        if block in local_shared_blocks:
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[dof]] = {
                        "masters": constraint_data[dof]["masters"],
                        "coeffs": constraint_data[dof]["coeffs"],
                        "owners": constraint_data[dof]["owners"]}
                else:
                    send_ghost_masters[proc] = {
                        loc_to_glob[dof]:
                        {"masters": constraint_data[dof]["masters"],
                         "coeffs": constraint_data[dof]["coeffs"],
                         "owners": constraint_data[dof]["owners"]}}
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=33)

    ghost_recv = []
    local_blocks_size = V.dofmap.index_map.size_local
    for dof in slave_dofs[num_local_dofs:]:
        g_block = dof//bs - local_blocks_size
        owner = ghost_owners[g_block]
        ghost_recv.append(owner)
        del owner, g_block
    ghost_recv = set(ghost_recv)
    ghost_dofs = {dof: {} for dof in slave_dofs[num_local_dofs:]}
    for owner in ghost_recv:
        from_owner = comm.recv(source=owner, tag=33)
        for dof in slave_dofs[num_local_dofs:]:
            if loc_to_glob[dof] in from_owner.keys():
                ghost_dofs[dof]["masters"] = \
                    from_owner[loc_to_glob[dof]]["masters"]
                ghost_dofs[dof]["coeffs"] = \
                    from_owner[loc_to_glob[dof]]["coeffs"]
                ghost_dofs[dof]["owners"] = \
                    from_owner[loc_to_glob[dof]]["owners"]
    # Flatten out arrays
    slaves = []
    masters = []
    coeffs = []
    owners = []
    offsets = [0]
    for dof in constraint_data.keys():
        slaves.append(dof)
        masters.extend(constraint_data[dof]["masters"])
        coeffs.extend(constraint_data[dof]["coeffs"])
        owners.extend(constraint_data[dof]["owners"])
        offsets.append(len(masters))

    for dof in ghost_dofs.keys():
        slaves.append(dof)
        masters.extend(ghost_dofs[dof]["masters"])
        coeffs.extend(ghost_dofs[dof]["coeffs"])
        owners.extend(ghost_dofs[dof]["owners"])
        offsets.append(len(masters))

    cc = dolfinx_mpc.cpp.mpc.ContactConstraint(
        V._cpp_object, slaves, num_local_dofs)
    cc.add_masters(masters, coeffs, owners, offsets)
    return cc

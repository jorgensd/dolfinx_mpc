import numpy as np
import dolfinx.fem
import dolfinx.geometry
import dolfinx_mpc.cpp


def create_periodic_condition(V, mt, tag, relation, bcs, scale=1):
    tdim = V.mesh.topology.dim
    bs = V.dofmap.index_map_bs
    size_local = V.dofmap.index_map.size_local
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(), dtype=np.int64)
    x = V.tabulate_dof_coordinates()
    comm = V.mesh.mpi_comm()
    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(bc.dof_indices[0])

    slave_entities = mt.indices[mt.values == tag]
    slave_dofs = dolfinx.fem.locate_dofs_topological(V, mt.dim, slave_entities)
    # Filter out Dirichlet BC dofs
    slave_dofs = slave_dofs[np.isin(slave_dofs, bc_dofs, invert=True)]
    num_local_dofs = len(slave_dofs[slave_dofs < bs * size_local])
    # Compute coordinates where each slave has to evaluate its masters
    tree = dolfinx.geometry.BoundingBoxTree(V.mesh, dim=tdim)
    global_tree = tree.compute_global_tree(comm)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    loc2glob_cell = np.array(cell_map.global_indices(), dtype=np.int64)
    del cell_map
    master_coordinates = relation(x[slave_dofs].T).T

    constraint_data = {slave_dof: {} for slave_dof in slave_dofs[:num_local_dofs]}
    remaining_slaves = list(slave_dofs[:num_local_dofs])
    num_remaining = len(remaining_slaves)
    search_data = {}
    for (slave_dof, master_coordinate) in zip(slave_dofs[:num_local_dofs], master_coordinates):
        block_idx = slave_dof % bs
        procs = [0]
        if comm.size > 1:
            procs = np.array(dolfinx.geometry.compute_collisions_point(global_tree, master_coordinate), dtype=np.int32)
        # Check if masters can be local
        search_globally = True
        masters_, coeffs_, owners_ = [], [], []
        if comm.rank in procs:
            possible_cells = np.array(dolfinx.geometry.compute_collisions_point(
                tree, master_coordinate), dtype=np.int32)
            if len(possible_cells) > 0:
                glob_cells = loc2glob_cell[possible_cells]
                is_owned = np.logical_and(cmin <= glob_cells, glob_cells < cmax)
                verified_candidate = dolfinx.cpp.geometry.select_colliding_cells(
                    V.mesh, list(possible_cells[is_owned]), master_coordinate, 1)
                if len(verified_candidate) > 0:
                    basis_values = dolfinx_mpc.cpp.mpc.get_basis_functions(
                        V._cpp_object, master_coordinate, verified_candidate[0])
                    cell_dofs = V.dofmap.cell_dofs(verified_candidate[0])
                    for local_idx, dof in enumerate(cell_dofs):
                        if not np.isclose(basis_values[local_idx, block_idx], 0):
                            block = dof // bs
                            rem = dof % bs
                            if block < size_local:
                                owners_.append(comm.rank)
                            else:
                                owners_.append(ghost_owners[block - size_local])
                            masters_.append(loc_to_glob[block] * bs + rem)
                            coeffs_.append(scale * basis_values[local_idx, block_idx])
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
                    possible_cells = np.array(dolfinx.geometry.compute_collisions_point(
                        tree, coordinate), dtype=np.int32)
                    if len(possible_cells) > 0:
                        glob_cells = loc2glob_cell[possible_cells]
                        is_owned = np.logical_and(cmin <= glob_cells, glob_cells < cmax)
                        verified_candidate = dolfinx.cpp.geometry.select_colliding_cells(
                            V.mesh, list(possible_cells[is_owned]), coordinate, 1)
                        if len(verified_candidate) > 0:
                            basis_values = dolfinx_mpc.cpp.mpc.get_basis_functions(
                                V._cpp_object, coordinate, verified_candidate[0])
                            cell_dofs = V.dofmap.cell_dofs(verified_candidate[0])
                            for idx, dof in enumerate(cell_dofs):
                                if not np.isclose(basis_values[idx, block_idx], 0):
                                    block = dof // bs
                                    rem = dof % bs
                                    if block < size_local:
                                        o_.append(comm.rank)
                                    else:
                                        o_.append(ghost_owners[block - size_local])
                                    m_.append(loc_to_glob[block] * bs + rem)
                                    c_.append(scale * basis_values[idx, block_idx])
                    if len(m_) > 0:
                        out_data[slave_dof] = {"masters": m_, "coeffs": c_, "owners": o_}
                comm.send(out_data, dest=proc, tag=22)
    for proc in recv_procs:
        master_info = comm.recv(source=proc, tag=22)
        for dof in master_info.keys():
            # First come first serve principle
            if dof in remaining_slaves:
                constraint_data[dof]["masters"] = master_info[dof]["masters"]
                constraint_data[dof]["coeffs"] = master_info[dof]["coeffs"]
                constraint_data[dof]["owners"] = master_info[dof]["owners"]
                remaining_slaves.remove(dof)
                num_remaining -= 1
    assert(num_remaining == 0)

    # Get the shared indices per block and send all dofs
    send_ghost_masters = {}
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    shared_blocks = np.array(list(shared_indices.keys()))
    local_shared_blocks = shared_blocks[shared_blocks < size_local]
    del shared_blocks
    for (i, dof) in enumerate(slave_dofs[:num_local_dofs]):
        block = dof // bs
        rem = dof % bs
        if block in local_shared_blocks:
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[block] * bs + rem] = {
                        "masters": constraint_data[dof]["masters"],
                        "coeffs": constraint_data[dof]["coeffs"],
                        "owners": constraint_data[dof]["owners"]}
                else:
                    send_ghost_masters[proc] = {loc_to_glob[block] * bs + rem:
                                                {"masters": constraint_data[dof]["masters"],
                                                 "coeffs": constraint_data[dof]["coeffs"],
                                                 "owners": constraint_data[dof]["owners"]}}
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=33)

    ghost_recv = []
    for dof in slave_dofs[num_local_dofs:]:
        g_block = dof // bs - size_local
        owner = ghost_owners[g_block]
        ghost_recv.append(owner)
        del owner, g_block
    ghost_recv = set(ghost_recv)
    ghost_dofs = {dof: {} for dof in slave_dofs[num_local_dofs:]}
    for owner in ghost_recv:
        from_owner = comm.recv(source=owner, tag=33)
        for dof in slave_dofs[num_local_dofs:]:
            block = dof // bs
            rem = dof % bs
            glob_dof = loc_to_glob[block] * bs + rem
            if glob_dof in from_owner.keys():
                ghost_dofs[dof]["masters"] = from_owner[glob_dof]["masters"]
                ghost_dofs[dof]["coeffs"] = from_owner[glob_dof]["coeffs"]
                ghost_dofs[dof]["owners"] = from_owner[glob_dof]["owners"]
    # Flatten out arrays
    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    for dof in constraint_data.keys():
        slaves.append(dof)
        masters.extend(constraint_data[dof]["masters"])
        coeffs.extend(constraint_data[dof]["coeffs"])
        owners.extend(constraint_data[dof]["owners"])
        offsets.append(len(masters))

    ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners, ghost_offsets = [], [], [], [], [0]
    for dof in ghost_dofs.keys():
        ghost_slaves.append(dof)
        ghost_masters.extend(ghost_dofs[dof]["masters"])
        ghost_coeffs.extend(ghost_dofs[dof]["coeffs"])
        ghost_owners.extend(ghost_dofs[dof]["owners"])
        ghost_offsets.append(len(ghost_masters))

    return ((slaves, ghost_slaves), (masters, ghost_masters),
            (coeffs, ghost_coeffs), (owners, ghost_owners),
            (offsets, ghost_offsets))

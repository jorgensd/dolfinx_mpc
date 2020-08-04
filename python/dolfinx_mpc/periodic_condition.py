import numpy as np
import dolfinx.fem
import dolfinx.geometry
import dolfinx_mpc.cpp


def create_periodic_condition(V, mt, tag, relation, bcs):
    tdim = V.mesh.topology.dim
    fdim = tdim - 1
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

    slave_facets = mt.indices[mt.values == tag]
    if V.num_sub_spaces() == 0:
        slave_blocks = dolfinx.fem.locate_dofs_topological(
            V, fdim, slave_facets)[:, 0]
        # Filter out Dirichlet BC dofs
        slave_blocks = slave_blocks[np.isin(
            slave_blocks, bc_dofs, invert=True)]
        num_local_blocks = len(slave_blocks[slave_blocks
                                            < local_size])
    else:
        # Use zeroth sub space to get the blocks and
        # relevant coordinates
        slave_blocks = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V.sub(0).collapse()), slave_facets)[:, 0]
        raise NotImplementedError("Not implemented yet")
    # Compute coordinates where each slave has to evaluate its masters
    tree = dolfinx.geometry.BoundingBoxTree(V.mesh, dim=tdim)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    loc2glob_cell = np.array(cell_map.global_indices(False),
                             dtype=np.int64)
    del cell_map
    master_coordinates = relation(x[slave_blocks].T).T

    # If data
    constraint_data = {slave_block: {}
                       for slave_block in slave_blocks[:num_local_blocks]}
    remaining_slaves = list(slave_blocks[:num_local_blocks])
    num_remaining = len(remaining_slaves)
    search_data = {}
    for (slave_block, master_coordinate) in zip(
            slave_blocks[:num_local_blocks], master_coordinates):
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
                    for k in range(bs):
                        for local_idx, dof in enumerate(cell_dofs):
                            if not np.isclose(basis_values[local_idx, k], 0):
                                if dof < local_size:
                                    owners_.append(comm.rank)
                                else:
                                    owners_.append(
                                        ghost_owners[dof//bs-local_size//bs])
                                masters_.append(loc_to_glob[dof])
                                coeffs_.append(basis_values[local_idx, k])
            if len(masters_) > 0:
                constraint_data[slave_block]["masters"] = masters_
                constraint_data[slave_block]["coeffs"] = coeffs_
                constraint_data[slave_block]["owners"] = owners_
                remaining_slaves.remove(slave_block)
                num_remaining -= 1
                search_globally = False
        # If masters not found on this processor
        if search_globally:
            other_procs = procs[procs != comm.rank]
            for proc in other_procs:
                if proc in search_data.keys():
                    search_data[proc][slave_block] = master_coordinate
                else:
                    search_data[proc] = {slave_block: master_coordinate}

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
                for block in in_data.keys():
                    coordinate = in_data[block]
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
                            for k in range(bs):
                                for idx, dof in enumerate(cell_dofs):
                                    if not np.isclose(basis_values[idx, k], 0):
                                        if dof < local_size:
                                            o_.append(comm.rank)
                                        else:
                                            o_.append(ghost_owners[
                                                dof//bs-local_size//bs])
                                        m_.append(loc_to_glob[dof])
                                        c_.append(basis_values[idx, k])
                    if len(m_) > 0:
                        out_data[block] = {"masters": m_,
                                           "coeffs": c_, "owners": o_}
                comm.send(out_data, dest=proc, tag=22)
    for proc in recv_procs:
        master_info = comm.recv(source=proc, tag=22)
        for block in master_info.keys():
            # First come first serve principle
            if block in remaining_slaves:
                constraint_data[block]["masters"] = \
                    master_info[block]["masters"]
                constraint_data[block]["coeffs"] = master_info[block]["coeffs"]
                constraint_data[block]["owners"] = master_info[block]["owners"]
                remaining_slaves.remove(block)
                num_remaining -= 1
    assert(num_remaining == 0)

    # Send ghost data
    send_ghost_masters = {}
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    for (i, block) in enumerate(slave_blocks[:num_local_blocks]):
        if block in shared_indices.keys():
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[bs*block]] = {
                        "masters": constraint_data[block]["masters"],
                        "coeffs": constraint_data[block]["coeffs"],
                        "owners": constraint_data[block]["owners"]}
                else:
                    send_ghost_masters[proc] = {
                        loc_to_glob[bs*block]:
                        {"masters": constraint_data[block]["masters"],
                         "coeffs": constraint_data[block]["coeffs"],
                         "owners": constraint_data[block]["owners"]}}
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=33)

    ghost_recv = []
    for block in slave_blocks[num_local_blocks:]:
        g_block = int(block - local_size/bs)
        owner = ghost_owners[g_block]
        ghost_recv.append(owner)
        del owner
    ghost_recv = set(ghost_recv)
    ghost_block = {block: {} for block in slave_blocks[num_local_blocks:]}
    for owner in ghost_recv:
        from_owner = comm.recv(source=owner, tag=33)
        for block in slave_blocks[num_local_blocks:]:
            if loc_to_glob[block*bs] in from_owner.keys():
                ghost_block[block]["masters"] = \
                    from_owner[loc_to_glob[block*bs]]["masters"]
                ghost_block[block]["coeffs"] = \
                    from_owner[loc_to_glob[block*bs]]["coeffs"]
                ghost_block[block]["owners"] = \
                    from_owner[loc_to_glob[block*bs]]["owners"]
    # Flatten out arrays
    slaves = []
    masters = []
    coeffs = []
    owners = []
    offsets = [0]
    for block in constraint_data.keys():
        slaves.append(block)
        masters.extend(constraint_data[block]["masters"])
        coeffs.extend(constraint_data[block]["coeffs"])
        owners.extend(constraint_data[block]["owners"])
        offsets.append(len(masters))
    for block in ghost_block.keys():
        slaves.append(block)
        masters.extend(ghost_block[block]["masters"])
        coeffs.extend(ghost_block[block]["coeffs"])
        owners.extend(ghost_block[block]["owners"])
        offsets.append(len(masters))

    cc = dolfinx_mpc.cpp.mpc.ContactConstraint(
        V._cpp_object, slaves, num_local_blocks)
    cc.add_masters(masters, coeffs, owners, offsets)
    return cc

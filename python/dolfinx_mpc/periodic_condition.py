from typing import Callable, List

import dolfinx
import numpy as np
from petsc4py import PETSc

import dolfinx_mpc.cpp


def create_periodic_condition_topological(V: dolfinx.FunctionSpace, mt: dolfinx.MeshTags, marker: int,
                                          relation: Callable[[np.ndarray], np.ndarray], bcs: List[dolfinx.DirichletBC],
                                          scale: PETSc.ScalarType):
    """
    Create periodic condition for all dofs in MeshTag with given marker:
       u(x_i) = scale * u(relation(x_i))
    for all x_i on marked entities.
    """
    slave_entities = mt.indices[mt.values == marker]
    slave_blocks = dolfinx.fem.locate_dofs_topological(V, mt.dim, slave_entities, remote=True)
    return _create_periodic_condition(V, slave_blocks, relation, bcs, scale)


def create_periodic_condition_geometrical(V: dolfinx.FunctionSpace, indicator: Callable[[np.ndarray], np.ndarray],
                                          relation: Callable[[np.ndarray], np.ndarray],
                                          bcs: List[dolfinx.DirichletBC], scale: PETSc.ScalarType):
    """
    Create a periodic condition for all degrees of freedom's satisfying indicator(x):
       u(x_i) = scale * u(relation(x_i)) for all x_i where indicator(x_i) == True
    """
    slave_blocks = dolfinx.fem.locate_dofs_geometrical(V, indicator)
    return _create_periodic_condition(V, slave_blocks, relation, bcs, scale)


def _create_periodic_condition(V: dolfinx.FunctionSpace, slave_blocks: np.ndarray,
                               relation: Callable[[np.ndarray], np.ndarray],
                               bcs: List[dolfinx.DirichletBC], scale: PETSc.ScalarType):
    """
    Create a periodic condition condition on all input slave blocks,
    x_i = x(slave_block)
    x_j = x(relation(x_i))
    u(x_i) = scale * u(x_j)
    """
    tdim = V.mesh.topology.dim
    bs = V.dofmap.index_map_bs
    size_local = V.dofmap.index_map.size_local
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    imap = V.dofmap.index_map
    x = V.tabulate_dof_coordinates()
    comm = V.mesh.mpi_comm()

    # Filter out Dirichlet BC dofs
    bc_dofs = []
    for bc in bcs:
        bc_indices, _ = bc._cpp_object.dof_indices()
        bc_dofs.extend(bc_indices)
    slave_blocks = slave_blocks[np.isin(slave_blocks, bc_dofs, invert=True)]
    num_local_blocks = len(slave_blocks[slave_blocks < size_local])

    # Compute coordinates where each slave has to evaluate its masters
    # FIXME: could be reduced to the set of boundary entities
    tree = dolfinx.geometry.BoundingBoxTree(V.mesh, tdim, padding=1e-15)
    global_tree = tree.create_global_tree(comm)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    master_coordinates = relation(x[slave_blocks].T).T
    master_procs = dolfinx.geometry.compute_collisions(global_tree, master_coordinates)
    possible_local_cells = dolfinx.geometry.compute_collisions(tree, master_coordinates)
    verified_candidates = dolfinx.geometry.compute_colliding_cells(V.mesh, possible_local_cells, master_coordinates)
    del possible_local_cells

    constraint_data = {slave * bs + j: {} for slave in slave_blocks[:num_local_blocks] for j in range(bs)}
    remaining_slaves = list(constraint_data.keys())

    num_remaining = len(remaining_slaves)
    search_data = {}
    for i, (slave_block, master_coordinate) in enumerate(zip(slave_blocks[:num_local_blocks], master_coordinates)):
        for block_idx in range(bs):
            slave_dof = slave_block * bs + block_idx
            procs = master_procs.links(i)

            # Check if masters can be local
            search_globally = True
            masters_, coeffs_, owners_ = [], [], []
            if comm.rank in procs:
                verified_candidate = verified_candidates.links(i)
                is_owned = verified_candidate < cell_map.size_local
                verified_candidate = verified_candidate[is_owned]
                if len(verified_candidate) > 0:
                    basis_values = dolfinx_mpc.cpp.mpc.get_basis_functions(
                        V._cpp_object, master_coordinate, verified_candidate[0])
                    cell_blocks = V.dofmap.cell_dofs(verified_candidate[0])
                    for local_idx, block in enumerate(cell_blocks):
                        for rem in range(bs):
                            if not np.isclose(basis_values[local_idx * bs + rem, block_idx], 0):
                                if block < size_local:
                                    owners_.append(comm.rank)
                                else:
                                    owners_.append(ghost_owners[block - size_local])
                                masters_.append(imap.local_to_global([block])[0] * bs + rem)
                                coeffs_.append(scale * basis_values[local_idx * bs + rem, block_idx])
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
                dofs = np.asarray(list(search_data[proc].keys()), dtype=np.int32)
                coordinates = np.zeros((len(dofs), 3), dtype=np.float64)
                for i, dof in enumerate(dofs):
                    coordinates[i] = search_data[proc][dof]
                comm.send(dofs, dest=proc, tag=11)
                comm.send(coordinates, dest=proc, tag=12)
                recv_procs.append(proc)
            else:
                comm.send(None, dest=proc, tag=11)
                comm.send(None, dest=proc, tag=12)
    # Receive search data
    for proc in range(comm.size):
        if proc != comm.rank:
            in_dofs = comm.recv(source=proc, tag=11)
            in_coords = comm.recv(source=proc, tag=12)

            out_data = {}
            if in_dofs is not None:
                local_cells = dolfinx.geometry.compute_collisions(tree, in_coords)
                verified_cells = dolfinx.geometry.compute_colliding_cells(V.mesh, local_cells, in_coords)
                del local_cells
                for i, slave_dof in enumerate(in_dofs):
                    block_idx = slave_dof % bs
                    m_, c_, o_ = [], [], []
                    owned_cells = verified_cells.links(i)[verified_cells.links(i) < cell_map.size_local]
                    if len(owned_cells) > 0:

                        basis_values = dolfinx_mpc.cpp.mpc.get_basis_functions(
                            V._cpp_object, in_coords[i], owned_cells[0])
                        cell_blocks = V.dofmap.cell_dofs(owned_cells[0])
                        for idx, block in enumerate(cell_blocks):
                            for rem in range(bs):
                                if not np.isclose(basis_values[idx * bs + rem, block_idx], 0):
                                    if block < size_local:
                                        o_.append(comm.rank)
                                    else:
                                        o_.append(ghost_owners[block - size_local])
                                    m_.append(imap.local_to_global([block])[0] * bs + rem)
                                    c_.append(scale * basis_values[idx * bs + rem, block_idx])
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
    shared_blocks = np.fromiter(shared_indices.keys(), dtype=np.int32)
    local_shared_blocks = shared_blocks[shared_blocks < size_local]
    del shared_blocks
    for (i, block) in enumerate(slave_blocks[:num_local_blocks]):
        for rem in range(bs):
            dof = block * bs + rem
            if block in local_shared_blocks:
                for proc in shared_indices[block]:
                    if proc in send_ghost_masters.keys():
                        send_ghost_masters[proc][imap.local_to_global([block])[0] * bs + rem] = {
                            "masters": constraint_data[dof]["masters"],
                            "coeffs": constraint_data[dof]["coeffs"],
                            "owners": constraint_data[dof]["owners"]}
                    else:
                        send_ghost_masters[proc] = {imap.local_to_global([block])[0] * bs + rem:
                                                    {"masters": constraint_data[dof]["masters"],
                                                     "coeffs": constraint_data[dof]["coeffs"],
                                                     "owners": constraint_data[dof]["owners"]}}
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=33)

    ghost_recv = []
    for block in slave_blocks[num_local_blocks:]:
        owner = ghost_owners[block - size_local]
        ghost_recv.append(owner)
        del owner, block
    ghost_recv = set(ghost_recv)
    ghost_dofs = {block * bs + j: {} for block in slave_blocks[num_local_blocks:] for j in range(bs)}
    for owner in ghost_recv:
        from_owner = comm.recv(source=owner, tag=33)
        for block in slave_blocks[num_local_blocks:]:
            glob_block = imap.local_to_global([block])[0]
            for rem in range(bs):
                glob_dof = glob_block * bs + rem
                if glob_dof in from_owner.keys():
                    dof = block * bs + rem
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
    # Add in ghost data
    for dof in ghost_dofs.keys():
        slaves.append(dof)
        masters.extend(ghost_dofs[dof]["masters"])
        coeffs.extend(ghost_dofs[dof]["coeffs"])
        owners.extend(ghost_dofs[dof]["owners"])
        offsets.append(len(masters))

    return (np.asarray(slaves, dtype=np.int32), np.asarray(masters, dtype=np.int64),
            np.asarray(coeffs, dtype=PETSc.ScalarType), np.asarray(owners, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32))

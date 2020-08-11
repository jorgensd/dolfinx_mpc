import dolfinx.fem as fem
import numpy as np
import dolfinx.common as common


def create_slip_condition(V, normal, n_to_W, facet_info,
                          bcs=[]):
    timer = common.Timer("~MPC: Create slip condition")
    if isinstance(V, tuple):
        W, Wsub = V
    else:
        W, Wsub = V, None
    mesh = W.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    comm = mesh.mpi_comm()

    meshtag, marker = facet_info

    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(bc.dof_indices[:, 0])

    marked_facets = meshtag.indices[meshtag.values == marker]
    # Find dofs for each sub-space, and determine which are in Dirichlet BCS
    # and are local
    if Wsub is None:
        raise NotImplementedError("not implemented for non-sub spaces")
    else:
        x = W.tabulate_dof_coordinates()
        facet_dofs = fem.locate_dofs_topological(
            (Wsub, normal.function_space), fdim, marked_facets)
        local_dofs = facet_dofs
        # Info from parent space
        W_global_indices = np.array(W.dofmap.index_map.global_indices(False))
        W_ghost_owners = W.dofmap.index_map.ghost_owner_rank()
        W_bs = W.dofmap.index_map.block_size
        W_local_size = W.dofmap.index_map.size_local

        # Normal vector with ghost values
        n_vec = normal.x.array()
        # Determine which dofs are located at the same coordinate
        # and create a mapping for the local slaves
        pairs = []
        for i in range(len(local_dofs[:, 0])):
            pairs.append(np.flatnonzero(np.isclose(np.linalg.norm(
                x[local_dofs[:, 0]]-x[local_dofs[i, 0]], axis=1), 0)))
        # Check if any dofs are on this processor
        slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
        ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners,\
            ghost_offsets = [], [], [], [], [0]

        if len(pairs) > 0:
            unique_pairs = np.unique(pairs, axis=0)

            for pair in unique_pairs:
                normal_dofs = facet_dofs[pair, 1]
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
                                W_global_indices[n_to_W[normal_dofs[index]]])
                            pair_coeffs.append(coeff)
                            if n_to_W[normal_dofs[index]] < W_local_size*W_bs:
                                pair_owners.append(comm.rank)
                            else:
                                pair_owners.append(
                                    W_ghost_owners[
                                        n_to_W[normal_dofs[index]]//W_bs
                                        - W_local_size])
                    # If all coeffs are 0 (normal aligned with axis),
                    # append one to enforce Dirichlet BC 0
                    if len(pair_masters) == 0:
                        master = n_to_W[W_global_indices[
                            normal_dofs[master_indices[0]]]]
                        pair_masters = [normal_dofs[master_indices[0]]]
                        pair_coeffs = [-n_vec[normal_dofs[master_indices[0]]] /
                                       n_vec[normal_dofs[slave_index]]]
                        if master < W_local_size*W_bs:
                            pair_owners.append(comm.rank)
                        else:
                            pair_owners.append(
                                ghost_owners[master//W_bs - W_local_size])
                    if n_to_W[normal_dofs[slave_index]] < W_local_size*W_bs:
                        slaves.append(n_to_W[normal_dofs[slave_index]])
                        masters.extend(pair_masters)
                        coeffs.extend(pair_coeffs)
                        offsets.append(len(masters))
                        owners.extend(pair_owners)
                    else:
                        ghost_slaves.append(n_to_W[normal_dofs[slave_index]])
                        ghost_masters.extend(pair_masters)
                        ghost_coeffs.extend(pair_coeffs)
                        ghost_offsets.append(len(ghost_masters))
                        ghost_owners.extend(pair_owners)

    timer.stop()
    return ((slaves, ghost_slaves),
            (masters, ghost_masters), (coeffs, ghost_coeffs),
            (owners, ghost_owners), (offsets, ghost_offsets))

import dolfinx.fem as fem
import numpy as np
import dolfinx.common as common


def create_slip_condition(V, normal, n_to_W, meshtag_info, bcs=[]):
    timer = common.Timer("~MPC: Create slip condition")
    if isinstance(V, tuple):
        W, Wsub = V
    else:
        W, Wsub = V, None
    mesh = W.mesh
    comm = mesh.mpi_comm()

    meshtag, marker = meshtag_info
    n_vec = np.array(normal.x.array, dtype=type(normal.x.array[0]))

    # Info from parent space
    W_imap = W.dofmap.index_map
    W_ghost_owners = W_imap.ghost_owner_rank()
    W_bs = W.dofmap.index_map_bs
    W_local_size = W_imap.size_local

    # Output arrays
    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    ghost_slaves, ghost_masters, ghost_coeffs, ghost_owners, ghost_offsets = [], [], [], [], [0]
    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_indices, _ = bc.dof_indices()
        bc_dofs.extend(bc_indices)

    marked_entities = meshtag.indices[meshtag.values == marker]

    # Locate dofs in W on facets and create mapping from normal space to W
    if Wsub is None:
        # This return the entity of the block, has to be unravelled
        entity_dofs = fem.locate_dofs_topological(W, meshtag.dim, marked_entities)

        def normal_to_W(x):
            return normal_dofs[x]

    else:
        # Locate dofs in normal-sub space
        def normal_to_W(x):
            return n_to_W[normal_dofs[x]]
        entity_dofs = fem.locate_dofs_topological(normal.function_space, meshtag.dim, marked_entities)
        bs_n = normal.function_space.dofmap.index_map_bs

    for block in entity_dofs:
        if Wsub is None:
            normal_dofs = [block * W_bs + k for k in range(W_bs)]
        else:
            normal_dofs = [block * bs_n + k for k in range(bs_n)]

        # Determine slave by largest normal component
        slave_index = np.argmax(np.abs(n_vec[normal_dofs]))
        pair_masters, pair_coeffs, pair_owners = [], [], []
        if not normal_to_W(slave_index) in bc_dofs:
            master_indices = np.delete(np.arange(normal.function_space.num_sub_spaces()), slave_index)
            for index in master_indices:
                coeff = -n_vec[normal_dofs[index]] / n_vec[normal_dofs[slave_index]]
                if not np.isclose(coeff, 0):
                    pair_masters.append(W_imap.local_to_global([normal_to_W(index)
                                                                // W_bs])[0] * W_bs + normal_to_W(index) % W_bs)
                    pair_coeffs.append(coeff)
                    if normal_to_W(index) < W_local_size * W_bs:
                        pair_owners.append(comm.rank)
                    else:
                        pair_owners.append(W_ghost_owners[normal_to_W(index) // W_bs - W_local_size])
            # If all coeffs are 0 (normal aligned with axis),
            # append one to enforce Dirichlet BC 0
            if len(pair_masters) == 0:
                master = normal_to_W(master_indices[0])
                pair_masters = [W_imap.local_to_global([master // W_bs])[0] * W_bs + master % W_bs]
                pair_coeffs = [-n_vec[master_indices[0]] / n_vec[normal_dofs[slave_index]]]
                if master < W_local_size * W_bs:
                    pair_owners.append(comm.rank)
                else:
                    pair_owners.append(W_ghost_owners[master // W_bs - W_local_size])
            if normal_to_W(slave_index) < W_local_size * W_bs:
                slaves.append(normal_to_W(slave_index))
                masters.extend(pair_masters)
                coeffs.extend(pair_coeffs)
                offsets.append(len(masters))
                owners.extend(pair_owners)
            else:
                ghost_slaves.append(normal_to_W(slave_index))
                ghost_masters.extend(pair_masters)
                ghost_coeffs.extend(pair_coeffs)
                ghost_offsets.append(len(ghost_masters))
                ghost_owners.extend(pair_owners)

    timer.stop()
    return ((slaves, ghost_slaves), (masters, ghost_masters), (coeffs, ghost_coeffs),
            (owners, ghost_owners), (offsets, ghost_offsets))

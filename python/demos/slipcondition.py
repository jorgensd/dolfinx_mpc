import dolfinx.fem as fem
import dolfinx_mpc
import numpy as np
import dolfinx_mpc.cpp as cpp


def slip_condition(V, normal, n_to_W, facet_info,
                   bcs=[]):
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

    # n_vec = normal.vector.getArray()
    # del n

    # Find dofs for each sub-space, and determine which are in Dirichlet BCS
    # and are local
    if Wsub is None:
        raise NotImplementedError("not implemented for non-sub spaces")
    else:
        x = W.tabulate_dof_coordinates()
        facet_dofs = fem.locate_dofs_topological(
            (Wsub, normal.function_space), tdim - 1, marked_facets)
        local_dofs = facet_dofs[facet_dofs[:, 1] < normal.function_space
                                .dofmap.index_map.size_local]
        pairs = []
        n_vec = normal.vector.getArray()
        # Determine which dofs are located at the same coordinate
        # and create a mapping for the local slaves
        for i in range(len(local_dofs[:, 0])):
            pairs.append(np.flatnonzero(np.isclose(np.linalg.norm(
                x[local_dofs[:, 0]]-x[local_dofs[i, 0]], axis=1), 0)))
        unique_pairs = np.unique(pairs, axis=0)

        slaves, masters, coeffs, offsets = [], [], [], [0]
        for pair in unique_pairs:
            normal_dofs = facet_dofs[pair, 1]
            # Determine slave by largest normal component
            slave_index = np.argmax(np.abs(n_vec[normal_dofs]))
            pair_masters, pair_coeffs = [], []
            if not n_to_W[normal_dofs[slave_index]] in bc_dofs:
                master_indices = np.delete(
                    np.arange(normal.function_space.num_sub_spaces()),
                    slave_index)
                for index in master_indices:
                    coeff = -(n_vec[normal_dofs[index]] /
                              n_vec[normal_dofs[slave_index]])
                    if not np.isclose(coeff, 0):
                        pair_masters.append(n_to_W[normal_dofs[index]])
                        pair_coeffs.append(coeff)
                # If all coeffs are 0 (normal aligned with spatial coordinate,
                # append one to enforce Dirichlet BC 0
                if len(pair_masters) == 0:
                    pair_masters = [n_to_W[normal_dofs[master_indices[0]]]]
                    pair_coeffs = [-n_vec[normal_dofs[master_indices[0]]] /
                                   n_vec[normal_dofs[slave_index]]]
                slaves.append(n_to_W[normal_dofs[slave_index]])
                masters.extend(pair_masters)
                coeffs.extend(pair_coeffs)
                offsets.append(len(masters))
    owner_ranks = np.full(len(masters), comm.rank)
    ghost_dofs = facet_dofs[facet_dofs[:, 1] <=
                            normal.function_space.dofmap.index_map.size_local]
    print("num ghosts at facets", comm.rank, ghost_dofs.shape)
    print(comm.rank, facet_dofs.shape, ghost_dofs.shape)
    # # print(ghost_dofs)
    # shared_indices = cpp.mpc.compute_shared_indices(W._cpp_object)
    # # print(shared_indices)
    # for slave in slaves:
    #     block = slave//W.dofmap.index_map.block_size
    #     if block in shared_indices.keys():
    #         print(comm.rank, slave, block, shared_indices[block])
    # print()
    # Create contact constraint
    cc = cpp.mpc.ContactConstraint(
        W._cpp_object, slaves, len(slaves))
    cc.add_masters(masters, coeffs, owner_ranks, offsets)
    return cc

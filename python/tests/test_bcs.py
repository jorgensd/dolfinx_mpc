from mpi4py import MPI

import dolfinx
import dolfinx_mpc
import numpy as np

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim -1, exterior_facets)

def periodic_map(x):
    out = x.copy()
    out[0] -= 1
    return out

def periodic_locator(x):
    return np.isclose(x[0], 1) 

def bc_locator(x):
    return np.isclose(x[0], 0)&  ((x[1] > 0.4) | (x[1] < 0.15))

bc_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, bc_locator)
bc_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim-1, bc_facets)
u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(lambda x: x[0])
bc = dolfinx.fem.dirichletbc(u_bc,  bc_dofs)
bcs = [bc]

mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, periodic_locator, periodic_map, bcs=[])

dof_indices = bc.dof_indices()[0]
is_slave = dolfinx.la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.int32)
is_slave.array[mpc._slaves] = 1

# Need to determine if master dofs align with Dirichlet dofs.
# We do this by passing 

own_order = np.argsort(mpc._owners)
dest_ranks, dest_size = np.unique(mpc._owners[own_order], return_counts=True)
dofs_out = mpc._masters[own_order]
reverse_map = np.zeros(len(own_order), dtype=np.int32)
reverse_map[own_order] = np.arange(len(own_order), dtype=np.int32)
slave_to_master = domain.comm.Create_dist_graph(
    [domain.comm.rank], [len(dest_ranks)], dest_ranks.tolist(), reorder=False
)
source, dest, _ = slave_to_master.Get_dist_neighbors()
recv_size = np.zeros(len(source), dtype=np.int32)
slave_to_master.Neighbor_alltoall(dest_size.astype(np.int32), recv_size)


recv_masters = np.full(sum(recv_size), -1, dtype=np.int64)

s_msg = [dofs_out, dest_size, MPI.INT64_T]
r_msg = [recv_masters, recv_size, MPI.INT64_T]
slave_to_master.Neighbor_alltoallv(s_msg, r_msg)
assert (recv_masters >= 0).all()
slave_to_master.Free()

# Convert to local index on process
master_blocks = (recv_masters // V.dofmap.index_map_bs).astype(np.int64)

master_rems = recv_masters % V.dofmap.index_map_bs
local_dofs = V.dofmap.index_map.global_to_local(master_blocks)*V.dofmap.index_map_bs + master_rems

is_master = np.zeros(V.dofmap.index_map.size_local*V.dofmap.index_map_bs, dtype=np.int32)
is_master[local_dofs] = True

bc_indicator = dolfinx.la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.int8)
for bc in bcs:
    bc_indicator.array[bc.dof_indices()[0]] = 1
bc_indicator_arr = bc_indicator.array[:len(is_master)]

should_be_replaced = is_master & bc_indicator_arr

replace_signal = should_be_replaced[local_dofs].astype(np.int8)

master_to_slave = domain.comm.Create_dist_graph_adjacent(
    dest, source, reorder=False
)
move_to_rhs = np.zeros(sum(dest_size), dtype=np.int8)
s_msg = [replace_signal, recv_size, MPI.INT8_T]
r_msg = [move_to_rhs, dest_size, MPI.INT8_T]
master_to_slave.Neighbor_alltoallv(s_msg, r_msg)
master_to_slave.Free()

move_idx = np.flatnonzero(move_to_rhs[reverse_map])
# Zero out all coefficients that are for constants, as they will be handled elsewhere
# However, we need these dofs in the extended map to move dirichlet values
mpc._coeffs[move_idx] = 0
# old_masters = mpc._masters[move_idx].copy()
# coeffs_deleted_dofs = mpc._coeffs[move_idx].copy()

# positions = np.searchsorted(mpc._offsets, move_idx)
# remove_bins, num_remove_per_offset = np.unique(positions, return_counts=True)
# data_per_bin = np.zeros(len(mpc._offsets)-1, dtype=np.int32)
# data_per_bin[remove_bins] = num_remove_per_offset
# mpc._masters = np.delete(mpc._masters, move_idx)
# mpc._coeffs = np.delete(mpc._coeffs, move_idx)
# mpc._owners = np.delete(mpc._owners, move_idx)
# mpc._offsets = mpc._offsets[1:] - np.cumsum(data_per_bin)
mpc.finalize()
#new_bc = dolfinx.fem.dirichletbc


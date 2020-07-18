
import dolfinx_mpc.cpp
import numpy as np

# from IPython import embed
from mpi4py import MPI
from helpers_contact import (compute_masters_from_global,
                             compute_local_master_coeffs,
                             locate_dofs_colliding_with_cells,
                             recv_bb_collisions,
                             recv_masters_from_other_proc,
                             send_bb_collisions, select_masters)
import dolfinx
import dolfinx.common as common
import dolfinx.cpp as cpp
import dolfinx.geometry as geometry
import dolfinx.fem as fem
import dolfinx.io as io
# import dolfinx.log as log
from stacked_cube import mesh_3D_rot


comm = MPI.COMM_WORLD
theta = np.pi/5
if comm.size == 1:
    mesh_3D_rot(theta)
# Read in mesh
with io.XDMFFile(comm, "mesh_hex_cube{0:.2f}.xdmf".format(theta),
                 "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    mesh.name = "mesh_hex_{0:.2f}".format(theta)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim, tdim)
    mesh.topology.create_connectivity(fdim, tdim)

    ct = xdmf.read_meshtags(mesh, "Grid")

with io.XDMFFile(comm, "facet_hex_cube{0:.2f}.xdmf".format(theta),
                 "r") as xdmf:
    mt = xdmf.read_meshtags(mesh, "Grid")

top_cube_marker = 2

# log.set_log_level(log.LogLevel.INFO)

# Create functionspace
V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                       dtype=np.int64)
bs = V.dofmap.index_map.block_size
local_size = V.dofmap.index_map.size_local*bs
x = V.tabulate_dof_coordinates()

# Bottom mesh top facets
slave_facets = mt.indices[mt.values == 4]

# Compute approximate normal vector for slave facets
nh = dolfinx_mpc.facet_normal_approximation(V, mt, 4)
n_vec = nh.vector.getArray()
# Find the masters from the same block as the slave (they will be local)
loc_masters_at_slave = []
for i in range(tdim-1):
    Vi = V.sub(i).collapse()
    loc_master_i_dofs = fem.locate_dofs_topological(
        (V.sub(i), Vi), fdim, slave_facets)[:, 0]
    # Remove ghosts
    loc_master = loc_master_i_dofs[loc_master_i_dofs < local_size]
    loc_masters_at_slave.append(loc_master)


# Local slaves and their corresponding coordinates
Vi = V.sub(tdim-1).collapse()
slaves_loc = fem.locate_dofs_topological(
    (V.sub(tdim-1), Vi), fdim, slave_facets)[:, 0]
# Remove ghosts
loc_slaves = slaves_loc[slaves_loc < local_size]
loc_slaves = loc_slaves.reshape((len(loc_slaves), 1))
loc_coords = x[loc_slaves]
# Compute local master coefficients for the local slaves
masters_local, coeffs_local = compute_local_master_coeffs(V, loc_slaves,
                                                          loc_masters_at_slave,
                                                          n_vec)


# Top mesh bottom facets
master_facets = mt.indices[mt.values == 9]

# Find the master cells from the facet tag
mesh.topology.create_connectivity(fdim, tdim)
facet_to_cell = mesh.topology.connectivity(fdim, tdim)
local_master_cells = []
for facet in master_facets:
    local_master_cells.extend(facet_to_cell.links(facet))
unique_master_cells = np.unique(local_master_cells)
# Loop over local slaves to see if we can find a corresponding master cell
# that its coordinate is close to
tree = geometry.BoundingBoxTree(mesh, tdim)

# Find local masters for slaves that are local on this process
masters_for_local_slave = locate_dofs_colliding_with_cells(
    mesh, loc_slaves, loc_coords, unique_master_cells, tree)

# Temporary storage of slave and cell info
# cc = dolfinx_mpc.cpp.mpc.ContactConstraint(V._cpp_object, slaves_loc)


# Find which cells the local slaves are in (locally)
cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local
cells_local_slave = []
offsets = [0]
cell_to_loc_slaves = {}


possible_recv = send_bb_collisions(V, loc_slaves, loc_coords, n_vec)

slaves_glob, owners, coords_glob, normals_glob = recv_bb_collisions()

# If received slaves, find possible masters
if len(slaves_glob) > 0:
    masters_for_slaves = compute_masters_from_global(V, slaves_glob,
                                                     coords_glob, normals_glob,
                                                     owners,
                                                     unique_master_cells,
                                                     tree)
    # Communicators that might later have to receive from master
    narrow_slave_recv = []
    # Send master info to slave processor
    for proc in set(owners):
        if proc in masters_for_slaves.keys():
            MPI.COMM_WORLD.send(masters_for_slaves[proc],
                                dest=proc, tag=1)
            narrow_slave_recv.append(proc)
        else:
            MPI.COMM_WORLD.send({}, dest=proc, tag=1)


loc_slaves_flat = loc_slaves.T[0]
glob_slaves = loc_to_glob[loc_slaves_flat]
global_masters, narrow_receive = recv_masters_from_other_proc(
    loc_slaves_flat, glob_slaves, possible_recv)


masters_from_global = select_masters(global_masters)

# Write cell partitioning to file
indices = np.arange(num_cells_local)
values = MPI.COMM_WORLD.rank*np.ones(num_cells_local, np.int32)
ct = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, values)
ct.name = "cells"
with io.XDMFFile(MPI.COMM_WORLD, "cf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)

with io.XDMFFile(MPI.COMM_WORLD, "n.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(nh)


print(MPI.COMM_WORLD.rank, "Num Local slaves:", len(loc_slaves))
# common.list_timings(MPI.COMM_WORLD,
#                     [dolfinx.common.TimingType.wall])

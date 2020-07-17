import dolfinx.common as common
from IPython import embed
import numpy as np
# from IPython import embed
from mpi4py import MPI

import dolfinx
import dolfinx.geometry as geometry
import dolfinx.io as io
import dolfinx.cpp as cpp
import dolfinx_mpc.cpp
# import dolfinx.log as log
from chris_mesh import mesh_3D_rot


comm = MPI.COMM_WORLD
theta = 0
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
# Bottom mesh top facets
slave_facets = mt.indices[mt.values == 4]
# Top mesh bottom facets
master_facets = mt.indices[mt.values == 9]
# log.set_log_level(log.LogLevel.INFO)

# Create functionspace
V = dolfinx.FunctionSpace(mesh, ("CG", 1))
loc_to_glob = V.dofmap.index_map.global_indices(False)
x = V.tabulate_dof_coordinates()

# Local slaves and their corresponding coordinates
slaves_loc = dolfinx.fem.locate_dofs_topological(V, fdim, slave_facets)
points = x[slaves_loc]

# Temporary storage of slave and cell info
cc = dolfinx_mpc.cpp.mpc.ContactConstraint(V._cpp_object, slaves_loc)


# Find which cells the local slaves are in (locally)
cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local
cells_local_slave = []
offsets = [0]
cell_to_loc_slaves = {}

# common.list_timings(MPI.COMM_WORLD,
#                     [dolfinx.common.TimingType.wall])
facettree = geometry.BoundingBoxTree(mesh, dim=fdim)


# Find all processors where a slave can be located
# according to the bounding box tree.
slaves_to_send = {}
slave_coords_to_send = {}
for dof, point in zip(slaves_loc, points):
    # Get processor number for a point
    procs = dolfinx_mpc.cpp.mpc.compute_process_collisions(
        facettree._cpp_object, point.T)
    for proc in procs:
        if proc != MPI.COMM_WORLD.rank:
            if proc not in slaves_to_send.keys():
                slaves_to_send[proc] = [loc_to_glob[dof[0]]]
            else:
                slaves_to_send[proc].append(loc_to_glob[dof[0]])
            if proc not in slave_coords_to_send.keys():
                slave_coords_to_send[proc] = [point[0]]
            else:
                slave_coords_to_send[proc].append(point[0])

# Send information about possible collisions with slave to the other processors
for proc in range(MPI.COMM_WORLD.size):
    if proc in slaves_to_send.keys():
        MPI.COMM_WORLD.send({"slaves": slaves_to_send[proc],
                             "coords": slave_coords_to_send[proc]},
                            dest=proc, tag=0)
    elif proc != MPI.COMM_WORLD.rank:
        MPI.COMM_WORLD.send({}, dest=proc, tag=0)

# Receive possible collisions (potential master dofs) from other processors
slaves_glob = []
coords_glob = None
slave_owner = []
owners = np.ones(0, dtype=np.int32)
for i in range(MPI.COMM_WORLD.size):
    if i != MPI.COMM_WORLD.rank:
        data = MPI.COMM_WORLD.recv(source=i, tag=0)
        if len(data) > 0:
            slaves_glob.extend(data["slaves"])
            owners = np.concatenate(
                (owners, np.full(len(data["slaves"]), i,
                                 dtype=np.int32)), axis=0)
            if coords_glob is None:
                coords_glob = data["coords"]
            else:
                coords_glob = np.vstack([coords_glob, data["coords"]])

# Write cell partitioning to file
indices = np.arange(num_cells_local)
values = MPI.COMM_WORLD.rank*np.ones(num_cells_local, np.int32)
ct = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, values)
ct.name = "cells"
with io.XDMFFile(MPI.COMM_WORLD, "cf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)

tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)


print(MPI.COMM_WORLD.rank, "Num Local slaves:", len(slaves_loc),
      "Num Global slaves:", len(slaves_glob),
      " Num other procs:", len(set(owners)))

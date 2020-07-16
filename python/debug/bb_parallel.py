import numpy as np
from IPython import embed
from mpi4py import MPI

import dolfinx
import dolfinx.geometry as geometry
import dolfinx.io as io
import dolfinx.log as log
from chris_mesh import mesh_3D_rot

import struct

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
f0 = mt.indices[mt.values == 4]
# Top mesh bottom facets
f1 = mt.indices[mt.values == 9]
# log.set_log_level(log.LogLevel.INFO)
# Create functionspace and locate dofs
V = dolfinx.FunctionSpace(mesh, ("CG", 1))
loc_to_glob = V.dofmap.index_map.global_indices(False)
x = V.tabulate_dof_coordinates()
loc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, f0)
points = x[loc_dofs]
facettree = geometry.BoundingBoxTree(mesh, dim=fdim)
embed()
slaves_loc = []
slaves_to_send = {}
slave_coords_to_send = {}
for dof, point in zip(loc_dofs, points):
    # Get processor number for a point
    procs = dolfinx.cpp.geometry.compute_process_collisions(
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
    slaves_loc.append(loc_to_glob[dof[0]])


for proc in range(MPI.COMM_WORLD.size):
    if proc in slaves_to_send.keys():
        MPI.COMM_WORLD.send({"slaves": slaves_to_send[proc],
                             "coords": slave_coords_to_send[proc]},
                            dest=proc, tag=0)
    elif proc != MPI.COMM_WORLD.rank:
        MPI.COMM_WORLD.send({}, dest=proc, tag=0)

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
                (owners, np.ones(len(data["slaves"]), dtype=np.int32)), axis=0)
            if coords_glob is None:
                coords_glob = data["coords"]
            else:
                coords_glob = np.vstack([coords_glob, data["coords"]])

cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local
indices = np.arange(num_cells_local)
values = MPI.COMM_WORLD.rank*np.ones(num_cells_local, np.int32)
ct = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, values)
ct.name = "cells"
with io.XDMFFile(MPI.COMM_WORLD, "cf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)

print(MPI.COMM_WORLD.rank, "Local slaves:", len(slaves_loc))
print(MPI.COMM_WORLD.rank, "Global slaves:",
      len(slaves_glob))
if len(slaves_glob) > 0:
    print(MPI.COMM_WORLD.rank, slaves_glob[0], coords_glob[0], owners[0])

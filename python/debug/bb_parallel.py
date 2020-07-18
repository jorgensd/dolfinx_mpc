import dolfinx_mpc.cpp
import numpy as np
from IPython import embed
# from IPython import embed
from mpi4py import MPI

import dolfinx
import dolfinx.common as common
import dolfinx.cpp as cpp
import dolfinx.geometry as geometry
import dolfinx.fem as fem
import dolfinx.io as io
# import dolfinx.log as log
from chris_mesh import mesh_3D_rot


def locate_dofs_colliding_with_cells(dofs, coords, cells, tree):
    """
    Function returning dictionary of which dofs that are colliding with
    a set of cells
    """
    cell_map = mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    l2g_cells = np.array(cell_map.global_indices(False), dtype=np.int64)
    dof_to_cell = {}
    for i, (dof, coord) in enumerate(zip(dofs, coords)):
        possible_collisions = np.array(
            geometry.compute_collisions_point(tree, coord.T))
        if len(possible_collisions) > 0:
            # Check if possible cells are in local range and if
            # they are in the supplied list of local cells
            contains_cells = np.isin(possible_collisions, cells)
            is_owned = np.logical_and(cmin <= l2g_cells[possible_collisions],
                                      l2g_cells[possible_collisions] < cmax)
            candidates = possible_collisions[np.logical_and(
                is_owned, contains_cells)]
            # Check if actual cell in mesh is within a distance of 1e-14
            cell = dolfinx.cpp.geometry.select_colliding_cells(
                mesh, list(candidates), coord.T, 1)
            if len(cell) > 0:
                dof_to_cell[i] = cell[0]
    return dof_to_cell


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

# log.set_log_level(log.LogLevel.INFO)

# Create functionspace
V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
loc_to_glob = V.dofmap.index_map.global_indices(False)
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
Vi = V.sub(tdim-2).collapse()
slaves_loc = fem.locate_dofs_topological(
    (V.sub(i), Vi), fdim, slave_facets)[:, 0]
# Remove ghosts
loc_slaves = slaves_loc[slaves_loc < local_size]
loc_slaves = loc_slaves.reshape((len(loc_slaves), 1))
loc_coords = x[loc_slaves]

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
    loc_slaves, loc_coords, unique_master_cells, tree)

# Temporary storage of slave and cell info
# cc = dolfinx_mpc.cpp.mpc.ContactConstraint(V._cpp_object, slaves_loc)


# Find which cells the local slaves are in (locally)
cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local
cells_local_slave = []
offsets = [0]
cell_to_loc_slaves = {}

# Find all processors where a slave can be located
# according to the bounding box tree.
facettree = geometry.BoundingBoxTree(mesh, dim=fdim)
slaves_to_send = {}
slave_coords_to_send = {}
normals_to_send = {}

for dof, point in zip(loc_slaves, loc_coords):
    # Zip coordinates for slave a 3D array
    b_off = dof % bs
    block_dofs = np.array([dof-b_off+i for i in range(bs)])
    n = list(n_vec[block_dofs].T[0])

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
            if proc not in normals_to_send.keys():
                normals_to_send[proc] = [n]
            else:
                normals_to_send[proc].append(n)

# Send information about possible collisions with slave to the other processors
possible_recv = []
for proc in range(MPI.COMM_WORLD.size):
    if proc in slaves_to_send.keys():
        MPI.COMM_WORLD.send({"slaves": slaves_to_send[proc],
                             "coords": slave_coords_to_send[proc],
                             "normals": normals_to_send[proc]},
                            dest=proc, tag=0)
        # Cache proc for later receive
        possible_recv.append(proc)
    elif proc != MPI.COMM_WORLD.rank:
        MPI.COMM_WORLD.send({}, dest=proc, tag=0)

# Receive possible collisions (potential master dofs) from other processors
slaves_glob = []
coords_glob = None
normals_glob = None
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
            if normals_glob is None:
                normals_glob = data["normals"]
            else:
                normals_glob = np.vstack([normals_glob, data["normals"]])

# Check if proc received anything
masters_for_glob_slave = {}
for proc in range(MPI.COMM_WORLD.size):
    masters_for_glob_slave[proc] = {}

# If received slaves, find possible masters
if len(slaves_glob) > 0:
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()

    # Find local cells that can possibly contain masters
    slaves_array = np.array(slaves_glob).reshape((len(slaves_glob), 1))
    cells_with_masters_for_global_slave = locate_dofs_colliding_with_cells(
        slaves_array, coords_glob, unique_master_cells, tree)

    masters_for_slaves = {}
    # Loop through slave dofs and compute local master contributions
    for index in cells_with_masters_for_global_slave.keys():
        cell = cells_with_masters_for_global_slave[index]
        coord = coords_glob[index]
        basis_values = dolfinx_mpc.cpp.mpc.\
            get_basis_functions(V._cpp_object,
                                coord, cell)
        cell_dofs = V.dofmap.cell_dofs(cell)
        masters_ = []
        owners_ = []
        coeffs_ = []
        for k in range(tdim):
            for local_idx, dof in enumerate(cell_dofs):
                coeff = basis_values[local_idx, k]
                coeff *= (normals_glob[index][k] /
                          normals_glob[index][tdim-1])
                if not np.isclose(coeff, 0):
                    if dof < local_size:
                        owners_.append(comm.rank)
                    else:
                        # If master i ghost, find its owner
                        owners_.append(
                            ghost_owners[dof//bs -
                                         local_size//bs])

                    masters_.append(dof)
                    coeffs_.append(coeff)
        if owners[index] in masters_for_slaves.keys():
            masters_for_slaves[owners[index]
                               ][slaves_glob[index]] = {"masters": masters_,
                                                        "coeffs": coeffs_,
                                                        "owners": owners_}

        else:
            masters_for_slaves[owners[index]] = {slaves_glob[index]:
                                                 {"masters": masters_,
                                                  "coeffs": coeffs_,
                                                  "owners": owners_}}

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

narrow_master_recv = []
for proc in possible_recv:
    data = MPI.COMM_WORLD.recv(source=proc, tag=1)
    if len(data.keys()) == 0:
        narrow_master_recv.append(proc)
    print("{0:d} received from {1:d}, len data {2:d}".format(
        MPI.COMM_WORLD.rank, proc, len(data.keys())))

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


tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)


print(MPI.COMM_WORLD.rank, "Num Local slaves:", len(slaves_loc),
      "Num Global slaves:", len(slaves_glob),
      " Num other procs:", len(set(owners)))
# common.list_timings(MPI.COMM_WORLD,
#                     [dolfinx.common.TimingType.wall])

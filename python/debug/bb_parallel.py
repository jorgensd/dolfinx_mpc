
import dolfinx_mpc.cpp
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from IPython import embed
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.common as common
import dolfinx.fem as fem
import dolfinx.geometry as geometry
import dolfinx.io as io
import dolfinx.log as log
from helpers_contact import (compute_masters_from_global,
                             compute_masters_local,
                             compute_masters_local_block, flatten_ghosts,
                             gather_masters_for_local_slaves,
                             locate_dofs_colliding_with_cells,
                             recv_bb_collisions, recv_masters, select_masters,
                             send_bb_collisions, send_masters)
from stacked_cube import mesh_3D_rot

# Generate mesh in serial and load it
comm = MPI.COMM_WORLD
theta = np.pi/5
if comm.size == 1:
    mesh_3D_rot(theta)
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

# log.set_log_level(log.LogLevel.INFO)

# Create functionspace
V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))


# Helper for orienting traction
r_matrix = pygmsh.helpers.rotation_matrix(
    [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)

g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
g = dolfinx.Constant(mesh, g_vec)

# Define boundary conditions
# Bottom boundary is fixed in all directions
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)

bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

# Top boundary has a given deformation normal to the interface
g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
g = dolfinx.Constant(mesh, g_vec)


def top_v(x):
    values = np.empty((3, x.shape[1]))
    values[0] = g_vec[0]
    values[1] = g_vec[1]
    values[2] = g_vec[2]
    return values


u_top = dolfinx.function.Function(V)
u_top.interpolate(top_v)

top_facets = mt.indices[np.flatnonzero(mt.values == 3)]
top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
bc_top = fem.DirichletBC(u_top, top_dofs)

bcs = [bc_bottom, bc_top]

# Elasticity parameters
E = 1.0e3
nu = 0
mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))


def sigma(v):
    # Stress computation
    return (2.0 * mu * ufl.sym(ufl.grad(v)) +
            lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))


# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
# NOTE: Traction deactivated until we have a way of fixing nullspace
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt,
                 subdomain_id=3)
lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v)*ufl.dx\
    + ufl.inner(g, v)*ds

# Extract structures needed for contact detection
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


# Locate slaves topologically, remove all slaves not owned by the processor
Vi = V.sub(tdim-1).collapse()
slaves_loc = fem.locate_dofs_topological(
    (V.sub(tdim-1), Vi), fdim, slave_facets)[:, 0]
loc_slaves_flat = slaves_loc[slaves_loc < local_size]
ghost_slaves = slaves_loc[local_size <= slaves_loc]
ghost_coords = x[ghost_slaves]
loc_slaves = loc_slaves_flat.reshape((len(loc_slaves_flat), 1))
# get the global slave indices and the dof coordinates
glob_slaves = loc_to_glob[loc_slaves_flat]
loc_coords = x[loc_slaves]

# Compute local master coefficients for the local slaves
masters_block = compute_masters_local_block(V, loc_slaves_flat,
                                            loc_masters_at_slave,
                                            n_vec)


# Find the master cells from facet tag of master interfaces
master_facets = mt.indices[mt.values == 9]
mesh.topology.create_connectivity(fdim, tdim)
facet_to_cell = mesh.topology.connectivity(fdim, tdim)
local_master_cells = []
for facet in master_facets:
    local_master_cells.extend(facet_to_cell.links(facet))
unique_master_cells = np.unique(local_master_cells)

# Find local masters for slaves that are local on this process
tree = geometry.BoundingBoxTree(mesh, tdim)
local_slave_to_cell = locate_dofs_colliding_with_cells(
    mesh, loc_slaves, loc_coords, unique_master_cells, tree)

# Find coefficients for masters located on this processors
local_masters_interface = compute_masters_local(V, loc_slaves_flat, loc_coords,
                                                n_vec, local_slave_to_cell)
# FIXME: HERE IN NEW REWRITE of BB parallel

# Compute collisions of slaves with bounding boxes on other processors
# and send these processors the slaves, coordinates and normals
possible_recv = send_bb_collisions(V, loc_slaves, loc_coords, n_vec, tag=0)

# Recieve info from other processors about possible collisions
slaves_glob, owners, coords_glob, normals_glob = recv_bb_collisions(tag=0)

# If received slaves, find possible masters
if len(slaves_glob) > 0:
    masters_for_slaves = compute_masters_from_global(V, slaves_glob,
                                                     coords_glob, normals_glob,
                                                     owners,
                                                     unique_master_cells,
                                                     tree)
    narrow_slave_send = send_masters(masters_for_slaves, owners, tag=1)

# Receive masters from other procs
global_masters, narrow_slave_recv = recv_masters(
    loc_slaves_flat, glob_slaves, possible_recv, tag=1)

# Select masters if getting them from multiple procs
masters_from_global = select_masters(global_masters)

# Gather all masters for local slave in a 1D lists
m_loc, c_loc, o_loc, offsets = gather_masters_for_local_slaves(
    loc_slaves_flat, loc_to_glob,
    masters_block, local_masters_interface, masters_from_global)

# Initialize Contact constraint (computes which cells contains slaves)
num_loc_slaves = len(loc_slaves_flat)

# Get shared indices for every ghost on the processor
shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
bs = V.dofmap.index_map.block_size

# Write master data for ghosted slaves per processor
share_masters = {}
for (i, dof) in enumerate(loc_slaves_flat):
    # Map slave to block to lookup if shared_index
    block = dof // bs
    if block in shared_indices.keys():
        for proc in shared_indices[block]:
            if proc in share_masters.keys():
                share_masters[proc][loc_to_glob[dof]] = {
                    "masters": m_loc[offsets[i]:offsets[i+1]],
                    "coeffs": c_loc[offsets[i]:offsets[i+1]],
                    "owners": o_loc[offsets[i]:offsets[i+1]]}
            else:
                share_masters[proc] = {
                    loc_to_glob[dof]:
                    {"masters": m_loc[offsets[i]:offsets[i+1]],
                     "coeffs": c_loc[offsets[i]:offsets[i+1]],
                     "owners": o_loc[offsets[i]:offsets[i+1]]}}
# Send masters to ghost processors
for proc in share_masters.keys():
    MPI.COMM_WORLD.send(share_masters[proc], dest=proc, tag=5)

# Receive masters from ghost processors
ghost_ranks = V.dofmap.index_map.ghost_owner_rank()
l_size = V.dofmap.index_map.size_local

# Loop through ghosted slaves to find the unique set of procs
ghost_recv = []
for slave in ghost_slaves:
    block = slave//bs - l_size
    owner = ghost_ranks[block]
    ghost_recv.append(owner)
ghost_recv = set(ghost_recv)

# Receive masters for ghosted slaves
ghost_masters = {}
for owner in ghost_recv:
    data = MPI.COMM_WORLD.recv(source=owner, tag=5)
    ghost_masters.update(data)


# Flatten received ghosts for masters as array
ghost_masters, ghost_coeffs, ghost_owners, offsets_ghosts = flatten_ghosts(
    loc_to_glob[ghost_slaves], ghost_masters)
offsets_ghosts += len(m_loc)

# Gather all masters and add them to contact constraint
all_masters = np.hstack([m_loc, ghost_masters])
all_coeffs = np.hstack([c_loc, ghost_coeffs])
all_owners = np.hstack([o_loc, ghost_owners])
all_offsets = np.hstack([offsets[:-1], offsets_ghosts])

cc = dolfinx_mpc.cpp.mpc.ContactConstraint(
    V._cpp_object, slaves_loc, num_loc_slaves)
cc.add_masters(all_masters, all_coeffs, all_owners, all_offsets)


# Assemble matrix
# log.set_log_level(log.LogLevel.INFO)

A = dolfinx_mpc.assemble_matrix_local(a, cc, bcs=bcs)

b = dolfinx_mpc.assemble_vector_local(lhs, cc)
# Apply boundary conditions
fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b, bcs)

# Solve Linear problem
opts = PETSc.Options()
opts["ksp_rtol"] = 1.0e-8
opts["pc_type"] = "gamg"
opts["pc_gamg_type"] = "agg"
opts["pc_gamg_coarse_eq_limit"] = 1000
opts["pc_gamg_sym_graph"] = True
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["matptap_via"] = "scalable"
opts["pc_gamg_square_graph"] = 2
opts["pc_gamg_threshold"] = 0.02
# opts["help"] = None # List all available options
# opts["ksp_view"] = None # List progress of solver

# Create functionspace and build near nullspace
Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                              cc.dofmap())
Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vmpc)
A.setNearNullSpace(null_space)

solver = PETSc.KSP().create(comm)
solver.setFromOptions()
solver.setOperators(A)
uh = b.copy()
uh.set(0)
with dolfinx.common.Timer("MPC: Solve"):
    solver.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)

# Back substitute to slave dofs
with dolfinx.common.Timer("MPC: Backsubstitute"):
    dolfinx_mpc.backsubstitution_local(cc, uh)

it = solver.getIterationNumber()
if comm.rank == 0:
    print("Number of iterations: {0:d}".format(it))

# Write solution to file
u_h = dolfinx.Function(Vmpc)
u_h.vector.setArray(uh.array)
u_h.name = "u_{0:s}_{1:.2f}".format("hex", theta)
outfile = io.XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(u_h, 0.0,
                       "Xdmf/Domain/"
                       + "Grid[@Name='{0:s}'][1]"
                       .format(mesh.name))

# Write cell partitioning to file
# tdim = mesh.topology.dim
cell_map = mesh.topology.index_map(tdim)
num_cells_local = cell_map.size_local
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


common.list_timings(MPI.COMM_WORLD,
                    [dolfinx.common.TimingType.wall])

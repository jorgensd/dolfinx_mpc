
import dolfinx_mpc.cpp
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.common as common
import dolfinx.fem as fem
import dolfinx.io as io
from stacked_cube import mesh_3D_rot

# import dolfinx.log as log


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

cc = dolfinx_mpc.create_contact_condition(V, mt, 4, 9)

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
outfile = io.XDMFFile(MPI.COMM_WORLD, "output_new.xdmf", "w")
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


common.list_timings(MPI.COMM_WORLD,
                    [dolfinx.common.TimingType.wall])

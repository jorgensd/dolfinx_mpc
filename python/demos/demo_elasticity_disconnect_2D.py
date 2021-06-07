# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Create constraint between two bodies that are not in contact

from petsc4py import PETSc
import dolfinx.fem as fem
import ufl
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import gmsh
import numpy as np
from mpi4py import MPI

# Mesh parameters for creating a mesh consisting of two disjoint rectangles
right_tag = 1
left_tag = 2


gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.clear()

    left_rectangle = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    right_rectangle = gmsh.model.occ.addRectangle(1.1, 0, 0, 1, 1)

    gmsh.model.occ.synchronize()

    # Add physical tags for volumes
    gmsh.model.addPhysicalGroup(2, [right_rectangle], tag=right_tag)
    gmsh.model.setPhysicalName(2, right_tag, "Right square")
    gmsh.model.addPhysicalGroup(2, [left_rectangle], tag=left_tag)
    gmsh.model.setPhysicalName(2, left_tag, "Left square")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")
mesh, ct = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, cell_data=True, gdim=2)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()


V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1

DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
left_cells = ct.indices[ct.values == left_tag]
right_cells = ct.indices[ct.values == right_tag]
left_dofs = fem.locate_dofs_topological(DG0, tdim, left_cells)
right_dofs = fem.locate_dofs_topological(DG0, tdim, right_cells)

# Elasticity parameters
E_right = 1e2
E_left = 1e2
nu_right = 0.3
nu_left = 0.1
mu = dolfinx.Function(DG0)
lmbda = dolfinx.Function(DG0)
with mu.vector.localForm() as local:
    local.array[left_dofs] = E_left / (2 * (1 + nu_left))
    local.array[right_dofs] = E_right / (2 * (1 + nu_right))
with lmbda.vector.localForm() as local:
    local.array[left_dofs] = E_left * nu_left / ((1 + nu_left) * (1 - 2 * nu_left))
    local.array[right_dofs] = E_right * nu_right / ((1 + nu_right) * (1 - 2 * nu_right))


# Stress computation
def sigma(v):
    return (2.0 * mu * ufl.sym(ufl.grad(v))
            + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))


# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
a = ufl.inner(sigma(u), ufl.grad(v)) * dx
x = ufl.SpatialCoordinate(mesh)
rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * dx


def push(x):
    values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 0.1
    return values


u_push = dolfinx.Function(V)
u_push.interpolate(push)
u_push.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bc_push = dolfinx.DirichletBC(u_push, dofs)
u_fix = dolfinx.Function(V)
u_fix.vector.set(0)
bc_fix = dolfinx.DirichletBC(u_fix, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 2.1)))
bcs = [bc_push, bc_fix]


def gather_dof_coordinates(V, dofs):
    """
    Distributes the dof coordinates of this subset of dofs to all processors
    """
    x = V.tabulate_dof_coordinates()
    local_dofs = dofs[dofs < V.dofmap.index_map.size_local * V.dofmap.index_map_bs]
    coords = x[local_dofs]
    num_nodes = len(coords)
    glob_num_nodes = MPI.COMM_WORLD.allreduce(num_nodes, op=MPI.SUM)
    recvbuf = None
    if MPI.COMM_WORLD.rank == 0:
        recvbuf = np.zeros(3 * glob_num_nodes, dtype=np.float64)
    sendbuf = coords.reshape(-1)
    sendcounts = np.array(MPI.COMM_WORLD.gather(len(sendbuf), 0))
    MPI.COMM_WORLD.Gatherv(sendbuf, (recvbuf, sendcounts), root=0)
    glob_coords = MPI.COMM_WORLD.bcast(recvbuf, root=0).reshape((-1, 3))
    return glob_coords


# Create pairs of dofs at each boundary
V0 = V.sub(0).collapse()

facets_r = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1))
dofs_r = fem.locate_dofs_topological(V0, fdim, facets_r)
facets_l = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.1))
dofs_l = fem.locate_dofs_topological(V0, fdim, facets_l)

# Given the local coordinates of the dofs, distribute them on all processors
nodes = []
nodes.append(gather_dof_coordinates(V0, dofs_r))
nodes.append(gather_dof_coordinates(V0, dofs_l))
pairs = []
for x in nodes[0]:
    for y in nodes[1]:
        if np.isclose(x[1], y[1]):
            pairs.append([x, y])
            break

mpc = dolfinx_mpc.MultiPointConstraint(V)
for i, pair in enumerate(pairs):
    sl, ms, co, ow, off = dolfinx_mpc.utils.create_point_to_point_constraint(
        V, pair[0], pair[1], vector=[1, 0])
    mpc.add_constraint(V, sl, ms, co, ow, off)
mpc.finalize()

null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())
num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
with dolfinx.common.Timer("~Contact: Assemble matrix ({0:d})".format(num_dofs)):
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
with dolfinx.common.Timer("~Contact: Assemble vector ({0:d})".format(num_dofs)):
    b = dolfinx_mpc.assemble_vector(rhs, mpc)

fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
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

A.setNearNullSpace(null_space)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setFromOptions()
uh = b.copy()
uh.set(0)
with dolfinx.common.Timer("~Contact: Solve old"):
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

it = solver.getIterationNumber()

unorm = uh.norm()
if MPI.COMM_WORLD.rank == 0:
    print("Number of iterations: {0:d}".format(it))

# Write solution to file
u_h = dolfinx.Function(mpc.function_space())
u_h.vector.setArray(uh.array)
u_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_h.name = "u"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity_disconnect_2D.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    # xdmf.write_meshtags(ct)
    # xdmf.write_meshtags(ft)
    xdmf.write_function(u_h)

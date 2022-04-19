# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Create constraint between two bodies that are not in contact


import gmsh
import numpy as np
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_geometrical,
                         locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import (create_point_to_point_constraint,
                               gmsh_model_to_mesh, rigid_motions_nullspace)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, Measure, SpatialCoordinate, TestFunction,
                 TrialFunction, grad, inner, sym, tr)

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
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    # Generate mesh
    gmsh.model.mesh.generate(2)

    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")

mesh, ct = gmsh_model_to_mesh(gmsh.model, cell_data=True, gdim=2)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

with XDMFFile(mesh.comm, "test.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
V = VectorFunctionSpace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1

# Locate cells with different elasticity parameters
DG0 = FunctionSpace(mesh, ("DG", 0))
left_cells = ct.indices[ct.values == left_tag]
right_cells = ct.indices[ct.values == right_tag]
left_dofs = locate_dofs_topological(DG0, tdim, left_cells)
right_dofs = locate_dofs_topological(DG0, tdim, right_cells)

# Elasticity parameters
E_right = 1e2
E_left = 1e2
nu_right = 0.3
nu_left = 0.1
mu = Function(DG0)
lmbda = Function(DG0)
with mu.vector.localForm() as local:
    local.array[left_dofs] = E_left / (2 * (1 + nu_left))
    local.array[right_dofs] = E_right / (2 * (1 + nu_right))
with lmbda.vector.localForm() as local:
    local.array[left_dofs] = E_left * nu_left / ((1 + nu_left) * (1 - 2 * nu_left))
    local.array[right_dofs] = E_right * nu_right / ((1 + nu_right) * (1 - 2 * nu_right))


# Stress computation
def sigma(v):
    return (2.0 * mu * sym(grad(v))
            + lmbda * tr(sym(grad(v))) * Identity(len(v)))


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=mesh, subdomain_data=ct)
a = inner(sigma(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
rhs = inner(Constant(mesh, PETSc.ScalarType((0, 0))), v) * dx

# Set boundary conditions
u_push = np.array([0.1, 0], dtype=PETSc.ScalarType)
dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bc_push = dirichletbc(u_push, dofs, V)
u_fix = np.array([0, 0], dtype=PETSc.ScalarType)
bc_fix = dirichletbc(u_fix, locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 2.1)), V)
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
V0, _ = V.sub(0).collapse()

facets_r = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1))
dofs_r = locate_dofs_topological(V0, fdim, facets_r)
facets_l = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.1))
dofs_l = locate_dofs_topological(V0, fdim, facets_l)

# Given the local coordinates of the dofs, distribute them on all processors
nodes = [gather_dof_coordinates(V0, dofs_r), gather_dof_coordinates(V0, dofs_l)]
pairs = []
for x in nodes[0]:
    for y in nodes[1]:
        if np.isclose(x[1], y[1]):
            pairs.append([x, y])
            break

mpc = MultiPointConstraint(V)
for i, pair in enumerate(pairs):
    sl, ms, co, ow, off = create_point_to_point_constraint(
        V, pair[0], pair[1], vector=[1, 0])
    mpc.add_constraint(V, sl, ms, co, ow, off)
mpc.finalize()

# Add back once PETSc release has added fix for
# https://gitlab.com/petsc/petsc/-/issues/1149
# petsc_options = {"ksp_rtol": 1.0e-8,
#                  "ksp_type": "cg",
#                  "pc_type": "gamg",
#                  "pc_gamg_type": "agg",
#                  "pc_gamg_coarse_eq_limit": 1000,
#                  "pc_gamg_sym_graph": True,
#                  "pc_gamg_square_graph": 2,
#                  "pc_gamg_threshold": 0.02,
#                  "mg_levels_ksp_type": "chebyshev",
#                  "mg_levels_pc_type": "jacobi",
#                  "mg_levels_esteig_ksp_type": "cg",
#                  #  "matptap_via": "scalable",
#                  "ksp_view": None,
#                  "help": None,
#                  "ksp_monitor": None
#                  }
petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

problem = LinearProblem(a, rhs, mpc, bcs=bcs, petsc_options=petsc_options)

# Build near nullspace
null_space = rigid_motions_nullspace(mpc.function_space)
problem.A.setNearNullSpace(null_space)
u_h = problem.solve()

it = problem.solver.getIterationNumber()

unorm = u_h.vector.norm()
if MPI.COMM_WORLD.rank == 0:
    print("Number of iterations: {0:d}".format(it))

# Write solution to file
u_h.name = "u"
with XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity_disconnect_2D.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

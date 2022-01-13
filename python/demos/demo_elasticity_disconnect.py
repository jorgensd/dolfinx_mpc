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
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx_mpc import (MultiPointConstraint, LinearProblem)
from dolfinx_mpc.utils import (create_point_to_point_constraint,
                               determine_closest_block, gmsh_model_to_mesh,
                               rigid_motions_nullspace)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, Measure, SpatialCoordinate, TestFunction,
                 TrialFunction, as_vector, grad, inner, sym, tr)

# Mesh parameters for creating a mesh consisting of two spheres,
# Sphere(r2)\Sphere(r1) and Sphere(r_0)
r0, r0_tag = 0.4, 1
r1, r1_tag = 0.5, 2
r2, r2_tag = 0.8, 3
outer_tag = 1
inner_tag = 2
assert(r0 < r1 and r1 < r2)


gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.clear()

    # Create Sphere(r2)\Sphere(r1)
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, r2)
    mid_sphere = gmsh.model.occ.addSphere(0, 0, 0, r1)
    hollow_sphere = gmsh.model.occ.cut([(3, outer_sphere)], [(3, mid_sphere)])
    # Create Sphere(r0)
    inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, r0)

    gmsh.model.occ.synchronize()

    # Add physical tags for volumes
    gmsh.model.addPhysicalGroup(hollow_sphere[0][0][0], [hollow_sphere[0][0][1]], tag=outer_tag)
    gmsh.model.setPhysicalName(hollow_sphere[0][0][0], 1, "Hollow sphere")
    gmsh.model.addPhysicalGroup(3, [inner_sphere], tag=inner_tag)
    gmsh.model.setPhysicalName(3, 2, "Inner sphere")

    # Add physical tags for surfaces
    r1_surface, r2_surface = [], []
    hollow_boundary = gmsh.model.getBoundary(hollow_sphere[0], oriented=False)
    inner_boundary = gmsh.model.getBoundary([(3, inner_sphere)], oriented=False)
    for boundary in hollow_boundary:
        bbox = gmsh.model.getBoundingBox(boundary[0], boundary[1])
        if np.isclose(max(bbox), r1):
            r1_surface.append(boundary[1])
        elif np.isclose(max(bbox), r2):
            r2_surface.append(boundary[1])
    gmsh.model.addPhysicalGroup(inner_boundary[0][0], [inner_boundary[0][1]], r0_tag)
    gmsh.model.setPhysicalName(inner_boundary[0][0], r0_tag, "Inner boundary")
    gmsh.model.addPhysicalGroup(2, r1_surface, r1_tag)
    gmsh.model.setPhysicalName(2, r1_tag, "Mid boundary")
    gmsh.model.addPhysicalGroup(2, r2_surface, r2_tag)
    gmsh.model.setPhysicalName(2, r2_tag, "Outer boundary")

    # Set mesh resolution
    res_inner = r0 / 5
    res_outer = (r1 + r2) / 5
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "NodesList", [p0])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res_inner)
    gmsh.model.mesh.field.setNumber(2, "LcMax", res_outer)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r1)
    gmsh.model.mesh.field.add("Threshold", 3)
    gmsh.model.mesh.field.setNumber(3, "IField", 1)
    gmsh.model.mesh.field.setNumber(3, "LcMin", res_outer)
    gmsh.model.mesh.field.setNumber(3, "LcMax", res_outer)
    gmsh.model.mesh.field.setNumber(3, "DistMin", r1)
    gmsh.model.mesh.field.setNumber(3, "DistMax", r2)
    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)
    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.setOrder(2)


mesh, ct, ft = gmsh_model_to_mesh(gmsh.model, facet_data=True, cell_data=True)

gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

V = VectorFunctionSpace(mesh, ("Lagrange", 1))

tdim = mesh.topology.dim
fdim = tdim - 1

DG0 = FunctionSpace(mesh, ("DG", 0))
outer_cells = ct.indices[ct.values == outer_tag]
inner_cells = ct.indices[ct.values == inner_tag]
outer_dofs = locate_dofs_topological(DG0, tdim, outer_cells)
inner_dofs = locate_dofs_topological(DG0, tdim, inner_cells)

# Elasticity parameters
E_outer = 1e3
E_inner = 1e5
nu_outer = 0.3
nu_inner = 0.1
mu = Function(DG0)
lmbda = Function(DG0)
with mu.vector.localForm() as local:
    local.array[inner_dofs] = E_inner / (2 * (1 + nu_inner))
    local.array[outer_dofs] = E_outer / (2 * (1 + nu_outer))
with lmbda.vector.localForm() as local:
    local.array[inner_dofs] = E_inner * nu_inner / ((1 + nu_inner) * (1 - 2 * nu_inner))
    local.array[outer_dofs] = E_outer * nu_outer / ((1 + nu_outer) * (1 - 2 * nu_outer))


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
rhs = inner(Constant(mesh, PETSc.ScalarType((0, 0, 0))), v) * dx
rhs += inner(Constant(mesh, PETSc.ScalarType((0.01, 0.02, 0))), v) * dx(outer_tag)
rhs += inner(as_vector(PETSc.ScalarType((0, 0, -9.81e-2))), v) * dx(inner_tag)


# Create dirichletbc
owning_processor, bc_dofs = determine_closest_block(V, -np.array([-r2, 0, 0]))
bc_dofs = [] if bc_dofs is None else bc_dofs

u_fixed = np.array([0, 0, 0], dtype=PETSc.ScalarType)
bc_fixed = dirichletbc(u_fixed, np.asarray(bc_dofs, dtype=np.int32), V)
bcs = [bc_fixed]

# Create point to point constraints
mpc = MultiPointConstraint(V)
signs = [-1, 1]
axis = [0, 1]
for i in axis:
    for s in signs:
        r0_point = np.zeros(3)
        r1_point = np.zeros(3)
        r0_point[i] = s * r0
        r1_point[i] = s * r1
        sl, ms, co, ow, off = create_point_to_point_constraint(V, r1_point, r0_point)
        mpc.add_constraint(V, sl, ms, co, ow, off)
mpc.finalize()

# Create nullspace
null_space = rigid_motions_nullspace(mpc.function_space)

petsc_options = {"ksp_rtol": 1.0e-8, "pc_type": "gamg", "pc_gamg_type": "agg",
                 "pc_gamg_coarse_eq_limit": 1000, "pc_gamg_sym_graph": True,
                 "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
                 "mg_levels_esteig_ksp_type": "cg", "matptap_via": "scalable",
                 "pc_gamg_square_graph": 2, "pc_gamg_threshold": 0.02
                 # ,"help": None, "ksp_view": None
                 }
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
with XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity_disconnect.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

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
    hollow_boundary = gmsh.model.getBoundary(hollow_sphere[0])
    inner_boundary = gmsh.model.getBoundary([(3, inner_sphere)])
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


mesh, ct, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, facet_data=True, cell_data=True)

gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

tdim = mesh.topology.dim
fdim = tdim - 1

DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
outer_cells = ct.indices[ct.values == outer_tag]
inner_cells = ct.indices[ct.values == inner_tag]
outer_dofs = fem.locate_dofs_topological(DG0, tdim, outer_cells)
inner_dofs = fem.locate_dofs_topological(DG0, tdim, inner_cells)

# Elasticity parameters
E_outer = 1e3
E_inner = 1e5
nu_outer = 0.3
nu_inner = 0.1
mu = dolfinx.Function(DG0)
lmbda = dolfinx.Function(DG0)
with mu.vector.localForm() as local:
    local.array[inner_dofs] = E_inner / (2 * (1 + nu_inner))
    local.array[outer_dofs] = E_outer / (2 * (1 + nu_outer))
with lmbda.vector.localForm() as local:
    local.array[inner_dofs] = E_inner * nu_inner / ((1 + nu_inner) * (1 - 2 * nu_inner))
    local.array[inner_dofs] = E_outer * nu_outer / ((1 + nu_outer) * (1 - 2 * nu_outer))


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
rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v) * dx
rhs += ufl.inner(dolfinx.Constant(mesh, (0.01, 0.02, 0)), v) * dx(outer_tag)
rhs += ufl.inner(ufl.as_vector((0, 0, -9.81e-2)), v) * dx(inner_tag)


owning_processor, bc_dofs = dolfinx_mpc.utils.determine_closest_block(V, -np.array([-r2, 0, 0]))
u_fixed = dolfinx.Function(V)
u_fixed.vector.set(0)
u_fixed.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if bc_dofs is None:
    bc_dofs = []

bc_fixed = dolfinx.DirichletBC(u_fixed, bc_dofs)
bcs = [bc_fixed]
mpc = dolfinx_mpc.MultiPointConstraint(V)


signs = [-1, 1]
axis = [0, 1]
for i in axis:
    for s in signs:
        r0_point = np.zeros(3)
        r1_point = np.zeros(3)
        r0_point[i] = s * r0
        r1_point[i] = s * r1

        with dolfinx.common.Timer("~Contact: Create node-node constraint"):
            sl, ms, co, ow, off = dolfinx_mpc.utils.create_point_to_point_constraint(V, r1_point, r0_point)
        with dolfinx.common.Timer("~Contact: Add data and finialize MPC"):
            mpc.add_constraint(V, sl, ms, co, ow, off)

mpc.finalize()
null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())
num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
with dolfinx.common.Timer("~Contact: Assemble matrix ({0:d})".format(num_dofs)):
    A = dolfinx_mpc.assemble_matrix_cpp(a, mpc, bcs=bcs)
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
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

it = solver.getIterationNumber()

unorm = uh.norm()
if MPI.COMM_WORLD.rank == 0:
    print("Number of iterations: {0:d}".format(it))

# Write solution to file
u_h = dolfinx.Function(mpc.function_space())
u_h.vector.setArray(uh.array)
u_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
u_h.name = "u"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity_disconnect.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    # xdmf.write_meshtags(ct)
    # xdmf.write_meshtags(ft)
    xdmf.write_function(u_h)

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Create constraint between two bodies that are not in contact

import dolfinx.cpp as cpp
import dolfinx.geometry as geometry
from IPython import embed
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
    gmsh.model.addPhysicalGroup(hollow_sphere[0][0][0], [hollow_sphere[0][0][1]], tag=1)
    gmsh.model.setPhysicalName(hollow_sphere[0][0][0], 1, "Hollow sphere")
    gmsh.model.addPhysicalGroup(3, [inner_sphere], tag=2)
    gmsh.model.setPhysicalName(3, 2, "Inner sphere")

    # Add physical tags for surfaces
    r1_surface, r2_surface = [], []
    hollow_boundary = gmsh.model.getBoundary(hollow_sphere[0:1])
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

mesh, ct, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, facet_data=True, cell_data=True)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define boundary conditions


def outer_displacement(x):
    return x


u_bc = dolfinx.function.Function(V)
u_bc.interpolate(outer_displacement)
u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

tdim = mesh.topology.dim
fdim = tdim - 1
outer_facets = ft.indices[np.flatnonzero(ft.values == r2_tag)]
bottom_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)
bc_outer = fem.DirichletBC(u_bc, bottom_dofs)

bcs = [bc_outer]

# Elasticity parameters
E = 1.0e3
nu = 0
mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

# Stress computation


def sigma(v):
    return (2.0 * mu * ufl.sym(ufl.grad(v))
            + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))


# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
# NOTE: Traction deactivated until we have a way of fixing nullspace
rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v) * ufl.dx


def determine_closest_dofs(point):
    """
    Determine the closest dofs (in a single block) to a point and the distance
    """
    bb_tree = geometry.BoundingBoxTree(mesh, tdim)
    midpoint_tree = bb_tree.create_midpoint_tree(mesh)
    # Find facet closest
    closest_cell_data = geometry.compute_closest_entity(bb_tree, midpoint_tree, mesh, point)
    closest_cell, min_distance = closest_cell_data[0][0], closest_cell_data[1][0]
    cell_imap = mesh.topology.index_map(tdim)
    # Set distance high if cell is not owned
    if cell_imap.size_local * cell_imap.block_size <= closest_cell:
        min_distance = 1e5
    # Find processor with cell closest to point
    global_distances = MPI.COMM_WORLD.allgather(min_distance)
    owning_processor = np.argmin(global_distances)

    dofmap = V.dofmap
    imap = dofmap.index_map
    ghost_owner = imap.ghost_owner_rank()
    block_size = imap.block_size
    local_max = imap.size_local * block_size
    # Determine which block of dofs is closest
    min_distance = max(min_distance, 1e5)
    minimal_distance_block = None
    min_dof_owner = owning_processor
    if MPI.COMM_WORLD.rank == owning_processor:
        x = V.tabulate_dof_coordinates()
        cell_dofs = dofmap.cell_dofs(closest_cell)
        for dof in cell_dofs:
            # Only do distance computation for one node per block
            if dof % block_size == 0:
                block = dof // block_size
                distance = np.linalg.norm(cpp.geometry.compute_distance_gjk(point, x[dof]))
                if distance < min_distance:
                    # If cell owned by processor, but not the closest dof
                    if dof < local_max:
                        minimal_distance_block = block
                    else:
                        min_dof_owner = ghost_owner[block - imap.size_local]
                        minimal_distance_block = block
                    min_distance = distance
        return owning_processor, [minimal_distance_block * block_size + i for i in range(block_size)]

    if min_dof_owner != owning_processor:
        raise NotImplementedError("Not implemented as there is no test case for this")
    else:
        return owning_processor, None


mpc = dolfinx_mpc.MultiPointConstraint(V)


def create_point_to_point_constraint(V, slave_point, master_point):
    # Determine which processor owns the dof closest to the slave and master point
    slave_proc, slave_dofs = determine_closest_dofs(slave_point)
    master_proc, master_dofs = determine_closest_dofs(master_point)

    # Create local to global mapping and map masters
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False), dtype=np.int64)
    global_masters = loc_to_glob[master_dofs]
    local_slaves, ghost_slaves = [], []
    local_masters, ghost_masters = [], []
    local_coeffs, ghost_coeffs = [], []
    local_owners, ghost_owners = [], []
    local_offsets, ghost_offsets = [], []

    if slave_proc == master_proc:
        if MPI.COMM_WORLD.rank == slave_proc:
            local_masters = slave_dofs
            local_masters = master_dofs
            local_owners = np.full(len(local_masters), master_proc, dtype=np.int32)
            local_coeffs = np.ones(len(local_masters), dtype=np.int32)
            local_offsets = np.arange(0, len(local_masters) + 1, dtype=np.int32)
    else:
        if MPI.COMM_WORLD.rank == master_proc:
            MPI.COMM_WORLD.send(global_masters, dest=slave_proc, tag=10)

        if MPI.COMM_WORLD.rank == slave_proc:
            global_masters = MPI.COMM_WORLD.recv(source=master_proc, tag=10)

    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    imap = V.dofmap.index_map
    ghost_processors = []
    if MPI.COMM_WORLD.rank == slave_proc:
        local_slaves = np.array(slave_dofs, dtype=np.int32)
        global_slaves = loc_to_glob[local_slaves]
        local_masters = global_masters
        local_owners = np.full(len(local_masters), master_proc, dtype=np.int32)
        local_coeffs = np.ones(len(local_masters), dtype=np.int32)
        local_offsets = np.arange(0, len(local_masters) + 1, dtype=np.int32)
        if slave_dofs[0] / imap.block_size in shared_indices.keys():
            ghost_processors = list(shared_indices[slave_dofs[0] / imap.block_size])

    # Broadcast processors containg slave
    ghost_processors = MPI.COMM_WORLD.bcast(ghost_processors, root=slave_proc)

    if MPI.COMM_WORLD.rank == slave_proc:
        for proc in ghost_processors:
            MPI.COMM_WORLD.send(global_slaves, dest=proc, tag=20 + proc)
            MPI.COMM_WORLD.send(local_coeffs, dest=proc, tag=30 + proc)
            MPI.COMM_WORLD.send(local_owners, dest=proc, tag=40 + proc)
            MPI.COMM_WORLD.send(local_masters, dest=proc, tag=50 + proc)
            MPI.COMM_WORLD.send(local_offsets, dest=proc, tag=60 + proc)

    # Receive data for ghost slaves
    if np.isin(MPI.COMM_WORLD.rank, ghost_processors):
        # Convert recieved slaves to the corresponding ghost index
        recv_slaves = MPI.COMM_WORLD.recv(source=slave_proc, tag=20 + MPI.COMM_WORLD.rank)
        ghost_coeffs = MPI.COMM_WORLD.recv(source=slave_proc, tag=30 + MPI.COMM_WORLD.rank)
        ghost_owners = MPI.COMM_WORLD.recv(source=slave_proc, tag=40 + MPI.COMM_WORLD.rank)
        ghost_masters = MPI.COMM_WORLD.recv(source=slave_proc, tag=50 + MPI.COMM_WORLD.rank)
        ghost_offsets = MPI.COMM_WORLD.recv(source=slave_proc, tag=60 + MPI.COMM_WORLD.rank)

        ghosts = imap.indices(True)[imap.size_local * imap.block_size:]
        ghost_slaves = np.zeros(len(recv_slaves), dtype=np.int32)
        for i, slave in enumerate(recv_slaves):
            idx = np.argwhere(ghosts == slave)
            ghost_slaves[i] = imap.size_local * imap.block_size + idx
    return (local_slaves, ghost_slaves), (local_masters, ghost_masters), (local_coeffs, ghost_coeffs),\
        (local_owners, ghost_owners), (local_offsets, ghost_offsets)


r0_point = np.array([r0, 0, 0])
r1_point = np.array([r1, 0, 0])

with dolfinx.common.Timer("~Contact: Create node-node constraint"):
    sl, ms, co, ow, off = create_point_to_point_constraint(V, r0_point, r1_point)

with dolfinx.common.Timer("~Contact: Add data and finialize MPC"):
    mpc.add_constraint(V, sl, ms, co, ow, off)

r0_point = np.array([-r0, 0, 0])
r1_point = np.array([-r1, 0, 0])

with dolfinx.common.Timer("~Contact: Create node-node constraint"):
    sl, ms, co, ow, off = create_point_to_point_constraint(V, r0_point, r1_point)

with dolfinx.common.Timer("~Contact: Add data and finialize MPC"):
    mpc.add_constraint(V, sl, ms, co, ow, off)
print(MPI.COMM_WORLD.rank, sl, ms, co, ow, off)
mpc.finalize()


null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())

with dolfinx.common.Timer("~Contact: Assemble matrix ({0:d})".format(V.dim)):
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)

with dolfinx.common.Timer("~Contact: Assemble vector ({0:d})".format(V.dim)):
    b = dolfinx_mpc.assemble_vector(rhs, mpc)

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
u_h.name = "u"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity_disconnect.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    # xdmf.write_meshtags(ct)
    # xdmf.write_meshtags(ft)
    xdmf.write_function(u_h)

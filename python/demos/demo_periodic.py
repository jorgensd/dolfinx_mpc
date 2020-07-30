# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
#
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

# Create mesh and finite element
N = 12
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def DirichletBoundary(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))


facets = dolfinx.mesh.locate_entities_boundary(mesh, 1,
                                               DirichletBoundary)
topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
bcs = [bc]


def PeriodicBoundary(x):
    return np.isclose(x[1], 1)


facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, PeriodicBoundary)
mt = dolfinx.MeshTags(mesh, mesh.topology.dim-1,
                      facets, np.full(len(facets), 2, dtype=np.int32))


def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0]
    out_x[1] = 1-x[1]
    out_x[2] = x[2]
    return out_x


def create_periodic_condition(V, mt, tag, relation, bcs):
    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local * bs
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                           dtype=np.int64)
    x = V.tabulate_dof_coordinates()
    comm = V.mesh.mpi_comm()
    # Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
    # Dirichlet BC dofs.
    bc_dofs = []
    for bc in bcs:
        bc_dofs.extend(bc.dof_indices[:, 0])

    slave_facets = mt.indices[mt.values == tag]
    if V.num_sub_spaces() == 0:
        slave_blocks = dolfinx.fem.locate_dofs_topological(
            V, fdim, slave_facets)[:, 0]
        # Filter out Dirichlet BC dofs
        slave_blocks = slave_blocks[np.isin(
            slave_blocks, bc_dofs, invert=True)]
        num_local_blocks = len(slave_blocks[slave_blocks
                                            < local_size])
    else:
        # Use zeroth sub space to get the blocks and
        # relevant coordinates
        slave_blocks = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V.sub(0).collapse()), slave_facets)[:, 0]
        raise NotImplementedError("Not implemented yet")
    # Compute coordinates where each slave has to evaluate its masters
    tree = dolfinx.geometry.BoundingBoxTree(mesh, dim=tdim)
    cell_map = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    loc2glob_cell = np.array(cell_map.global_indices(False),
                             dtype=np.int64)
    del cell_map
    master_coordinates = relation(x[slave_blocks].T).T

    # If data
    constraint_data = {slave_block: {}
                       for slave_block in slave_blocks[:num_local_blocks]}
    remaining_slaves = list(slave_blocks[:num_local_blocks])
    num_remaining = len(remaining_slaves)
    search_data = {}
    for (slave_block, master_coordinate) in zip(
            slave_blocks[:num_local_blocks], master_coordinates):
        procs = np.array(
            dolfinx_mpc.cpp.mpc.compute_process_collisions(
                tree._cpp_object, master_coordinate), dtype=np.int32)
        # Check if masters can be local
        search_globally = True
        masters_, coeffs_, owners_ = [], [], []
        if comm.rank in procs:
            possible_cells = \
                np.array(
                    dolfinx.geometry.compute_collisions_point(
                        tree, master_coordinate), dtype=np.int32)
            if len(possible_cells) > 0:
                glob_cells = loc2glob_cell[possible_cells]
                is_owned = np.logical_and(
                    cmin <= glob_cells, glob_cells < cmax)
                verified_candidate = \
                    dolfinx.cpp.geometry.select_colliding_cells(
                        V.mesh, list(possible_cells[is_owned]),
                        master_coordinate, 1)
                if len(verified_candidate) > 0:
                    basis_values = \
                        dolfinx_mpc.cpp.mpc.get_basis_functions(
                            V._cpp_object, master_coordinate,
                            verified_candidate[0])
                    cell_dofs = V.dofmap.cell_dofs(verified_candidate[0])
                    for k in range(bs):
                        for local_idx, dof in enumerate(cell_dofs):
                            if not np.isclose(basis_values[local_idx, k], 0):
                                if dof < local_size:
                                    owners_.append(comm.rank)
                                else:
                                    owners_.append(
                                        ghost_owners[dof//bs-local_size//bs])
                                masters_.append(loc_to_glob[dof])
                                coeffs_.append(basis_values[local_idx, k])
            if len(masters_) > 0:
                constraint_data[slave_block]["masters"] = masters_
                constraint_data[slave_block]["coeffs"] = coeffs_
                constraint_data[slave_block]["owners"] = owners_
                remaining_slaves.remove(slave_block)
                num_remaining -= 1
                search_globally = False
        # If masters not found on this processor
        if search_globally:
            other_procs = procs[procs != comm.rank]
            for proc in other_procs:
                if proc in search_data.keys():
                    search_data[proc][slave_block] = master_coordinate
                else:
                    search_data[proc] = {slave_block: master_coordinate}

    # Send search data
    recv_procs = []
    for proc in range(comm.size):
        if proc != comm.rank:
            if proc in search_data.keys():
                comm.send(search_data[proc], dest=proc, tag=11)
                recv_procs.append(proc)
            else:
                comm.send(None, dest=proc, tag=11)
    # Receive search data
    for proc in range(comm.size):
        if proc != comm.rank:
            in_data = comm.recv(source=proc, tag=11)
            out_data = {}
            if in_data is not None:
                for block in in_data.keys():
                    coordinate = in_data[block]
                    m_, c_, o_ = [], [], []
                    possible_cells = \
                        np.array(
                            dolfinx.geometry.compute_collisions_point(
                                tree, coordinate), dtype=np.int32)
                    if len(possible_cells) > 0:
                        glob_cells = loc2glob_cell[possible_cells]
                        is_owned = np.logical_and(
                            cmin <= glob_cells, glob_cells < cmax)
                        verified_candidate = \
                            dolfinx.cpp.geometry.select_colliding_cells(
                                V.mesh, list(possible_cells[is_owned]),
                                coordinate, 1)
                        if len(verified_candidate) > 0:
                            basis_values = \
                                dolfinx_mpc.cpp.mpc.get_basis_functions(
                                    V._cpp_object, coordinate,
                                    verified_candidate[0])
                            cell_dofs = V.dofmap.cell_dofs(
                                verified_candidate[0])
                            for k in range(bs):
                                for idx, dof in enumerate(cell_dofs):
                                    if not np.isclose(basis_values[idx, k], 0):
                                        if dof < local_size:
                                            o_.append(comm.rank)
                                        else:
                                            o_.append(ghost_owners[
                                                dof//bs-local_size//bs])
                                        m_.append(loc_to_glob[dof])
                                        c_.append(basis_values[idx, k])
                    if len(m_) > 0:
                        out_data[block] = {"masters": m_,
                                           "coeffs": c_, "owners": o_}
                comm.send(out_data, dest=proc, tag=22)
    for proc in recv_procs:
        master_info = comm.recv(source=proc, tag=22)
        for block in master_info.keys():
            # First come first serve principle
            if block in remaining_slaves:
                constraint_data[block]["masters"] = \
                    master_info[block]["masters"]
                constraint_data[block]["coeffs"] = master_info[block]["coeffs"]
                constraint_data[block]["owners"] = master_info[block]["owners"]
                remaining_slaves.remove(block)
                num_remaining -= 1
    assert(num_remaining == 0)

    # Send ghost data
    send_ghost_masters = {}
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(V._cpp_object)
    for (i, block) in enumerate(slave_blocks[:num_local_blocks]):
        if block in shared_indices.keys():
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[bs*block]] = {
                        "masters": constraint_data[block]["masters"],
                        "coeffs": constraint_data[block]["coeffs"],
                        "owners": constraint_data[block]["owners"]}
                else:
                    send_ghost_masters[proc] = {
                        loc_to_glob[bs*block]:
                        {"masters": constraint_data[block]["masters"],
                         "coeffs": constraint_data[block]["coeffs"],
                         "owners": constraint_data[block]["owners"]}}
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=33)

    ghost_recv = []
    for block in slave_blocks[num_local_blocks:]:
        g_block = int(block - local_size/bs)
        owner = ghost_owners[g_block]
        ghost_recv.append(owner)
        del owner
    ghost_recv = set(ghost_recv)
    ghost_block = {block: {} for block in slave_blocks[num_local_blocks:]}
    for owner in ghost_recv:
        from_owner = comm.recv(source=owner, tag=33)
        for block in slave_blocks[num_local_blocks:]:
            if loc_to_glob[block*bs] in from_owner.keys():
                ghost_block[block]["masters"] = \
                    from_owner[loc_to_glob[block*bs]]["masters"]
                ghost_block[block]["coeffs"] = \
                    from_owner[loc_to_glob[block*bs]]["coeffs"]
                ghost_block[block]["coeffs"] = \
                    from_owner[loc_to_glob[block*bs]]["coeffs"]
    # Flatten out arrays
    print(comm.rank, ghost_block)


create_periodic_condition(V, mt, 2, periodic_relation, bcs)
exit(1)
# Create MPC (map all left dofs to right
dof_at = dolfinx_mpc.dof_close_to


def slave_locater(i, N):
    return lambda x: dof_at(x, [1, float(i/N)])


def master_locater(i, N):
    return lambda x: dof_at(x, [0, float(i/N)])


s_m_c = {}
for i in range(1, N):
    s_m_c[slave_locater(i, N)] = {master_locater(i, N): 1}

(slaves, masters,
 coeffs, offsets, owner_ranks) = dolfinx_mpc.slave_master_structure(V, s_m_c)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

x = ufl.SpatialCoordinate(mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0]*ufl.sin(5.0*ufl.pi*x[1]) \
    + 1.0*ufl.exp(-(dx*dx + dy*dy)/0.02)

lhs = ufl.inner(f, v)*ufl.dx

# Compute solution
u = dolfinx.Function(V)
u.name = "uh"
mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                               masters, coeffs, offsets,
                                               owner_ranks)
# Setup MPC system

A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
b = dolfinx_mpc.assemble_vector(lhs, mpc)

# Apply boundary conditions
dolfinx.fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)

# Solve Linear problem
solver = PETSc.KSP().create(MPI.COMM_WORLD)

if complex:
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
else:
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-6
    opts["pc_type"] = "hypre"
    opts['pc_hypre_type'] = 'boomeramg'
    opts["pc_hypre_boomeramg_max_iter"] = 1
    opts["pc_hypre_boomeramg_cycle_type"] = "v"
    # opts["pc_hypre_boomeramg_print_statistics"] = 1
    solver.setFromOptions()

solver.setOperators(A)

uh = b.copy()
uh.set(0)
start = time.time()
solver.solve(b, uh)
# solver.view()
end = time.time()
it = solver.getIterationNumber()

print("Solver time: {0:.2e}, Iterations {1:d}".format(end-start, it))

uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)

# Back substitute to slave dofs
dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

# Create functionspace and function for mpc vector
Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                              mpc.mpc_dofmap())
Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

# Write solution to file
u_h = dolfinx.Function(Vmpc)
u_h.vector.setArray(uh.array)
u_h.name = "u_mpc"
outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                              "results/demo_periodic.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(u_h)


print("----Verification----")
# --------------------VERIFICATION-------------------------
A_org = dolfinx.fem.assemble_matrix(a, bcs)

A_org.assemble()
L_org = dolfinx.fem.assemble_vector(lhs)
dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver.setOperators(A_org)
u_ = dolfinx.Function(V)
start = time.time()
solver.solve(L_org, u_.vector)
end = time.time()

it = solver.getIterationNumber()
print("Org solver time: {0:.2e}, Iterations {1:d}".format(end-start, it))
u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
u_.name = "u_unconstrained"
outfile.write_function(u_)

# Transfer data from the MPC problem to numpy arrays for comparison
A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

# Create global transformation matrix
K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
                                                   masters, coeffs,
                                                   offsets)
# Create reduced A
A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
reduced_A = np.matmul(np.matmul(K.T, A_global), K)


# Created reduced L
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)
# Solve linear system
d = np.linalg.solve(reduced_A, reduced_L)
# Back substitution to full solution vector
uh_numpy = np.dot(K, d)

# compare LHS, RHS and solution with reference values

dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

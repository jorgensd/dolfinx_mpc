# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import resource
import time
from contextlib import ExitStack

import h5py
import numpy as np
from petsc4py import PETSc

import dolfinx
import dolfinx.common
import dolfinx.io
import dolfinx.la
import dolfinx.log
import dolfinx.mesh
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
from bench_elasticity import build_elastic_nullspace
from mpi4py import MPI

opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-5
opts["pc_type"] = "hypre"
opts['pc_hypre_type'] = 'boomeramg'
opts["pc_hypre_boomeramg_max_iter"] = 1
opts["pc_hypre_boomeramg_cycle_type"] = "v"
opts["pc_hypre_boomeramg_print_statistics"] = 1

# opts["ksp_type"] = "cg"
# opts["ksp_rtol"] = 1.0e-10
# opts["pc_type"] = "gamg"
# opts["pc_gamg_type"] = "agg"
# opts["pc_gamg_coarse_eq_limit"] = 1000
# opts["pc_gamg_sym_graph"] = True
# opts["mg_levels_ksp_type"] = "chebyshev"
# opts["mg_levels_pc_type"] = "jacobi"
# opts["mg_levels_esteig_ksp_type"] = "cg"
# opts["matptap_via"] = "scalable"

def demo_elasticity(r_lvl=1, outfile=None):
    N = 3
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    for i in range(r_lvl):
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        mesh = dolfinx.mesh.refine(mesh, redistribute=True)
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
        N *= 2
    fdim = mesh.topology.dim - 1
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def boundaries(x):
        return np.isclose(x[0], np.finfo(float).eps)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                                   boundaries)
    topological_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
    bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc]

    def slaves_locater(x):
        return np.logical_and(np.isclose(x[0], 1), np.isclose(x[2], 0))

    def master_locater(x):
        return np.logical_and(np.isclose(x[0], 1), np.isclose(x[2], 1))

    # Find all masters and slaves
    slave_dofs = dolfinx.fem.locate_dofs_geometrical(
        (V.sub(2), V.sub(2).collapse()), slaves_locater).T[0]
    master_dofs = dolfinx.fem.locate_dofs_geometrical(
        (V.sub(2), V.sub(2).collapse()), master_locater).T[0]

    def create_master_slave_map(master_dofs, slave_dofs):
        timer = dolfinx.common.Timer("MPC: Create slave-master relationship")
        x = V.tabulate_dof_coordinates()
        local_min = (V.dofmap.index_map.local_range[0]
                     * V.dofmap.index_map.block_size)
        local_max = (V.dofmap.index_map.local_range[1]
                     * V.dofmap.index_map.block_size)
        x_slaves = x[slave_dofs, :]
        x_masters = x[master_dofs, :]
        masters = np.zeros(N+1, dtype=np.int64)
        slaves = np.zeros(N+1, dtype=np.int64)
        for i in range(N+1):
            if len(slave_dofs) > 0:
                slave_coord = [1, i/N, 0]
                slave_diff = np.abs(x_slaves-slave_coord).sum(axis=1)
                if np.isclose(slave_diff[slave_diff.argmin()], 0):
                    # Only add if owned by processor
                    if (slave_dofs[slave_diff.argmin()]
                            < local_max - local_min):
                        slaves[i] = (slave_dofs[slave_diff.argmin()]
                                     + local_min)
                if len(master_dofs) > 0:
                    master_coord = [1, i/N, 1]
                    master_diff = np.abs(x_masters-master_coord).sum(axis=1)
                    if np.isclose(master_diff[master_diff.argmin()], 0):
                        # Only add if owned by processor
                        if (master_dofs[master_diff.argmin()] <
                                local_max-local_min):
                            masters[i] = (master_dofs[master_diff.argmin()]
                                          + local_min)
        timer.stop()
        return slaves, masters
    snew, mnew = create_master_slave_map(master_dofs, slave_dofs)
    masters = sum(MPI.COMM_WORLD.allgather(mnew))
    slaves = sum(MPI.COMM_WORLD.allgather(snew))
    offsets = np.array(range(len(slaves)+1), dtype=np.int64)
    coeffs = 0.2*np.ones(len(masters), dtype=np.float64)

    # Create traction meshtag

    def traction_boundary(x):
        return np.isclose(x[0], 1)
    t_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                                     traction_boundary)
    facet_values = np.ones(len(t_facets), dtype=np.int32)
    mt = dolfinx.MeshTags(mesh, fdim, t_facets, facet_values)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.1
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    g = dolfinx.Constant(mesh, (0, 0, -1e2))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # x = ufl.SpatialCoordinate(mesh)
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(g, v)*ufl.ds(domain=mesh,
                                 subdomain_data=mt, subdomain_id=1)

    # Create MPC
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
    null_space = build_elastic_nullspace(Vmpc)
    A.setNearNullSpace(null_space)

    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # def monitor(ksp, its, rnorm, r_lvl=-1):
    #     if MPI.COMM_WORLD.rank == 0:
    #         print("{}: Iteration: {}, rel. residual: {}"
    #               .format(r_lvl, its, rnorm))

    # def pmonitor(ksp, its, rnorm):
    #     return monitor(ksp, its, rnorm, r_lvl=r_lvl)

    # Solve Linear problem

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    # solver.setMonitor(pmonitor)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    start = time.time()
    solver.solve(b, uh)
    end = time.time()
    #print(r_lvl, MPI.COMM_WORLD.rank, end-start)
    
    # solver.view()
    mem = sum(MPI.COMM_WORLD.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    it = solver.getIterationNumber()
    if outfile is not None:
        d_set = f.get("its")
        d_set[r_lvl] = it
        d_set = f.get("num_dofs")
        d_set[r_lvl] = V.dim
        d_set = f.get("num_slaves")
        d_set[r_lvl] = len(slaves)
        d_set = f.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end-start
    if MPI.COMM_WORLD.rank == 0:
        print("Rlvl {0:d}, Iterations {1:d}".format(r_lvl, it))
        print("Rlvl {0:d}, Max usage {1:d} (kb), #dofs {2:d}".format(r_lvl, mem, V.dim))

    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc"
    # outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
    #                               "results/bench_elasticity.xdmf", "w")
    # outfile.write_mesh(mesh)
    # outfile.write_function(u_h)

    # Generate reference matrices and unconstrained solution
    # A_org = dolfinx.fem.assemble_matrix(a, bcs)
    # A_org.assemble()
    # null_space_org = build_elastic_nullspace(V)
    # A_org.setNearNullSpace(null_space_org)

    # L_org = dolfinx.fem.assemble_vector(lhs)
    # dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    # L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
    #                   mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(L_org, bcs)

    # solver = PETSc.KSP().create(MPI.COMM_WORLD)
    # solver.setFromOptions()
    # solver.setOperators(A_org)
    # u_ = dolfinx.Function(V)
    # solver.solve(L_org, u_.vector)
    # u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
    #                       mode=PETSc.ScatterMode.FORWARD)
    # # u_.name = "u_unconstrained"
    # # outfile.write_function(u_)
    # # outfile.close()

    # # Create global transformation matrix
    # K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
    #                                                    masters, coeffs,
    #                                                    offsets)

    # # Create reduced A
    # A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    # reduced_A = np.matmul(np.matmul(K.T, A_global), K)

    # # Created reduced L
    # vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    # reduced_L = np.dot(K.T, vec)

    # # Solve linear system
    # d = np.linalg.solve(reduced_A, reduced_L)

    # # Back substitution to full solution vector
    # uh_numpy = np.dot(K, d)

    # # Transfer data from the MPC problem to numpy arrays for comparison
    # A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    # mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # # Compare LHS, RHS and solution with reference values
    # dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    # dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    # print(max(abs(uh.array-uh_numpy[uh.owner_range[0]:
    #                                       uh.owner_range[1]])))
    # assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
    #                                       uh.owner_range[1]], atol=1e-6)


    # for i in range(len(masters)):
    #     if uh.owner_range[0] < masters[i] and masters[i] < uh.owner_range[1]:
    #         print("-"*10, i, "-"*10)
    #         print("MASTER DOF, MPC {0:.4e} Unconstrained {1:.4e}"
    #               .format(uh.array[masters[i]-uh.owner_range[0]],
    #                       u_.vector.array[masters[i]-uh.owner_range[0]]))
    #         print("Slave (given as master*coeff) {0:.4e}".
    #               format(uh.array[masters[i]-uh.owner_range[0]]*coeffs[0]))

    #     if uh.owner_range[0] < slaves[i] and slaves[i] < uh.owner_range[1]:
    #         print("SLAVE  DOF, MPC {0:.4e} Unconstrained {1:.4e}"
    #               .format(uh.array[slaves[i]-uh.owner_range[0]],
    #                       u_.vector.array[slaves[i]-uh.owner_range[0]]))


if __name__ == "__main__":
    n_level = 6
    f = h5py.File('bench_edge_output.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    f.create_dataset("its", (n_level,), dtype=np.int32)
    f.create_dataset("num_dofs", (n_level,), dtype=np.int32)
    f.create_dataset("num_slaves", (n_level,), dtype=np.int32)
    f.create_dataset("solve_time", (n_level, MPI.COMM_WORLD.size), dtype=np.float64)
    for i in range(n_level):
        if MPI.COMM_WORLD.rank == 0:
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                            "Run {0:1d} in progress".format(i))
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
        demo_elasticity(i, outfile=f)
        # dolfinx.common.list_timings(
        #     MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    f.close()

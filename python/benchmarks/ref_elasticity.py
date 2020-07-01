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

# View solver info
# opts['ksp_view'] = None

# List all options
# opts['help'] = None

# opts["ksp_rtol"] = 1.0e-10
# opts["pc_type"] = "gamg"
# opts["pc_gamg_type"] = "classical"
# opts["pc_gamg_coarse_eq_limit"] = 1000
# opts["pc_gamg_sym_graph"] = True
# opts["mg_levels_ksp_type"] = "chebyshev"
# opts["mg_levels_pc_type"] = "jacobi"
# opts["mg_levels_esteig_ksp_type"] = "cg"
# opts["matptap_via"] = "scalable"




def demo_elasticity(r_lvl=1, outfile=None):
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
    for i in range(r_lvl):
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        mesh = dolfinx.mesh.refine(mesh, redistribute=True)
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

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
    lhs = ufl.inner(g, v)*ufl.ds(domain=mesh,subdomain_data=mt, subdomain_id=1)
    if MPI.COMM_WORLD.rank == 0:
        print("Problem size {0:d} ".format(V.dim))

    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()
    null_space_org = build_elastic_nullspace(V)
    A_org.setNearNullSpace(null_space_org)

    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

    # def org_monitor(ksp, its, rnorm, r_lvl=-1):
    #     if MPI.COMM_WORLD.rank == 0:
    #         print("Orginal: {}: Iteration: {}, rel. residual: {}"
    #              .format(r_lvl, its, rnorm))
    #         pass

    # def org_pmonitor(ksp, its, rnorm):
    #     return org_monitor(ksp, its, rnorm, r_lvl=r_lvl)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)
    start = time.time()
    solver.solve(L_org, u_.vector)
    end = time.time()
    #solver.view()
    it = solver.getIterationNumber()
    if outfile is not None:
        d_set = f.get("its")
        d_set[r_lvl] = it
        d_set = f.get("num_dofs")
        d_set[r_lvl] = V.dim
        d_set = f.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end-start
    if MPI.COMM_WORLD.rank == 0:
        print("Refinement level {0:d}, Iterations {1:d}".format(r_lvl, it))

    mem = sum(MPI.COMM_WORLD.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if MPI.COMM_WORLD.rank ==0:
        print("{1:d}: Max usage after trad. solve {0:d} (kb)".format(mem, r_lvl))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    u_.name = "u_unconstrained"

if __name__ == "__main__":
    n_level = 6
    f = h5py.File('ref_output.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    f.create_dataset("its", (n_level,), dtype=np.int32)
    f.create_dataset("num_dofs", (n_level,), dtype=np.int32)
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

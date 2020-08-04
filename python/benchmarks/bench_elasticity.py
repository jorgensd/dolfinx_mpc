# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import resource
import sys
import time

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
from mpi4py import MPI


def bench_elasticity_one(out_xdmf=None, r_lvl=0, out_hdf5=None,
                         xdmf=False, boomeramg=False, kspview=False):
    N = 3
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
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

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(g, v)*ufl.ds(domain=mesh,
                                 subdomain_data=mt, subdomain_id=1)

    # Create MPC
    with dolfinx.common.Timer("~Elasticity: Init constraint (old)"):
        dof_at = dolfinx_mpc.dof_close_to
        s_m_c = {lambda x: dof_at(x, [1, 0, 0]): {
            lambda x: dof_at(x, [1, 0, 1]): 0.5}}
        # , lambda x: dof_at(x, [1, 1, 0]): {
        # lambda x: dof_at(x, [1, 1, 1]): 0.5}}
        (slaves, masters,
         coeffs, offsets,
         owner_ranks) = dolfinx_mpc.slave_master_structure(V, s_m_c,
                                                           2, 2)
        mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                       masters, coeffs,
                                                       offsets, owner_ranks)

    with dolfinx.common.Timer("~Elasticity: Init constraint"):
        def l2b(li):
            return np.array(li, dtype=np.float64).tobytes()
        s_m_c_new = {l2b([1, 0, 0]): {l2b([1, 0, 1]): 0.5}}
        cc = dolfinx_mpc.create_dictionary_constraint(V, s_m_c_new, 2, 2)

    # Setup MPC system
    with dolfinx.common.Timer("~Elasticity: Assemble (old)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(lhs, mpc)
    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    with dolfinx.common.Timer("~Elasticity: Assemble"):
        Acc = dolfinx_mpc.assemble_matrix_local(a, cc, bcs=bcs)
        bcc = dolfinx_mpc.assemble_vector_local(lhs, cc)
    # Apply boundary conditions
    dolfinx.fem.apply_lifting(bcc, [a], [bcs])
    bcc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(bcc, bcs)

    # Create functionspace and function for mpc vector

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    if boomeramg:
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-5
        opts["pc_type"] = "hypre"
        opts['pc_hypre_type'] = 'boomeramg'
        opts["pc_hypre_boomeramg_max_iter"] = 1
        opts["pc_hypre_boomeramg_cycle_type"] = "v"
        # opts["pc_hypre_boomeramg_print_statistics"] = 1
    else:
        opts["ksp_rtol"] = 1.0e-10
        opts["pc_type"] = "gamg"
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_coarse_eq_limit"] = 1000
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_esteig_ksp_type"] = "cg"
        opts["matptap_via"] = "scalable"
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    with dolfinx.common.Timer("~Elasticity: Solve problem old"):
        Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                      mpc.mpc_dofmap())
        Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
        null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vmpc)
        A.setNearNullSpace(null_space)
        solver.setFromOptions()
        solver.setOperators(A)
        # Solve linear problem
        uh = b.copy()
        uh.set(0)
        start = time.time()
        solver.solve(b, uh)
        end = time.time()
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)

    with dolfinx.common.Timer("~Elasticity: Solve problem"):
        Vcc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                     cc.dofmap())
        Vcc = dolfinx.FunctionSpace(None, V.ufl_element(), Vcc_cpp)
        null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vcc)
        Acc.setNearNullSpace(null_space)
        solver.setFromOptions()
        solver.setOperators(Acc)
        # Solve linear problem
        uhcc = bcc.copy()
        uhcc.set(0)
        start = time.time()
        solver.solve(bcc, uhcc)
        end = time.time()
        uhcc.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

    with dolfinx.common.Timer("~Elasticity: Backsubstitute (old)"):
        # Back substitute to slave dofs
        dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    with dolfinx.common.Timer("~Elasticity: Backsubstitute"):
        # Back substitute to slave dofs
        dolfinx_mpc.backsubstitution_local(cc, uhcc)

    it = solver.getIterationNumber()
    if kspview:
        solver.view()

    # Print max usage of summary
    mem = sum(MPI.COMM_WORLD.allgather(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if MPI.COMM_WORLD.rank == 0:
        print("Rlvl {0:d}, Iterations {1:d}".format(r_lvl, it))
        print("Rlvl {0:d}, Max usage {1:d} (kb), #dofs {2:d}".format(
            r_lvl, mem, V.dim))

    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = V.dim
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end-start

    if xdmf:
        # Write solution to file
        u_h = dolfinx.Function(Vmpc)
        u_h.vector.setArray(uh.array)
        u_h.name = "u_mpc"

        u_hcc = dolfinx.Function(Vcc)
        u_hcc.vector.setArray(uhcc.array)
        u_hcc.name = "u_mpc_cc"

        fname = "results/bench_elasticity_{0:d}.xdmf".format(r_lvl)
        outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                      fname, "w")
        outfile.write_mesh(mesh)
        outfile.write_function(u_h)
        outfile.write_function(u_hcc)
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", default=1, type=np.int8, dest="n_ref",
                        help="Number of spatial refinements")
    parser.add_argument('--xdmf', action='store_true', dest="xdmf",
                        help="XDMF-output of function (Default false)")
    parser.add_argument('--timings', action='store_true', dest="timings",
                        help="List timings (Default false)")
    parser.add_argument('--kspview', action='store_true', dest="kspview",
                        help="View PETSc progress")
    parser.add_argument("-o", default='elasticity_one.hdf5', dest="hdf5",
                        help="Name of HDF5 output file")
    solver_parser = parser.add_mutually_exclusive_group(required=False)
    solver_parser.add_argument('--boomeramg', dest='boomeramg', default=True,
                               action='store_true',
                               help="Use BoomerAMG preconditioner (Default)")
    solver_parser.add_argument('--gamg', dest='boomeramg',
                               action='store_false',
                               help="Use PETSc GAMG preconditioner")
    args = parser.parse_args()
    thismodule = sys.modules[__name__]
    n_ref = timings = boomeramg = kspview = degree = hdf5 = xdmf = tetra = None

    for key in vars(args):
        setattr(thismodule, key, getattr(args, key))

    N = n_ref + 1

    # Setup hd5f output file
    h5f = h5py.File(hdf5, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    sd = h5f.create_dataset(
        "solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if boomeramg else "GAMG"
    sd.attrs["solver"] = np.string_(solver)

    # Loop over refinement levels
    for i in range(N):
        if MPI.COMM_WORLD.rank == 0:
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                            "Run {0:1d} in progress".format(i))
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
        bench_elasticity_one(r_lvl=i, out_hdf5=h5f, xdmf=xdmf,
                             boomeramg=boomeramg, kspview=kspview)
        if timings and i == N-1:
            dolfinx.common.list_timings(
                MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    h5f.close()

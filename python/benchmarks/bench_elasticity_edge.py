# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import resource
import sys
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
from mpi4py import MPI


def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [dolfinx.cpp.la.create_vector(
        V.dofmap.index_map) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm())
                     for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(gdim):
            dofs = V.sub(i).dofmap.list
            basis[i][dofs.array()] = 1.0

        # Build rotational null space basis
        if gdim == 2:
            V.sub(0).set_x(basis[2], -1.0, 1)
            V.sub(1).set_x(basis[2], 1.0, 0)
        elif gdim == 3:
            V.sub(0).set_x(basis[3], -1.0, 1)
            V.sub(1).set_x(basis[3], 1.0, 0)

            V.sub(0).set_x(basis[4], 1.0, 2)
            V.sub(2).set_x(basis[4], -1.0, 0)

            V.sub(2).set_x(basis[5], 1.0, 1)
            V.sub(1).set_x(basis[5], -1.0, 2)

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


def bench_elasticity_edge(out_xdmf=None, r_lvl=0, out_hdf5=None,
                          xdmf=False, boomeramg=False, kspview=False):
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
        owner_ranks = np.zeros(N+1, dtype=np.int64)
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
                            owner_ranks[i] = MPI.COMM_WORLD.rank
        timer.stop()
        return slaves, masters, owner_ranks

    # Find all masters and slaves
    masters = []
    slaves = []
    master_ranks = []
    for j in range(mesh.topology.dim):
        Vj = V.sub(j).collapse()
        slave_dofs = dolfinx.fem.locate_dofs_geometrical(
            (V.sub(j), Vj), slaves_locater).T[0]
        master_dofs = dolfinx.fem.locate_dofs_geometrical(
            (V.sub(j), Vj), master_locater).T[0]

        snew, mnew, mranks = create_master_slave_map(master_dofs, slave_dofs)
        mastersj = sum(MPI.COMM_WORLD.allgather(mnew))
        slavesj = sum(MPI.COMM_WORLD.allgather(snew))
        ownersj = sum(MPI.COMM_WORLD.allgather(mranks))
        masters.append(mastersj)
        slaves.append(slavesj)
        master_ranks.append(ownersj)
    masters = np.hstack(masters)
    slaves = np.hstack(slaves)
    master_ranks = np.hstack(master_ranks)
    offsets = np.array(range(len(slaves)+1), dtype=np.int64)
    coeffs = 0.5*np.ones(len(masters), dtype=np.float64)

    # Create traction meshtag

    def traction_boundary(x):
        return np.isclose(x[0], 1)
    t_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                                     traction_boundary)
    facet_values = np.ones(len(t_facets), dtype=np.int32)
    mt = dolfinx.MeshTags(mesh, fdim, t_facets, facet_values)

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
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets, master_ranks)
    # Setup MPC system
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Assembling".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Create nullspace for elasticity problem and assign to matrix
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
        opts["ksp_rtol"] = 1.0e-8
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

    # Setup PETSc solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)

    # Solve Linear problem
    uh = b.copy()
    uh.set(0)
    start = time.time()
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Solving".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    with dolfinx.common.Timer("Ref solve") as t:
        solver.solve(b, uh)
    end = time.time()

    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    if kspview:
        solver.view()

    mem = sum(MPI.COMM_WORLD.allgather(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    it = solver.getIterationNumber()

    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = V.dim
        d_set = out_hdf5.get("num_slaves")
        d_set[r_lvl] = len(slaves)
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end-start

    if MPI.COMM_WORLD.rank == 0:
        print("Rlvl {0:d}, Iterations {1:d}".format(r_lvl, it))
        print("Rlvl {0:d}, Max usage {1:d} (kb), #dofs {2:d}".format(
            r_lvl, mem, V.dim))

    if xdmf:
        # Write solution to file
        u_h = dolfinx.Function(Vmpc)
        u_h.vector.setArray(uh.array)
        u_h.name = "u_mpc"
        fname = "results/bench_elasticity_edge_{0:d}.xdmf".format(r_lvl)
        outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                      fname, "w")
        outfile.write_mesh(mesh)
        outfile.write_function(u_h)
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
    solver_parser.add_argument('--gamg', dest='boomeramg', action='store_false',
                               help="Use PETSc GAMG preconditioner")
    args = parser.parse_args()
    thismodule = sys.modules[__name__]
    for key in vars(args):
        setattr(thismodule, key, getattr(args, key))
    N = n_ref + 1

    h5f = h5py.File('bench_edge_output.hdf5', 'w',
                    driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    h5f.create_dataset("num_slaves", (N,), dtype=np.int32)
    sd = h5f.create_dataset("solve_time",
                            (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if boomeramg else "GAMG"
    sd.attrs["solver"] = np.string_(solver)

    for i in range(N):
        if MPI.COMM_WORLD.rank == 0:
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                            "Run {0:1d} in progress".format(i))
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        bench_elasticity_edge(r_lvl=i, out_hdf5=h5f, xdmf=xdmf,
                              boomeramg=boomeramg, kspview=kspview)
        if timings:
            dolfinx.common.list_timings(MPI.COMM_WORLD,
                                        [dolfinx.common.TimingType.wall])
    h5f.close()

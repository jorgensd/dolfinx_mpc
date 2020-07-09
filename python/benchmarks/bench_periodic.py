# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPC.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import argparse
import sys
import time

import h5py
import numpy as np
from petsc4py import PETSc

import dolfinx
import dolfinx.common
import dolfinx.io
import dolfinx.log
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
from dolfinx.mesh import refine
from mpi4py import MPI


def demo_periodic3D(tetra, out_xdmf=None, r_lvl=0, out_hdf5=None,
                    xdmf=False, boomeramg=False, kspview=False, degree=1):
    # Create mesh and function space
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Create/Refine mesh".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    if tetra:
        ct = dolfinx.cpp.mesh.CellType.tetrahedron
    else:
        ct = dolfinx.cpp.mesh.CellType.hexahedron
    # Tet setup
    N = 3
    for i in range(r_lvl):
        # mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        # mesh = refine(mesh, redistribute=True)
        N *= 2
    with dolfinx.common.Timer("Create mesh"):
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N, ct)

    V = dolfinx.FunctionSpace(mesh, ("CG", degree))
    M = N*degree

    # Create Dirichlet boundary condition
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def DirichletBoundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0),
                                           np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0),
                                           np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = dolfinx.fem.locate_dofs_geometrical(
        V, DirichletBoundary)
    bc = dolfinx.fem.DirichletBC(u_bc, geometrical_dofs)
    bcs = [bc]

    def slaves_locater(x):
        """
        Helper function to check if a coordinate is located at a slave position
        """
        eps = 1e-14
        not_edges_y = np.logical_and(x[1] > eps, x[1] < 1-eps)
        not_edges_z = np.logical_and(x[2] > eps, x[2] < 1-eps)
        not_edges = np.logical_and(not_edges_y, not_edges_z)
        return np.logical_and(np.isclose(x[0], 1), not_edges)

    def master_locater(x):
        """
        Helper function to check if a coordinate is located at a slave position
        """
        eps = 1e-14
        not_edges_y = np.logical_and(x[1] > eps, x[1] < 1-eps)
        not_edges_z = np.logical_and(x[2] > eps, x[2] < 1-eps)
        not_edges = np.logical_and(not_edges_y, not_edges_z)
        return np.logical_and(np.isclose(x[0], 0), not_edges)

    # Find all masters and slaves
    slave_dofs = dolfinx.fem.locate_dofs_geometrical(V, slaves_locater).T[0]
    master_dofs = dolfinx.fem.locate_dofs_geometrical(V, master_locater).T[0]

    def create_master_slave_map(master_dofs, slave_dofs):
        """
        Helper function that creates the relationship between the correct
        master and slave nodes, where (1,i/M,j/M)->(0,i/M,j/M)
        This function is much more efficient than calling
        dolfinx.fem.locate_dofs_geometrical in a loop
        """
        timer = dolfinx.common.Timer("MPC: Create slave-master relationship")
        x = V.tabulate_dof_coordinates()
        local_min = (V.dofmap.index_map.local_range[0]
                     * V.dofmap.index_map.block_size)
        local_max = (V.dofmap.index_map.local_range[1]
                     * V.dofmap.index_map.block_size)
        x_slaves = x[slave_dofs, :]
        x_masters = x[master_dofs, :]
        Dy = int(M)
        Dz = int(M)

        masters = np.zeros((Dy-1)*(Dz-1), dtype=np.int64)
        slaves = np.zeros((Dy-1)*(Dz-1), dtype=np.int64)
        master_rank = np.zeros((Dy-1)*(Dz-1), dtype=np.int64)
        for i in range(1, Dy):
            for j in range(1, Dz):
                if len(slave_dofs) > 0:
                    slave_coord = [1, i/M, j/M]
                    slave_diff = np.abs(x_slaves-slave_coord).sum(axis=1)
                    if np.isclose(slave_diff[slave_diff.argmin()], 0):
                        # Only add if owned by processor
                        if (slave_dofs[slave_diff.argmin()]
                                < local_max - local_min):
                            slaves[(i-1)*(Dz-1)+j -
                                   1] = (slave_dofs[slave_diff.argmin()]
                                         + local_min)
                if len(master_dofs) > 0:
                    master_coord = [0, i/M, j/M]
                    master_diff = np.abs(x_masters-master_coord).sum(axis=1)
                    if np.isclose(master_diff[master_diff.argmin()], 0):
                        # Only add if owned by processor
                        if (master_dofs[master_diff.argmin()] <
                                local_max-local_min):
                            masters[(i-1)*(Dz-1)+j -
                                    1] = (master_dofs[master_diff.argmin()]
                                          + local_min)
                            master_rank[(i-1)*(Dz-1)+j -
                                        1] = MPI.COMM_WORLD.rank

        timer.stop()
        return slaves, masters, master_rank

    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Create master-slave map".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    snew, mnew, mrankloc = create_master_slave_map(master_dofs, slave_dofs)
    masters = sum(MPI.COMM_WORLD.allgather(mnew))
    slaves = sum(MPI.COMM_WORLD.allgather(snew))
    mrank = sum(MPI.COMM_WORLD.allgather(mrankloc))

    offsets = np.array(range(len(slaves)+1), dtype=np.int64)
    coeffs = 0.8*np.ones(len(masters), dtype=np.float64)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    dx = x[0] - 0.9
    dy = x[1] - 0.5
    dz = x[2] - 0.1
    f = x[0]*ufl.sin(5.0*ufl.pi*x[1]) \
        + 1.0*ufl.exp(-(dx*dx + dy*dy + dz*dz)/0.02)

    lhs = ufl.inner(f, v)*ufl.dx

    # Setup multi-point constraint
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets,
                                                   mrank)
    # Assemble LHS and RHS with multi-point constraint
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Assemble matrix".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    with dolfinx.common.Timer("MPC: Assemble matrix (Total time)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)

    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Assembling vector".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    with dolfinx.common.Timer("MPC: Assemble vector (Total time)"):
        b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Apply lifting".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    with dolfinx.common.Timer("MPC: Apply lifting"):
        dolfinx.fem.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, bcs)

    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Prepare solver".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    with dolfinx.common.Timer("MPC: Prepare solver"):
        # Create nullspace
        nullspace = PETSc.NullSpace().create(constant=True)
        PETSc.Mat.setNearNullSpace(A, nullspace)

        # Set PETSc solver options
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
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-12
            opts["pc_type"] = "gamg"
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_sym_graph"] = True

            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "richardson"
            opts["mg_levels_pc_type"] = "sor"
        # opts["help"] = None # List all available options
        # opts["ksp_view"] = None # List progress of solver

        # Create solver, set operator and options
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setFromOptions()
        solver.setOperators(A)

    # Solve linear problem
    uh = b.copy()
    uh.set(0)
    start = time.time()
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Solving".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    with dolfinx.common.Timer("MPC: Solve"):
        solver.solve(b, uh)
    end = time.time()
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run {0:1d}: Backsubstitute".format(r_lvl))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    # Back substitute to slave dofs
    with dolfinx.common.Timer("MPC: Backsubstitute"):
        dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    if kspview:
        solver.view()

    # Output information to HDF5
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

    # Output solution to XDMF
    if xdmf:
        if tetra:
            ext = "tet"
        else:
            ext = "hex"

        # Create function space with correct index map for MPC
        Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                      mpc.mpc_dofmap())
        Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
        u_h = dolfinx.Function(Vmpc)
        u_h.vector.setArray(uh.array)

        # Name formatting of functions
        mesh.name = "mesh_" + ext
        u_h.name = "u_" + ext

        fname = "results/bench_periodic3d_{0:d}_{1:s}.xdmf".format(r_lvl, ext)
        out_xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w")
        out_xdmf.write_mesh(mesh)
        out_xdmf.write_function(u_h, 0.0,
                                "Xdmf/Domain/"
                                + "Grid[@Name='{0:s}'][1]"
                                .format(mesh.name))
        out_xdmf.close()


if __name__ == "__main__":
    # Set Argparser defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", default=1, type=np.int8, dest="n_ref",
                        help="Number of spatial refinements")
    parser.add_argument("--degree", default=1, type=np.int8, dest="degree",
                        help="CG Function space degree")
    parser.add_argument('--xdmf', action='store_true', dest="xdmf",
                        help="XDMF-output of function (Default false)")
    parser.add_argument('--timings', action='store_true', dest="timings",
                        help="List timings (Default false)")
    parser.add_argument('--kspview', action='store_true', dest="kspview",
                        help="View PETSc progress")
    parser.add_argument("-o", default='periodic_output.hdf5', dest="hdf5",
                        help="Name of HDF5 output file")
    ct_parser = parser.add_mutually_exclusive_group(required=False)
    ct_parser.add_argument('--tet', dest='tetra', action='store_true',
                           help="Tetrahedron elements")
    ct_parser.add_argument('--hex', dest='tetra', action='store_false',
                           help="Hexahedron elements")
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

    # Prepare output HDF5 file
    h5f = h5py.File(hdf5, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    h5f.create_dataset("num_slaves", (N,), dtype=np.int32)
    sd = h5f.create_dataset(
        "solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if boomeramg else "GAMG"
    ct = "Tet" if tetra else "Hex"
    sd.attrs["solver"] = np.string_(solver)
    sd.attrs["degree"] = np.string_(str(int(degree)))
    sd.attrs["ct"] = np.string_(ct)

    # Loop over refinements
    for i in range(N):
        if MPI.COMM_WORLD.rank == 0:
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                            "Run {0:1d} in progress".format(i))
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        demo_periodic3D(tetra, r_lvl=i, out_hdf5=h5f,
                        xdmf=xdmf, boomeramg=boomeramg, kspview=kspview,
                        degree=int(degree))

        # List_timings
        if timings and i == N-1:
            dolfinx.common.list_timings(
                MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    h5f.close()


# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
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
import ufl
from dolfinx.mesh import refine
from mpi4py import MPI


def reference_periodic(tetra, out_xdmf=None, r_lvl=0, out_hdf5=None,
                       xdmf=False, boomeramg=False, kspview=False):
    # Create mesh and finite element
    if tetra:
        # Tet setup
        N = 3
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        for i in range(r_lvl):
            mesh = refine(mesh, redistribute=True)
            N *= 2
        V = dolfinx.FunctionSpace(mesh, ("CG", 2))
        M = 2*N
    else:
        # Hex setup
        N = 3
        V = dolfinx.FunctionSpace(mesh, ("CG", 1))
        for i in range(r_lvl):
            N *= 2
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N,
                                    dolfinx.cpp.mesh.CellType.hexahedron)
        M = N

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

    # Assemble LHS, RHS and apply lifting
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

    # Create PETSc nullspace
    nullspace = PETSc.NullSpace().create(constant=True)
    PETSc.Mat.setNearNullSpace(A_org, nullspace)

    # Set PETSc options
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

    # Initialize PETSc solver, set options and operator
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A_org)

    # Solve linear problem
    u_ = dolfinx.Function(V)
    start = time.time()
    with dolfinx.common.Timer("Solve") as t:
        solver.solve(L_org, u_.vector)
    end = time.time()
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    if kspview:
        solver.view()

    it = solver.getIterationNumber()
    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = V.dim
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

        fname = "results/reference_periodic_{0:d}_{1:s}.xdmf".format(i, ext)
        u_.name = "u_" + ext + "_unconstrained"
        out_periodic = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                           fname, "w")
        out_periodic.write_function(u_, 0.0,
                                    "Xdmf/Domain/"
                                    + "Grid[@Name='{0:s}'][1]"
                                    .format(mesh.name))
        out_periodic.close()


if __name__ == "__main__":
    # Set Argparser defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", default=1, type=np.int8, dest="n_ref",
                        help="Number of spatial refinements")
    parser.add_argument('--xdmf', action='store_true', dest="xdmf",
                        help="XDMF-output of function (Default false)")
    parser.add_argument('--timings', action='store_true', dest="timings",
                        help="List timings (Default false)")
    parser.add_argument('--kspview', action='store_true', dest="kspview",
                        help="View PETSc progress")
    parser.add_argument("-o", default='periodic_ref_output.hdf5', dest="hdf5",
                        help="Name of HDF5 output file")
    ct_parser = parser.add_mutually_exclusive_group(required=False)
    ct_parser.add_argument('--tet', dest='tetra', action='store_true',
                           help="Tetrahedron elements", default=True)
    ct_parser.add_argument('--hex', dest='tetra', action='store_false',
                           help="Hexahedron elements")
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

    h5f = h5py.File('periodic_ref_output.hdf5', 'w',
                    driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
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

        reference_periodic(tetra, r_lvl=i, out_hdf5=h5f,
                           xdmf=xdmf, boomeramg=boomeramg, kspview=kspview)

        if timings:
            dolfinx.common.list_timings(
                MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    h5f.close()

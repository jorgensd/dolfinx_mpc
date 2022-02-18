
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
# SPDX-License-Identifier:    MIT


import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import perf_counter

import h5py
import numpy as np
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Function, FunctionSpace, dirichletbc, form,
                         locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile
from dolfinx.log import LogLevel, log, set_log_level
from dolfinx.mesh import CellType, create_unit_cube, refine
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad,
                 inner, pi, sin)


def reference_periodic(tetra: bool, r_lvl: int = 0, out_hdf5: h5py.File = None,
                       xdmf: bool = False, boomeramg: bool = False, kspview: bool = False,
                       degree: int = 1):
    # Create mesh and finite element
    if tetra:
        # Tet setup
        N = 3
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
        for i in range(r_lvl):
            mesh.topology.create_entities(mesh.topology.dim - 2)
            mesh = refine(mesh, redistribute=True)
            N *= 2
    else:
        # Hex setup
        N = 3
        for i in range(r_lvl):
            N *= 2
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)

    V = FunctionSpace(mesh, ("CG", degree))

    # Create Dirichlet boundary condition

    def dirichletboundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0),
                                           np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0),
                                           np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = locate_dofs_geometrical(V, dirichletboundary)
    bc = dirichletbc(PETSc.ScalarType(0), geometrical_dofs, V)
    bcs = [bc]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    x = SpatialCoordinate(mesh)
    dx_ = x[0] - 0.9
    dy_ = x[1] - 0.5
    dz_ = x[2] - 0.1
    f = x[0] * sin(5.0 * pi * x[1]) + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) / 0.02)
    rhs = inner(f, v) * dx

    # Assemble rhs, RHS and apply lifting
    bilinear_form = form(a)
    linear_form = form(rhs)
    A_org = assemble_matrix(bilinear_form, bcs)
    A_org.assemble()
    L_org = assemble_vector(linear_form)
    apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L_org, bcs)

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
    u_ = Function(V)
    start = perf_counter()
    with Timer("Solve"):
        solver.solve(L_org, u_.vector)
    end = perf_counter()
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    if kspview:
        solver.view()

    it = solver.getIterationNumber()
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = num_dofs
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end - start

    if MPI.COMM_WORLD.rank == 0:
        print("Rlvl {0:d}, Iterations {1:d}".format(r_lvl, it))

    # Output solution to XDMF
    if xdmf:
        ext = "tet" if tetra else "hex"
        fname = "results/reference_periodic_{0:d}_{1:s}.xdmf".format(
            r_lvl, ext)
        u_.name = "u_" + ext + "_unconstrained"
        with XDMFFile(MPI.COMM_WORLD, fname, "w") as out_periodic:
            out_periodic.write_mesh(mesh)
            out_periodic.write_function(u_, 0.0,
                                        "Xdmf/Domain/"
                                        + "Grid[@Name='{0:s}'][1]"
                                        .format(mesh.name))


if __name__ == "__main__":
    # Set Argparser defaults
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
    solver_parser.add_argument('--gamg', dest='boomeramg',
                               action='store_false',
                               help="Use PETSc GAMG preconditioner")

    args = parser.parse_args()
    thismodule = sys.modules[__name__]
    n_ref = timings = boomeramg = kspview = degree = hdf5 = xdmf = tetra = None

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
    ct = "Tet" if tetra else "Hex"
    sd.attrs["solver"] = np.string_(solver)
    sd.attrs["degree"] = np.string_(str(int(degree)))
    sd.attrs["ct"] = np.string_(ct)
    for i in range(N):
        if MPI.COMM_WORLD.rank == 0:
            set_log_level(LogLevel.INFO)
            log(LogLevel.INFO, "Run {0:1d} in progress".format(i))
            set_log_level(LogLevel.ERROR)

        reference_periodic(tetra, r_lvl=i, out_hdf5=h5f,
                           xdmf=xdmf, boomeramg=boomeramg, kspview=kspview,
                           degree=int(degree))

        if timings and i == N - 1:
            list_timings(MPI.COMM_WORLD, [TimingType.wall])
    h5f.close()

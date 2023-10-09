# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import resource
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import Optional

import basix.ufl
import h5py
import numpy as np
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Constant, Function, dirichletbc, form, functionspace,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile
from dolfinx.log import LogLevel, log, set_log_level
from dolfinx.mesh import (CellType, create_unit_cube, locate_entities_boundary,
                          meshtags, refine)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, ds, dx, grad, inner, sym, tr)

from dolfinx_mpc.utils import rigid_motions_nullspace


def ref_elasticity(tetra: bool = True, r_lvl: int = 0, out_hdf5: Optional[h5py.File] = None,
                   xdmf: bool = False, boomeramg: bool = False, kspview: bool = False, degree: int = 1):
    if tetra:
        N = 3 if degree == 1 else 2
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    else:
        N = 3
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
    for i in range(r_lvl):
        # set_log_level(LogLevel.INFO)
        N *= 2
        if tetra:
            mesh = refine(mesh, redistribute=True)
        else:
            mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
        # set_log_level(LogLevel.ERROR)
    N = degree * N
    fdim = mesh.topology.dim - 1
    el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, el)

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    u_bc.vector.destroy()

    def boundaries(x):
        return np.isclose(x[0], np.finfo(float).eps)
    facets = locate_entities_boundary(mesh, fdim, boundaries)
    topological_dofs = locate_dofs_topological(V, fdim, facets)
    bc = dirichletbc(u_bc, topological_dofs)
    bcs = [bc]

    # Create traction meshtag
    def traction_boundary(x):
        return np.isclose(x[0], 1)
    t_facets = locate_entities_boundary(mesh, fdim, traction_boundary)
    facet_values = np.ones(len(t_facets), dtype=np.int32)
    arg_sort = np.argsort(t_facets)
    mt = meshtags(mesh, fdim, t_facets[arg_sort], facet_values[arg_sort])

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e4)
    nu = 0.1
    mu = Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    g = Constant(mesh, PETSc.ScalarType((0, 0, -1e2)))
    x = SpatialCoordinate(mesh)
    f = Constant(mesh, PETSc.ScalarType(1e4)) * \
        as_vector((0, -(x[2] - 0.5)**2, (x[1] - 0.5)**2))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v)))

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    rhs = inner(g, v) * ds(domain=mesh, subdomain_data=mt, subdomain_id=1) + inner(f, v) * dx

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if MPI.COMM_WORLD.rank == 0:
        print("Problem size {0:d} ".format(num_dofs))

    # Generate reference matrices and unconstrained solution
    bilinear_form = form(a)
    A_org = assemble_matrix(bilinear_form, bcs)
    A_org.assemble()
    null_space_org = rigid_motions_nullspace(V)
    A_org.setNearNullSpace(null_space_org)

    linear_form = form(rhs)
    L_org = assemble_vector(linear_form)
    apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L_org, bcs)
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
        opts["pc_gamg_square_graph"] = 2
        opts["pc_gamg_threshold"] = 0.02
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    # Create solver, set operator and options
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A_org)

    # Solve linear problem
    u_ = Function(V)
    start = perf_counter()
    with Timer("Ref solve"):
        solver.solve(L_org, u_.vector)
    end = perf_counter()
    u_.x.scatter_forward()

    if kspview:
        solver.view()

    it = solver.getIterationNumber()
    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = num_dofs
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = end - start

    if MPI.COMM_WORLD.rank == 0:
        print("Refinement level {0:d}, Iterations {1:d}".format(r_lvl, it))

    # List memory usage
    mem = sum(MPI.COMM_WORLD.allgather(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if MPI.COMM_WORLD.rank == 0:
        print("{1:d}: Max usage after trad. solve {0:d} (kb)"
              .format(mem, r_lvl))

    if xdmf:

        # Name formatting of functions
        u_.name = "u_unconstrained"
        outdir = Path("results")
        outdir.mkdir(exist_ok=True, parents=True)

        fname = outdir / "ref_elasticity_{0:d}.xdmf".format(r_lvl)
        with XDMFFile(MPI.COMM_WORLD, fname, "w") as out_xdmf:
            out_xdmf.write_mesh(mesh)
            out_xdmf.write_function(u_, 0.0, "Xdmf/Domain/Grid[@Name='{0:s}'][1]".format(mesh.name))


if __name__ == "__main__":
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
    parser.add_argument("-o", default='elasticity_ref.hdf5', dest="hdf5",
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

    N = args.n_ref + 1

    # Setup hd5f output file
    h5f = h5py.File(args.hdf5, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    sd = h5f.create_dataset("solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if args.boomeramg else "GAMG"
    ct = "Tet" if args.tetra else "Hex"
    sd.attrs["solver"] = np.string_(solver)
    sd.attrs["degree"] = np.string_(str(int(args.degree)))
    sd.attrs["ct"] = np.string_(ct)

    # Loop over refinement levels
    for i in range(N):
        if MPI.COMM_WORLD.rank == 0:
            set_log_level(LogLevel.INFO)
            log(LogLevel.INFO,
                "Run {0:1d} in progress".format(i))
            set_log_level(LogLevel.ERROR)
        ref_elasticity(tetra=args.tetra, r_lvl=i, out_hdf5=h5f,
                       xdmf=args.xdmf, boomeramg=args.boomeramg, kspview=args.kspview,
                       degree=args.degree)
        if args.timings and i == N - 1:
            list_timings(MPI.COMM_WORLD, [TimingType.wall])
    h5f.close()

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import resource
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import basix.ufl
import h5py
import numpy as np
from dolfinx import default_scalar_type
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc, form,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, create_unit_cube, locate_entities_boundary,
                          meshtags)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, ds, dx, grad, inner, sym, tr)

from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)
from dolfinx_mpc.utils import log_info, rigid_motions_nullspace


def bench_elasticity_edge(tetra: bool = True, r_lvl: int = 0, out_hdf5=None, xdmf: bool = False,
                          boomeramg: bool = False, kspview: bool = False, degree: int = 1, info: bool = False):
    N = 3
    for i in range(r_lvl):
        N *= 2
    ct = CellType.tetrahedron if tetra else CellType.hexahedron
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, ct)

    el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), int(degree), shape=(mesh.geometry.dim,))
    V = functionspace(mesh, el)

    # Generate Dirichlet BC (Fixed)
    u_bc = Function(V)
    u_bc.x.array[:] = 0

    def boundaries(x):
        return np.isclose(x[0], 0, 500 * np.finfo(x.dtype).resolution)
    fdim = mesh.topology.dim - 1
    facets = locate_entities_boundary(mesh, fdim, boundaries)
    topological_dofs = locate_dofs_topological(V, fdim, facets)
    bc = dirichletbc(u_bc, topological_dofs)
    bcs = [bc]

    def PeriodicBoundary(x):
        return np.logical_and(np.isclose(x[0], 1), np.isclose(x[2], 0))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = x[0]
        out_x[1] = x[1]
        out_x[2] = x[2] + 1
        return out_x
    with Timer("~Elasticity: Initialize MPC"):
        edim = mesh.topology.dim - 2
        edges = locate_entities_boundary(mesh, edim, PeriodicBoundary)
        arg_sort = np.argsort(edges)
        periodic_mt = meshtags(mesh, edim, edges[arg_sort], np.full(len(edges), 2, dtype=np.int32))

        mpc = MultiPointConstraint(V)
        mpc.create_periodic_constraint_topological(V, periodic_mt, 2, periodic_relation, bcs,
                                                   scale=default_scalar_type(0.5))
        mpc.finalize()

    # Create traction meshtag

    def traction_boundary(x):
        return np.isclose(x[0], 1)
    t_facets = locate_entities_boundary(mesh, fdim, traction_boundary)
    facet_values = np.ones(len(t_facets), dtype=np.int32)
    arg_sort = np.argsort(t_facets)
    mt = meshtags(mesh, fdim, t_facets[arg_sort], facet_values)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.1
    mu = Constant(mesh, default_scalar_type(E / (2.0 * (1.0 + nu))))
    lmbda = Constant(mesh, default_scalar_type(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    g = Constant(mesh, default_scalar_type((0, 0, -1e2)))
    x = SpatialCoordinate(mesh)
    f = Constant(mesh, default_scalar_type(1e3)) * as_vector((0, -(x[2] - 0.5)**2, (x[1] - 0.5)**2))

    # Stress computation
    def epsilon(v):
        return sym(grad(v))

    def sigma(v):
        return (2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(len(v)))

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    rhs = inner(g, v) * ds(domain=mesh, subdomain_data=mt, subdomain_id=1) + inner(f, v) * dx

    # Setup MPC system
    if info:
        log_info(f"Run {r_lvl}: Assembling matrix and vector")
    bilinear_form = form(a)
    linear_form = form(rhs)
    with Timer("~Elasticity: Assemble LHS and RHS"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
        b = assemble_vector(linear_form, mpc)

    # Create nullspace for elasticity problem and assign to matrix
    null_space = rigid_motions_nullspace(mpc.function_space)
    A.setNearNullSpace(null_space)

    # Apply boundary conditions
    apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, bcs)

    opts = PETSc.Options()  # type: ignore
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

    # Setup PETSc solver
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    solver.setFromOptions()  # type: ignore

    if info:
        log_info(f"Run {r_lvl}: Solving")

    with Timer("~Elasticity: Solve problem") as timer:
        solver.setOperators(A)
        uh = Function(mpc.function_space)
        uh.x.array[:] = 0
        solver.solve(b, uh.vector)
        uh.x.scatter_forward()
        mpc.backsubstitution(uh)
        solver_time = timer.elapsed()
    if kspview:
        solver.view()

    mem = sum(MPI.COMM_WORLD.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    it = solver.getIterationNumber()

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = num_dofs
        d_set = out_hdf5.get("num_slaves")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = mpc.num_local_slaves
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = solver_time[0]
    if info:
        log_info(f"Lvl: {r_lvl}, Its: {it}, max Mem: {mem}, dim(V): {num_dofs}")

    if xdmf:
        # Write solution to file
        u_h = Function(mpc.function_space)
        u_h.vector.setArray(uh.array)
        u_h.name = "u_mpc"
        results = Path("results").absolute()
        results.mkdir(exist_ok=True)
        fname = results / f"bench_elasticity_edge_{r_lvl}.xdmf"
        with XDMFFile(mesh.comm, fname, "w") as outfile:
            outfile.write_mesh(mesh)
            outfile.write_function(u_h)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nref", default=1, type=np.int8, dest="n_ref", help="Number of spatial refinements")
    parser.add_argument("--degree", default=1, type=np.int8, dest="degree", help="CG Function space degree")
    parser.add_argument('--xdmf', action='store_true', dest="xdmf", help="XDMF-output of function (Default false)")
    parser.add_argument('--timings', action='store_true', dest="timings", help="List timings (Default false)")
    parser.add_argument('--info', action='store_true', dest="info",
                        help="Set loglevel to info (Default false)", default=False)
    parser.add_argument('--kspview', action='store_true', dest="kspview", help="View PETSc progress")
    ct_parser = parser.add_mutually_exclusive_group(required=False)
    ct_parser.add_argument('--tet', dest='tetra', action='store_true', help="Tetrahedron elements")
    ct_parser.add_argument('--hex', dest='tetra', action='store_false', help="Hexahedron elements")
    solver_parser = parser.add_mutually_exclusive_group(required=False)
    solver_parser.add_argument('--boomeramg', dest='boomeramg', default=True, action='store_true',
                               help="Use BoomerAMG preconditioner (Default)")
    solver_parser.add_argument('--gamg', dest='boomeramg', action='store_false',
                               help="Use PETSc GAMG preconditioner")
    args = parser.parse_args()
    N = args.n_ref + 1
    out_file = Path('output/ench_edge_output.hdf5').absolute()
    out_file.parent.mkdir(exist_ok=True)
    h5f = h5py.File(out_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    h5f.create_dataset("num_slaves", (N, MPI.COMM_WORLD.size), dtype=np.int32)
    sd = h5f.create_dataset("solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if args.boomeramg else "GAMG"
    ct = "Tet" if args.tetra else "Hex"
    sd.attrs["solver"] = np.string_(solver)
    sd.attrs["degree"] = np.string_(str(int(args.degree)))
    sd.attrs["ct"] = np.string_(ct)

    for i in range(N):
        log_info(f"Run {i} in progress")
        bench_elasticity_edge(tetra=args.tetra, r_lvl=i, out_hdf5=h5f, xdmf=args.xdmf,
                              boomeramg=args.boomeramg, kspview=args.kspview,
                              degree=args.degree, info=args.info)

        if args.timings and i == N - 1:
            list_timings(MPI.COMM_WORLD, [TimingType.wall])
    h5f.close()

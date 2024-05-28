# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

import resource
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Optional

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import h5py
import numpy as np
from dolfinx import default_real_type, default_scalar_type
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_cube, locate_entities_boundary, meshtags, refine
from ufl import Identity, TestFunction, TrialFunction, ds, dx, grad, inner, sym, tr

from dolfinx_mpc import MultiPointConstraint, apply_lifting, assemble_matrix, assemble_vector
from dolfinx_mpc.utils import log_info, rigid_motions_nullspace


def bench_elasticity_one(
    r_lvl: int = 0,
    out_hdf5: Optional[h5py.File] = None,
    xdmf: bool = False,
    boomeramg: bool = False,
    kspview: bool = False,
):
    N = 3
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    for i in range(r_lvl):
        mesh.topology.create_entities(mesh.topology.dim - 2)
        mesh = refine(mesh, redistribute=True)

    fdim = mesh.topology.dim - 1
    el = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    V = functionspace(mesh, el)

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = Function(V)
    with u_bc.x.petsc_vec.localForm() as u_local:
        u_local.set(0.0)
    u_bc.x.petsc_vec.destroy()

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

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.1
    mu = Constant(mesh, default_scalar_type(E / (2.0 * (1.0 + nu))))
    lmbda = Constant(mesh, default_scalar_type(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    g = Constant(mesh, (0, 0, -1e2))

    # Stress computation
    def sigma(v):
        return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v))

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    rhs = inner(g, v) * ds(domain=mesh, subdomain_data=mt, subdomain_id=1)

    # Create MPC
    with Timer("~Elasticity: Init constraint"):

        def l2b(li):
            return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

        s_m_c = {l2b([1, 0, 0]): {l2b([1, 0, 1]): 0.5}}
        mpc = MultiPointConstraint(V)
        mpc.create_general_constraint(s_m_c, 2, 2)
        mpc.finalize()

    # Setup MPC system
    bilinear_form = form(a)
    linear_form = form(rhs)
    with Timer("~Elasticity: Assemble LHS and RHS"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
        b = assemble_vector(linear_form, mpc)
    # Apply boundary conditions
    apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, bcs)

    # Create functionspace and function for mpc vector

    # Solve Linear problem
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    opts = PETSc.Options()  # type: ignore
    if boomeramg:
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-5
        opts["pc_type"] = "hypre"
        opts["pc_hypre_type"] = "boomeramg"
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

    with Timer("~Elasticity: Solve problem") as timer:
        null_space = rigid_motions_nullspace(mpc.function_space)
        A.setNearNullSpace(null_space)
        solver.setFromOptions()
        solver.setOperators(A)
        # Solve linear problem
        uh = b.copy()
        uh.set(0)
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        mpc.backsubstitution(uh)
        solver_time = timer.elapsed()

    it = solver.getIterationNumber()
    if kspview:
        solver.view()

    # Print max usage of summary
    mem = sum(MPI.COMM_WORLD.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if MPI.COMM_WORLD.rank == 0:
        print(f"Rlvl {r_lvl}, Iterations {it}")
        print(f"Rlvl {r_lvl}, Max usage {mem} (kb), #dofs {num_dofs}")

    if out_hdf5 is not None:
        d_set = out_hdf5.get("its")
        d_set[r_lvl] = it
        d_set = out_hdf5.get("num_dofs")
        d_set[r_lvl] = num_dofs
        d_set = out_hdf5.get("solve_time")
        d_set[r_lvl, MPI.COMM_WORLD.rank] = solver_time[0]
    if xdmf:
        # Write solution to file
        u_h = Function(mpc.function_space)
        u_h.x.petsc_vec.setArray(uh.array)
        u_h.name = "u_mpc"
        outdir = Path("results")
        outdir.mkdir(exist_ok=True, parents=True)
        fname = outdir / f"bench_elasticity_{r_lvl}.xdmf"
        with XDMFFile(mesh.comm, fname, "w") as outfile:
            outfile.write_mesh(mesh)
            outfile.write_function(u_h)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nref", default=1, type=int, dest="n_ref", help="Number of spatial refinements")
    parser.add_argument("--xdmf", action="store_true", dest="xdmf", help="XDMF-output of function (Default false)")
    parser.add_argument("--timings", action="store_true", dest="timings", help="List timings (Default false)")
    parser.add_argument("--kspview", action="store_true", dest="kspview", help="View PETSc progress")
    parser.add_argument("-o", default="elasticity_one.hdf5", dest="hdf5", help="Name of HDF5 output file")
    solver_parser = parser.add_mutually_exclusive_group(required=False)
    solver_parser.add_argument(
        "--boomeramg",
        dest="boomeramg",
        default=True,
        action="store_true",
        help="Use BoomerAMG preconditioner (Default)",
    )
    solver_parser.add_argument("--gamg", dest="boomeramg", action="store_false", help="Use PETSc GAMG preconditioner")
    args = parser.parse_args()

    N = args.n_ref + 1

    # Setup hd5f output file
    h5f = h5py.File(args.hdf5, "w", driver="mpio", comm=MPI.COMM_WORLD)
    h5f.create_dataset("its", (N,), dtype=np.int32)
    h5f.create_dataset("num_dofs", (N,), dtype=np.int32)
    sd = h5f.create_dataset("solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if args.boomeramg else "GAMG"
    sd.attrs["solver"] = np.string_(solver)

    # Loop over refinement levels
    for i in range(N):
        log_info(f"Run {i} in progress")
        bench_elasticity_one(r_lvl=i, out_hdf5=h5f, xdmf=args.xdmf, boomeramg=args.boomeramg, kspview=args.kspview)
        if args.timings and i == N - 1:
            list_timings(MPI.COMM_WORLD, [TimingType.wall])
    h5f.close()

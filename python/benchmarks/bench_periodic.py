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
# SPDX-License-Identifier:    MIT


import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import h5py
import numpy as np
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (DirichletBC, Function, FunctionSpace,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, MeshTags, create_unit_cube,
                          locate_entities_boundary)
from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)
from dolfinx_mpc.utils import log_info
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad,
                 inner, pi, sin)


def demo_periodic3D(tetra, r_lvl=0, out_hdf5=None,
                    xdmf=False, boomeramg=False, kspview=False, degree=1):
    # Create mesh and function space
    log_info(f"Run {r_lvl}: Create mesh")
    ct = CellType.tetrahedron if tetra else CellType.hexahedron
    # Tet setup
    N = 3
    for i in range(r_lvl):
        N *= 2
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, ct)

    V = FunctionSpace(mesh, ("CG", degree))

    # Create Dirichlet boundary condition
    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def dirichletboundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = locate_dofs_geometrical(V, dirichletboundary)
    bc = DirichletBC(u_bc, geometrical_dofs)
    bcs = [bc]

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    log_info(f"Run {r_lvl}: Create MultiPoint Constraint {num_dofs}")
    with Timer("~Periodic: Initialize periodic constraint"):
        facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
        mt = MeshTags(mesh, mesh.topology.dim - 1, facets, np.full(len(facets), 2, dtype=np.int32))
        mpc = MultiPointConstraint(V)
        mpc.create_periodic_constraint_topological(mt, 2, periodic_relation, bcs)
        mpc.finalize()

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    x = SpatialCoordinate(mesh)
    dx_ = x[0] - 0.9
    dy_ = x[1] - 0.5
    dz_ = x[2] - 0.1
    f = x[0] * sin(5.0 * pi * x[1]) + 1.0 * exp(-(dx_**2 + dy_**2 + dz_**2) / 0.02)

    rhs = inner(f, v) * dx

    # Assemble LHS and RHS with multi-point constraint

    log_info(f"Run {r_lvl}: Assemble matrix")
    with Timer(f"~Periodic {r_lvl}: Assemble matrix"):
        A = assemble_matrix(a, mpc, bcs=bcs)
    with Timer(f"~Periodic {r_lvl}: Assemble matrix (cached)"):
        A = assemble_matrix(a, mpc, bcs=bcs)

    log_info(f"Run {r_lvl}: Assembling vector")
    with Timer(f"~Periodic: {r_lvl} Assemble vector (Total time)"):
        b = assemble_vector(rhs, mpc)

    # Apply boundary conditions
    log_info(f"Run {r_lvl}: Apply lifting")
    apply_lifting(b, [a], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Create nullspace
    nullspace = PETSc.NullSpace().create(constant=True)

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

    # Solve linear problem
    log_info(f"Run {r_lvl}: Solving")
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    with Timer("~Periodic: Solve") as timer:
        # Create solver, set operator and options
        PETSc.Mat.setNearNullSpace(A, nullspace)
        uh = b.copy()
        solver.setFromOptions()
        solver.setOperators(A)
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        mpc.backsubstitution(uh)
        solver_time = timer.elapsed()
        if kspview:
            solver.view()

    # Output information to HDF5
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

    if MPI.COMM_WORLD.rank == 0:
        print(f"Rlvl {r_lvl}, Iterations {it}")

    # Output solution to XDMF
    if xdmf:
        # Create function space with correct index map for MPC
        u_h = Function(mpc.function_space)
        u_h.vector.setArray(uh.array)

        # Name formatting of functions
        ext = "tet" if tetra else "hex"
        mesh.name = f"mesh_{ext}"
        u_h.name = f"u_{ext}"
        fname = f"results/bench_periodic3d_{r_lvl}_{ext}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fname, "w") as out_xdmf:
            out_xdmf.write_mesh(mesh)
            out_xdmf.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")


if __name__ == "__main__":
    # Set Argparser defaults
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nref", default=1, type=np.int8, dest="n_ref", help="Number of spatial refinements")
    parser.add_argument("--degree", default=1, type=np.int8, dest="degree", help="CG Function space degree")
    parser.add_argument('--xdmf', action='store_true', dest="xdmf",
                        help="XDMF-output of function (Default false)")
    parser.add_argument('--timings', action='store_true', dest="timings", help="List timings (Default false)")
    parser.add_argument('--kspview', action='store_true', dest="kspview", help="View PETSc progress")
    parser.add_argument("-o", default='periodic_output.hdf5', dest="hdf5", help="Name of HDF5 output file")
    ct_parser = parser.add_mutually_exclusive_group(required=False)
    ct_parser.add_argument('--tet', dest='tetra', action='store_true', help="Tetrahedron elements")
    ct_parser.add_argument('--hex', dest='tetra', action='store_false', help="Hexahedron elements")
    solver_parser = parser.add_mutually_exclusive_group(required=False)
    solver_parser.add_argument('--boomeramg', dest='boomeramg', default=True,
                               action='store_true', help="Use BoomerAMG preconditioner (Default)")
    solver_parser.add_argument('--gamg', dest='boomeramg', action='store_false',
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
    h5f.create_dataset("num_slaves", (N, MPI.COMM_WORLD.size), dtype=np.int32)
    sd = h5f.create_dataset("solve_time", (N, MPI.COMM_WORLD.size), dtype=np.float64)
    solver = "BoomerAMG" if boomeramg else "GAMG"
    ct = "Tet" if tetra else "Hex"
    sd.attrs["solver"] = np.string_(solver)
    sd.attrs["degree"] = np.string_(str(int(degree)))
    sd.attrs["ct"] = np.string_(ct)

    # Loop over refinements
    for i in range(N):
        log_info(f"Run {i} in progress")
        demo_periodic3D(tetra, r_lvl=i, out_hdf5=h5f, xdmf=xdmf,
                        boomeramg=boomeramg, kspview=kspview, degree=int(degree))

        # List_timings
        if timings and i == N - 1:
            list_timings(MPI.COMM_WORLD, [TimingType.wall])
    h5f.close()

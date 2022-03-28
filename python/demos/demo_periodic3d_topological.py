
# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
#
# Copyright (C) Jørgen S. Dokken 2020-2022.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as fem
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.io import VTXWriter
from dolfinx.mesh import (CellType, create_unit_cube, locate_entities_boundary,
                          meshtags)
from dolfinx_mpc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector, dx,
                 exp, grad, inner, pi, sin)

# Get PETSc int and scalar types
complex_mode = True if np.dtype(PETSc.ScalarType).kind == 'c' else False


def demo_periodic3D(celltype):
    # Create mesh and finite element
    if celltype == CellType.tetrahedron:
        # Tet setup
        N = 10
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
        V = fem.VectorFunctionSpace(mesh, ("CG", 1))
    else:
        # Hex setup
        N = 10
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
        V = fem.VectorFunctionSpace(mesh, ("CG", 2))

    def dirichletboundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], 1)))

    # Create Dirichlet boundary condition
    zero = PETSc.ScalarType([0, 0, 0])
    geometrical_dofs = fem.locate_dofs_geometrical(V, dirichletboundary)
    bc = fem.dirichletbc(zero, geometrical_dofs, V)
    bcs = [bc]

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
    arg_sort = np.argsort(facets)
    mt = meshtags(mesh, mesh.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x
    with Timer("~~Periodic: Compute mpc condition"):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_topological(V.sub(0), mt, 2, periodic_relation, bcs, 1)
        mpc.finalize()
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    x = SpatialCoordinate(mesh)
    dx_ = x[0] - 0.9
    dy_ = x[1] - 0.5
    dz_ = x[2] - 0.1
    f = as_vector((x[0] * sin(5.0 * pi * x[1])
                   + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) / 0.02), 0.1 * dx_ * dz_, 0.1 * dx_ * dy_))

    rhs = inner(f, v) * dx

    if complex_mode:
        rtol = 1e-16
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    else:
        rtol = 1e-8
        petsc_options = {"ksp_type": "cg", "ksp_rtol": rtol, "pc_type": "hypre", "pc_hypre_typ": "boomeramg",
                         "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v",
                         "pc_hypre_boomeramg_print_statistics": 1}
    problem = LinearProblem(a, rhs, mpc, bcs, petsc_options=petsc_options)
    u_h = problem.solve()

    # --------------------VERIFICATION-------------------------
    print("----Verification----")
    u_ = fem.Function(V)
    u_.x.array[:] = 0
    org_problem = fem.petsc.LinearProblem(a, rhs, u=u_, bcs=bcs, petsc_options=petsc_options)
    with Timer("~Periodic: Unconstrained solve"):
        org_problem.solve()
        it = org_problem.solver.getIterationNumber()
    print(f"Unconstrained solver iterations: {it}")

    # Write solutions to file
    ext = "tet" if celltype == CellType.tetrahedron else "hex"
    u_.name = "u_" + ext + "_unconstrained"

    # NOTE: Workaround as tabulate dof coordinates does not like extra ghosts
    u_out = fem.Function(V)
    old_local = u_out.x.map.size_local * u_out.x.bs
    old_ghosts = u_out.x.map.num_ghosts * u_out.x.bs
    mpc_local = u_h.x.map.size_local * u_h.x.bs
    assert(old_local == mpc_local)
    u_out.x.array[:old_local + old_ghosts] = u_h.x.array[:mpc_local + old_ghosts]
    u_out.name = "u_" + ext
    fname = f"results/demo_periodic3d_{ext}.bp"
    out_periodic = VTXWriter(MPI.COMM_WORLD, fname, u_out)
    out_periodic.write(0)
    out_periodic.close()

    root = 0
    with Timer("~Demo: Verification"):
        dolfinx_mpc.utils.compare_mpc_lhs(org_problem.A, problem.A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(org_problem.b, problem.b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(org_problem.A, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(org_problem.b, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(u_h.vector, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc, rtol=rtol)


if __name__ == "__main__":
    for celltype in [CellType.hexahedron, CellType.tetrahedron]:
        demo_periodic3D(celltype)
    list_timings(MPI.COMM_WORLD, [TimingType.wall])

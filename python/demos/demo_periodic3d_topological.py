
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
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as fem
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, MeshTags, create_unit_cube,
                          locate_entities_boundary)
from dolfinx_mpc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad,
                 inner, pi, sin)

# Get PETSc int and scalar types
complex_mode = True if np.dtype(PETSc.ScalarType).kind == 'c' else False


def demo_periodic3D(celltype, out_periodic):
    # Create mesh and finite element
    if celltype == CellType.tetrahedron:
        # Tet setup
        N = 4
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
        V = fem.FunctionSpace(mesh, ("CG", 2))
    else:
        # Hex setup
        N = 12
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
        V = fem.FunctionSpace(mesh, ("CG", 1))

    # Create Dirichlet boundary condition
    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def dirichletboundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = fem.locate_dofs_geometrical(V, dirichletboundary)
    bc = fem.DirichletBC(u_bc, geometrical_dofs)
    bcs = [bc]

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
    mt = MeshTags(mesh, mesh.topology.dim - 1, facets, np.full(len(facets), 2, dtype=np.int32))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x

    with Timer("~Periodic: New init"):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
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
    f = x[0] * sin(5.0 * pi * x[1]) \
        + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) / 0.02)

    rhs = inner(f, v) * dx

    if complex_mode:
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    else:
        petsc_options = {"ksp_type": "cg", "ksp_rtol": 1e-6, "pc_type": "hypre", "pc_hypre_typ": "boomeramg",
                         "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v",
                         "pc_hypre_boomeramg_print_statistics": 1}
    problem = LinearProblem(a, rhs, mpc, bcs, petsc_options=petsc_options)
    u_h = problem.solve()

    # Write solution to file
    if celltype == CellType.tetrahedron:
        ext = "tet"
    else:
        ext = "hex"

    mesh.name = "mesh_" + ext
    u_h.name = "u_" + ext

    out_periodic.write_mesh(mesh)
    out_periodic.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")

    # --------------------VERIFICATION-------------------------
    print("----Verification----")
    org_problem = fem.LinearProblem(a, rhs, bcs=bcs, petsc_options=petsc_options)
    with Timer("~Periodic: Unconstrained solve"):
        u_ = org_problem.solve()
        it = org_problem.solver.getIterationNumber()
    print(f"Unconstrained solver iterations: {it}")
    u_.name = "u_" + ext + "_unconstrained"
    out_periodic.write_function(u_, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")

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
            assert np.allclose(uh_numpy, u_mpc)


if __name__ == "__main__":
    fname = "results/demo_periodic3d.xdmf"
    out_periodic = XDMFFile(MPI.COMM_WORLD, fname, "w")
    for celltype in [CellType.tetrahedron, CellType.hexahedron]:
        demo_periodic3D(celltype, out_periodic)
    out_periodic.close()
    list_timings(MPI.COMM_WORLD, [TimingType.wall])

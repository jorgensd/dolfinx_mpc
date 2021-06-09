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
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

# Create mesh and finite element
N = 50
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = dolfinx.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def DirichletBoundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, DirichletBoundary)
topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
bcs = [bc]


def PeriodicBoundary(x):
    return np.isclose(x[0], 1)


facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, PeriodicBoundary)
mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, np.full(len(facets), 2, dtype=np.int32))


def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


with dolfinx.common.Timer("~PERIODIC: Initialize MPC"):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint(mt, 2, periodic_relation, bcs)
    mpc.finalize()

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

x = ufl.SpatialCoordinate(mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0] * ufl.sin(5.0 * ufl.pi * x[1]) + 1.0 * ufl.exp(-(dx * dx + dy * dy) / 0.02)

rhs = ufl.inner(f, v) * ufl.dx


# Setup MPC system
with dolfinx.common.Timer("~PERIODIC: Initialize varitional problem"):
    problem = dolfinx_mpc.LinearProblem(a, rhs, mpc, bcs=bcs)

solver = problem.solver

if complex:
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
else:
    opts = PETSc.Options()
    prefix = solver.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "cg"
    opts[f"{prefix}ksp_rtol"] = 1.0e-6
    opts[f"{prefix}pc_type"] = "hypre"
    opts[f'{prefix}pc_hypre_type'] = 'boomeramg'
    opts[f"{prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    # opts["pc_hypre_boomeramg_print_statistics"] = 1
    solver.setFromOptions()


with dolfinx.common.Timer("~PERIODIC: Assemble and solve MPC problem"):
    uh = problem.solve()
    solver.view()
    it = solver.getIterationNumber()
    print("Constrained solver iterations {0:d}".format(it))

# Write solution to file
uh.name = "u_mpc"
outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/demo_periodic.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(uh)


print("----Verification----")
# --------------------VERIFICATION-------------------------
A_org = dolfinx.fem.assemble_matrix(a, bcs)

A_org.assemble()
L_org = dolfinx.fem.assemble_vector(rhs)
dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver.setOperators(A_org)
u_ = dolfinx.Function(V)
solver.solve(L_org, u_.vector)

it = solver.getIterationNumber()
print("Unconstrained solver iterations {0:d}".format(it))
dolfinx.cpp.la.scatter_forward(u_.x)
u_.name = "u_unconstrained"
outfile.write_function(u_)

root = 0
comm = mesh.mpi_comm()
with dolfinx.common.Timer("~Demo: Verification"):
    dolfinx_mpc.utils.compare_MPC_LHS(A_org, problem._A, mpc, root=root)
    dolfinx_mpc.utils.compare_MPC_RHS(L_org, problem._b, mpc, root=root)

    # Gather LHS, RHS and solution on one process
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
    u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.vector, root=root)

    if MPI.COMM_WORLD.rank == root:
        KTAK = K.T * A_csr * K
        reduced_L = K.T @ L_np
        # Solve linear system
        d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
        # Back substitution to full solution vector
        uh_numpy = K @ d
        assert np.allclose(uh_numpy, u_mpc)
dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

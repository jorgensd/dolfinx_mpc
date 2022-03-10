# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
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
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector, dx,
                 exp, grad, inner, pi, sin)

# Get PETSc int and scalar types
complex_mode = True if np.dtype(PETSc.ScalarType).kind == 'c' else False

# Create mesh and finite element
NX = 50
NY = 100
mesh = create_unit_square(MPI.COMM_WORLD, NX, NY)
V = fem.VectorFunctionSpace(mesh, ("CG", 1))


def dirichletboundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


# Create Dirichlet boundary condition
facets = locate_entities_boundary(mesh, 1, dirichletboundary)
topological_dofs = fem.locate_dofs_topological(V, 1, facets)
zero = np.array([0, 0], dtype=PETSc.ScalarType)
bc = fem.dirichletbc(zero, topological_dofs, V)
bcs = [bc]


def periodic_boundary(x):
    return np.isclose(x[0], 1)


def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


with Timer("~PERIODIC: Initialize MPC"):
    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs)
    mpc.finalize()

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx

x = SpatialCoordinate(mesh)
dx_ = x[0] - 0.9
dy_ = x[1] - 0.5
f = as_vector((x[0] * sin(5.0 * pi * x[1]) + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_) / 0.02), 0.3 * x[1]))

rhs = inner(f, v) * dx


# Setup MPC system
with Timer("~PERIODIC: Initialize varitional problem"):
    problem = LinearProblem(a, rhs, mpc, bcs=bcs)

solver = problem.solver

# Give PETSc solver options a unique prefix
solver_prefix = "dolfinx_mpc_solve_{}".format(id(solver))
solver.setOptionsPrefix(solver_prefix)


if complex_mode:
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
else:
    petsc_options = {"ksp_type": "cg", "ksp_rtol": 1e-6, "pc_type": "hypre", "pc_hypre_typ": "boomeramg",
                     "pc_hypre_boomeramg_max_iter": 1, "pc_hypre_boomeramg_cycle_type": "v"  # ,
                     # "pc_hypre_boomeramg_print_statistics": 1
                     }

# Set PETSc options
opts = PETSc.Options()
opts.prefixPush(solver_prefix)
if petsc_options is not None:
    for k, v in petsc_options.items():
        opts[k] = v
opts.prefixPop()
solver.setFromOptions()


with Timer("~PERIODIC: Assemble and solve MPC problem"):
    uh = problem.solve()
    # solver.view()
    it = solver.getIterationNumber()
    print("Constrained solver iterations {0:d}".format(it))

# Write solution to file
uh.name = "u_mpc"
outfile = XDMFFile(MPI.COMM_WORLD, "results/demo_periodic_geometrical.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(uh)

print("----Verification----")
# --------------------VERIFICATION-------------------------
bilinear_form = fem.form(a)
A_org = fem.petsc.assemble_matrix(bilinear_form, bcs)
A_org.assemble()

linear_form = fem.form(rhs)
L_org = fem.petsc.assemble_vector(linear_form)
fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(L_org, bcs)
solver.setOperators(A_org)
u_ = fem.Function(V)
solver.solve(L_org, u_.vector)

it = solver.getIterationNumber()
print("Unconstrained solver iterations {0:d}".format(it))
u_.x.scatter_forward()
u_.name = "u_unconstrained"
outfile.write_function(u_)

root = 0
comm = mesh.comm
with Timer("~Demo: Verification"):
    dolfinx_mpc.utils.compare_mpc_lhs(A_org, problem._A, mpc, root=root)
    dolfinx_mpc.utils.compare_mpc_rhs(L_org, problem._b, mpc, root=root)

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
list_timings(MPI.COMM_WORLD, [TimingType.wall])

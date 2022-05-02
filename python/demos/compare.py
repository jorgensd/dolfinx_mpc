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
from dolfinx.common import Timer, TimingType, list_timings, timing
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad,
                 inner, pi, sin)

# Get PETSc int and scalar types
complex_mode = True if np.dtype(PETSc.ScalarType).kind == 'c' else False

# Create mesh and finite element
NX = 100
NY = 10000
mesh = create_unit_square(MPI.COMM_WORLD, NX, NY)
V = fem.FunctionSpace(mesh, ("CG", 1))


def dirichletboundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


# Create Dirichlet boundary condition
facets = locate_entities_boundary(mesh, 1, dirichletboundary)
topological_dofs = fem.locate_dofs_topological(V, 1, facets)
bc = fem.dirichletbc(PETSc.ScalarType(0), topological_dofs, V)
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
    with Timer("~PERIODIC: NewConstraint"):
        mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint(
            V._cpp_object, periodic_boundary, periodic_relation, [bc._cpp_object], 1)
    mpc.add_constraint_from_mpc_data(V, mpc_data)
    mpc.finalize()
    # old
    mpc2 = MultiPointConstraint(V)
    with Timer("~PERIODIC: OldConstraint"):
        mpc2.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs, 1)
    mpc2.finalize()

t_new = timing("~PERIODIC: NewConstraint")
t_old = timing("~PERIODIC: OldConstraint")
print(f"{mesh.comm.rank}: New {t_new[1]} Old {t_old[1]} New/Old {t_new[1]/t_old[1]}")
print(len(mpc.slaves))
list_timings(mesh.comm, [TimingType.wall])

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx

x = SpatialCoordinate(mesh)
dx_ = x[0] - 0.9
dy_ = x[1] - 0.5
f = x[0] * sin(5.0 * pi * x[1]) + 1.0 * exp(-(dx_ * dx_ + dy_ * dy_) / 0.02)

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

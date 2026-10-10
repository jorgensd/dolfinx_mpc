# This demo program solves the vector-valued Poisson equation
#
#     - div grad u(x, y) = f(x, y),   u = (u_1, u_2)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions, on both components, at x = 0, 1.
#
# Copyright (C) Jørgen S. Dokken 2020-2022.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import dolfinx.fem.petsc
import numpy as np
import scipy.sparse.linalg
from dolfinx import default_real_type, default_scalar_type
from dolfinx.common import Timer, list_timings
from dolfinx.fem import Function
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dx,
    exp,
    grad,
    inner,
    pi,
    sin,
)

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint

# Get PETSc int and scalar types
complex_mode = True if np.dtype(default_scalar_type).kind == "c" else False

# Create mesh and finite element
NX = 50
NY = 100
comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, NX, NY, dtype=default_real_type)
V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
tol = 250 * np.finfo(default_real_type).resolution


def dirichletboundary(x):
    return np.logical_or(np.isclose(x[1], 0, atol=tol), np.isclose(x[1], 1, atol=tol))


# Create Dirichlet boundary condition
facets = locate_entities_boundary(mesh, 1, dirichletboundary)
topological_dofs = fem.locate_dofs_topological(V, 1, facets)
zero = np.array([0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(zero, topological_dofs, V)
bcs = [bc]


def periodic_boundary(x):
    return np.isclose(x[0], 1, atol=tol)


def periodic_relation(x):
    out_x = np.zeros_like(x)
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


petsc_options: dict[str, str | int | float | bool]
if complex_mode or default_scalar_type == np.float32:
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
else:
    petsc_options = {
        "ksp_type": "cg",
        "ksp_rtol": 1e-6,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_max_iter": 1,
        "pc_hypre_boomeramg_cycle_type": "v",
    }
petsc_options["ksp_error_if_not_converged"] = True

# Setup MPC system
with Timer("~PERIODIC: Initialize varitional problem"):
    problem = LinearProblem(a, rhs, mpc, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix="periodic_mpc_")

with Timer("~PERIODIC: Assemble and solve MPC problem"):
    uh = problem.solve()
    assert isinstance(uh, Function)
    it = problem.solver.getIterationNumber()
if comm.rank == 0:
    print("Constrained solver iterations {0:d}".format(it))

# --------------------VERIFICATION-------------------------
# The same problem without the constraint, with a solver of its own
problem_org = dolfinx.fem.petsc.LinearProblem(
    a, rhs, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix="periodic_unconstrained_"
)
u_ = problem_org.solve()
assert isinstance(u_, Function)
it = problem_org.solver.getIterationNumber()
if comm.rank == 0:
    print("----Verification----")
    print("Unconstrained solver iterations {0:d}".format(it))
A_org, L_org = problem_org.A, problem_org.b

# Write solutions to file
outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
uh.name = "u_mpc"
u_.name = "u_unconstrained"
with XDMFFile(comm, outdir / "demo_periodic_geometrical.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_function(uh)
    outfile.write_function(u_)

root = 0
with Timer("~Demo: Verification"):
    dolfinx_mpc.utils.compare_mpc_lhs(A_org, problem.A, mpc, root=root)
    dolfinx_mpc.utils.compare_mpc_rhs(L_org, problem.b, mpc, root=root)
    is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
    scipy_dtype = np.complex128 if is_complex else np.float64
    # Gather LHS, RHS and solution on one process
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
    u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)

    if comm.rank == root:
        KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
        reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
        # Solve linear system
        d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
        # Back substitution to full solution vector
        uh_numpy = K.astype(scipy_dtype) @ d.astype(scipy_dtype)
        assert np.allclose(uh_numpy.astype(u_mpc.dtype), u_mpc, atol=float(tol))
list_timings(comm)
del problem, problem_org, A_org, L_org
PETSc.garbage_cleanup(comm)

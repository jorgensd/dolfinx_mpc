# Copyright (C) 2021-2022 fmonteghetti, Connor D. Pierce, and Jorgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
# This demo solves the Laplace eigenvalue problem
#
#     - div grad u(x, y) = lambda*u(x, y)
#
# on the unit square with two sets of boundary conditions:
#
#   u(x=0) = u(x=1) and u(y=0) = u(y=1) = 0,
#
# or
#
#   u(x=0) = u(x=1) and u(y=0) = u(y=1).
#
# The weak form reads
#
#       (grad(u),grad(v)) = lambda * (u,v),
#
# which leads to the generalized eigenvalue problem
#
#       A * U = lambda * B * U,
#
# where A and B are real symmetric positive definite matrices. The generalized
# eigenvalue problem is solved using SLEPc and the computed eigenvalues are
# compared to the exact ones.
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import numpy as np
from dolfinx import default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
from slepc4py import SLEPc
from ufl import TestFunction, TrialFunction, dx, grad, inner

from dolfinx_mpc import MultiPointConstraint, assemble_matrix


def print0(string: str):
    """Print on rank 0 only"""
    if MPI.COMM_WORLD.rank == 0:
        print(string)


def monitor_EPS_short(EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int):
    """
    Concise monitor for EPS.solve().

    Parameters
    ----------
    eps
        Eigenvalue Problem Solver class.
    it
       Current iteration number.
    nconv
       Number of converged eigenvalue.
    eig
       Eigenvalues
    err
       Computed errors.
    it_skip
        Iteration skip.

    """
    if it == 1:
        print0("******************************")
        print0("***  SLEPc Iterations...   ***")
        print0("******************************")
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    """Print summary of solution results."""
    print0("\n******************************")
    print0("*** SLEPc Solution Results ***")
    print0("******************************")
    its = EPS.getIterationNumber()
    print0(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print0(f"Converged eigenpairs: {nconv}")

    if nconv > 0:
        # Create the results vectors
        vr, vi = EPS.getOperators()[0].createVecs()
        print0("\nConverged eigval.  Error ")
        print0("----------------- -------")
        for i in range(nconv):
            k = EPS.getEigenpair(i, vr, vi)
            error = EPS.computeError(i)
            if not np.isclose(k.imag, 0.0):
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")


def EPS_get_spectrum(
    EPS: SLEPc.EPS, mpc: MultiPointConstraint
) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:  # type: ignore
    """Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
    Parameters
    ----------
    EPS
       The SLEPc solver
    mpc
       The multipoint constraint

    Returns
    -------
        Tuple consisting of: List of complex converted eigenvalues,
         lists of converted eigenvectors (real part) and (imaginary part)
    """
    # Get results in lists
    eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
    eigvec_r = list()
    eigvec_i = list()
    V = mpc.function_space
    for i in range(EPS.getConverged()):
        vr = fem.Function(V)
        vi = fem.Function(V)

        EPS.getEigenvector(i, vr.x.petsc_vec, vi.x.petsc_vec)
        eigvec_r.append(vr)
        eigvec_i.append(vi)  # Sort by increasing real parts
    idx = np.argsort(np.real(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(
    A: PETSc.Mat,  # type: ignore
    B: PETSc.Mat,  # type: ignore
    problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
    solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
    nev: int = 10,
    tol: float = 1e-7,
    max_it: int = 10,
    target: float = 0.0,
    shift: float = 0.0,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> SLEPc.EPS:
    """
     Solve generalized eigenvalue problem A*x=lambda*B*x using shift-and-invert
     as spectral transform method.

     Parameters
     ----------
     A
        The matrix A
     B
        The matrix B
     problem_type
        The problem type, for options see: https://bit.ly/3gM5pth
    solver:
        Solver type, for options see: https://bit.ly/35LDcMG
     nev
         Number of requested eigenvalues.
     tol
        Tolerance for slepc solver
     max_it
        Maximum number of iterations.
     target
        Target eigenvalue. Also used for sorting.
     shift
        Shift 'sigma' used in shift-and-invert.

     Returns
     -------
     EPS
        The SLEPc solver
    """

    # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS()
    EPS.create(comm=comm)
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)
    # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev)
    # Set solver
    EPS.setType(solver)
    # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target)  # sorting
    # set tolerance and max iterations
    EPS.setTolerances(tol=tol, max_it=max_it)
    # Set up shift-and-invert
    # Only work if 'whichEigenpairs' is 'TARGET_XX'
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)
    EPS.setST(ST)
    # set monitor
    it_skip = 1
    EPS.setMonitor(lambda eps, it, nconv, eig, err: monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS


def assemble_and_solve(boundary_condition: List[str] = ["dirichlet", "periodic"], Nev: int = 10):
    """
    Assemble and solve the Laplace eigenvalue problem on the unit square with
    the prescribed boundary conditions.

    Parameters
    ----------
    boundary_condition
        First item describes b.c. on {x=0} and {x=1}
        Second item describes b.c. on {y=0} and {y=1}
    Nev
        Number of requested eigenvalues. The default is 10.
    """
    comm = MPI.COMM_WORLD
    # Create mesh and finite element
    N = 50
    mesh = create_unit_square(comm, N, N)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    fdim = mesh.topology.dim - 1

    bcs = []
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
    pbc_slave_to_master_maps = []

    def generate_pbc_slave_to_master_map(i):
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[i] = x[i] - 1
            return out_x

        return pbc_slave_to_master_map

    def generate_pbc_is_slave(i):
        return lambda x: np.isclose(x[i], 1)

    def generate_pbc_is_master(i):
        return lambda x: np.isclose(x[i], 0)

    # Parse boundary conditions
    for i, bc_type in enumerate(boundary_condition):
        if bc_type == "dirichlet":
            u_bc = fem.Function(V)
            u_bc.x.array[:] = 0

            def dirichletboundary(x):
                return np.logical_or(np.isclose(x[i], 0), np.isclose(x[i], 1))

            facets = locate_entities_boundary(mesh, fdim, dirichletboundary)
            topological_dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(u_bc, topological_dofs))
        elif bc_type == "periodic":
            pbc_directions.append(i)
            pbc_slave_tags.append(i + 2)
            pbc_is_slave.append(generate_pbc_is_slave(i))
            pbc_is_master.append(generate_pbc_is_master(i))
            pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

            facets = locate_entities_boundary(mesh, fdim, pbc_is_slave[-1])
            arg_sort = np.argsort(facets)
            pbc_meshtags.append(
                meshtags(
                    mesh,
                    fdim,
                    facets[arg_sort],
                    np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32),
                )
            )

    # Create MultiPointConstraint object
    mpc = MultiPointConstraint(V)
    N_pbc = len(pbc_directions)
    for i in range(N_pbc):
        if N_pbc > 1:

            def pbc_slave_to_master_map(x):
                out_x = pbc_slave_to_master_maps[i](x)
                idx = pbc_is_slave[(i + 1) % N_pbc](x)
                out_x[pbc_directions[i]][idx] = np.nan
                return out_x
        else:
            pbc_slave_to_master_map = pbc_slave_to_master_maps[i]

        mpc.create_periodic_constraint_topological(V, pbc_meshtags[i], pbc_slave_tags[i], pbc_slave_to_master_map, bcs)
    if len(pbc_directions) > 1:
        # Map intersection(slaves_x, slaves_y) to intersection(masters_x, masters_y),
        # i.e. map the slave dof at (1, 1) to the master dof at (0, 0)
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[0] = x[0] - 1
            out_x[1] = x[1] - 1
            idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
            out_x[0][~idx] = np.nan
            out_x[1][~idx] = np.nan
            return out_x

        mpc.create_periodic_constraint_topological(V, pbc_meshtags[1], pbc_slave_tags[1], pbc_slave_to_master_map, bcs)
    mpc.finalize()
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    b = inner(u, v) * dx
    mass_form = fem.form(a)
    stiffness_form = fem.form(b)

    # Diagonal values for slave and Dirichlet DoF
    # The generalized eigenvalue problem will have spurious eigenvalues at
    # lambda_spurious = diagval_A/diagval_B. Here we choose lambda_spurious=1e4,
    # which is far from the region of interest.
    diagval_A = 1e2
    diagval_B = 1e-2
    tol = float(5e2 * np.finfo(default_scalar_type).resolution)

    A = assemble_matrix(mass_form, mpc, bcs=bcs, diagval=diagval_A)
    B = assemble_matrix(stiffness_form, mpc, bcs=bcs, diagval=diagval_B)
    EPS = solve_GEP_shiftinvert(
        A,
        B,
        problem_type=SLEPc.EPS.ProblemType.GHEP,
        solver=SLEPc.EPS.Type.KRYLOVSCHUR,
        nev=Nev,
        tol=tol,
        max_it=10,
        target=1.5,
        shift=1.5,
        comm=comm,
    )
    (eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)
    # update slave DoF
    for i in range(len(eigval)):
        eigvec_r[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_r[i])
        eigvec_i[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_i[i])
    print0(f"Computed eigenvalues:\n {np.around(eigval,decimals=2)}")

    # Save all eigenvectors
    suffix = "".join([bc_type[0] for bc_type in boundary_condition])
    outdir = Path("results")
    outdir.mkdir(exist_ok=True, parents=True)
    with XDMFFile(mesh.comm, outdir / f"eigenvector_{suffix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        for i, e_vec in enumerate(eigvec_r):
            xdmf.write_function(e_vec, i)


def print_exact_eigenvalues(boundary_condition: List[str], N: int):
    L = [1, 1]
    if boundary_condition[0] == "dirichlet":
        ev_x = [(n * np.pi / L[0]) ** 2 for n in range(1, N + 1)]
    elif boundary_condition[0] == "periodic":
        ev_x = [(n * 2 * np.pi / L[0]) ** 2 for n in range(-N, N)]
    if boundary_condition[1] == "dirichlet":
        ev_y = [(n * np.pi / L[1]) ** 2 for n in range(1, N + 1)]
    elif boundary_condition[1] == "periodic":
        ev_y = [(n * 2 * np.pi / L[1]) ** 2 for n in range(-N, N)]
    ev_ex = np.sort([r + q for r in ev_x for q in ev_y])
    ev_ex = ev_ex[0:N]
    print0(f"Exact eigenvalues (repeated with multiplicity):\n {np.around(ev_ex,decimals=2)}")


# Dirichlet boundary condition on {x=0} and {x=1}
# Dirichlet boundary condition on {y=0} and {y=1}
assemble_and_solve(["dirichlet", "dirichlet"], 10)
print_exact_eigenvalues(["dirichlet", "dirichlet"], 10)

# Dirichlet boundary condition on {x=0} and {x=1}
# Periodic boundary condition: {y=0} -> {y=1}
assemble_and_solve(["dirichlet", "periodic"], 10)
print_exact_eigenvalues(["dirichlet", "periodic"], 10)

# Periodic boundary condition: {x=0} -> {x=1}
# Dirichlet boundary condition on {y=0} and {y=1}
assemble_and_solve(["periodic", "dirichlet"], 10)
print_exact_eigenvalues(["periodic", "dirichlet"], 10)

# Periodic boundary condition: {x=0} -> {x=1}
# Periodic boundary condition: {y=0} -> {y=1}
assemble_and_solve(["periodic", "periodic"], 10)
print_exact_eigenvalues(["periodic", "periodic"], 10)

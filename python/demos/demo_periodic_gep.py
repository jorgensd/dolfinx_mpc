# Copyright (C) 2021-2022 fmonteghetti and Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
# This demo, adapted from 'demo_periodic.py', solves
#
#     - div grad u(x, y) = lambda*u(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
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

from typing import List, Tuple

import dolfinx.fem as fem
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
from dolfinx_mpc import MultiPointConstraint, assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import TestFunction, TrialFunction, dx, grad, inner


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
    if (it == 1):
        print0('******************************')
        print0('***  SLEPc Iterations...   ***')
        print0('******************************')
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    """ Print summary of solution results. """
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
            if k.imag != 0.0:
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")


def EPS_get_spectrum(EPS: SLEPc.EPS,
                     mpc: MultiPointConstraint) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:
    """ Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
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

        EPS.getEigenvector(i, vr.vector, vi.vector)
        eigvec_r.append(vr)
        eigvec_i.append(vi)    # Sort by increasing real parts
    idx = np.argsort(np.real(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(A: PETSc.Mat, B: PETSc.Mat,
                          problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
                          solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev: int = 10, tol: float = 1e-7, max_it: int = 10,
                          target: float = 0.0, shift: float = 0.0) -> SLEPc.EPS:
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
    EPS.create(comm=MPI.COMM_WORLD)
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
    EPS.setMonitor(lambda eps, it, nconv, eig, err:
                   monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS


# Create mesh and finite element
N = 50
mesh = create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = fem.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def dirichletboundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


fdim = mesh.topology.dim - 1
facets = locate_entities_boundary(mesh, fdim, dirichletboundary)
topological_dofs = fem.locate_dofs_topological(V, fdim, facets)
bc = fem.dirichletbc(u_bc, topological_dofs)
bcs = [bc]


def periodicboundary(x):
    return np.isclose(x[0], 1)


facets = locate_entities_boundary(mesh, fdim, periodicboundary)
arg_sort = np.argsort(facets)
mt = meshtags(mesh, fdim, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))


def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


mpc = MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, mt, 2, periodic_relation, bcs)
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
A = assemble_matrix(mass_form, mpc, bcs=bcs, diagval=diagval_A)
B = assemble_matrix(stiffness_form, mpc, bcs=bcs, diagval=diagval_B)
Nev = 10  # number of requested eigenvalues
EPS = solve_GEP_shiftinvert(A, B, problem_type=SLEPc.EPS.ProblemType.GHEP,
                            solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                            nev=Nev, tol=1e-7, max_it=10,
                            target=1.5, shift=1.5)
(eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)
# update slave DoF
for i in range(len(eigval)):
    mpc.backsubstitution(eigvec_r[i].vector)
    mpc.backsubstitution(eigvec_i[i].vector)
print0(f"Computed eigenvalues:\n {np.around(eigval,decimals=2)}")
# Exact eigenvalues
l_exact = list(set([(i * np.pi)**2 + (2 * j * np.pi)**2 for i in range(Nev) for j in range(Nev)]))
l_exact.remove(0)
l_exact.sort()
print0(f"Exact eigenvalues:\n {np.around(l_exact[0:Nev-1],decimals=2)}")


# Save first eigenvector
with XDMFFile(MPI.COMM_WORLD, "results/eigenvector_0.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(eigvec_r[0])

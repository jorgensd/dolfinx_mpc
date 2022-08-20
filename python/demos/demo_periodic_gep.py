# Copyright (C) 2021-2022 fmonteghetti and Jorgen S. Dokken
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
                          target: float = 0.0, shift: float = 0.0,
                          comm: MPI.Intracomm = MPI.COMM_WORLD) -> SLEPc.EPS:
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
    EPS.setMonitor(lambda eps, it, nconv, eig, err:
                   monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS


def assemble_and_solve(boundary_condition: List[str] = ["dirichlet","periodic"],
                       Nev: int = 10):
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
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    fdim = mesh.topology.dim - 1
    
    bcs = []
    # Dirichlet boundary condition on {y=0} and {y=1}
    if boundary_condition[1] == "dirichlet":
        u_bc = fem.Function(V)
        u_bc.x.array[:]  =0
        def dirichletboundary(x):
            return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
        facets = locate_entities_boundary(mesh, fdim, dirichletboundary)
        topological_dofs = fem.locate_dofs_topological(V, fdim, facets)
        bc = fem.dirichletbc(u_bc, topological_dofs)
        bcs = [bc]
    
    # Periodic boundary condition: {x=0} -> {x=1}
    if boundary_condition[0]=="periodic":
        pbc_hor_slave_tag = 2
        pbc_hor_is_slave = lambda x: np.isclose(x[0],1) # identifier for slave boundary
        pbc_hor_is_master = lambda x: np.isclose(x[0],0) # identifier for master boundary
        def pbc_hor_slave_to_master_map(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - 1
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x
        facets = locate_entities_boundary(mesh, fdim, pbc_hor_is_slave)
        arg_sort = np.argsort(facets)
        pbc_hor_mt = meshtags(mesh, fdim, facets[arg_sort], np.full(len(facets), pbc_hor_slave_tag, dtype=np.int32))
        
    # Periodic boundary condition: {y=0} -> {y=1}
    if boundary_condition[1]=="periodic":
        pbc_vert_slave_tag = 3
        pbc_vert_is_slave = lambda x: np.isclose(x[1],1)
        def pbc_vert_slave_to_master_map(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - 1
            out_x[2] = x[2]
            return out_x
        facets = locate_entities_boundary(mesh, fdim, pbc_vert_is_slave)
        arg_sort = np.argsort(facets)
        pbc_vert_mt = meshtags(mesh, fdim, facets[arg_sort], np.full(len(facets), pbc_vert_slave_tag, dtype=np.int32))
        
    # Create MultiPointConstraint object
    mpc = MultiPointConstraint(V)
    if boundary_condition[0]=="periodic":
        mpc.create_periodic_constraint_topological(V, pbc_hor_mt, 
                                                   pbc_hor_slave_tag,
                                                   pbc_hor_slave_to_master_map, 
                                                   bcs)
    if boundary_condition[1]=="periodic":
        if boundary_condition[0]=="periodic":
            # Redefine slave_to_master map to exclude dofs that are slave or 
            # master of the previous periodic boundary condition
            tmp = pbc_vert_slave_to_master_map
            def pbc_vert_slave_to_master_map(x):
                out_x = tmp(x)
                idx = pbc_hor_is_slave(x) + pbc_hor_is_master(x)
                out_x[0][idx] = np.nan
                return out_x
        mpc.create_periodic_constraint_topological(V, pbc_vert_mt, 
                                                   pbc_vert_slave_tag,
                                                   pbc_vert_slave_to_master_map, 
                                                   bcs)
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
    EPS = solve_GEP_shiftinvert(A, B, problem_type=SLEPc.EPS.ProblemType.GHEP,
                                solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                                nev=Nev, tol=1e-7, max_it=10,
                                target=1.5, shift=1.5,comm=comm)
    (eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)
    # update slave DoF
    for i in range(len(eigval)):
        eigvec_r[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_r[i].vector)
        eigvec_i[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_i[i].vector)
    print0(f"Computed eigenvalues:\n {np.around(eigval,decimals=2)}")

    # Save first eigenvector
    with XDMFFile(MPI.COMM_WORLD, "results/eigenvector_0.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(eigvec_r[0])
    
def print_exact_eigenvalues(boundary_condition: List[str], N: int):
    L = [1,1]
    if boundary_condition[0]=="periodic":
        ev_x  = [(n*2*np.pi/L[0])**2 for n in range(-N,N)]
    if boundary_condition[1]=="dirichlet":
        ev_y = [(n*np.pi/L[1])**2 for n in range(1,N+1)]
    elif boundary_condition[1]=="periodic":
        ev_y  = [(n*2*np.pi/L[1])**2 for n in range(-N,N)]
    ev_ex = np.sort([r+q for r in ev_x for q in ev_y])
    ev_ex = ev_ex[0:N]
    print0(f"Exact eigenvalues (repeated with multiplicity):\n {np.around(ev_ex,decimals=2)}")


# Periodic boundary condition: {x=0} -> {x=1}
# Dirichlet boundary condition on {y=0} and {y=1}
assemble_and_solve(["periodic","dirichlet"],10)
print_exact_eigenvalues(["periodic","dirichlet"],10)

# Periodic boundary condition: {x=0} -> {x=1}
# Periodic boundary condition: {y=0} -> {y=1}
assemble_and_solve(["periodic","periodic"],10)
print_exact_eigenvalues(["periodic","periodic"],10)
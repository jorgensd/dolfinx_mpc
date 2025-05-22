# Copyright (C) 2025
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import numpy.testing as nt
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc

# Skip all tests if SLEPc is not available
slepc4py = pytest.importorskip("slepc4py")
from slepc4py import SLEPc


def test_standard_eigenvalue_problem():
    """Test standard eigenvalue problem A*x = λ*x with periodic boundary conditions."""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Define variational form for Laplacian
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Create multipoint constraint for periodic BC
    def periodic_relation(x):
        out_x = np.copy(x)
        out_x[0] = 1 - x[0]
        return out_x

    def periodic_condition(x):
        return np.isclose(x[0], 0)

    # Create meshtags for periodic boundary
    facets = locate_entities_boundary(mesh, 1, periodic_condition)
    arg_sort = np.argsort(facets)
    mt = meshtags(mesh, 1, facets[arg_sort], np.full(len(facets), 1, dtype=np.int32))

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, mt, 1, periodic_relation, bcs=[])
    mpc.finalize()

    # Create and solve eigenvalue problem
    problem = dolfinx_mpc.EigenProblem(a, mpc, slepc_options={"eps_nev": 5})
    eigenvalues, eigenvectors = problem.solve()

    # Check that we got the requested number of eigenvalues
    assert len(eigenvalues) >= 5
    assert len(eigenvectors) == len(eigenvalues)

    # Check that eigenvalues are real and positive (for Laplacian)
    for eigval in eigenvalues:
        assert np.imag(eigval) < 1e-10
        assert np.real(eigval) > 0

    # Check orthogonality of eigenvectors (approximately)
    # Due to the MPC, we can't check exact orthogonality easily,
    # but eigenvectors should at least be normalized
    for eigvec in eigenvectors:
        norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(eigvec, eigvec) * ufl.dx)))
        nt.assert_allclose(norm, 1.0, rtol=1e-2)


def test_generalized_eigenvalue_problem():
    """Test generalized eigenvalue problem A*x = λ*B*x."""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(mesh, ("Lagrange", 2))

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Stiffness matrix (A)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Mass matrix (B)
    b = ufl.inner(u, v) * ufl.dx

    # Create boundary conditions (fix left edge)
    def left_boundary(x):
        return np.isclose(x[0], 0)
    
    left_facets = locate_entities_boundary(mesh, 1, left_boundary)
    left_dofs = fem.locate_dofs_topological(V, 1, left_facets)
    bc = fem.dirichletbc(default_scalar_type(0), left_dofs, V)

    # Create simple MPC (no constraints, just to test the interface)
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.finalize()

    # Create and solve generalized eigenvalue problem
    problem = dolfinx_mpc.EigenProblem(
        a, mpc, b=b, bcs=[bc],
        slepc_options={
            "eps_type": "gen_hermitian",
            "eps_nev": 10,
            "st_type": "sinvert",
            "eps_target": 0.0
        }
    )
    eigenvalues, eigenvectors = problem.solve()

    # Check that we got eigenvalues
    assert len(eigenvalues) > 0
    assert len(eigenvectors) == len(eigenvalues)

    # For the Laplacian with mass matrix, eigenvalues should be real and positive
    for eigval in eigenvalues:
        assert np.imag(eigval) < 1e-10
        assert np.real(eigval) > 0

    # Check that eigenvalues are sorted (SLEPc usually returns them sorted)
    real_eigenvalues = [np.real(ev) for ev in eigenvalues]
    assert all(real_eigenvalues[i] <= real_eigenvalues[i+1] for i in range(len(real_eigenvalues)-1))


def test_eigenvalue_problem_with_complex_mpc():
    """Test eigenvalue problem with more complex multi-point constraints."""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    m = ufl.inner(u, v) * ufl.dx

    # Create periodic constraints in both directions
    def periodic_relation_x(x):
        out_x = np.copy(x)
        out_x[0] = 1 - x[0]
        return out_x

    def periodic_condition_x(x):
        return np.isclose(x[0], 0)

    def periodic_relation_y(x):
        out_x = np.copy(x)
        out_x[1] = 1 - x[1]
        return out_x

    def periodic_condition_y(x):
        return np.logical_and(np.isclose(x[1], 0), np.logical_not(np.isclose(x[0], 0)))

    # Create meshtags for periodic boundaries
    facets_x = locate_entities_boundary(mesh, 1, periodic_condition_x)
    arg_sort_x = np.argsort(facets_x)
    mt_x = meshtags(mesh, 1, facets_x[arg_sort_x], np.full(len(facets_x), 1, dtype=np.int32))
    
    facets_y = locate_entities_boundary(mesh, 1, periodic_condition_y)
    arg_sort_y = np.argsort(facets_y)
    mt_y = meshtags(mesh, 1, facets_y[arg_sort_y], np.full(len(facets_y), 2, dtype=np.int32))

    # Create MPC with periodic BCs in both directions
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, mt_x, 1, periodic_relation_x, bcs=[])
    mpc.create_periodic_constraint_topological(V, mt_y, 2, periodic_relation_y, bcs=[])
    mpc.finalize()

    # Solve eigenvalue problem
    problem = dolfinx_mpc.EigenProblem(
        a, mpc, b=m,
        slepc_options={
            "eps_type": "gen_hermitian",
            "eps_nev": 15,
            "st_type": "sinvert",
            "eps_target": 0.0
        }
    )
    eigenvalues, eigenvectors = problem.solve(nev=15)

    # Verify results
    assert len(eigenvalues) >= 15
    
    # The smallest eigenvalue should be close to 0 (constant mode)
    min_eigenvalue = min(np.real(ev) for ev in eigenvalues)
    nt.assert_allclose(min_eigenvalue, 0.0, atol=1e-8)

    # Check error estimates
    n_conv = problem.get_number_converged()
    assert n_conv >= 15
    
    for i in range(min(5, n_conv)):
        error = problem.error_estimate(i)
        assert error < 1e-6


def test_eigenvalue_problem_options():
    """Test that solver options are properly set."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.finalize()

    # Test with custom options
    custom_options = {
        "eps_type": "arnoldi",
        "eps_nev": 3,
        "eps_tol": 1e-10,
        "eps_max_it": 200,
    }
    
    problem = dolfinx_mpc.EigenProblem(a, mpc, slepc_options=custom_options)
    eigenvalues, eigenvectors = problem.solve()

    # Should get at least the requested number of eigenvalues
    assert len(eigenvalues) >= 3


if __name__ == "__main__":
    test_standard_eigenvalue_problem()
    test_generalized_eigenvalue_problem()
    test_eigenvalue_problem_with_complex_mpc()
    test_eigenvalue_problem_options()
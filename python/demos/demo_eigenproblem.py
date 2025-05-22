"""
Eigenvalue problem demo using EigenProblem class
=================================================

This demo illustrates how to:
- Use the EigenProblem class for solving eigenvalue problems with MPCs
- Solve both standard (A*x = λ*x) and generalized (A*x = λ*B*x) eigenvalue problems
- Apply multi-point constraints in eigenvalue computations
"""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, io, mesh
from dolfinx.mesh import locate_entities_boundary

import dolfinx_mpc

# Create mesh and function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
V = fem.functionspace(domain, ("Lagrange", 2))

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define bilinear forms
# Stiffness form (Laplacian)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Mass form
m = ufl.inner(u, v) * ufl.dx

# Create periodic boundary conditions using MPC
def periodic_relation(x):
    out_x = np.copy(x)
    out_x[0] = 1 - x[0]
    return out_x

def periodic_condition(x):
    return np.isclose(x[0], 0)

# Also add a Dirichlet BC on bottom edge
def bottom(x):
    return np.isclose(x[1], 0)

bottom_facets = locate_entities_boundary(domain, 1, bottom)
bottom_dofs = fem.locate_dofs_topological(V, 1, bottom_facets)
bc = fem.dirichletbc(default_scalar_type(0), bottom_dofs, V)

# Create meshtags for periodic boundary
periodic_facets = locate_entities_boundary(domain, 1, periodic_condition)
arg_sort = np.argsort(periodic_facets)
mt = mesh.meshtags(domain, 1, periodic_facets[arg_sort], np.full(len(periodic_facets), 1, dtype=np.int32))

# Create multi-point constraint
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, mt, 1, periodic_relation, bcs=[], scale=default_scalar_type(1))
mpc.finalize()

print("Number of MPC constraints:", mpc.num_local_slaves)

# Example 1: Standard eigenvalue problem
print("\n" + "="*60)
print("Example 1: Standard eigenvalue problem (A*x = λ*x)")
print("="*60)

problem1 = dolfinx_mpc.EigenProblem(
    a, mpc, bcs=[bc],
    slepc_options={
        "eps_type": "krylovschur",
        "eps_nev": 10,
        "eps_tol": 1e-10,
    }
)

eigenvalues1, eigenvectors1 = problem1.solve()

print(f"\nFound {len(eigenvalues1)} eigenvalues:")
for i, eigval in enumerate(eigenvalues1[:5]):
    print(f"  λ_{i+1} = {np.real(eigval):.6f}")

# Example 2: Generalized eigenvalue problem
print("\n" + "="*60)
print("Example 2: Generalized eigenvalue problem (A*x = λ*B*x)")
print("="*60)

problem2 = dolfinx_mpc.EigenProblem(
    a, mpc, b=m, bcs=[bc],
    slepc_options={
        "eps_type": "gen_hermitian",
        "eps_nev": 10,
        "st_type": "sinvert",
        "eps_target": 0.0,
    }
)

eigenvalues2, eigenvectors2 = problem2.solve()

print(f"\nFound {len(eigenvalues2)} eigenvalues:")
for i, eigval in enumerate(eigenvalues2[:5]):
    print(f"  λ_{i+1} = {np.real(eigval):.6f}")

# Save eigenvectors for visualization
with io.XDMFFile(domain.comm, "eigenmodes.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    
    # Save first 3 eigenmodes from generalized problem
    for i in range(min(3, len(eigenvectors2))):
        eigenvectors2[i].name = f"eigenmode_{i+1}"
        xdmf.write_function(eigenvectors2[i], i)

print("\nEigenmodes saved to eigenmodes.xdmf")

# Example 3: Using solve with specific number of eigenvalues
print("\n" + "="*60)
print("Example 3: Solving for specific number of eigenvalues")
print("="*60)

# Solve for exactly 5 eigenvalues
eigenvalues3, eigenvectors3 = problem2.solve(nev=5)
print(f"Requested 5 eigenvalues, got {len(eigenvalues3)}")

# Check error estimates
print("\nError estimates for computed eigenvalues:")
for i in range(problem2.get_number_converged()):
    error = problem2.error_estimate(i)
    print(f"  Eigenvalue {i+1}: error = {error:.2e}")

print("\nDemo completed successfully!")
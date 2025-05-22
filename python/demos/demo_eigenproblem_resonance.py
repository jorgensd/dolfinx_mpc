"""
Demo: Finding resonant modes in structures with bonded parts using EigenProblem
================================================================================

This demo shows how to use the EigenProblem class to find resonant modes 
(eigenfrequencies and eigenmodes) of a structure composed of multiple parts
connected through multi-point constraints.

We consider a composite beam made of two materials bonded together, and find
its natural frequencies and mode shapes.
"""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, io, mesh, plot
from dolfinx.mesh import CellType, create_box, locate_entities_boundary

import dolfinx_mpc

# Material properties
E1 = 200e9   # Young's modulus for material 1 (steel) [Pa]
E2 = 70e9    # Young's modulus for material 2 (aluminum) [Pa]
rho1 = 7850  # Density for material 1 [kg/m³]
rho2 = 2700  # Density for material 2 [kg/m³]
nu = 0.3     # Poisson's ratio (same for both materials)

# Geometry
L = 1.0      # Total length [m]
H = 0.1      # Height [m]
W = 0.05     # Width [m]

# Create mesh: two beams side by side
mesh1 = create_box(
    MPI.COMM_WORLD,
    [[0, 0, 0], [L/2, H, W]],
    [10, 4, 2],
    cell_type=CellType.hexahedron
)
mesh2 = create_box(
    MPI.COMM_WORLD,
    [[L/2, 0, 0], [L, H, W]],
    [10, 4, 2],
    cell_type=CellType.hexahedron
)

# For simplicity, we'll use a single mesh in this demo
# In practice, you might have separate meshes that need to be connected
mesh_full = create_box(
    MPI.COMM_WORLD,
    [[0, 0, 0], [L, H, W]],
    [20, 4, 2],
    cell_type=CellType.hexahedron
)

# Create function space (vector for displacement)
V = fem.functionspace(mesh_full, ("Lagrange", 2, (3,)))

# Define material subdomains
def material1(x):
    return x[0] <= L/2 + 1e-10

def material2(x):
    return x[0] >= L/2 - 1e-10

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Strain and stress
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, E):
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

# Create material property functions
Q = fem.functionspace(mesh_full, ("DG", 0))
E = fem.Function(Q)
rho = fem.Function(Q)

# Assign material properties
cells1 = mesh.locate_entities(mesh_full, 3, material1)
cells2 = mesh.locate_entities(mesh_full, 3, material2)

E.x.array[:] = E2  # Default to material 2
E.x.array[cells1] = E1
rho.x.array[:] = rho2  # Default to material 2
rho.x.array[cells1] = rho1

# Define variational forms
# Stiffness matrix (K)
a = ufl.inner(sigma(u, E), epsilon(v)) * ufl.dx

# Mass matrix (M)
m = rho * ufl.inner(u, v) * ufl.dx

# Boundary conditions: Fixed at left end
def left_boundary(x):
    return np.isclose(x[0], 0)

left_facets = locate_entities_boundary(mesh_full, 2, left_boundary)
left_dofs = fem.locate_dofs_topological(V, 2, left_facets)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), left_dofs, V)

# Create MPC for bonding interface (if using separate meshes)
# For this demo with a single mesh, we create a simple MPC
mpc = dolfinx_mpc.MultiPointConstraint(V)

# In a real application with separate meshes, you would create constraints
# at the interface between the two parts here
# Example (commented out):
# def interface_locator(x):
#     return np.isclose(x[0], L/2)
# mpc.create_contact_constraint(V, interface_locator, ...)

mpc.finalize()

# Create and solve eigenvalue problem
print("Setting up eigenvalue problem...")
problem = dolfinx_mpc.EigenProblem(
    a, mpc, b=m, bcs=[bc],
    slepc_options={
        "eps_type": "gen_hermitian",
        "eps_nev": 10,                    # Number of eigenvalues to compute
        "st_type": "sinvert",              # Spectral transformation
        "st_shift": 0.0,                   # Shift for spectral transformation
        "eps_target": 0.0,                 # Target eigenvalue
        "eps_tol": 1e-8,                   # Tolerance
        "eps_monitor": "",                 # Monitor convergence
    }
)

print("Solving eigenvalue problem...")
eigenvalues, eigenmodes = problem.solve()

# Convert eigenvalues to frequencies
# For structural dynamics: ω² = λ, f = ω/(2π)
frequencies = []
for i, eigval in enumerate(eigenvalues):
    omega_squared = np.real(eigval)
    if omega_squared > 0:
        freq = np.sqrt(omega_squared) / (2 * np.pi)
        frequencies.append(freq)
        print(f"Mode {i+1}: f = {freq:.2f} Hz (ω² = {omega_squared:.2e})")
    else:
        frequencies.append(0.0)
        print(f"Mode {i+1}: Rigid body mode (ω² = {omega_squared:.2e})")

# Save results for visualization
with io.XDMFFile(mesh_full.comm, "resonant_modes.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_full)
    
    # Save first few eigenmodes
    for i in range(min(5, len(eigenmodes))):
        eigenmode = eigenmodes[i]
        eigenmode.name = f"Mode_{i+1}_freq_{frequencies[i]:.2f}Hz"
        xdmf.write_function(eigenmode, i)

# Also save to VTK for alternative visualization
with io.VTKFile(mesh_full.comm, "resonant_modes.pvd", "w") as vtk:
    for i in range(min(5, len(eigenmodes))):
        eigenmode = eigenmodes[i]
        eigenmode.name = f"Mode_{i+1}"
        vtk.write_function(eigenmode, i)

print("\nResults saved to resonant_modes.xdmf and resonant_modes.pvd")
print("View in ParaView to see the mode shapes")

# Verify orthogonality of eigenmodes with respect to mass matrix
print("\nVerifying orthogonality of eigenmodes...")
for i in range(min(3, len(eigenmodes))):
    for j in range(i+1, min(3, len(eigenmodes))):
        # Compute (φᵢ, M φⱼ)
        form = fem.form(rho * ufl.inner(eigenmodes[i], eigenmodes[j]) * ufl.dx)
        mass_ij = fem.assemble_scalar(form)
        print(f"<φ_{i+1}, M φ_{j+1}> = {mass_ij:.2e}")

print("\nDemo completed successfully!")
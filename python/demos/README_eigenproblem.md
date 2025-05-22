# EigenProblem Class

The `EigenProblem` class provides a user-friendly interface for solving eigenvalue problems with multi-point constraints, similar to how `LinearProblem` simplifies solving linear systems.

## Features

- Supports both standard eigenvalue problems (A*x = λ*x) and generalized eigenvalue problems (A*x = λ*B*x)
- Seamlessly integrates with multi-point constraints (MPCs)
- Handles Dirichlet boundary conditions
- Uses SLEPc as the backend solver
- Provides sensible default solver parameters while allowing full customization

## Installation

To use the `EigenProblem` class, you need to install `slepc4py`:

```bash
pip install dolfinx_mpc[eigen]
```

or separately:

```bash
pip install slepc4py
```

## Basic Usage

```python
import dolfinx_mpc

# Create your forms
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Stiffness
m = ufl.inner(u, v) * ufl.dx                       # Mass (optional)

# Create and finalize MPC
mpc = dolfinx_mpc.MultiPointConstraint(V)
# ... add constraints ...
mpc.finalize()

# Solve eigenvalue problem
problem = dolfinx_mpc.EigenProblem(a, mpc, b=m, bcs=[bc])
eigenvalues, eigenvectors = problem.solve()
```

## Examples

See the demo files:
- `demo_eigenproblem.py`: Basic usage with periodic boundary conditions
- `demo_eigenproblem_resonance.py`: Finding resonant modes in composite structures

## Use Cases

This class is particularly useful for:
- Modal analysis of structures
- Finding resonant frequencies in composite materials
- Stability analysis
- Vibration analysis of structures with multiple bonded parts
- Any eigenvalue problem involving multi-point constraints
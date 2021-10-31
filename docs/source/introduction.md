# Introduction

## Overview

DOLFINx-MPC is a set of additional operations that can be used to implement multi point constraints (MPCs) for models written in [DOLFINx](https://github.com/FEniCS/dolfinx).

DOLFINx is a software for solving partial differential equations (PDEs) using the finite element method. DOLFINx uses the [Unified Form Language](https://github.com/FEniCS/ufl) (UFL) to specify variational forms in a near to mathematical syntax.

The variational forms are then in turn assembled into a set of linear equations of the form $Au=b$, where $A$ is a $N\times N$ matrix, $u$ is a vector containing the $N$ degrees of freedom, and $b$ a vector of size $N$.

This package allows a user to add constraints to the linear system on the form:

$ u_j = \sum_{i=1, i\neq j}^N \alpha_i u_i$ for any $j\in [1, N]$.

Some of the boundary conditions that fall under this category is:
- Slip conditions: $u\cdot n= 0$ on $\partial \Omega$.
- Periodic conditons $u\vert_{\partial \Omega_1} = \alpha u\vert_{\partial \Omega_2}$.

## Directory structure
- `cpp/` Contains the C++ implementation of the multi point constraint class, and assemblers form matrices and vectors.
- `python/dolfinx_mpc` Contains the Python interface of the package.
- `python/dolfinx_mpc/numba` Optional package with of the custom MPC assemblers implemented with [numba](https://numba.pydata.org/).
- `python/dolfinx_mpc/utils` is an optional package containing several convenice functions for IO, testing and creating special constraints.
- `python/dolfinx_mpc/demos` contains a set of demos illustrating how to use the package with DOLFINx.
- `python/dolfinx_mpc/benchmarks` contains several benchmarks to investigate the scalability of the implemented functions
- `python/dolfinx_mpc/tests` contains all unit tests for verifying the implementation of the multi point constraints.
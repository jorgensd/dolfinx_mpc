# Multi-point constraints with FEniCS-X
![CI status](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/test_mpc.yml/badge.svg)

[![SonarCloud](https://sonarcloud.io/images/project_badges/sonarcloud-orange.svg)](https://sonarcloud.io/summary/new_code?id=jorgensd_dolfinx_mpc)

Author: Jørgen S. Dokken

This library contains an add-on to FEniCSx enabling the possibilities
of enforce multi-point constraints, such as 

$$u_i =\sum_{j=0,i \neq j}^n \alpha_j u_j, i\in I_N,$$

where $I_N$ is the set of degrees of freedom to constrain.

This can be used to for instance enforce slip conditions strongly.

Consider a linear system of the form 
$Au=b$, with the additional constraints written on the form $K\hat u=u$, where $K$ is a prolongation matrix, $\hat u$ is the vector of unknowns excluding the $I_N$ entries. 

We then solve the system 
$K^T A K \hat u = K^T b$, where $K^T A K$ is symmetric if $A$ was symmetric.
(For complex numbers, we use the Hermitian transpose and solve the system $\overline{K^T} A K \hat u = \overline{K^T} b$, where $\overline{K^T}$ is the complex conjugate of $K^T$, and $\overline{K^T} A K$ is Hermitian if $A$ was Hermitian.)

If we include boundary conditions on the form $u=g$, we 
assemble the system
$K^TAK\hat u = K^T(b-A\hat g)$ where $A\hat g$ is an extension of the boundary condition $g$ to all degrees of freedom.

The library performs custom matrix and vector assembly adding the extra constraints to the set of linear equations. All assemblies are local to the process, and no MPI communication except when setting up the multi point constraints.

These assemblers are written in C++, but have equivalent Python assemblers in the optional `dolfinx_mpc.numba` module.

# Documentation
Documentation at [https://jorgensd.github.io/dolfinx_mpc](https://jorgensd.github.io/dolfinx_mpc)

# Installation

## Conda
The DOLFINx MPC package is now on Conda.
The C++ library can be found under [libdolfinx_mpc](https://anaconda.org/conda-forge/libdolfinx_mpc) and the Python library under
[dolfinx_mpc](https://anaconda.org/conda-forge/dolfinx_mpc). If you have any issues with these installations, add an issue at [dolfinx_mpc feedstock](https://github.com/conda-forge/dolfinx_mpc-feedstock).

## Docker

Version 0.6.1.post1 is available as an docker image at [Github Packages](https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared ghcr.io/jorgensd/dolfinx_mpc:v0.6.1.post1
```
To change to complex mode run `source dolfinx-complex-mode`.
Similarly, to change back to real mode, call `source dolfinx-real-mode`.

## Source

To install the latest version (main branch), you need to install the latest release of [DOLFINx](https://github.com/FEniCS/dolfinx).
Easiest way to install DOLFINx is to use docker. The DOLFINx docker images goes under the name [dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx).
Remember to use an appropriate tag to get the correct version of DOLFINx, i.e. (`:nightly` or `:vx.y.z`).

To install the `dolfinx_mpc`-library run the following code from this directory:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir cpp/
ninja -j3 install -C build-dir
python3 -m pip install python/. --upgrade
```
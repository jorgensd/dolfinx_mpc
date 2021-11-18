# Multi-point constraints with FEniCS-X
![CI status](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/test_mpc.yml/badge.svg)

Author: Jørgen S. Dokken

This library contains an add-on to FEniCS_X enabling the possibilities
of enforce multi-point constraints, such as 
$u_i =\sum_{j=0,i{\not=}j}^n \alpha_j u_j, i\in I_N$

This can be used to for instance enforce slip conditions strongly.

Consider a linear system of the form 
$Au=b$, with the additional constraints written on the form $K\hat u=u$, where $K$ is a prolongation matrix, $\hat u$ is the vector of unknowns excluding the $I_N$ entries. 

We then solve the system 
$K^T A K \hat u = K^T b$, where $K^T A K$ is symmetric if $A$ was symmetric.

If we include boundary conditions on the form $u=g$, we 
assemble the system
$K^TAK\hat u = K^T(b-A\hat g)$ where $A\hat g$ is an extension of the boundary condition $g$ to all degrees of freedom.

The library performs custom matrix and vector assembly adding the extra constraints to the set of linear equations. All assemblies are local to the process, and no MPI communication except when setting up the multi point constraints.

These assemblers are written in C++, but have equivalent Python assemblers in the optional `dolfinx_mpc.numba` module.


# Installation

Version 0.3.0 is available as an docker image at [DockerHub](https://hub.docker.com/r/dokken92/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared dokken92/dolfinx_mpc:0.3.0
```

To install the latest version (master branch), you need to install the latest release of [dolfinx](https://github.com/FEniCS/dolfinx).
Easiest way to install dolfinx is to use docker. The dolfinx docker images goes under the name [dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx)

To install the `dolfinx_mpc`-library run the following code from this directory:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir ../cpp/
ninja -j3 install -C build-dir
pip3 install python/. --upgrade
```
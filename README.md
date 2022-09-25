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

If we include boundary conditions on the form $u=g$, we 
assemble the system
$K^TAK\hat u = K^T(b-A\hat g)$ where $A\hat g$ is an extension of the boundary condition $g$ to all degrees of freedom.

The library performs custom matrix and vector assembly adding the extra constraints to the set of linear equations. All assemblies are local to the process, and no MPI communication except when setting up the multi point constraints.

These assemblers are written in C++, but have equivalent Python assemblers in the optional `dolfinx_mpc.numba` module.


# Installation

A Docker image can be found in the [Packages](https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc) tab.

## Source

To install the latest version (main branch), you need to install the latest release of [dolfinx](https://github.com/FEniCS/dolfinx).
Easiest way to install dolfinx is to use docker. The dolfinx docker images goes under the name [dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx)

To install the `dolfinx_mpc`-library run the following code from this directory:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir cpp/
ninja -j3 install -C build-dir
pip3 install python/. --upgrade
```

## Dockerhub

Version 0.5.0 is available as an docker image at [DockerHub](https://hub.docker.com/r/dokken92/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared dokken92/dolfinx_mpc:v0.5.0
```

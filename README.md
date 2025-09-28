# Multi-point constraints with FEniCS-X
[![Github Pages](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/deploy-pages.yml/badge.svg?branch=main)](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/deploy-pages.yml)
[![SonarCloud](https://sonarcloud.io/images/project_badges/sonarcloud-orange.svg)](https://sonarcloud.io/summary/new_code?id=jorgensd_dolfinx_mpc)

[Code Coverage Report](https://jsdokken.com/dolfinx_mpc/code-coverage-report/index.html)

Author: Jørgen S. Dokken

This library contains an add-on to FEniCSx enabling the possibilities
of enforce multi-point constraints, such as 

$$u_i =\sum_{j=0,i \neq j}^n \alpha_j u_j, i\in I_N,$$

where $I_N$ is the set of degrees of freedom to constrain.

This can be used to for instance enforce slip conditions strongly.

Consider a linear system of the form 
$Au=b$, with the additional constraints written on the form ${K\hat{u}=u}$, where $K$ is a prolongation matrix, $\hat{u}$ is the vector of unknowns excluding the $I_N$ entries. 

We then solve the system 
${K^T A K \hat{u} = K^T b}$, where $K^T A K$ is symmetric if $A$ was symmetric.
For complex numbers, we use the Hermitian transpose and solve the system ${\overline{K^T} A K \hat{u} = \overline{K^T} b}$, where $\overline{K^T}$ is the complex conjugate of $K^T$, and $\overline{K^T} A K$ is Hermitian if $A$ was Hermitian.

If we include boundary conditions on the form $u=g$, we 
assemble the system
${K^TAK\hat{u} = K^T(b-A\hat{g})}$ where ${A\hat{g}}$ is an extension of the boundary condition $g$ to all degrees of freedom.

The library performs custom matrix and vector assembly adding the extra constraints to the set of linear equations. All assemblies are local to the process, and no MPI communication except when setting up the multi point constraints.

These assemblers are written in C++, but have equivalent Python assemblers in the optional `dolfinx_mpc.numba` module.

# Documentation
Documentation at [https://jorgensd.github.io/dolfinx_mpc](https://jorgensd.github.io/dolfinx_mpc)

# Installation

## Spack
DOLFINx MPC is on spack as both a [C++ package](https://packages.spack.io/package.html?name=dolfinx-mpc) (dolfinx-mpc) and a [Python package](https://packages.spack.io/package.html?name=py-dolfinx-mpc) (py-dolfinx-mpc).
First, clone the spack repository and enable spack

```bash
git clone --depth=2 https://github.com/spack/spack.git
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

Next create an environment:

```bash
spack env create mpc_env
spack env activate mpc_env
```
Find the compilers on the system
```bash
spack compiler find
```

and install the relevant package

### C++
```bash
spack add dolfinx-mpc@v0.9.3 ^mpich ^petsc+mumps+hypre
spack concretize
spack install
```

### Python
```bash
spack add py-dolfinx-mpc@v0.9.3 ^mpich ^petsc+mumps+hypre ^py-fenics-dolfinx+petsc4py
spack add py-scipy py-pytest py-gmsh
spack concretize
spack install
```

Finally, note that spack needs some packages already installed on your system. On a clean ubuntu container for example one need to install the following packages before running spack
```bash
apt update && apt install gcc unzip git python3-dev g++ gfortran xz-utils libzip2 -y
```

## Conda
The DOLFINx MPC package is now on Conda.
The C++ library can be found under [libdolfinx_mpc](https://anaconda.org/conda-forge/libdolfinx_mpc) and the Python library under
[dolfinx_mpc](https://anaconda.org/conda-forge/dolfinx_mpc). If you have any issues with these installations, add an issue at [dolfinx_mpc feedstock](https://github.com/conda-forge/dolfinx_mpc-feedstock).

## Docker

Version 0.9.0 is available as an docker image at [Github Packages](https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared ghcr.io/jorgensd/dolfinx_mpc:v0.9.0
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
python3 -m pip -v install --config-settings=cmake.build-type="Release" --no-build-isolation ./python -U
```

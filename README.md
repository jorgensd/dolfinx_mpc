# Multi-point constraints with FEniCS-X
[![Github Pages](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/deploy-pages.yml/badge.svg?branch=main)](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/deploy-pages.yml)
[![SonarCloud](https://sonarcloud.io/images/project_badges/sonarcloud-orange.svg)](https://sonarcloud.io/summary/new_code?id=jorgensd_dolfinx_mpc)

[Code Coverage Report](https://jsdokken.com/dolfinx_mpc/code-coverage-report/index.html)

Author: Jørgen S. Dokken

This library is an add-on to [DOLFINx](https://github.com/FEniCS/dolfinx) for imposing
multi-point constraints, linear relations between degrees of freedom,

$$
u_s = \sum_j c_{sj}\, u_{m_j} + g_s,
$$

where each constrained, *slave*, degree of freedom $u_s$ is given by *master* degrees of freedom
$u_{m_j}$, with known coefficients $c_{sj}$ and an offset $g_s$. A master may carry a Dirichlet
condition, and its value is then part of the offset. The constraints are eliminated from the
system as it is assembled, so the solution satisfies them exactly.

<!-- The book's front page, index.md, shows its gallery here -->

The library provides:

- **Slip conditions**, $u\cdot n = 0$ on boundaries not aligned with the axes;
- **Periodic conditions**, also with a phase, as in Floquet–Bloch conditions;
- **Contact** conditions between non-matching interfaces, with or without slip;
- **Integral conditions**, such as a prescribed mean value, boundary average or flow rate;
- **Spiders**, which tie degrees of freedom to a point, rigidly (RBE2) or as a weighted average
  (RBE3);
- **Constraints between spaces and meshes**, with masters in another block of the system, such as
  a field on a submesh tied to the trace of its parent;
- **General constraints**, from a map between degrees of freedom, or from slaves, masters and
  coefficients given directly.

The constraints apply to linear and nonlinear problems, to block systems, assembled as nest or
single matrices, for real and complex scalars, in serial and in parallel. The assembly is written in
C++, with a Python interface. How the constraints are eliminated is described in the theory section
of the [documentation](https://jorgensd.github.io/dolfinx_mpc/docs/elimination.html).


# Documentation
Documentation at [https://jorgensd.github.io/dolfinx_mpc](https://jorgensd.github.io/dolfinx_mpc)

<!-- The book's front page, index.md, continues from here: it is the documentation itself -->

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

Get the relevant packages repos (FEniCS and Scientific Computing)
```bash
spack repo add https://github.com/FEniCS/spack-fenics.git
spack repo add https://github.com/scientificcomputing/spack_repos.git
```

and install the relevant package

### C++
```bash

spack add dolfinx-mpc@0.10 ^mpich ^petsc+mumps+hypre
spack concretize
spack install
```

### Python
```bash
spack add py-dolfinx-mpc@0.10 ^mpich ^petsc+mumps+hypre ^py-fenics-dolfinx+petsc4py
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

Version 0.10.1 is available as an docker image at [Github Packages](https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared ghcr.io/jorgensd/dolfinx_mpc:v0.10.1
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

# Multi-point constraints with FEniCS-X
![CI status](https://github.com/jorgensd/dolfinx_mpc/actions/workflows/test_mpc.yml/badge.svg)

Author: Jørgen S. Dokken

This library contains an add-on to FEniCS_X enabling the possibilities
of enforce multi-point constraints, such as $`u_i =
\sum_{j=0,i{\not=}j}^n \alpha_j u_j`$.

This can be used to for instance enforce slip conditions strongly. To
add multi-point constraints, we eliminate the "slave" degrees of freedom
on an element level, keeping the symmetry of the original system.

The custom assembler is written in python, using numba for just in time
compilation.

This library is dependent on C++ code, which does local modifications to
dolfinx::la::SparsityPattern and dolfinx::common::IndexMap to be able to
use PETSc matrices.

# Installation

Version 0.1.0 is available as an docker image at [DockerHub](https://hub.docker.com/r/dokken92/dolfinx_mpc)
and can be ran using
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared dokken92/dolfinx_mpc:0.1.0
```

To install the latest version (master branch), you need to install the latest release of [dolfinx](https://github.com/FEniCS/dolfinx).
Easiest way to install dolfinx is to use docker. The dolfinx docker images goes under the name [dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx)

To install the `dolfinx_mpc`-library run the following code from this directory:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build-dir ../cpp/
ninja -j3 install -C build-dir
pip3 install python/. --upgrade
```
# Multi-point constraints with FEniCS-X

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

Requirements: dolfinx
Easiest way to install dolfinx is to use docker.
The dolfinx docker images goes under the name [dolfinx/real](https://hub.docker.com/r/dolfinx/real) and [dolfinx/complex](https://hub.docker.com/r/dolfinx/complex).

To install the library run the following code from this directory:
```
rm -rf build
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/
ninja -j3 install
cd ../python
pip3 install . --upgrade
```
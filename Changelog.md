# Changelog

## main
- **API**:
  - Move `dolfinx_mpc.cpp.mpc.create_*_contact_condition` to `dolfinx_mpc.MultiPointConstraint.create_*_contact_condition`.
  - New default assembler: The default for `assemble_matrix` and `assemble_vector` is now C++ implementations. The numba implementations can be accessed through the submodule `dolfinx_mpc.numba`.
  - New submodule: `dolfinx_mpc.numba`. This module contains the `assemble_matrix` and `assemble_vector` that uses numba.
  - The `mpc_data` is fully rewritten, now the data is accessible as properties `slaves`, `masters`, `owners`, `coeffs` and `offsets`.
  - The `MultiPointConstraint` class has been rewritten, with the following functions changing
    - The `add_constraint` function now only accept single arrays of data, instead of tuples of (owned, ghost) data.
    - `slave_cells` does now longer exist as it can be gotten implicitly from `cell_to_slaves`.

- **Performance**:
  - The C++ assembler has been fully rewritten and has a "Insert number" speed-up for the `demo_contact_3D.py` with `XXX` dofs.
- **DOLFINX API-changes**:
  - Updates to match dolfinx implementation of exterior facet integrals
  - Updated user-interface of `dolfinx.Constant`, explicitly casting scalar-type with `PETSc.ScalarType`.
  - Various internal changes to handle new `dolfinx.DirichletBC` without class inheritance
  - Various internal changes to handle new way of JIT-compliation of `dolfinx::fem::Form_{scalar_type}`

## 0.3.0 (25.08.2021)
- Minor internal changes

## 0.2.0 (06.08.2021)
- Add new MPC constraint: Periodic boundary condition constrained geometrically. See `demo_periodic_geometrical.py` for use-case.
- New: `demo_periodic_gep.py` proposed and initally implemented by [fmonteghetti](https://github.com/fmonteghetti) using SLEPc for eigen-value problems. 
  This demo illustrates the usage of the new `diagval` keyword argument in the `assemble_matrix` class. 

- **API**:
  - Renaming and clean-up of `assemble_matrix` in C++
  - Renaming of Periodic constraint due to additional geometrical constraint, `mpc.create_periodic_constraint` -> `mpc.create_periodic_constraint_geometrical/topological`.
  - Introduce new class `dolfinx_mpc.LinearProblem` mimicking the DOLFINx class (Usage illustrated in `demo_periodic_geometrical.py`)
  - Additional `kwarg` `b: PETSc.Vec` for `assemble_vector` to be able to re-use Vector.
  - Additional `kwargs`: `form_compiler_parameters` and `jit_parameters` to `assemble_matrix`, `assemble_vector`, to allow usage of fast math etc.  

- **Performance**:
  - Slip condition constructor moved to C++ (Speedup for large problems)
  - Use scipy sparse matrices for verification
- **Misc**:
  - Update GMSH code in demos to be compatible with [GMSH 4.8.4](https://gitlab.onelab.info/gmsh/gmsh/-/tags/gmsh_4_8_4).
- **DOLFINX API-changes**:
  - `dolfinx.cpp.la.scatter_forward(x)` is replaced by `x.scatter_forward()`
  - Various interal updates to match DOLFINx API (including dof transformations moved outside of ffcx kernel)


## 0.1.0 (11.05.2021)
- First tagged release of dolfinx_mpc, compatible with [DOLFINx 0.1.0](https://github.com/FEniCS/dolfinx/releases/tag/0.1.0).

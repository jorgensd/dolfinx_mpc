# Changelog

## main
- **API**:
  - `assemble_matrix` and `assemble_vector` now only accepts compiled DOLFINx forms as opposed to `ufl`-forms. `LinearProblem` still accepts `ufl`-forms
  - `dolfinx_mpc.utils.create_normal_approximation` now takes in the meshtag and the marker, instead of the marked entities
  - No longer direct access to dofmap and indexmap of MPC, now collected through the `dolfinx_mpc.MultiPointConstraint.function_space`.
  - Introducing custom lifting operator: `dolfinx_mpc.apply_lifting`. Resolves a bug that woul occur if one had a non-zero Dirichlet BC on the same cell as a slave degree of freedom.
    However, one can still not use Dirichlet dofs as slaves or masters in a multi point constraint.
  - Move `dolfinx_mpc.cpp.mpc.create_*_contact_condition` to `dolfinx_mpc.MultiPointConstraint.create_*_contact_condition`.
  - New default assembler: The default for `assemble_matrix` and `assemble_vector` is now C++ implementations. The numba implementations can be accessed through the submodule `dolfinx_mpc.numba`.
  - New submodule: `dolfinx_mpc.numba`. This module contains the `assemble_matrix` and `assemble_vector` that uses numba.
  - The `mpc_data` is fully rewritten, now the data is accessible as properties `slaves`, `masters`, `owners`, `coeffs` and `offsets`.
  - The `MultiPointConstraint` class has been rewritten, with the following functions changing
    - The `add_constraint` function now only accept single arrays of data, instead of tuples of (owned, ghost) data.
    - `slave_cells` does now longer exist as it can be gotten implicitly from `cell_to_slaves`.

- **Performance**:
  - Major rewrite of periodic boundary conditions. On average at least a 5 x performance speed-up.
  - The C++ assembler has been fully rewritten.
  - Various improvements to `ContactConstraint`.

- **Bugs**
  - Resolved issue where `create_facet_normal_approximation` would give you a 0 normal for a surface dof it was not owned by any of the cells with facets on the surface.

- **DOLFINX API-changes**:
  - `dolfinx.fem.DirichletBC` -> `dolfinx.fem.dirichletbc`
  - `dolfinx.fem.Form` -> `dolfinx.fem.form`
  - Updates to use latest import schemes from dolfinx, including `UnitSquareMesh` -> `create_unit_square`.
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

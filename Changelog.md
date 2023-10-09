# Changelog

## v0.7.0

- **API**:
  - Change input of `dolfinx_mpc.MultiPointConstraint.homogenize` and `dolfinx_mpc.backsubstitution` to `dolfinx.fem.Function` instead of `PETSc.Vec`.
  - Add support for more floating types (`dtype` can now be specified as input for coefficients). In theory this means one could support single precision. Not thoroughly tested.
    - This resulted in a minor refactoring of the pybindings, meaning that tte class `dolfinx_mpc.cpp.mpc.MultiPointConstraint` is replaced by `dolfinx_mpc.cpp.mpc.MultiPointConstraint_{dtype}`
  - Casting scalar-type with `dolfinx.default_scalar_type` instead of `PETSc.ScalarType`
  - Remove usage of `VectorFunctionSpace`. Use blocked basix element instead.
- **DOLFINX API-changes**:
  - Use `dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))` instead of `dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))` as the latter is being deprecated.
  - Use `basix.ufl.element` in favor of `ufl.FiniteElement` as the latter is deprecated in DOLFINx.

## v0.6.1 (30.01.2023)

- Fixes for CI
- Add auto-publishing CI
- Fixes for `h5py` installation

## v0.6.0 (27.01.2023)

- Remove `dolfinx::common::impl::copy_N` in favor of `std::copy_n` by @jorgensd in #24
- Improving and fixing `demo_periodic_gep.py` by @fmonteghetti in #22 and @conpierce8 in #30
- Remove xtensor by @jorgensd in #25
- Complex valued periodic constraint (scale) by @jorgensd in #34
- Implement Hermitian pre-multiplication by @conpierce8 in #38
- Fixes for packaging by @mirk in #41, #42, #4
- Various updates to dependencies

## v0.5.0 (12.08.2022)

- Minimal C++ standard is now [C++20](https://en.cppreference.com/w/cpp/20)
- Deprecating GMSH IO functions from `dolfinx_mpc.utils`, see: [DOLFINx PR: 2261](https://github.com/FEniCS/dolfinx/pull/2261) for details.
- Various API changes in DOLFINx relating to `dolfinx.common.IndexMap`.
- Made code [mypy](https://mypy.readthedocs.io/en/stable/)-compatible (tests added to CI).
- Made code [PEP-561](https://peps.python.org/pep-0561/) compatible.

## v0.4.0 (30.04.2022)

- **API**:

  - **New feature**: Support for nonlinear problems (by @nate-sime) for mpc, see `test_nonlinear_assembly.py` for usage
  - Updated user interface for `dolfinx_mpc.create_slip_constraint`. See documentation for details.
  - **New feature**: Support for periodic constraints on sub-spaces. See `dolfinx_mpc.create_periodic_constraint` for details.
  - **New feature**: `assemble_matrix_nest` and `assemble_vector_nest` by @nate-sime allows for block assembly of rectangular matrices, with different MPCs applied for rows and columns. This is highlighed in `demo_stokes_nest.py`
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

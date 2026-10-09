# Changelog

## main
- **New feature**: chained constraints, where a master is itself a slave ([#280](https://github.com/jorgensd/dolfinx_mpc/issues/280)). `MultiPointConstraint.finalize` and `finalize_multipointconstraints` take `resolve_chains=False` and `round_limit=None`. With `resolve_chains=True`, the row of a slave is substituted wherever it is a master, also across blocks, by rounds of one exchange with the owners of the masters, until no master is a slave: $u_s = \sum_m c_{sm}u_m + g_s$ with $u_m = \sum_n c_{mn}u_n + g_m$ becomes $u_s = \sum_n c_{sm}c_{mn}u_n + c_{sm}g_m + g_s$. Substituted rows are ordered by block and global dof, with repeated masters merged, so the result does not depend on the partition. A cycle, or a chain longer than `round_limit` (default: the global number of slaves), raises a `ValueError` on every process. The rows supplied are kept: `all_coefficients`, `update_coefficients`, `scale_coefficients`, and so `update_rbe2` and `update_rbe3`, act on them and substitute the chains again, updating the rows of every block finalized together. The offsets $g_m$ of the substituted slaves follow `update_constants`. The default keeps the previous behaviour: a master that is a slave raises, now pointing to `resolve_chains`. New test: `python/tests/test_chained_constraints.py`. `impl::send_rows` moved from `point_basis.h` to `mpi_utils.h`.
- **Bug fix**: `create_general_constraint` (`create_dictionary_constraint`) could leave masters unresolved (index and owner `-1`) on ghost copies of a slave in parallel, when the master is owned by a process on which the slave is also a ghost: ghost entries were only completed from owned and non-local entries. The assembled system was then wrong without any error, while `backsubstitution` still satisfied the constraint. Ghost entries are now also completed from the ghost entries of the other processes. New test: `python/tests/test_dictionary_constraint.py`.
- **Change**: `create_general_constraint` (`create_dictionary_constraint`) exchanges the masters with a single `MPI_Allreduce` instead of gathering everything on process 0 and merging owned, ghost and non-local copies of each slave pairwise. Every copy of a slave, owned or ghost, gets its masters from the same reduced data, and the coefficients are read from each process's own dictionary, in the scalar type of the constraint. The input is checked: every process must pass the same dictionary, in the same order, with the same sub spaces, and a master point that no process has now raises a `ValueError`, where it previously gave the master `-1` silently. Both errors, and those of locating the slaves and masters, are raised on every process. The slaves and masters are located for all points at once, with a sorted search over the dof coordinates, instead of one `locate_dofs_geometrical` per point, which is 200 to 700 times faster for a few hundred points. In a blocked space, a point holds a dof of every component, so the sub spaces must be given; the block was previously taken as a dof.
- **New feature**: coefficients of a finalized `MultiPointConstraint` can be changed without recreating it ([#36](https://github.com/jorgensd/dolfinx_mpc/issues/36), [#154](https://github.com/jorgensd/dolfinx_mpc/issues/154)), e.g. for Floquet-Bloch conditions $u(x_s) = e^{ik\cdot L}u(x_m)$ swept over $k$. `scale_coefficients(scale)` multiplies the masters of each slave by a factor given as a scalar, a UFL expression (compiled to a `dolfinx.fem.Expression`), a compiled `Expression` or a `Function`, interpolated into the constraint's space; repeated calls compound. `update_coefficients(coeffs)` replaces all coefficients, in the layout of the new `all_coefficients()`/`all_masters()`, which include masters eliminated by a Dirichlet condition. Both update those eliminated masters and recompute the constraint offset. The masters are fixed at creation, so `create_periodic_constraint_*` accepts `coefficient_tol=0` to keep every basis value instead of cutting small ones. New demo: `python/demos/demo_scaling_coefficients.py`.
- **New feature**: `MultiPointConstraint.finalize` accepts an optional `filter`. A master whose coefficient satisfies $|c_{sj}| < \mathrm{filter}\cdot\max_k|c_{sk}|$, relative to the largest coefficient of that same slave, is discarded. Such a master contributes nothing to the constraint but still costs a ghost, a row of the sparsity pattern and an entry in every element matrix modification. Filtering is done in C++, is local and adds no communication; the default (`None`) keeps every master, reproducing the previous behaviour.
- **New feature**: `dolfinx_mpc.create_integral_constraint` and the corresponding `MultiPointConstraint.add_integral_constraint(weight_form, value, bcs=None, coefficient_tol=None)` express a scalar integral condition $L(u) = \gamma$, for any linear functional $L$ given as a UFL form, as a single-slave affine constraint: the degree of freedom with the largest coefficient becomes the slave, every other degree of freedom in the support of $L$ becomes a master with coefficient $-w_i/w_s$ ($w=$ the assembled vector of $L$), and $\gamma/w_s$ is written into `rhs_coeffs`. `weight_form`'s test function must be in the function space of the constraint, checked eagerly. Typical uses are a mean-value constraint $\int_\Omega u\,\mathrm{d}x=\gamma$ (removing the null space of a pure Neumann problem) or a boundary/flow-rate constraint $\int_\Gamma u\cdot n\,\mathrm{d}s=\gamma$, both alternatives to a real-space Lagrange multiplier that eliminate an unknown rather than adding one. Collective, and communicates only to the processes holding the chosen slave.
- **New demos**: `python/demos/demo_mean_value_constraint.py`, `python/demos/demo_boundary_average_constraint.py` and `python/demos/demo_flow_rate_constraint.py` show `create_integral_constraint` used for a mean-value, a boundary-average and a Stokes flow-rate constraint respectively, each verified against a `basix.ufl.real_element` solve and measuring the trade-off: for a cell integral every dof becomes a master, so $K^HAK$ is essentially full and its condition number is inflated by a factor $\sim M$; for a facet integral the master set is only the boundary and the constraint is the cheaper of the two formulations.
- **New feature**: Affine multi-point constraints: `MultiPointConstraint` now supports affine constraints of the form $x = K x_{\text{red}} + g$ via the new optional bcs and rhs_coeffs arguments. Time-dependent boundary data is supported via `MultiPointConstraint.update_constants()`. For manual linear assembly, use the new `dolfinx_mpc.apply_mpc_lifting` function (handled automatically by `LinearProblem`). NonlinearProblem automatically handles affine constraints and Dirichlet conditions without requiring any API changes. Passing neither of the new arguments reproduces the previous homogeneous behaviour. Note: Numba assemblers currently raise `NotImplementedError` for inhomogeneous constraints. For a full mathematical derivation of the offset $g$ and the linear/nonlinear solver paths, see the [theory document](./docs/nonlinear_mpc.md). 
- **Error handling**: The `MultiPointConstraint` constructor now throws `invalid_argument` if a dof is both a slave and Dirichlet-constraine. That was previously accepted silently.
- **New feature**: interior facet integrals (`dS`) are supported by the matrix and vector
  assemblers, lifting and the sparsity pattern, including on the interface between two
  subdomains, where an argument exists on one side of the facet only. New demo:
  `python/demos/demo_mortar_subdomains.py`, mortar coupling of two subdomains through a
  multiplier on the interface. Not supported by the numba assemblers.
- **Assembly takes dof markers**, following DOLFINx [#4583](https://github.com/FEniCS/dolfinx/pull/4583).
  `dolfinx_mpc::assemble_matrix` gained an overload taking `dof_marker0`/`dof_marker1`; the
  `bcs` overload remains as a convenience wrapper that rebuilds them per call. The new
  `dolfinx_mpc.BCData` caches markers and diagonal rows, keyed by function space, so one
  instance serves an operator, its preconditioner and its Jacobian. `LinearProblem` and
  `NonlinearProblem` hold one and reuse it, rather than rebuilding on every solve or Newton
  iteration.
- **Lifting takes dof markers and values.** `dolfinx_mpc::apply_lifting` is one template taking
  `bc_markers1`/`bc_values1` per block, plus a `bcs` wrapper; the four per-scalar-type overloads
  are gone. `dolfinx_mpc.apply_lifting` gained `bc_data`; markers are cached, values are re-read
  on every call. A condition now applies to block `j` if it is defined on (a subspace of)
  that block's trial space, so a flat list of conditions is accepted.
- **New feature**: `dolfinx_mpc.finalize_multipointconstraints(mpcs, filter=None)` finalizes the
  constraints of several function spaces together, for instance the blocks of a
  `ufl.MixedFunctionSpace`, each on its own mesh. The checks that need communication are reduced once
  for all of them. `MultiPointConstraint.finalize` is the one-space case and behaves as before; in C++,
  `create_multipointconstraints` is the factory and the constructor delegates to it. New errors, raised
  identically on every process: meshes on communicators of different size or rank order, and a master
  that is itself a slave, which `backsubstitution` does not resolve, and a slave constrained more than
  once, as when two periodic conditions share a corner, which previously failed on the offsets in serial
  and hung in parallel. A master without a local index in the extended space, which would previously
  have given wrong results silently, is also rejected.
- **New feature**: `MultiPointConstraint.input_space`, the function space a constraint was created with, on which forms
  and Dirichlet conditions are stated, while `function_space` is its extension by the masters after finalizing.
  `ufl.MixedFunctionSpace(*(mpc.input_space for mpc in mpcs))` gives the spaces in the order of the blocks.
- **Fix**: a Dirichlet condition on a block that no form touches, such as a spider tied to a body only by a constraint,
  now gets its diagonal and its value in the right-hand side. The sparsity pattern reserves the whole owned diagonal of
  such a block, and the assembly and `LinearProblem`/`NonlinearProblem` take the block's space from its constraint.
  The diagonal was missing before, which made the system singular, and no error was raised.
- **Error handling**: assembly raises `ValueError` on every process if a Dirichlet condition given to it constrains a
  master that the constraint kept. Give the condition to the constraint of the master's space as well,
  `MultiPointConstraint(V, bcs=...)`, so that the master is eliminated into the offset. Previously the slave's entries
  were added to the constrained row, a silently wrong solution.
- **New feature**: rigid spiders (RBE2). `MultiPointConstraint.add_rbe2_topological` and `add_rbe2_geometrical` tie
  dofs to the rigid-body motion of points of a spider mesh (`dolfinx_mpc.create_spider_mesh`), and `update_rbe2`
  recomputes the coefficients after the meshes move. C++: `dolfinx_mpc::create_rbe2`, `update_rbe2` and
  `locate_spiders`. Blocked vector assembly skips a block without a linear form. New demos:
  `python/demos/demo_spider.py`, and `python/demos/demo_spider_hinge.py`, a double pendulum of two beams on two pins,
  each pin a pair of spiders joined by a spring stiff in all but rotation about the pin.
  The spider utilities are in `dolfinx_mpc.spider`: `create_spider_mesh` takes the points on the first process (the
  others pass none or the same, checked on every process), keeps coinciding points as distinct spiders and agrees on one
  coordinate type; `create_spider_pair` relates spider k of two spider meshes of the same size, as an entity map for
  forms coupling them, such as springs; `move(mesh, u)` moves a mesh, or the spiders by their translations, by a
  displacement. The scalar type of a constraint is set once, by `MultiPointConstraint(V, dtype=...)`, which defaults
  to the precision of the mesh.
- **New feature**: flexible spiders (RBE3). `MultiPointConstraint.add_rbe3_topological` and `add_rbe3_geometrical`, on
  the constraint of a spider mesh, tie each spider to the weighted least-squares rigid fit of its feet, which may be on
  several meshes; `update_rbe3` recomputes the coefficients after the meshes move. C++: `dolfinx_mpc::create_rbe3` and
  `update_rbe3`. New demo: `python/demos/demo_rbe3.py`.
- **New feature**: constraints between a submesh and its parent. `MultiPointConstraint.create_submesh_constraint(V,
  master_space, entity_map)` ties every dof of `V` in a cell related through the entity map of
  `dolfinx.mesh.create_submesh` to `master_space` evaluated there, for submeshes of codimension 0 or 1, either way
  round, and subspaces on either side. The related cell is a table lookup, so no geometric search is done.
  `finalize_multipointconstraints` now also accepts a subspace of a block as master space. C++:
  `dolfinx_mpc::create_submesh_constraint`. New demo: `python/demos/demo_bulk_surface.py`, a bulk problem with a
  surface diffusion equation on part of its boundary, checked against a Lagrange multiplier. The demos in the
  documentation are now grouped by theme.
- **Periodic and contact constraints share one implementation.** `dolfinx_mpc::evaluate_basis_at_points`
  ([cpp/point_basis.h](cpp/point_basis.h)) finds a cell containing each slave point, on this process or another
  located through a global bounding-box tree, and evaluates the basis there; each constraint keeps only how it
  picks its slaves and builds a row. The constraints are unchanged at 1-4 processes. Contact no longer sends
  every unfound point to every process with master facets. A slave without masters in
  `create_contact_slip_condition`, or in `create_contact_inelastic_condition` without `allow_missing_masters`,
  now raises on every process: in parallel the slip condition previously kept the slave silently, and the
  inelastic one raised on one process and hung the others. Removed from the C++ API: `recv_data`,
  `send_master_data_to_owner`, `append_master_data`, `create_neighborhood_comms` and the vector overload of
  `create_owner_to_ghost_comm`.
- **Change**: one pair of tolerances for every constraint. `distance_tol` is the largest distance from a point to the
  cell or point it is matched to, and the padding of the bounding boxes; `coefficient_tol` drops a master whose
  coefficient is below `coefficient_tol` times the largest of its slave, and `0` keeps every master. Both are
  keyword arguments of `create_periodic_constraint_geometrical`/`_topological`, `create_contact_slip_condition` and
  `create_contact_inelastic_condition`; `create_general_constraint`, `create_dictionary_constraint` and `close_to` take
  `distance_tol`, and `add_integral_constraint`/`create_integral_constraint` and `create_submesh_constraint`, which
  looks its cells up rather than searching for them, take `coefficient_tol`. Both default to 500 machine epsilon, of the mesh's coordinate type for distances and of the constraint's real type for coefficients
  (C++: `dolfinx_mpc::default_tolerance<U>()`), rather than of `dolfinx.default_real_type`: a single precision mesh in a
  double precision build had a tolerance below its rounding, and slaves whose image was found in no cell were silently
  left unconstrained. What changes in the results:
  - the contact constraints dropped coefficients below the absolute value `1e-6`, which no argument could change;
  - the periodic `tol` was at once the padding, a squared distance and an absolute cut;
  - the contact default `eps2` was `1e-20`, which a distance computed in single precision cannot reach;
  - the dictionary constraint matched coordinates to within `1000` times the resolution of the mesh's type, and
    `close_to` to that of `dolfinx.default_real_type`.

  The pull-back of a slave point into a non-affine cell stops at `max(distance_tol², 500 eps)` rather than at the squared
  distance, which rounding could keep it from reaching. In C++, the eight `create_periodic_condition_*` overloads are
  two templates taking `distance_tol` and `coefficient_tol`.
- **Deprecated**: the periodic `tol` (a value sets both new tolerances, `None` sets `coefficient_tol=0`), the contact
  `eps2` (`distance_tol = sqrt(eps2)`), the integral `rtol` (`coefficient_tol`), the submesh `tol` (`coefficient_tol`,
  `None` sets it to `0`) and the `atol` of `close_to` (`distance_tol`). They warn and map to the new arguments.
- **New feature**: masters in another block. `MultiPointConstraint.add_constraint` takes `master_space` (all masters in
  that space) or `master_blocks` (a block per master, its position in the list given to
  `finalize_multipointconstraints`), and the masters are in the global numbering of their block. The constraints are
  finalized together, the extended index map of each block gaining the masters every block puts there. Assembly,
  lifting and back-substitution put a master's contributions in its own block, so a block-diagonal form can give a
  reduced system with off-diagonal blocks. Such a system is assembled with `assemble_matrix`/`assemble_vector` on the
  array of forms, `kind="nest"` or monolithic; the matrix of a single form rejects a constraint with masters elsewhere.
  `backsubstitution` takes the functions of every block. A Dirichlet condition on a master is taken from the master's
  block. The sparsity pattern now covers every block of the system (`dolfinx_mpc::create_sparsity_patterns`). C++
  callers assemble a block with `dolfinx_mpc::assemble_matrix_blocks`, routing the masters' entries to the matrices of
  the system with `dolfinx_mpc::make_mat_add_blocks`.
- **New**: C++ tests (`cpp/test`, Catch2) and C++ demos (`cpp/demo`), built against the installed library and run in
  CI. The first of each covers masters in another block: the test checks every block against a dense `K^T A K` with
  `la::MatrixCSR`, and `demo_cross_block` solves such a system with PETSc.
- **Changed**: a nest matrix now always has its diagonal blocks, which receive the diagonal of their slaves also when the
  block has no form; previously such slave rows were empty. With masters in another block every block of the nest
  exists. `create_matrix_nest` is built in C++ (`dolfinx_mpc::create_matrix_nest`).
- **New feature**: monolithic matrices and vectors for a system with one constraint per block. `dolfinx_mpc.create_matrix`,
  `assemble_matrix`, `create_vector` and `assemble_vector` take a `kind`, as in `dolfinx.fem.petsc`: a single form gives a
  matrix of that PETSc type, an array of forms with `kind="nest"` (or a nested sequence of matrix types, one per block)
  a `nest` matrix, and any other kind a single matrix with the blocks one after another, the dofs of each ordered
  `[owned, ghosts]` with the ghosts of its extended index map. A diagonal block without a form still gets the diagonal
  entry of its slaves. `LinearProblem` accepts `kind` as well; without one a problem with several constraints stays
  `nest`, and `kind="mpi"` selects the monolithic layout. Element tensors are inserted with the block-size dispatch
  of DOLFINx, so a vector-valued block works in the local sub-matrix of a monolithic matrix. C++:
  `dolfinx_mpc::create_matrix_block`.
- `NonlinearProblem` accepts the same `kind`: with one constraint per block, `None` or `"mpi"` gives a monolithic
  Jacobian and residual, `"nest"` or a nested sequence of matrix types a nest. The SNES callbacks receive the block
  layout of a monolithic system and set it on the vectors SNES hands them, as DOLFINx does.
- **BUGFIX**: the residual of a blocked `NonlinearProblem`, nest included, called `dolfinx.fem.petsc._assign_block_data`,
  which DOLFINx removed. `demo_stokes_nonlinear_nest.py` failed in SNES because of it and is run in CI again.
- **BUGFIX**: `demo_linear_wave_problem.py` did not zero its right-hand side between time steps. Assembly has been additive since the
  assemblers stopped zeroing their output, so the vector accumulated and the solution grew exponentially. It zeroes the vector
  now, uses the unified API, and checks that the energy is conserved.
- **Deprecated**: `create_matrix_nest`, `assemble_matrix_nest`, `create_vector_nest` and `assemble_vector_nest`. They
  still work and warn; use the functions above with `kind="nest"`, which also select the layout from the matrix or
  vector passed in.
- **BUGFIX**: `NonlinearProblem` failed with a non-nest `P`, and with `J=None` for a single form.
- `LinearProblem` accepts `entity_maps`, as `NonlinearProblem` already did, so forms coupling two meshes need not be compiled by hand.
- **Forms coupling spaces on different meshes** — a space on a submesh coupled to one on its
  parent, as in a mortar or Lagrange multiplier formulation — can now be assembled with a
  multi point constraint. **BUGFIX**: the exterior facet kernels were passed a null facet
  permutation, so such a form segfaulted. **BUGFIX**: the sparsity pattern, the vector
  assembler and `lifting.h` each indexed one argument's dofmap with another argument's cell
  index; all three now use their own integration entities. None of this was reachable
  before, since every test and demo used same-mesh forms. Creating a pattern for a form with
  interior facet integrals now raises instead of returning one the assembler would refuse.
  New demo: `python/demos/demo_mortar.py`.
- **Diagonal entries are added per block rather than per form.** `dolfinx_mpc::assemble_matrix`
  no longer takes `diagval` and no longer writes the slave diagonal; use
  `dolfinx_mpc::insert_slave_diagonal` (bound as `insert_diagonal_slaves`) once per diagonal
  block. This fixes a block whose diagonal form is `None` receiving no diagonal on its slave
  rows, and a block appearing in several forms receiving one twice. The Python signatures are
  unchanged.
- **The assemblers are additive**, matching the DOLFINx convention: a matrix or vector
  supplied by the caller is no longer zeroed. Zero it yourself first (`A.zeroEntries()`,
  `dolfinx.la.petsc._zero_vector(b)`). An object the assembler creates itself is still
  returned zeroed.
- `MultiPointConstraint`'s cell-to-slave map now covers ghost cells rather than owned cells
  only, removing a precondition on which cells a form may name.
- **BUGFIX**: `distribute_ghost_data` wrote past the end of two `reserve`d-but-never-`resize`d
  vectors whose values were never read; removed.
- **BUGFIX**: `modify_mpc_vec` zeroed the slave entry of the element vector inside the loop
  over that slave's masters, so a slave with an *empty* master list kept its entry. The
  reduced residual then had non-zero slave rows, which left the solution correct but made a
  nonlinear (SNES) solve stagnate at a finite residual norm. A master list is empty whenever
  every master is eliminated by a Dirichlet condition.

## V0.11.0

- Make MPC Nonlinearproblem prefix deterministic. See [PR 245](https://github.com/jorgensd/dolfinx_mpc/pull/245)
- Fix Jacobian in `dolfinx_mpc.NonlinearProblem` by enforcing the constraint prior to assembly. See [PR 246](https://github.com/jorgensd/dolfinx_mpc/pull/246)

## V0.10.5

- **BUGFIX**: Fix inelastic contact conditions for scalar spaces, see [PR 241](https://github.com/jorgensd/dolfinx_mpc/pull/241).

## V0.10.4

- **BUGFIX**: Fix typo in 0.10.3 release

## V0.10.3

- **BUGFIX**: Fixing periodic conditions for mixed spaces with block size !=1, see [PR 237](https://github.com/jorgensd/dolfinx_mpc/pull/237).

## V0.10.2

- **Installation**: Improved installation of Python interface by @jhale

## v0.10.1

- **Bugs**: Insertion in non-square matrices fixed. No change to user API.

## v0.10.0
- **New demo**: Periodic conditions for a linear wave, see [demo_linear_wave_problem.py](./python/demos/demo_linear_wave_problem.py)
- **New feature**: Use of a preconditioner for `dolfinx_mpc.LinearProblem`, as well as allowing for `NEST` systems. See [demo_stokes.py](./python/demos/demo_stokes.py)
- **New demo**: How to use SNES with fieldsplitting for non-linear problems, see [demo_stokes_nonlinear_nest.py](./python/demos/demo_stokes_nonlinear_nest.py) for an example.
- **API**
  - Update how to use assembly into PETSc NEST matrices, see [test_rectangular_assembly.py](./python/tests/test_rectangular_assembly.py)
- **Bugs**
  - Fix backsubstitution [PR #155](https://github.com/jorgensd/dolfinx_mpc/pull/155)


## v0.9.0

- No major API changes, only following DOLFINx API changes

## v0.8.0

- **API**
  - Various shared pointers in C++ interface is changed to const references
  - Multipoint-constraint now accept `std::span` instead of vectors
  - Now using [nanobind](https://github.com/wjakob/nanobind) for Python bindings
  - Switch to `pyproject.toml`, **see installation notes** for updated instructions
- **DOLFINx API-changes**
  - `dolfinx.fem.FunctionSpaceBase` replaced by `dolfinx.fem.FunctionSpace`
  - `ufl.FiniteElement` and `ufl.VectorElement` is replaced by `basix.ufl.element`

## v0.7.2

- **New feature**: Add support for "scalar" inelastic contact conditions. This is a special case where you want to create a periodic constraint between two sets of facets, which might or might not align.

## v0.7.1

- Patch for Python 3.8
- Fix import order of `mpi4py`, `petsc4py`, `dolfinx` and `dolfinx_mpc`

## v0.7.0

- **API**:
  - Change input of `dolfinx_mpc.MultiPointConstraint.homogenize` and `dolfinx_mpc.backsubstitution` to `dolfinx.fem.Function` instead of `PETSc.Vec`.
  - **New feature**: Add support for more floating types (float32, float64, complex64, complex128). The floating type of a MPC is related to the mesh geometry.
    - This resulted in a minor refactoring of the pybindings, meaning that the class `dolfinx_mpc.cpp.mpc.MultiPointConstraint` is replaced by `dolfinx_mpc.cpp.mpc.MultiPointConstraint_{dtype}`
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

# # Periodic boundary conditions for a representative volume element
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT

# We import the required modules

# + tags=["hide-input"]
from __future__ import annotations

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot

from dolfinx_mpc import LinearProblem, MultiPointConstraint, dofs_at_point

# -

# This demo constrains a unit cell so it behaves as one periodic tile of an
# infinite microstructure under a prescribed macroscopic strain, the boundary
# condition used to compute the effective (homogenized) elastic response of a
# composite from its unit cell. It follows the corner-node formulation of the
# periodic boundary conditions of
# {cite}`homog-Danas2017` (Appendix B), first used in
# {cite}`homog-LopezPamiesGoudarziDanas2013`,
# specialised here to small-strain linear elasticity: where that formulation
# treats a general hyperelastic energy $\psi(\mathbf{F})$, we restrict to a
# linear material so that the boundary condition can be expressed as an affine
# multi-point constraint and solved directly with
# {py:class}`dolfinx_mpc.LinearProblem`, without a penalty term or a Newton
# solve. The same constraint structure, layered onto
# {py:class}`dolfinx_mpc.NonlinearProblem`, is how the general case would be
# treated.
#
# We consider a square cell $\Omega=(0,L)^2$.

# +
comm = MPI.COMM_WORLD
N = 32
L = 1.0
dtype = np.dtype(default_scalar_type)
# Crossed: each square is cut into four triangles, so the mesh has the symmetries of the square
domain = mesh.create_rectangle(
    comm, [[0, 0], [L, L]], [N, N], diagonal=mesh.DiagonalType.crossed, dtype=default_real_type
)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
H_bar = np.array([[0.02, 0.01], [0.0, -0.015]])
assert H_bar.shape[0] == domain.geometry.dim
assert H_bar.shape[1] == domain.geometry.dim

# -

# ## The periodicity condition
#
# On a square unit cell $\Omega=(0,L)^2$ the displacement is split into an
# affine part carrying the macroscopic strain and a periodic fluctuation,
#
# $$
# \begin{aligned}
# \mathbf{u}(\mathbf{X}) &= (\bar{\mathbf{F}}-\boldsymbol{\delta})\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),\\
# \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) &= \mathbf{u}^*(\mathbf{X}),
# \end{aligned}
# $$
#
# where $\bar{\mathbf{F}}=\boldsymbol{\delta}+\bar{\mathbf{H}}$ is the prescribed
# average deformation gradient, $\boldsymbol{\delta}$ the identity, and
# $\bar{\mathbf{H}}$ the average displacement gradient (`H_bar` in the code
# below). Labelling the corners $A=(0,0)$, $B=(L,0)$, $C=(L,L)$, $D=(0,L)$,
# fixing $\mathbf{u}^A=\mathbf{0}$ to remove the rigid body mode, and writing
#
# $$
# \mathbf{u}^A = \mathbf{0}
# $$ (eq:homog-uA)
#
# $$
# \mathbf{u}^B = (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}L\\0\end{pmatrix}
# $$ (eq:homog-uB)
#
# $$
# \mathbf{u}^D = (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}0\\L\end{pmatrix}
# $$ (eq:homog-uD)
#
# for the two corners that carry the strain, periodicity of $\mathbf{u}^*$
# reduces every other constraint to
#
# $$
# \mathbf{u}^{\text{RIGHT}} = \mathbf{u}^{\text{LEFT}} + \mathbf{u}^B
# $$ (eq:homog-right)
#
# $$
# \mathbf{u}^{\text{TOP}} = \mathbf{u}^{\text{BOTTOM}} + \mathbf{u}^D
# $$ (eq:homog-top)
#
# $$
# \mathbf{u}^C = \mathbf{u}^B + \mathbf{u}^D
# $$ (eq:homog-uC)

# We split the conditions into several sub-components, which each expose a feature of
# DOLFINx-MPC.
#
# ### Dirichlet conditions on the corners
# For three of the corners ({eq}`eq:homog-uA`, {eq}`eq:homog-uB` and
# {eq}`eq:homog-uD`) we will apply {py:class}`dolfinx.fem.DirichletBC`
# directly, which is the simplest way to fix a degree of freedom.
# We mark the corner vertices once with a {py:class}`dolfinx.mesh.MeshTags` object,
# and locate their degrees of freedom topologically.
# We create a convenience function for this


def create_dirichletbc(
    V: fem.FunctionSpace, tag: mesh.MeshTags, tags: tuple[int, ...] | int, value: np.ndarray
) -> fem.DirichletBC:
    """A Dirichlet condition fixing every degree of freedom associated with the entity closure
    of the tagged entities to ``value``.

    Args:
        V: The function space to constrain. This might be an un-collapsed subspace.
        tag: The mesh tag marking the entities to constrain.
        tags: The tag values of the entities to constrain.
        value: The value to fix the degrees of freedom to. Must have shape of a function in `V`
            in physical (not reference) space.

    Returns:
        A Dirichlet boundary condition.
    """
    # Find all dofs associated with given entities
    assert V.mesh.topology == tag.topology
    entities = tag.indices[np.isin(tag.values, tags)]
    # A subspace holds the value in its collapsed space
    is_sub = len(V.component()) > 0
    V_c = V.collapse()[0] if is_sub else V
    dofs = fem.locate_dofs_topological((V, V_c) if is_sub else V, tag.dim, entities)
    # Populate function with constant value
    fn = fem.Function(V_c)
    fn.interpolate(lambda x: np.repeat(value, x.shape[1]).reshape(-1, x.shape[1]))
    # Create and return BC
    return fem.dirichletbc(fn, dofs, V) if is_sub else fem.dirichletbc(fn, dofs)


# Furthermore, we create a function to locate a corner of the mesh with coordinates `(px, py)`.


def corner(px: float, py: float, atol: float = 500 * np.finfo(default_real_type).eps):
    """Indicator function for a single point, padded for a 3D coordinate array."""
    return lambda x: np.isclose(x[0], px, atol=atol) & np.isclose(x[1], py, atol=atol)


# With the helpers in place we can create the Dirichlet conditions on the corners.

# +
TAG_A, TAG_B, TAG_D = 1, 2, 3
corner_points = {TAG_A: (0, 0), TAG_B: (L, 0), TAG_D: (0, L)}
vertex_map = domain.topology.index_map(0)
vertex_values = np.full(vertex_map.size_local + vertex_map.num_ghosts, -1, dtype=np.int32)
for tag, point_ in corner_points.items():
    vertex_values[mesh.locate_entities_boundary(domain, 0, corner(*point_))] = tag
marked_vertices = np.flatnonzero(vertex_values != -1).astype(np.int32)
corner_tags = mesh.meshtags(domain, 0, marked_vertices, vertex_values[marked_vertices])
domain.topology.create_connectivity(0, domain.topology.dim)

u_B = H_bar @ np.array([L, 0.0])
u_D = H_bar @ np.array([0.0, L])
bc_A = create_dirichletbc(V, corner_tags, (TAG_A,), np.zeros(domain.geometry.dim))
bc_B = create_dirichletbc(V, corner_tags, (TAG_B,), u_B)
bc_D = create_dirichletbc(V, corner_tags, (TAG_D,), u_D)
bcs = [bc_A, bc_B, bc_D]
# -

# ## Periodic constraints with corner masters
# {eq}`eq:homog-right`, {eq}`eq:homog-top` and {eq}`eq:homog-uC` share one form. For a node
# $\mathbf{X}$ on RIGHT, TOP or at $C$, let $s_i=1$ if $X_i=L$ and $s_i=0$ otherwise, and let
# $\mathbf{X}^-=\mathbf{X}-L\mathbf{s}$ be its image on LEFT, BOTTOM or at $A$. Then
#
# $$
# \mathbf{u}(\mathbf{X}) = \mathbf{u}(\mathbf{X}^-) + s_1\,\mathbf{u}^B + s_2\,\mathbf{u}^D .
# $$ (eq:homog-periodic)
#
# We build it in two steps:
#
# 1. {py:meth}`create_periodic_constraint_geometrical
#    <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>` ties every such node
#    to its image, $\mathbf{u}(\mathbf{X}) = \mathbf{u}(\mathbf{X}^-)$, finding the image in
#    parallel. The corners $B$ and $D$ are not slaves: for them {eq}`eq:homog-periodic` is an
#    identity.
# 2. {py:meth}`add_master_from_point <dolfinx_mpc.MultiPointConstraint.add_master_from_point>`
#    adds the dofs at a point, times a coefficient, to the right-hand side of the constraints of
#    the slaves it marks, component by component. One call per direction adds the corner of that
#    direction, with coefficient $1$, to the nodes with $X_i=L$: $\mathbf{u}^B$ on RIGHT and at
#    $C$, then $\mathbf{u}^D$ on TOP and at $C$, so that $C$ gets both.
#
# The corners $B$ and $D$ carry Dirichlet conditions here. Passing the conditions as `bcs` to the
# constraint folds their values into its offset, so the constraint needs no constant term of its
# own. {eq}`eq:homog-uC` has the Dirichlet corner $A$ as its periodic master, which is folded the
# same way. Under stress control, below, the same constraint is used with $B$ and $D$ free.

# +
gdim = domain.geometry.dim
xdt = domain.geometry.x.dtype
# Tolerance of the checks below, from the precision of the mesh coordinates
atol = 50 * np.sqrt(np.finfo(xdt).resolution)
master_corners = [(L, 0.0), (0.0, L)]  # B and D: the corner one period from A in direction i


def periodic_nodes(x, atol: float = 500 * np.finfo(default_real_type).eps):
    """RIGHT, TOP and C: the nodes with some X_i = L, except the corners B and D."""
    shifted = np.isclose(x[0], L, atol=atol) | np.isclose(x[1], L, atol=atol)
    is_master_corner = [corner(*p, atol=atol)(x) for p in master_corners]
    return shifted & ~np.logical_or.reduce(is_master_corner)


def to_image(x, atol: float = 500 * np.finfo(default_real_type).eps):
    """X -> X^- = X - L s."""
    out = x.copy()
    out[:gdim][np.isclose(x[:gdim], L, atol=atol)] -= L
    return out


def periodic_cell_constraint(bcs: list[fem.DirichletBC]) -> MultiPointConstraint:
    """The constraint {eq}`eq:homog-periodic`, finalized, with the conditions `bcs` folded in."""
    mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, periodic_nodes, to_image, bcs, scale=dtype.type(1.0))
    for i, corner_point in enumerate(master_corners):  # s_i u^{corner} on the nodes with X_i = L

        def on_side(x, i=i):
            return periodic_nodes(x) & np.isclose(x[i], L, atol=500 * np.finfo(default_real_type).eps)

        mpc.add_master_from_point(V, on_side, 1.0, corner_point)
    mpc.finalize()  # collective: every rank must reach this
    return mpc


# -

# With the helpers in place we create the constraint for the Dirichlet conditions above.

mpc = periodic_cell_constraint(bcs)

# ## Verifying the mechanism: a homogeneous unit cell
#
# For a single homogeneous material the periodic fluctuation $\mathbf{u}^*$ must
# vanish identically -- a homogeneous medium looks the same from every unit
# cell, so there is nothing left to fluctuate -- and the exact solution is the
# affine field $\mathbf{u}(\mathbf{X})=\bar{\mathbf{H}}\mathbf{X}$ itself. That
# makes this the sharpest test of the constraint: any bug in the corner or edge
# handling shows up as a nonzero fluctuation.


# +
def elasticity_forms(V: fem.FunctionSpace, mu, lmbda):
    """Linear elasticity forms with (possibly spatially varying) Lame parameters."""
    domain = V.mesh
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    zero = fem.Constant(domain, np.zeros(2, dtype=default_scalar_type))
    L = ufl.inner(zero, v) * ufl.dx
    return a, L


E_uniform, nu = 10.0, 0.3
mu_uniform = fem.Constant(domain, default_scalar_type(E_uniform / (2 * (1 + nu))))
lmbda_uniform = fem.Constant(domain, default_scalar_type(E_uniform * nu / ((1 + nu) * (1 - 2 * nu))))
a, Lform = elasticity_forms(V, mu_uniform, lmbda_uniform)

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(a, Lform, mpc, bcs=bcs, petsc_options=petsc_options)
uh_homogeneous = problem.solve()
# -

# The bound to check against is machine precision, not discretization error:
# $\bar{\mathbf{H}}\mathbf{X}$ is itself linear in $\mathbf{X}$, so it lies
# *exactly* in the P1 space `V` is built from -- there is no interpolation gap
# for the mesh resolution to close. A correct implementation therefore
# reproduces it to floating-point roundoff regardless of $N$; anything larger,
# such as an error that shrinks with mesh refinement or scales with $\bar H$,
# would mean a real bug in the corner or edge constraint, not insufficient
# resolution. The error is compared with the size of the affine field, scaled by
# `atol`, $50$ times the square root of the resolution of the coordinate type of the
# mesh. That bound lies well above the round-off (measured in the $10^{-15}$ range
# in double precision), so it is robust to the round-off growing with the number of
# processes, and it holds in single precision as well.


# +
def assemble_scalar_global(form: fem.Form):
    """The value of a compiled scalar form, summed over all processes."""
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


x = ufl.SpatialCoordinate(domain)
u_affine = ufl.dot(ufl.as_tensor(H_bar), x)
diff = uh_homogeneous - u_affine
error = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_affine = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_affine, u_affine) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < atol * norm_affine
# -

# ## A heterogeneous microstructure
#
# The point of a periodic unit cell is to homogenize a microstructure, not a
# uniform block, so the second solve gives the matrix a stiff circular
# inclusion. Two checks, none of them requiring a closed-form solution, and a report:
#
# 1. With no macroscopic strain the cell carries no load, so the volume
#    averaged stress must vanish exactly whatever the microstructure.
# 2. Under an isotropic macroscopic strain the circular inclusion is symmetric
#    under a $90°$ rotation and under reflections, so the averaged stress must be
#    isotropic too -- a property of the geometry, not of any elastic constant. The
#    crossed mesh, and the inclusion made of its cells, share these symmetries, so
#    this holds to round-off. A mesh with all diagonals in one direction does not:
#    there $\bar\sigma_{12}$ is a discretization error, about $10^{-3}$ here.
# 3. Under a general macroscopic strain we report the homogenized stress of this
#    microstructure. The periodicity equations hold to solver precision by
#    construction: {py:meth}`dolfinx_mpc.MultiPointConstraint.backsubstitution`
#    enforces them after every solve.

Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q, dtype=dtype)
tdim = domain.topology.dim
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
# The owned cells of the inclusion, a disc of radius 0.25 at the centre of the cell
cells0 = np.flatnonzero((midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2).astype(np.int32)
E.interpolate(lambda x: np.full(x.shape[1], E_uniform))
E.interpolate(lambda x: np.full(x.shape[1], 50.0 * E_uniform), cells0=cells0)
E.x.scatter_forward()
mu_field = E / (2 * (1 + nu))
lmbda_field = E * nu / ((1 + nu) * (1 - 2 * nu))
a_het, L_het = elasticity_forms(V, mu_field, lmbda_field)

# The stress of the heterogeneous cell is averaged for the solutions of several problems, so its
# forms are compiled once, for a function in V that average_stress fills: the leading entries of
# a constraint space are those of V
u_avg = fem.Function(V, dtype=dtype)
eps_avg = ufl.sym(ufl.grad(u_avg))
sigma_avg_ufl = 2 * mu_field * eps_avg + lmbda_field * ufl.tr(eps_avg) * ufl.Identity(2)
sigma_forms = [[fem.form(sigma_avg_ufl[i, j] * ufl.dx, dtype=dtype) for j in range(2)] for i in range(2)]


def average_stress(uh: fem.Function) -> np.ndarray:
    """The volume average of the stress of the heterogeneous cell for the solution `uh`."""
    u_avg.x.array[:] = uh.x.array[: u_avg.x.array.size]
    return np.array([[assemble_scalar_global(s_ij) for s_ij in row] for row in sigma_forms]).real / L**2


# The parameter study below solves the heterogeneous cell for several macroscopic strains.
# Only the Dirichlet values of the corners $B$ and $D$ change with the strain, and
# {py:class}`dolfinx_mpc.LinearProblem` folds them into the constraint before every solve
# ({py:meth}`update_constants <dolfinx_mpc.MultiPointConstraint.update_constants>`). So one
# problem, on the constraint built above, serves every strain.


def set_corner_values(H_bar_case: np.ndarray):
    """Set the Dirichlet values {eq}`eq:homog-uB` and {eq}`eq:homog-uD` of B and D for `H_bar_case`."""
    for bc, corner_point in ((bc_B, (L, 0.0)), (bc_D, (0.0, L))):
        u_corner = H_bar_case @ np.array(corner_point)
        bc.g.interpolate(lambda x: np.tile(u_corner.reshape(-1, 1), x.shape[1]))


problem_het = LinearProblem(a_het, L_het, mpc, bcs=bcs, petsc_options=petsc_options)


def homogenized_stress(problem: LinearProblem, H_bar_case: np.ndarray) -> tuple[fem.Function, np.ndarray]:
    """Solve the heterogeneous cell for ``H_bar_case`` and return (uh, (s11, s22, s12)).

    The solution is a copy: the problem returns the same function at every solve."""
    set_corner_values(H_bar_case)
    uh = problem.solve()
    assert isinstance(uh, fem.Function)
    uh = uh.copy()
    sigma_avg = average_stress(uh)
    return uh, np.array([sigma_avg[0, 0], sigma_avg[1, 1], sigma_avg[0, 1]])


# ### Test case 1: No macroscopic strain

_, sigma_zero = homogenized_stress(problem_het, np.zeros((2, 2)))
if comm.rank == 0:
    print(f"----No macroscopic strain----\n  sigma_avg = {sigma_zero}  (should vanish)")
assert np.abs(sigma_zero).max() < atol

# ### Test case 2: Isotropic macroscopic strain

_, sigma_iso = homogenized_stress(problem_het, 0.02 * np.eye(2))
if comm.rank == 0:
    print(
        f"----Isotropic macroscopic strain----\n  s11={sigma_iso[0]:.5f}  s22={sigma_iso[1]:.5f}  "
        f"s12={sigma_iso[2]:.2e}  (should be isotropic: s11≈s22, s12≈0)"
    )
assert abs(sigma_iso[0] - sigma_iso[1]) < atol * abs(sigma_iso[0])
assert abs(sigma_iso[2]) < atol * abs(sigma_iso[0])

# ### Test case 3: General macroscopic strain

uh_general, sigma_general = homogenized_stress(problem_het, H_bar)
if comm.rank == 0:
    print(
        f"----General macroscopic strain----\n  s11={sigma_general[0]:.5f}  s22={sigma_general[1]:.5f}  "
        f"s12={sigma_general[2]:.5f}"
    )

# ## Visualization
#
# The mesh is partitioned in parallel, so each process holds only a piece of the
# field. Each one builds a PyVista grid over the cells it *owns*, and the grids
# are gathered onto one process and drawn into a single figure with common
# colour limits.


# + tags=["hide-input"]
def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str, root: int = 0):
    """Owned-cell PyVista grids with ``u`` attached, gathered on ``root``.

    Vector fields are padded to three components, as PyVista expects.
    """
    comm = V.mesh.comm
    bs = V.dofmap.index_map_bs
    tdim = V.mesh.topology.dim
    owned_cells = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned_cells))
    values = u.x.array.real[: grid.n_points * bs]
    if bs == 1:
        grid.point_data[name] = values
        magnitude = np.abs(values)
    else:
        padded = np.zeros((grid.n_points, 3))
        padded[:, :bs] = values.reshape(-1, bs)
        grid.point_data[name] = padded
        grid.set_active_vectors(name)
        magnitude = np.linalg.norm(padded, axis=1)
        grid.point_data[f"|{name}|"] = magnitude
    lo = comm.allreduce(float(magnitude.min()) if magnitude.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(magnitude.max()) if magnitude.size else -np.inf, op=MPI.MAX)
    return comm.gather(grid, root=root), [lo, hi]


# -

# `uh_general` lives in the constraint's extended space, which carries the
# master dofs as extra ghosts, so its array is longer than the original space in
# parallel. The extended index map keeps the original dofs first, so the
# leading entries are exactly the values of the original space.

# + tags=["hide-input"]
u_plot = fem.Function(V, dtype=dtype)
u_plot.x.array[:] = uh_general.x.array[: u_plot.x.array.size]
disp_pieces, disp_clim = gather_grids(u_plot, V, "u")


def gather_cell_data(field: fem.Function, name: str, root: int = 0):
    """Owned-cell PyVista grids with a cellwise-constant field, gathered on ``root``.

    `plot.vtk_mesh` needs a point layout, which a cellwise-constant (DG0) space
    does not have, so the grid is built from the mesh topology directly and the
    field is attached as `cell_data`, one value per owned cell.
    """
    domain = field.function_space.mesh
    comm = domain.comm
    tdim = domain.topology.dim
    owned_cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, tdim, owned_cells))
    values = field.x.array.real[: len(owned_cells)]
    grid.cell_data[name] = values
    lo = comm.allreduce(float(values.min()) if values.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(values.max()) if values.size else -np.inf, op=MPI.MAX)
    return comm.gather(grid, root=root), [lo, hi]


material_pieces, material_clim = gather_cell_data(E, "E")

w_plot = fem.Function(V, dtype=dtype)
w_plot.interpolate(lambda x: H_bar @ x[:2])
w_plot.x.array[:] = u_plot.x.array - w_plot.x.array  # periodic fluctuation u* = u - H_bar X
fluct_pieces, fluct_clim = gather_grids(w_plot, V, "w")

figure = Path("demo_periodic_homogenization.py").with_suffix(".png")
if comm.rank == 0:
    outline = pyvista.Rectangle([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0)])
    bar = {"fmt": "%.1e", "n_labels": 3, "position_x": 0.2, "width": 0.6}
    plotter = pyvista.Plotter(shape=(1, 3), window_size=[780, 420])
    plotter.subplot(0, 0)
    plotter.add_text(f"Microstructure\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)", font_size=10)
    for piece in material_pieces:
        plotter.add_mesh(
            piece,
            scalars="E",
            cmap="viridis",
            clim=material_clim,
            show_edges=False,
            scalar_bar_args={"n_labels": 2, "fmt": "%.0f", "position_x": 0.2, "width": 0.6},
        )
    plotter.view_xy()
    load = "prescribed H = [[{:g}, {:g}], [{:g}, {:g}]]".format(*H_bar.ravel())
    for col, (pieces, clim, name, title) in enumerate(
        [
            (disp_pieces, disp_clim, "u", f"Deformed cell\n{load}"),
            (fluct_pieces, fluct_clim, "w", "Periodic fluctuation\nu* = u - H X"),
        ],
        start=1,
    ):
        factor = 0.1 * L / clim[1]  # largest displacement drawn as 10 % of the cell size
        plotter.subplot(0, col)
        plotter.add_text(f"{title}\n(amplified x{factor:.0f})", font_size=10)
        for piece in pieces:
            plotter.add_mesh(
                piece.warp_by_vector(name, factor=factor),
                scalars=f"|{name}|",
                cmap="viridis",
                clim=clim,
                scalar_bar_args={**bar, "title": f"|{name}|"},
            )
        plotter.add_mesh(outline, style="wireframe", color="black", line_width=2)
        plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(figure)
    else:
        plotter.show(screenshot=figure)
# -


# ## Stress control
#
# So far the strain was prescribed and the average stress computed from the solution:
#
# $$
# \bar{\boldsymbol{\sigma}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\sigma}(\mathbf{u})~\mathrm{d}x,
# $$
#
# the quantity `homogenized_stress` returns. We now prescribe its value instead,
# $\bar{\boldsymbol{\sigma}}=\bar{\mathbf{S}}$, and the displacements $\mathbf{u}^B$ and
# $\mathbf{u}^D$ become unknowns.
#
# This changes how the corners enter the problem. The constraint is the same, with the corners as
# masters of the periodic constraints. Under strain control $\mathbf{u}^B$, $\mathbf{u}^D$ are
# known: they are Dirichlet values on the corners, folded into the constraint. Under stress control
# they are unknowns, free masters, and the prescribed stress enters the right-hand side of the
# equations as point forces on their degrees of freedom, derived below.
# Each corner degree of freedom gets either a prescribed displacement or a prescribed force, never
# both.
#
# | | strain control | stress control |
# |---|---|---|
# | $\mathbf{u}^B$, $\mathbf{u}^D$ | prescribed, Dirichlet conditions | unknowns, masters of the constraints |
# | periodic constraints | constant from $\mathbf{u}^B$, $\mathbf{u}^D$ | no constant |
# | right-hand side | no load | point forces $|\Gamma|\bar S_{ij}$ on the corners |

# ### Corner displacements
#
# In {eq}`eq:homog-uB` and {eq}`eq:homog-uD`, $\bar{\mathbf{F}}-\boldsymbol{\delta}=\bar{\mathbf{H}}$.
# Split it into a symmetric and a skew part,
#
# $$
# \begin{aligned}
# \bar{\mathbf{H}} &= \bar{\mathbf{E}} + \bar{\boldsymbol{\omega}},\\
# \bar{\mathbf{E}} &= \tfrac12\left(\bar{\mathbf{H}}+\bar{\mathbf{H}}^T\right),\\
# \bar{\boldsymbol{\omega}} &= \tfrac12\left(\bar{\mathbf{H}}-\bar{\mathbf{H}}^T\right)
# = \begin{pmatrix}0 & -\omega\\ \omega & 0\end{pmatrix}.
# \end{aligned}
# $$
#
# The symmetric part, the macroscopic strain $\bar{\mathbf{E}}$ of components $\bar E_{ij}$,
# produces stress. The skew part is a rigid rotation by the angle $\omega$, which does no work,
# so
#
# $$
# \begin{aligned}
# \mathbf{u}^B &= L\begin{pmatrix}\bar E_{11}\\ \bar E_{12}+\omega\end{pmatrix},\\
# \mathbf{u}^D &= L\begin{pmatrix}\bar E_{12}-\omega\\ \bar E_{22}\end{pmatrix}
# \end{aligned}
# $$
#
# describe the same stress for every $\omega$. We remove the rotation by choosing
# $\omega=-\bar E_{12}$, that is $u^B_2=0$. Then
#
# $$
# \begin{aligned}
# \mathbf{u}^B &= L\begin{pmatrix}\bar E_{11}\\ 0\end{pmatrix},\\
# \mathbf{u}^D &= L\begin{pmatrix}2\bar E_{12}\\ \bar E_{22}\end{pmatrix}.
# \end{aligned}
# $$ (eq:homog-sc-corners)

# ### The prescribed stress as nodal forces
#
# Let $\mathbf{T}=\boldsymbol{\sigma}\mathbf{n}$ be the traction on $\partial\Omega$. The stress
# is periodic, while the outward normals of opposite edges are opposite, so the tractions are
# anti-periodic: $\mathbf{T}(\mathbf{X}+L\mathbf{e}_1)=-\mathbf{T}(\mathbf{X})$ on RIGHT and LEFT,
# and likewise on TOP and BOTTOM. A field $\mathbf{v}$ that satisfies {eq}`eq:homog-periodic`
# with $\mathbf{v}^A=\mathbf{0}$ takes on RIGHT its values on LEFT plus $\mathbf{v}^B$, so
# the work of the tractions on the two edges cancels but for $\mathbf{v}^B$, and likewise on TOP
# and BOTTOM with $\mathbf{v}^D$:
#
# $$
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}~\mathrm{d}s
# = v^B_i\int_{\text{RIGHT}}T_i~\mathrm{d}s + v^D_i\int_{\text{TOP}}T_i~\mathrm{d}s .
# $$
#
# With $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, the divergence theorem gives
# $\int_{\partial\Omega}T_iX_j~\mathrm{d}s=\int_\Omega\sigma_{ij}~\mathrm{d}x=|\Omega|\,\bar\sigma_{ij}$.
# For $j=1$, $X_1=L$ on RIGHT and $X_1=0$ on LEFT, while on TOP and BOTTOM the anti-periodic
# tractions cancel at each $X_1$. Hence $\int_{\text{RIGHT}}T_i~\mathrm{d}s=|\Gamma|\bar\sigma_{i1}$, and
# likewise $\int_{\text{TOP}}T_i~\mathrm{d}s=|\Gamma|\bar\sigma_{i2}$, with $|\Gamma|=|\Omega|/L$ the length of
# an edge. The principle of virtual work, with $\bar\sigma_{ij}=\bar S_{ij}$, becomes
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})~\mathrm{d}x
# = |\Gamma|\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i\right)
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=0$. The prescribed stress is
# therefore a set of **nodal forces**: $|\Gamma|\bar S_{11}$ on $u^B_1$, $|\Gamma|\bar S_{12}$ on
# $u^D_1$ and $|\Gamma|\bar S_{22}$ on $u^D_2$. The reaction at $u^B_2$ is $|\Gamma|\bar S_{21}$,
# which equals $|\Gamma|\bar S_{12}$ as the stress is symmetric.
#
# | component | control |
# |---|---|
# | $u^B_1=L\bar E_{11}$ | force $|\Gamma|\bar S_{11}$ |
# | $u^B_2=0$ | removes the rigid rotation |
# | $u^D_1=2L\bar E_{12}$ | force $|\Gamma|\bar S_{12}$ |
# | $u^D_2=L\bar E_{22}$ | force $|\Gamma|\bar S_{22}$ |
#
# Here the cell is loaded by a combination of tension and shear, $\bar S_{11}$ and $\bar S_{12}$, with
# $\bar S_{22}=0$.

# ### Constraints with free masters
#
# The masters $B$ and $D$ are now degrees of freedom, and the constraint is the same as under
# strain control: {eq}`eq:homog-periodic`, built by `periodic_cell_constraint`. Only the
# conditions differ: $\mathbf{u}^A=\mathbf{0}$ and $u^B_2=0$. The Dirichlet master $u^B_2=0$ is
# folded into the constraint, while $u^B_1$, $u^D_1$ and $u^D_2$ stay masters.

# +
S_B = np.array([0.09, 0.0])  # prescribed S_11 (first entry; u^B_2 = 0 is a constraint)
S_D = np.array([0.05, 0.0])  # prescribed (S_12, S_22)
area = L  # area of a side of the cell, per unit thickness

bc_A_sc = create_dirichletbc(V, corner_tags, TAG_A, np.zeros(gdim))
bc_B1_sc = create_dirichletbc(V.sub(1), corner_tags, TAG_B, np.zeros(1))  # u^B_2 = 0
bcs_sc = [bc_A_sc, bc_B1_sc]
mpc_sc = periodic_cell_constraint(bcs_sc)
# -

# ### The nodal forces as vertex integrals
#
# The nodal forces are the work $|\Gamma|\bar S_{i1}v^B_i + |\Gamma|\bar S_{i2}v^D_i$, a vertex integral over
# the corners $B$ and $D$:
#
# $$
# \int_{\{B\}} |\Gamma|\bar S_{i1}\,v_i~\mathrm{d}P + \int_{\{D\}} |\Gamma|\bar S_{i2}\,v_i~\mathrm{d}P,
# $$
#
# added to the linear form with the measure `ufl.dP` on the corners tagged in `corner_tags`. Their degrees of freedom
# are masters of the constraint, so the forces stay in the reduced system. The force on
# $u^B_2$ is zero, and its Dirichlet condition would remove it anyway. The problem is then an
# ordinary {py:class}`dolfinx_mpc.LinearProblem`.

# +
dP = ufl.Measure("dP", domain=domain, subdomain_data=corner_tags)
# Column j is the force per area on the corner of direction j: (S_11, 0) on B and (S_12, S_22) on D.
# Not the stress tensor: the entry on u^B_2 is a placeholder, removed by its Dirichlet condition,
# where the reaction is S_21.
corner_loads = fem.Constant(domain, np.column_stack([S_B, S_D]).astype(default_scalar_type))
A_side = fem.Constant(domain, default_scalar_type(area))
w = ufl.TestFunction(V)
point_forces = A_side * (ufl.inner(corner_loads[:, 0], w) * dP(TAG_B) + ufl.inner(corner_loads[:, 1], w) * dP(TAG_D))


def solve_stress_control(a_ufl, L_ufl) -> fem.Function:
    problem_sc = LinearProblem(a_ufl, L_ufl + point_forces, mpc_sc, bcs=bcs_sc, petsc_options=petsc_options)
    uh = problem_sc.solve()
    del problem_sc  # its PETSc objects are destroyed by PETSc.garbage_cleanup at the end
    return uh


# The dofs of the corners B and D, and the process owning them, located once
corner_dofs = {point: dofs_at_point(V, point) for point in ((L, 0.0), (0.0, L))}


def value_at(u: fem.Function, point) -> np.ndarray:
    """The value of `u` at the dofs of V at the corner `point`, read by the process owning them and sent to all."""
    dofs, owner = corner_dofs[point]
    value = None
    if comm.rank == owner:
        value = u.x.array[dofs - V.dofmap.index_map.local_range[0] * V.dofmap.index_map_bs]
    return comm.bcast(value, root=owner)


# -

# ### Verification
#
# **Homogeneous cell.** The exact solution is affine,
# $\mathbf{u}=(X_1\mathbf{u}^B+X_2\mathbf{u}^D)/L$ with {eq}`eq:homog-sc-corners` and
#
# $$
# \begin{aligned}
# \begin{pmatrix}\bar E_{11}\\ \bar E_{22}\end{pmatrix}
# &= \begin{pmatrix}\lambda+2\mu & \lambda\\ \lambda & \lambda+2\mu\end{pmatrix}^{-1}
#   \begin{pmatrix}\bar S_{11}\\ \bar S_{22}\end{pmatrix},\\
# \bar E_{12} &= \frac{\bar S_{12}}{2\mu}.
# \end{aligned}
# $$
#
# It must be reproduced to round-off.

# +
uh_sc = solve_stress_control(a, Lform)
lam0, mu0 = float(lmbda_uniform.value.real), float(mu_uniform.value.real)
E_11_exact, E_22_exact = np.linalg.solve([[lam0 + 2 * mu0, lam0], [lam0, lam0 + 2 * mu0]], [S_B[0], S_D[1]])
E_12_exact = S_D[0] / (2 * mu0)
H_exact = np.array([[E_11_exact, 2 * E_12_exact], [0.0, E_22_exact]])
u_exact = fem.Function(V, dtype=dtype)
u_exact.interpolate(lambda x: H_exact @ x[:gdim])
diff = uh_sc - u_exact
error_sc = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_exact = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < atol * norm_exact
# -

# **Heterogeneous cell.** The strain is read from the free masters through
# {eq}`eq:homog-sc-corners`: $\bar E_{11}=u^B_1/L$, $\bar E_{12}=u^D_1/(2L)$,
# $\bar E_{22}=u^D_2/L$.
#
# In linear elasticity the same state is obtained by superposition of three strain-controlled
# solutions computed with `homogenized_stress`, for the unit strains $\bar E_{11}=1$,
# $\bar E_{22}=1$ and $\bar E_{12}=\bar E_{21}=1$. They give the homogenized stiffness, from
# which we solve for $\bar{\mathbf{E}}$ with $\bar{\boldsymbol{\sigma}}=\bar{\mathbf{S}}$. The two
# routes must give the same result, and the average stress must be the prescribed one.

# +
uh_sc = solve_stress_control(a_het, L_het)
sigma_sc = average_stress(uh_sc)
u_B_sc, u_D_sc = value_at(uh_sc, (L, 0.0)), value_at(uh_sc, (0.0, L))
E_bar_sc = np.array([[u_B_sc[0] / L, u_D_sc[0] / (2 * L)], [u_D_sc[0] / (2 * L), u_D_sc[1] / L]])  # eq:homog-sc-corners

# superposition: stiffness columns d(s11, s22, s12)/d(E_11, E_22, E_12), from unit symmetric strains
unit = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
C_hom = np.column_stack([homogenized_stress(problem_het, 1e-2 * Ek)[1] / 1e-2 for Ek in unit])  # rows: s11, s22, s12
E_sup = np.linalg.solve(C_hom, [S_B[0], S_D[1], S_D[0]])  # (E_11, E_22, E_12)
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  sigma_bar = [[{sigma_sc[0, 0]:.6e}, {sigma_sc[0, 1]:.6e}], [{sigma_sc[1, 0]:.6e}, {sigma_sc[1, 1]:.6e}]]")
    print(f"  E_bar     = [[{E_bar_sc[0, 0]:.6e}, {E_bar_sc[0, 1]:.6e}], [{E_bar_sc[1, 0]:.6e}, {E_bar_sc[1, 1]:.6e}]]")
    print(
        f"  superposition of strain-controlled solutions: E_11 = {E_sup[0]:.6e},"
        f" E_22 = {E_sup[1]:.6e}, E_12 = {E_sup[2]:.6e}"
    )
S_bar_sc = np.array([[S_B[0], S_D[0]], [S_D[0], S_D[1]]])
assert np.allclose(sigma_sc, S_bar_sc, rtol=0, atol=atol * np.abs(S_bar_sc).max())
E_bar_sup = np.array([[E_sup[0], E_sup[2]], [E_sup[2], E_sup[1]]])
assert np.allclose(E_bar_sc, E_bar_sup, rtol=0, atol=atol * np.abs(E_bar_sup).max())
# -

# The microstructure, the deformed cell under the prescribed stress over the outline of the
# undeformed cell, and the periodic fluctuation
# $\mathbf{u}^*=\mathbf{u}-(X_1\mathbf{u}^B+X_2\mathbf{u}^D)/L$.

# + tags=["hide-input"]
u_plot_sc = fem.Function(V, dtype=dtype)
u_plot_sc.x.array[:] = uh_sc.x.array[: u_plot_sc.x.array.size]
w_sc = fem.Function(V, dtype=dtype)
w_sc.interpolate(lambda x: np.outer(u_B_sc, x[0]) / L + np.outer(u_D_sc, x[1]) / L)
w_sc.x.array[:] = u_plot_sc.x.array - w_sc.x.array
sc_pieces, sc_clim = gather_grids(u_plot_sc, V, "u")
w_pieces, w_clim = gather_grids(w_sc, V, "w")
if comm.rank == 0:
    outline = pyvista.Rectangle([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0)])
    bar = {"fmt": "%.1e", "n_labels": 3, "position_x": 0.2, "width": 0.6}
    plotter = pyvista.Plotter(shape=(1, 3), window_size=[750, 520])
    plotter.subplot(0, 0)
    plotter.add_text(f"Microstructure\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)", font_size=10)
    for piece in material_pieces:
        plotter.add_mesh(
            piece,
            scalars="E",
            cmap="viridis",
            clim=material_clim,
            show_edges=False,
            scalar_bar_args={"n_labels": 2, "fmt": "%.0f", "position_x": 0.2, "width": 0.6},
        )
    plotter.view_xy()
    load = f"S11 = {S_B[0]:g}, S12 = {S_D[0]:g}, S22 = {S_D[1]:g} prescribed"
    for col, (pieces, clim, name, title) in enumerate(
        [
            (sc_pieces, sc_clim, "u", f"Deformed cell, stress control\n{load}"),
            (w_pieces, w_clim, "w", "Periodic fluctuation\nu* = u - (F - I) X"),
        ],
        start=1,
    ):
        factor = 0.1 * L / clim[1]  # largest displacement drawn as 10 % of the cell size
        plotter.subplot(0, col)
        plotter.add_text(f"{title}\n(amplified x{factor:.0f})", font_size=10)
        for piece in pieces:
            plotter.add_mesh(
                piece.warp_by_vector(name, factor=factor),
                scalars=f"|{name}|",
                cmap="viridis",
                clim=clim,
                scalar_bar_args={**bar, "title": f"|{name}|"},
            )
        plotter.add_mesh(outline, style="wireframe", color="black", line_width=2)
        plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot()
    else:
        plotter.show()
# -

# +
del problem, problem_het
PETSc.garbage_cleanup(comm)
# -

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homog-
# ```

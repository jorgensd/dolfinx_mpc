# # Periodic boundary conditions for a representative volume element
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT

# We import the required modules

# + tags=["hide-input"]
from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot

from dolfinx_mpc import LinearProblem, MultiPointConstraint

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
domain = mesh.create_rectangle(comm, [[0, 0], [L, L]], [N, N])

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
# \mathbf{u}(\mathbf{X}) = (\bar{\mathbf{F}}-\boldsymbol{\delta})\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),
# \qquad \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) = \mathbf{u}^*(\mathbf{X}),
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

# We split the conditions into several sub-components, which each expose a feature oF
# DOLFINx-MPC.
#
# ### Dirichlet conditions on the corners
# For three of the corners ({eq}`eq:homog-uA`, {eq}`eq:homog-uB` and
# {eq}`eq:homog-uD`) we will apply {py:class}`dolfinx.fem.DirichletBC`
# directly, which is the simplest way to fix a degree of freedom.

# +


def corner(px: float, py: float):
    """Indicator function for a single point, padded for a 3D coordinate array."""
    return lambda x: np.isclose(x[0], px) & np.isclose(x[1], py)


def dirichletbc_at_point(V: fem.FunctionSpace, indicator, value: np.ndarray) -> tuple[fem.DirichletBC, np.ndarray]:
    """A Dirichlet condition fixing every degree of freedom at one point to ``value``."""
    dofs = fem.locate_dofs_geometrical(V, indicator)
    fn = fem.Function(V)
    fn.interpolate(lambda x: np.tile(np.asarray(value, dtype=default_scalar_type).reshape(-1, 1), x.shape[1]))
    return fem.dirichletbc(fn, dofs), dofs


u_B = H_bar @ np.array([L, 0.0])
u_D = H_bar @ np.array([0.0, L])
bc_A, _ = dirichletbc_at_point(V, corner(0, 0), np.zeros(domain.geometry.dim))
bc_B, dofs_B = dirichletbc_at_point(V, corner(L, 0), u_B)
bc_D, dofs_D = dirichletbc_at_point(V, corner(0, L), u_D)
bcs = [bc_A, bc_B, bc_D]
# -

# ## Periodic constraints with non-zero constants
# {eq}`eq:homog-right` and {eq}`eq:homog-top` are periodic constraints with a nonzero constant
# in the equation. We handle these by supplying a {py:class}`dolfinx.fem.Function` `g` to
# the {py:class}`dolfinx_mpc.MultiPointConstraint` constructor, which is added to the
# right-hand side of the constraint equation.
# The function `g` is zero everywhere except on the open right and top edges,
# where it equals $\mathbf{u}^B$ or $\mathbf{u}^D$; the corners are excluded, as
# they are already fixed by the Dirichlet conditions above.
#
# ```{admonition} Alternative constraint construction
# :class: tip dropdown
# An alternative would be to add $\mathbf{u}^B$/$\mathbf{u}^D$ as an explicit
# master on every edge dof and let *Dirichlet-master folding* -- the mechanism
# the next constraint relies on -- absorb the offset instead of building `g` by
# hand. We use `g` here because
# {py:meth}`dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical`
# already does the parallel, geometric left/right (resp. top/bottom) dof pairing
# for every edge dof in bulk; repeating that pairing by hand just to attach one
# extra master per dof would be slower, not simpler.
# ```


# +
def right_edge(x):
    return np.isclose(x[0], L) & ~(np.isclose(x[1], 0.0) | np.isclose(x[1], L))


def top_edge(x):
    return np.isclose(x[1], L) & ~(np.isclose(x[0], 0.0) | np.isclose(x[0], L))


def offset(x):
    values = np.zeros((domain.geometry.dim, x.shape[1]), dtype=dtype)
    values[:, right_edge(x)] = u_B.reshape(-1, 1)
    values[:, top_edge(x)] = u_D.reshape(-1, 1)
    return values


g = fem.Function(V, dtype=dtype)
g.interpolate(offset)
g.x.scatter_forward()  # before finalize()

# -

# Next we create the periodic constraints with `g` as input to the
# {py:class}`dolfinx_mpc.MultiPointConstraint` constructor
# and the two edge-pair constraints with
# {py:meth}`dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical`.


# +
def to_left(x):
    out = x.copy()
    out[0] = x[0] - L
    return out


def to_bottom(x):
    out = x.copy()
    out[1] = x[1] - L
    return out


mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs, rhs_coeffs=g)
mpc.create_periodic_constraint_geometrical(V, right_edge, to_left, bcs, scale=dtype.type(1.0))
mpc.create_periodic_constraint_geometrical(V, top_edge, to_bottom, bcs, scale=dtype.type(1.0))
# -

# ### Periodic constraints including DirichletBC masters
# {eq}`eq:homog-uC` includes the two Dirichlet-constrained corners $B$ and $D$ as masters,
# so it needs to fold these into the constraint.
# This is done by passing the Dirichlet conditions as `bcs` to the constraint,
# which removes them from the master list and folds their values into the offset automatically.

# {py:meth}`create_general_constraint
# <dolfinx_mpc.MultiPointConstraint.create_general_constraint>` is the
# convenience wrapper for exactly this kind of single point-to-point coupling:
# it takes the slave/master relation by *coordinate* rather than by dof index,
# and resolves which rank owns each point internally, so no discovery code is
# needed here. It relates one scalar dof per call, so a vector point needs one
# call per component, matched via `subspace_slave`/`subspace_master`.

gdim = domain.geometry.dim
xdt = domain.geometry.x.dtype
# Tolerance of the checks below, from the precision of the mesh coordinates
atol = 50 * np.sqrt(np.finfo(xdt).resolution)
# Largest distance between a dof and the point it is located at, from the rounding of the coordinates
point_tol = 500 * np.finfo(xdt).eps * L
B_coord = np.array([L, 0.0, 0.0], dtype=xdt)
C_coord = np.array([L, L, 0.0], dtype=xdt)
D_coord = np.array([0.0, L, 0.0], dtype=xdt)
for c in range(gdim):
    mpc.create_general_constraint(
        {C_coord.tobytes(): {B_coord.tobytes(): 1.0, D_coord.tobytes(): 1.0}},
        subspace_slave=c,
        subspace_master=c,
    )

# Now that we have set up the full set of constraints we finalize them.

mpc.finalize()  # collective: every rank must reach this

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
# resolution. The bound `atol`, $50$ times the square root of the resolution of
# the coordinate type of the mesh, is deliberately loose around that roundoff
# floor (measured in the $10^{-15}$ range in double precision) rather than tight
# to it, so the check stays robust to the roundoff growing slightly with rank
# count from reduction-order effects in
# {py:meth}`comm.allreduce<mpi4py.MPI.Commm.allreduce>`s.
# and holds in single precision as well.

# +
x = ufl.SpatialCoordinate(domain)
u_affine = ufl.dot(ufl.as_tensor(H_bar), x)
diff = uh_homogeneous - u_affine
error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < atol
# -

# ## A heterogeneous microstructure
#
# The point of a periodic unit cell is to homogenize a microstructure, not a
# uniform block, so the second solve gives the matrix a stiff circular
# inclusion. Three checks, none of them requiring a closed-form solution:
#
# 1. With no macroscopic strain the cell carries no load, so the volume
#    averaged stress must vanish exactly whatever the microstructure.
# 2. Under an isotropic macroscopic strain the circular inclusion is symmetric
#    under a $90°$ rotation, so the averaged stress must be (nearly) isotropic
#    too -- a property of the geometry, not of any elastic constant.
# 3. The constraint machinery guarantees the periodicity equations hold to
#    solver precision by construction
#    ({py:meth}`dolfinx_mpc.MultiPointConstraint.backsubstitution`
#    enforces them after every solve); the payoff we report is the
#    homogenized stress this microstructure produces under a
#    general macroscopic strain.

Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q)
tdim = domain.topology.dim
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
inclusion = (midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2
E.x.array[: len(inclusion)] = np.where(inclusion, 50.0 * E_uniform, E_uniform)
mu_field = E / (2 * (1 + nu))
lmbda_field = E * nu / ((1 + nu) * (1 - 2 * nu))
a_het, L_het = elasticity_forms(V, mu_field, lmbda_field)


# The parameter study below solves the same constraint for several macroscopic
# strains, so the walkthrough above, corners, edge offset, periodicity, corner
# folding, is packaged here for reuse, parametrised by `H_bar_case`.


def build_constraint(H_bar_case: np.ndarray) -> tuple[MultiPointConstraint, list[fem.DirichletBC]]:
    """Repeat the constraint construction above for a different average strain."""
    u_B_case = H_bar_case @ np.array([L, 0.0])
    u_D_case = H_bar_case @ np.array([0.0, L])
    bc_A_case, _ = dirichletbc_at_point(V, corner(0, 0), np.zeros(domain.geometry.dim))
    bc_B_case, _ = dirichletbc_at_point(V, corner(L, 0), u_B_case)
    bc_D_case, _ = dirichletbc_at_point(V, corner(0, L), u_D_case)
    bcs_case = [bc_A_case, bc_B_case, bc_D_case]

    def offset_case(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dtype)
        values[:, right_edge(x)] = u_B_case.reshape(-1, 1)
        values[:, top_edge(x)] = u_D_case.reshape(-1, 1)
        return values

    g_case = fem.Function(V, dtype=dtype)
    g_case.interpolate(offset_case)

    mpc_case = MultiPointConstraint(V, dtype=dtype, bcs=bcs_case, rhs_coeffs=g_case)
    mpc_case.create_periodic_constraint_geometrical(V, right_edge, to_left, bcs_case, scale=dtype.type(1.0))
    mpc_case.create_periodic_constraint_geometrical(V, top_edge, to_bottom, bcs_case, scale=dtype.type(1.0))

    for c in range(gdim):
        mpc_case.create_general_constraint(
            {C_coord.tobytes(): {B_coord.tobytes(): 1.0, D_coord.tobytes(): 1.0}},
            subspace_slave=c,
            subspace_master=c,
        )

    g_case.x.scatter_forward()  # before finalize(): ghost copies of the offset must be up to date
    mpc_case.finalize()  # collective: every rank must reach this
    return mpc_case, bcs_case


# The convenience function for solving the constrained problem and computing the average stress
# is parameterized by `H_bar_case` below:


def homogenized_stress(H_bar_case: np.ndarray) -> tuple[fem.Function, np.ndarray]:
    """Solve the heterogeneous cell for ``H_bar_case`` and return (uh, sigma_avg)."""
    mpc_case, bcs_case = build_constraint(H_bar_case)
    uh = LinearProblem(a_het, L_het, mpc_case, bcs=bcs_case, petsc_options=petsc_options).solve()
    eps = ufl.sym(ufl.grad(uh))
    sigma = 2 * mu_field * eps + lmbda_field * ufl.tr(eps) * ufl.Identity(2)
    one = fem.Constant(domain, default_scalar_type(1.0))
    vol = comm.allreduce(fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM)
    components = [(0, 0), (1, 1), (0, 1)]
    sigma_avg = np.array(
        [comm.allreduce(fem.assemble_scalar(fem.form(sigma[i, j] * ufl.dx)), op=MPI.SUM) / vol for i, j in components]
    )
    return uh, sigma_avg


# ### Test case 1: No macroscopic strain

_, sigma_zero = homogenized_stress(np.zeros((2, 2)))
if comm.rank == 0:
    print(f"----No macroscopic strain----\n  sigma_avg = {sigma_zero}  (should vanish)")
assert np.abs(sigma_zero).max() < atol

# ### Test case 2: Isotropic macroscopic strain

_, sigma_iso = homogenized_stress(0.02 * np.eye(2))
if comm.rank == 0:
    print(
        f"----Isotropic macroscopic strain----\n  s11={sigma_iso[0]:.5f}  s22={sigma_iso[1]:.5f}  "
        f"s12={sigma_iso[2]:.2e}  (should be isotropic: s11≈s22, s12≈0)"
    )
assert abs(sigma_iso[0] - sigma_iso[1]) < 5e-3 * abs(sigma_iso[0])
assert abs(sigma_iso[2]) < 5e-3 * abs(sigma_iso[0])

# ### Test case 3: General macroscopic strain

uh_general, sigma_general = homogenized_stress(H_bar)
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
u_plot = fem.Function(V)
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

w_plot = fem.Function(V)
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
# We now prescribe the average stress $\bar{\mathbf{S}}$ instead of the strain. The
# displacements $\mathbf{u}^B$ and $\mathbf{u}^D$ become unknowns.

# ### Corner displacements
#
# In {eq}`eq:homog-uB` and {eq}`eq:homog-uD`, $\bar{\mathbf{F}}-\boldsymbol{\delta}=\bar{\mathbf{H}}$.
# Only its symmetric part, the macroscopic strain $\bar{\mathbf{E}}$, produces stress; the skew
# part is a rigid rotation, which does no work and is removed with $u^B_2=0$. Then
#
# $$
# \mathbf{u}^B = L\begin{pmatrix}\bar E_{11}\\ 0\end{pmatrix},\qquad
# \mathbf{u}^D = L\begin{pmatrix}2\bar E_{12}\\ \bar E_{22}\end{pmatrix}.
# $$ (eq:homog-sc-corners)

# ### The prescribed stress as nodal forces
#
# Let $\mathbf{T}=\boldsymbol{\sigma}\mathbf{n}$ be the traction on $\partial\Omega$. The
# tractions are anti-periodic, $\mathbf{T}(\mathbf{X}+L\mathbf{e}_1)=-\mathbf{T}(\mathbf{X})$ on
# RIGHT and LEFT, and likewise on TOP and BOTTOM. For a field $\mathbf{v}$ that satisfies the
# constraints {eq}`eq:homog-uA`–{eq}`eq:homog-uC`, the work of the tractions therefore reduces to
#
# $$
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}\,\mathrm{d}s
# = v^B_i\int_{\text{RIGHT}}T_i\,\mathrm{d}s + v^D_i\int_{\text{TOP}}T_i\,\mathrm{d}s .
# $$
#
# With $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, the divergence theorem gives
# $\int_{\partial\Omega}T_iX_j\,\mathrm{d}s=|\Omega|\,\bar\sigma_{ij}$, hence
# $\int_{\text{RIGHT}}T_i\,\mathrm{d}s=A\bar\sigma_{i1}$ and
# $\int_{\text{TOP}}T_i\,\mathrm{d}s=A\bar\sigma_{i2}$, with $A=|\Omega|/L$. The principle of
# virtual work, with $\bar\sigma_{ij}=\bar S_{ij}$, becomes
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})\,\mathrm{d}\Omega
# = A\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i\right)
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=0$. The prescribed stress is
# therefore a set of **nodal forces**: $A\bar S_{11}$ on $u^B_1$, $A\bar S_{12}$ on $u^D_1$ and
# $A\bar S_{22}$ on $u^D_2$. $\bar S_{21}=\bar S_{12}$ is the reaction at $u^B_2$.
#
# | component | control |
# |---|---|
# | $u^B_1=L\bar E_{11}$ | force $A\bar S_{11}$ |
# | $u^B_2=0$ | removes the rigid rotation |
# | $u^D_1=2L\bar E_{12}$ | force $A\bar S_{12}$ |
# | $u^D_2=L\bar E_{22}$ | force $A\bar S_{22}$ |
#
# Here the cell is loaded by a combination of tension and shear, $\bar S_{11}$ and $\bar S_{12}$, with
# $\bar S_{22}=0$.

# ### Constraints with free masters
#
# The masters $B$ and $D$ are now degrees of freedom. Each slave has two masters: its partner on
# LEFT and $B$, its partner on BOTTOM and $D$, or $B$ and $D$ for the corner $C$. These
# constraints are built with
# {py:meth}`create_general_constraint <dolfinx_mpc.MultiPointConstraint.create_general_constraint>`,
# one call per component. For the second component the Dirichlet master $u^B_2=0$ is folded into
# the constraint.

# +
from petsc4py import PETSc  # noqa: E402

import dolfinx.fem.petsc  # noqa: E402

import dolfinx_mpc  # noqa: E402

S_B = np.array([0.09, 0.0])  # prescribed S_11 (first entry; u^B_2 = 0 is a constraint)
S_D = np.array([0.05, 0.0])  # prescribed (S_12, S_22)
area = L  # area of a side of the cell, per unit thickness

bc_A_sc, _ = dirichletbc_at_point(V, corner(0, 0), np.zeros(gdim))
V1 = V.sub(1).collapse()[0]
dofs_B1 = fem.locate_dofs_geometrical((V.sub(1), V1), corner(L, 0))
bc_B1_sc = fem.dirichletbc(fem.Function(V1), dofs_B1, V.sub(1))  # u^B_2 = 0
bcs_sc = [bc_A_sc, bc_B1_sc]
mpc_sc = MultiPointConstraint(V, dtype=dtype, bcs=bcs_sc)
x_dofs = V.tabulate_dof_coordinates()


def gather_points(marker):
    return np.unique(np.round(np.vstack(comm.allgather(x_dofs[marker(x_dofs.T)])), 14), axis=0)


def key(p):
    return np.array([p[0], p[1], 0.0], dtype=xdt).tobytes()


right_points, top_points = gather_points(right_edge), gather_points(top_edge)
for c in range(gdim):
    slave_master = {key(p): {key((0.0, p[1])): 1.0, B_coord.tobytes(): 1.0} for p in right_points}
    slave_master.update({key(p): {key((p[0], 0.0)): 1.0, D_coord.tobytes(): 1.0} for p in top_points})
    slave_master[C_coord.tobytes()] = {B_coord.tobytes(): 1.0, D_coord.tobytes(): 1.0}
    mpc_sc.create_general_constraint(slave_master, subspace_slave=c, subspace_master=c)
mpc_sc.finalize()  # collective: every rank must reach this
# -

# ### Assembly with the nodal forces
#
# The system is assembled as in {py:class}`dolfinx_mpc.LinearProblem`. Once the Dirichlet
# conditions are applied, the nodal forces are added to the right-hand side at the dofs of $B$
# and $D$. These are masters of the constraint, so they belong to the reduced system, and they
# are neither Dirichlet dofs nor slaves.

# +
dofs_B = [fem.locate_dofs_geometrical((V.sub(c), V.sub(c).collapse()[0]), corner(L, 0))[0] for c in range(gdim)]
dofs_D = [fem.locate_dofs_geometrical((V.sub(c), V.sub(c).collapse()[0]), corner(0, L))[0] for c in range(gdim)]
n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs


def solve_stress_control(a_ufl, L_ufl) -> fem.Function:
    a_c, L_c = fem.form(a_ufl), fem.form(L_ufl)
    mpc_sc.update_constants()
    A_mat = dolfinx_mpc.assemble_matrix(a_c, mpc_sc, bcs=bcs_sc)
    A_mat.assemble()
    b = dolfinx_mpc.assemble_vector(L_c, mpc_sc)
    dolfinx_mpc.apply_lifting(b, [a_c], bcs=[bcs_sc], constraint=mpc_sc)
    dolfinx_mpc.apply_mpc_lifting(b, [a_c], constraint=mpc_sc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs_sc)
    b.array[dofs_B[0][dofs_B[0] < n_owned]] += area * S_B[0]  # A S_11 on u^B_1, on the owning process
    for c in range(gdim):  # A S_12 on u^D_1 and A S_22 on u^D_2
        b.array[dofs_D[c][dofs_D[c] < n_owned]] += area * S_D[c]
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_mat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setErrorIfNotConverged(True)
    uh = fem.Function(mpc_sc.function_space)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc_sc.homogenize(uh)
    mpc_sc.backsubstitution(uh)
    ksp.destroy(), A_mat.destroy(), b.destroy()
    return uh


def average_stress(uh, mu_, lmbda_) -> np.ndarray:
    eps = ufl.sym(ufl.grad(uh))
    sig = 2 * mu_ * eps + lmbda_ * ufl.tr(eps) * ufl.Identity(2)
    return (
        np.array(
            [
                [comm.allreduce(fem.assemble_scalar(fem.form(sig[i, j] * ufl.dx)), op=MPI.SUM) for j in range(gdim)]
                for i in range(gdim)
            ]
        )
        / L**2
    )


def value_at(uh, point) -> np.ndarray:
    nloc = V.dofmap.index_map.size_local
    i = np.flatnonzero(np.linalg.norm(x_dofs[:nloc, :2] - np.asarray(point), axis=1) < point_tol)
    local = uh.x.array[gdim * i[0] : gdim * i[0] + gdim].copy() if len(i) else None
    return next(v for v in comm.allgather(local) if v is not None)


# -

# ### Verification
#
# **Homogeneous cell.** The exact solution is affine,
# $\mathbf{u}=(X_1\mathbf{u}^B+X_2\mathbf{u}^D)/L$ with {eq}`eq:homog-sc-corners` and
#
# $$
# \begin{pmatrix}\bar E_{11}\\ \bar E_{22}\end{pmatrix}
# = \begin{pmatrix}\lambda+2\mu & \lambda\\ \lambda & \lambda+2\mu\end{pmatrix}^{-1}
#   \begin{pmatrix}\bar S_{11}\\ \bar S_{22}\end{pmatrix},\qquad
# \bar E_{12}=\frac{\bar S_{12}}{2\mu}.
# $$
#
# It must be reproduced to round-off.

# +
uh_sc = solve_stress_control(a, Lform)
lam0, mu0 = float(lmbda_uniform.value), float(mu_uniform.value)
E_11_exact, E_22_exact = np.linalg.solve([[lam0 + 2 * mu0, lam0], [lam0, lam0 + 2 * mu0]], [S_B[0], S_D[1]])
E_12_exact = S_D[0] / (2 * mu0)
H_exact = np.array([[E_11_exact, 2 * E_12_exact], [0.0, E_22_exact]])
u_exact = fem.Function(V)
u_exact.interpolate(lambda x: H_exact @ x[:gdim])
diff = uh_sc - u_exact
error_sc = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < atol
# -

# **Heterogeneous cell.** The strain is read from the free masters through
# {eq}`eq:homog-sc-corners`: $\bar E_{11}=u^B_1/L$, $\bar E_{12}=u^D_1/(2L)$,
# $\bar E_{22}=u^D_2/L$.
#
# In linear elasticity the same state is obtained by superposition of three strain-controlled
# solutions computed with `homogenized_stress`, for the unit strains $\bar E_{11}=1$,
# $\bar E_{22}=1$ and $\bar E_{12}=\bar E_{21}=1$. They give the homogenized stiffness, from
# which we solve for $\bar{\mathbf{E}}$ with $\bar{\boldsymbol{\sigma}}=\bar{\mathbf{S}}$. The two
# routes give the same result.

# +
uh_sc = solve_stress_control(a_het, L_het)
sigma_sc = average_stress(uh_sc, mu_field, lmbda_field)
u_B_sc, u_D_sc = value_at(uh_sc, (L, 0.0)), value_at(uh_sc, (0.0, L))
E_bar_sc = np.array([[u_B_sc[0] / L, u_D_sc[0] / (2 * L)], [u_D_sc[0] / (2 * L), u_D_sc[1] / L]])  # eq:homog-sc-corners

# superposition: stiffness columns d(s11, s22, s12)/d(E_11, E_22, E_12), from unit symmetric strains
unit = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
C_hom = np.column_stack([homogenized_stress(1e-2 * Ek)[1] / 1e-2 for Ek in unit])  # rows: s11, s22, s12
E_sup = np.linalg.solve(C_hom, [S_B[0], S_D[1], S_D[0]])  # (E_11, E_22, E_12)
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  sigma_bar = [[{sigma_sc[0, 0]:.6e}, {sigma_sc[0, 1]:.6e}], [{sigma_sc[1, 0]:.6e}, {sigma_sc[1, 1]:.6e}]]")
    print(f"  E_bar     = [[{E_bar_sc[0, 0]:.6e}, {E_bar_sc[0, 1]:.6e}], [{E_bar_sc[1, 0]:.6e}, {E_bar_sc[1, 1]:.6e}]]")
    print(
        f"  superposition of strain-controlled solutions: E_11 = {E_sup[0]:.6e},"
        f" E_22 = {E_sup[1]:.6e}, E_12 = {E_sup[2]:.6e}"
    )
# -

# The microstructure, the deformed cell under the prescribed stress over the outline of the
# undeformed cell, and the periodic fluctuation
# $\mathbf{u}^*=\mathbf{u}-(X_1\mathbf{u}^B+X_2\mathbf{u}^D)/L$.

# + tags=["hide-input"]
u_plot_sc = fem.Function(V)
u_plot_sc.x.array[:] = uh_sc.x.array[: u_plot_sc.x.array.size]
u_B_h, u_D_h = value_at(uh_sc, (L, 0.0)), value_at(uh_sc, (0.0, L))
w_sc = fem.Function(V)
w_sc.interpolate(lambda x: np.outer(u_B_h, x[0]) / L + np.outer(u_D_h, x[1]) / L)
w_sc.x.array[:] = u_plot_sc.x.array - w_sc.x.array
sc_pieces, sc_clim = gather_grids(u_plot_sc, V, "u")
w_pieces, w_clim = gather_grids(w_sc, V, "w")
material_pieces_sc, _ = gather_cell_data(E, "E")
if comm.rank == 0:
    outline = pyvista.Rectangle([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0)])
    bar = {"fmt": "%.1e", "n_labels": 3, "position_x": 0.2, "width": 0.6}
    plotter = pyvista.Plotter(shape=(1, 3), window_size=[750, 520])
    plotter.subplot(0, 0)
    plotter.add_text(f"Microstructure\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)", font_size=10)
    for piece in material_pieces_sc:
        plotter.add_mesh(
            piece,
            scalars="E",
            cmap="viridis",
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

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homog-
# ```

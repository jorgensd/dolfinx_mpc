# # Periodic boundary conditions for a representative volume element: finite strain
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT

# +
from __future__ import annotations

from mpi4py import MPI

import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot

import dolfinx_mpc
from dolfinx_mpc import MultiPointConstraint

# -

# This demo is the finite-strain version of the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`. The unit cell,
# the microstructure and the periodic constraints are the same; the material is hyperelastic,
# the macroscopic deformation is large, and the problem is solved with
# {py:class}`dolfinx_mpc.NonlinearProblem`. The constraints are linear in $\mathbf{u}$ also at
# finite strain, so they are built exactly as in the linear case. They follow the corner-node
# formulation of the periodic boundary conditions of
# {cite}`homognl-Danas2017` (Appendix B), first used in
# {cite}`homognl-LopezPamiesGoudarziDanas2013`.
#
# We consider a square cell $\Omega=(0,L)^2$.

# +
comm = MPI.COMM_WORLD
N = 32
L = 1.0
dtype = np.dtype(default_scalar_type)
domain = mesh.create_rectangle(comm, [[0, 0], [L, L]], [N, N])
gdim = tdim = domain.geometry.dim

V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
F_bar = np.eye(2) + 25.0 * np.array([[0.02, 0.01], [0.0, -0.015]])  # 25 x the strain of the linear demo
# -

# ## The periodicity condition
#
# The displacement is split into an affine part carrying the macroscopic deformation and a
# periodic fluctuation,
#
# $$
# \mathbf{u}(\mathbf{X}) = (\bar{\mathbf{F}}-\boldsymbol{\delta})\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),
# \qquad \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) = \mathbf{u}^*(\mathbf{X}),
# $$
#
# where $\bar{\mathbf{F}}$ is the average deformation gradient (`F_bar`), a full, non-symmetric
# tensor. With the corners $A=(0,0)$, $B=(L,0)$, $C=(L,L)$, $D=(0,L)$ and $\mathbf{u}^A=\mathbf{0}$,
#
# $$
# \mathbf{u}^A = \mathbf{0},\qquad
# \mathbf{u}^B = (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}L\\0\end{pmatrix},\qquad
# \mathbf{u}^D = (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}0\\L\end{pmatrix},
# $$ (eq:nl-corners)
#
# and periodicity of $\mathbf{u}^*$ reduces every other constraint to
#
# $$
# \mathbf{u}^{\text{RIGHT}} = \mathbf{u}^{\text{LEFT}} + \mathbf{u}^B,\qquad
# \mathbf{u}^{\text{TOP}} = \mathbf{u}^{\text{BOTTOM}} + \mathbf{u}^D,\qquad
# \mathbf{u}^C = \mathbf{u}^B + \mathbf{u}^D.
# $$ (eq:nl-periodic)
#
# ### Dirichlet conditions on the corners
#
# The corners $A$, $B$, $D$ are fixed with {py:class}`dolfinx.fem.DirichletBC`. The values are
# stored in functions, so that they can be updated during the load stepping.

# +


def corner(px: float, py: float):
    """Indicator function for a single point, padded for a 3D coordinate array."""
    return lambda x: np.isclose(x[0], px) & np.isclose(x[1], py)


def dirichletbc_at_point(V: fem.FunctionSpace, indicator) -> tuple[fem.DirichletBC, fem.Function]:
    """A Dirichlet condition on every degree of freedom at one point; its value is the function returned."""
    dofs = fem.locate_dofs_geometrical(V, indicator)
    fn = fem.Function(V)
    return fem.dirichletbc(fn, dofs), fn


def set_value(fn: fem.Function, value: np.ndarray):
    fn.interpolate(lambda x: np.tile(np.asarray(value, dtype=default_scalar_type).reshape(-1, 1), x.shape[1]))


bc_A, _ = dirichletbc_at_point(V, corner(0, 0))
bc_B, value_B = dirichletbc_at_point(V, corner(L, 0))
bc_D, value_D = dirichletbc_at_point(V, corner(0, L))
bcs = [bc_A, bc_B, bc_D]
# -

# ### Periodic constraints with non-zero constants
#
# The constant terms $\mathbf{u}^B$ and $\mathbf{u}^D$ of the edge constraints are supplied as a
# function `g` to the {py:class}`dolfinx_mpc.MultiPointConstraint` constructor: `g` equals
# $\mathbf{u}^B$ on the open right edge, $\mathbf{u}^D$ on the open top edge, and zero elsewhere.
# The edges are paired with
# {py:meth}`dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical`.


# +
def right_edge(x):
    return np.isclose(x[0], L) & ~(np.isclose(x[1], 0.0) | np.isclose(x[1], L))


def top_edge(x):
    return np.isclose(x[1], L) & ~(np.isclose(x[0], 0.0) | np.isclose(x[0], L))


def to_left(x):
    out = x.copy()
    out[0] = x[0] - L
    return out


def to_bottom(x):
    out = x.copy()
    out[1] = x[1] - L
    return out


g = fem.Function(V, dtype=dtype)


def set_macroscopic_deformation(F_case: np.ndarray):
    """Corner values {eq}`eq:nl-corners` and edge offset `g` for the average deformation F_case."""
    u_B = (F_case - np.eye(gdim)) @ np.array([L, 0.0])
    u_D = (F_case - np.eye(gdim)) @ np.array([0.0, L])
    set_value(value_B, u_B)
    set_value(value_D, u_D)

    def offset(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dtype)
        values[:, right_edge(x)] = u_B.reshape(-1, 1)
        values[:, top_edge(x)] = u_D.reshape(-1, 1)
        return values

    g.interpolate(offset)
    g.x.scatter_forward()


set_macroscopic_deformation(np.eye(gdim))
mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs, rhs_coeffs=g)
mpc.create_periodic_constraint_geometrical(V, right_edge, to_left, bcs, scale=dtype.type(1.0))
mpc.create_periodic_constraint_geometrical(V, top_edge, to_bottom, bcs, scale=dtype.type(1.0))
# -

# ### Periodic constraints including DirichletBC masters
#
# The corner equation $\mathbf{u}^C=\mathbf{u}^B+\mathbf{u}^D$ has the two Dirichlet corners as
# masters, which are folded into the constraint.

# +
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
mpc.finalize()  # collective: every rank must reach this
# -

# ## Hyperelastic material
#
# Both phases are compressible neo-Hookean, in plane strain, with first Piola-Kirchhoff stress
#
# $$
# \mathbf{P}(\mathbf{F}) = \mu\left(\mathbf{F}-\mathbf{F}^{-T}\right) + \lambda\ln J\,\mathbf{F}^{-T},
# \qquad \mathbf{F}=\boldsymbol{\delta}+\nabla\mathbf{u},\quad J=\det\mathbf{F}.
# $$
#
# The macroscopic stress is the volume average
#
# $$
# \bar{\mathbf{S}} = \frac{1}{|\Omega|}\int_\Omega \mathbf{P}(\mathbf{F})\,\mathrm{d}\Omega,
# $$
#
# which is not symmetric. The Young modulus is a cellwise constant function, uniform for the first test
# and with the stiff inclusion afterwards.

# +
E_uniform, nu = 10.0, 0.3
Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q)
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
inclusion = (midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


def set_young_modulus(E_inclusion: float):
    E.x.array[: len(inclusion)] = np.where(inclusion, E_inclusion, E_uniform)
    E.x.scatter_forward()


def piola(F):
    Finv_T = ufl.inv(F).T
    return mu * (F - Finv_T) + lmbda * ufl.ln(ufl.det(F)) * Finv_T


def average_stress(uh: fem.Function) -> np.ndarray:
    P = piola(ufl.Identity(gdim) + ufl.grad(uh))
    return (
        np.array(
            [
                [comm.allreduce(fem.assemble_scalar(fem.form(P[i, j] * ufl.dx)), op=MPI.SUM) for j in range(gdim)]
                for i in range(gdim)
            ]
        )
        / L**2
    )


x_dofs = V.tabulate_dof_coordinates()


def value_at(u: fem.Function, point) -> np.ndarray:
    nloc = V.dofmap.index_map.size_local
    i = np.flatnonzero(np.linalg.norm(x_dofs[:nloc, :2] - np.asarray(point), axis=1) < point_tol)
    local = u.x.array[gdim * i[0] : gdim * i[0] + gdim].copy() if len(i) else None
    return next(w for w in comm.allgather(local) if w is not None)


def average_deformation(u: fem.Function) -> np.ndarray:
    """F_bar from the corner displacements u^B, u^D (eq:nl-corners)."""
    return np.eye(gdim) + np.column_stack([value_at(u, (L, 0.0)), value_at(u, (0.0, L))]) / L


# -

# ## Nonlinear problem and load stepping
#
# The residual is $\int_\Omega\mathbf{P}(\mathbf{F}):\nabla\mathbf{v}\,\mathrm{d}\Omega$, with
# the unknown in the space of the constraint and the test and trial functions in `V`, the space of
# the Dirichlet conditions. The macroscopic deformation is applied in load steps; after the
# Dirichlet values and `g` change,
# {py:meth}`update_constants <dolfinx_mpc.MultiPointConstraint.update_constants>` must be called
# before solving, so that the constraint uses the new values.

# +
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-12,
    "snes_atol": 1e-11,
    "snes_max_it": 40,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def nonlinear_problem(constraint: MultiPointConstraint, bcs_: list[fem.DirichletBC], prefix: str):
    uh = fem.Function(constraint.function_space)
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    residual = ufl.inner(piola(ufl.Identity(gdim) + ufl.grad(uh)), ufl.grad(v)) * ufl.dx
    problem = dolfinx_mpc.NonlinearProblem(
        residual,
        uh,
        constraint,
        bcs=bcs_,
        J=ufl.derivative(residual, uh, du),
        petsc_options=petsc_options,
        petsc_options_prefix=prefix,
    )
    return problem, uh


def set_affine(uh: fem.Function, G: np.ndarray):
    """uh = G X; the leading entries of the constraint space are those of V, the extra ghosts are updated."""
    affine = fem.Function(V)
    affine.interpolate(lambda x_: G @ x_[:gdim])
    uh.x.array[: affine.x.array.size] = affine.x.array
    uh.x.scatter_forward()


def solve_in_steps(
    problem, uh: fem.Function, constraint: MultiPointConstraint, set_load, n_steps: int, G_prescribed: np.ndarray
):
    """Solve for the loads set_load(t), t = 1/n, ..., 1. The first Newton solve starts from the affine
    field of the prescribed part G_prescribed of F_bar - I, the next ones from a linear extrapolation of
    the two previous steps. Returns S_bar and F_bar at the final load."""
    set_affine(uh, G_prescribed / n_steps)
    previous = current = np.zeros_like(uh.x.array)  # converged solutions of the last two steps
    for k in range(1, n_steps + 1):
        if k > 1:
            uh.x.array[:] = 2 * current - previous
        set_load(k / n_steps)
        constraint.update_constants()
        problem.solve()
        previous, current = current, uh.x.array.copy()
    return average_stress(uh), average_deformation(uh)


problem, uh = nonlinear_problem(mpc, bcs, "strain_")


def homogenized_stress(F_case: np.ndarray, n_steps: int = 20):
    """Solve the cell for the average deformation F_case and return (uh, S_bar)."""
    S_bar, _ = solve_in_steps(
        problem,
        uh,
        mpc,
        lambda t: set_macroscopic_deformation(np.eye(gdim) + t * (F_case - np.eye(gdim))),
        n_steps,
        F_case - np.eye(gdim),
    )
    return uh, S_bar


# -

# ## Verifying the mechanism: a homogeneous unit cell
#
# For a homogeneous material the fluctuation vanishes and the exact solution is the affine field
# $\mathbf{u}=(\bar{\mathbf{F}}-\boldsymbol{\delta})\mathbf{X}$, whatever the material law. It lies
# in the P1 space, so it must be reproduced to round-off.

# +
set_young_modulus(E_uniform)
uh_homogeneous, _ = homogenized_stress(F_bar)
x = ufl.SpatialCoordinate(domain)
diff = uh_homogeneous - ufl.dot(ufl.as_tensor(F_bar - np.eye(gdim)), x)
error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < atol
# -

# ## A heterogeneous microstructure
#
# The matrix now contains a stiff circular inclusion, 50 times stiffer. As in the linear demo:
#
# 1. with no macroscopic deformation the average stress vanishes;
# 2. under an isotropic macroscopic stretch the average stress is (nearly) isotropic, by the
#    symmetry of the inclusion;
# 3. under a general $\bar{\mathbf{F}}$ we report $\bar{\mathbf{S}}$.

set_young_modulus(50.0 * E_uniform)

# #### Test case 1: No macroscopic deformation

_, S_zero = homogenized_stress(np.eye(gdim), n_steps=1)
if comm.rank == 0:
    print(f"----No macroscopic deformation----\n  S_bar = {S_zero.tolist()}  (should vanish)")
assert np.abs(S_zero).max() < atol

# #### Test case 2: Isotropic macroscopic stretch

_, S_iso = homogenized_stress((1.0 + 10 * 0.02) * np.eye(gdim))  # 10 x the linear demo
if comm.rank == 0:
    print(
        f"----Isotropic macroscopic stretch----\n  S11={S_iso[0, 0]:.5f}  S22={S_iso[1, 1]:.5f}  "
        f"S12={S_iso[0, 1]:.2e}  S21={S_iso[1, 0]:.2e}  (should be isotropic: S11≈S22, S12≈S21≈0)"
    )
assert abs(S_iso[0, 0] - S_iso[1, 1]) < 5e-3 * abs(S_iso[0, 0])
assert max(abs(S_iso[0, 1]), abs(S_iso[1, 0])) < 5e-3 * abs(S_iso[0, 0])

# #### Test case 3: General macroscopic deformation

uh_general, S_general = homogenized_stress(F_bar)
if comm.rank == 0:
    print(
        f"----General macroscopic deformation----\n  S11={S_general[0, 0]:.5f}  S22={S_general[1, 1]:.5f}  "
        f"S12={S_general[0, 1]:.5f}  S21={S_general[1, 0]:.5f}"
    )

# ## Visualization
#
# The deformation is large, so the deformed cell is drawn at true scale. The panels show the
# microstructure, the deformed cell over the outline of the undeformed cell, and the periodic
# fluctuation $\mathbf{u}^*=\mathbf{u}-(\bar{\mathbf{F}}-\boldsymbol{\delta})\mathbf{X}$.


# + tags=["hide-input"]
def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str, root: int = 0):
    """Owned-cell PyVista grids with ``u`` attached, gathered on ``root``."""
    bs = V.dofmap.index_map_bs
    owned_cells = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned_cells))
    padded = np.zeros((grid.n_points, 3))
    padded[:, :bs] = u.x.array.real[: grid.n_points * bs].reshape(-1, bs)
    grid.point_data[name] = padded
    magnitude = np.linalg.norm(padded, axis=1)
    grid.point_data[f"|{name}|"] = magnitude
    lo = comm.allreduce(float(magnitude.min()) if magnitude.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(magnitude.max()) if magnitude.size else -np.inf, op=MPI.MAX)
    return comm.gather(grid, root=root), [lo, hi]


def gather_cell_data(field: fem.Function, name: str, root: int = 0):
    """Owned-cell PyVista grids with a cellwise-constant field, gathered on ``root``."""
    owned_cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, tdim, owned_cells))
    grid.cell_data[name] = field.x.array.real[: len(owned_cells)]
    return comm.gather(grid, root=root)


def fluctuation(u: fem.Function, F_case: np.ndarray) -> tuple[fem.Function, fem.Function]:
    """The displacement in V and its periodic fluctuation u - (F_case - I) X."""
    u_V, w = fem.Function(V), fem.Function(V)
    u_V.x.array[:] = u.x.array[: u_V.x.array.size]
    w.interpolate(lambda x_: (F_case - np.eye(gdim)) @ x_[:gdim])
    w.x.array[:] = u_V.x.array - w.x.array
    return u_V, w


def fmt_F(F_case: np.ndarray) -> str:
    return f"[[{F_case[0, 0]:.2f}, {F_case[0, 1]:.2f}], [{F_case[1, 0]:.2f}, {F_case[1, 1]:.2f}]]"


bar = {"fmt": "%.1e", "n_labels": 3, "position_x": 0.2, "width": 0.6}
outline = pyvista.Rectangle([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0)])
material_title = f"Microstructure\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)"


def plot_cell(u: fem.Function, F_case: np.ndarray, titles: list[str], filename: str):
    """Microstructure, deformed cell (true scale) and fluctuation, with the load written in the titles."""
    u_V, w = fluctuation(u, F_case)
    u_pieces, u_clim = gather_grids(u_V, V, "u")
    w_pieces, w_clim = gather_grids(w, V, "w")
    material_pieces = gather_cell_data(E, "E")
    if comm.rank != 0:
        return
    factor_w = 0.1 * L / w_clim[1] if w_clim[1] < 0.05 * L else 1.0  # amplified only if small
    plotter = pyvista.Plotter(shape=(1, 3), window_size=[1500, 520])
    plotter.subplot(0, 0)
    plotter.add_text(material_title, font_size=10)
    for piece in material_pieces:
        plotter.add_mesh(
            piece,
            scalars="E",
            cmap="viridis",
            show_edges=False,
            scalar_bar_args={"n_labels": 2, "fmt": "%.0f", "position_x": 0.2, "width": 0.6},
        )
    plotter.view_xy()
    panels = [
        (u_pieces, u_clim, "u", 1.0, titles[0] + "\n(true scale)"),
        (
            w_pieces,
            w_clim,
            "w",
            factor_w,
            titles[1] + ("\n(true scale)" if factor_w == 1.0 else f"\n(amplified x{factor_w:.0f})"),
        ),
    ]
    for col, (pieces, clim, name, factor, title) in enumerate(panels, start=1):
        plotter.subplot(0, col)
        plotter.add_text(title, font_size=10)
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
        plotter.screenshot(filename)
    else:
        plotter.show()


plot_cell(
    uh_general,
    F_bar,
    [f"Deformed cell\nprescribed F = {fmt_F(F_bar)}", "Periodic fluctuation\nu* = u - (F - I) X"],
    "demo_periodic_homogenization_nl.png",
)
# -

# ## Stress control
#
# We now prescribe the average stress $\bar{\mathbf{S}}$ instead of the deformation. The
# displacements $\mathbf{u}^B$ and $\mathbf{u}^D$ become unknowns.
#
# This changes how the corners enter the problem. Under strain control $\mathbf{u}^B$, $\mathbf{u}^D$ are known:
# they are Dirichlet values on the corners, and enter the periodic constraints as their constant.
# Under stress control they are unknowns: the corners become masters of the periodic constraints,
# and the prescribed stress enters the right-hand side of the equations as point forces on their
# degrees of freedom, derived below. Each corner degree of freedom gets either a prescribed
# displacement or a prescribed force, never both.
#
# | | strain control | stress control |
# |---|---|---|
# | $\mathbf{u}^B$, $\mathbf{u}^D$ | prescribed, Dirichlet conditions | unknowns, masters of the constraints |
# | periodic constraints | constant from $\mathbf{u}^B$, $\mathbf{u}^D$ | no constant |
# | right-hand side | no load | point forces $A\bar S_{ij}$ on the corners |

# ### Corner displacements
#
# The energy of a neo-Hookean material is objective: a rigid rotation $\mathbf{R}$ of the deformed
# cell turns $\bar{\mathbf{F}}$ into $\mathbf{R}\bar{\mathbf{F}}$, without changing the energy or
# doing work. With $\bar{\mathbf{F}}$ unknown, nothing fixes this rotation. We remove it by
# choosing the $\mathbf{R}$ that turns $\bar{\mathbf{F}}\mathbf{e}_1$ onto $\mathbf{e}_1$, which
# makes $\bar F_{21}=0$, that is $u^B_2=0$. By {eq}`eq:nl-corners`,
#
# $$
# \mathbf{u}^B = L\begin{pmatrix}\bar F_{11}-1\\ 0\end{pmatrix},\qquad
# \mathbf{u}^D = L\begin{pmatrix}\bar F_{12}\\ \bar F_{22}-1\end{pmatrix}.
# $$ (eq:nl-sc-corners)

# ### The prescribed stress as nodal forces
#
# Let $\mathbf{T}=\mathbf{P}\mathbf{N}$ be the traction on $\partial\Omega$ in the reference
# configuration. The stress is periodic, while the outward normals $\mathbf{N}$ of opposite edges
# are opposite, so the tractions are anti-periodic,
# $\mathbf{T}(\mathbf{X}+L\mathbf{e}_1)=-\mathbf{T}(\mathbf{X})$ on RIGHT and LEFT, and likewise on
# TOP and BOTTOM. A field $\mathbf{v}$ that satisfies the constraints {eq}`eq:nl-periodic` with
# $\mathbf{v}^A=\mathbf{0}$ takes on RIGHT its values on LEFT plus $\mathbf{v}^B$, so the work of
# the tractions on the two edges cancels but for $\mathbf{v}^B$, and likewise on TOP and BOTTOM
# with $\mathbf{v}^D$:
#
# $$
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}\,\mathrm{d}S
# = v^B_i\int_{\text{RIGHT}}T_i\,\mathrm{d}S + v^D_i\int_{\text{TOP}}T_i\,\mathrm{d}S .
# $$
#
# With $\operatorname{Div}\mathbf{P}=\mathbf{0}$, the divergence theorem gives
# $\int_{\partial\Omega}T_iX_j\,\mathrm{d}S=\int_\Omega P_{ij}\,\mathrm{d}\Omega=|\Omega|\,\bar S_{ij}$.
# For $j=1$, $X_1=L$ on RIGHT and $X_1=0$ on LEFT, while on TOP and BOTTOM the anti-periodic
# tractions cancel at each $X_1$. Hence $\int_{\text{RIGHT}}T_i\,\mathrm{d}S=A\bar S_{i1}$, and
# likewise $\int_{\text{TOP}}T_i\,\mathrm{d}S=A\bar S_{i2}$, with $A=|\Omega|/L$ the length of an
# edge. The principle of virtual work, with $\bar{\mathbf{S}}$ prescribed, becomes
#
# $$
# \int_\Omega \mathbf{P}(\mathbf{F}):\nabla\mathbf{v}\,\mathrm{d}\Omega
# = A\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i\right)
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=0$. The prescribed stress is
# therefore a set of **nodal forces**:
#
# | component | control |
# |---|---|
# | $u^B_1=L(\bar F_{11}-1)$ | force $A\bar S_{11}$ |
# | $u^B_2=0$ | removes the rigid rotation |
# | $u^D_1=L\bar F_{12}$ | force $A\bar S_{12}$ |
# | $u^D_2=L(\bar F_{22}-1)$ | force $A\bar S_{22}$ |
#
# $\bar{\mathbf{S}}$ is not symmetric, and only three of its components are independent: the
# balance of moments requires $\bar{\mathbf{S}}\bar{\mathbf{F}}^T$ to be symmetric. The reaction at
# $u^B_2$ is $A\bar S_{21}$, which that balance determines from the other three.

# ### Constraints with free masters
#
# The masters $B$ and $D$ are now degrees of freedom. Each slave has two masters: its partner on
# LEFT and $B$, its partner on BOTTOM and $D$, or $B$ and $D$ for the corner $C$. These
# constraints are built with
# {py:meth}`create_general_constraint <dolfinx_mpc.MultiPointConstraint.create_general_constraint>`,
# one call per component. For the second component the Dirichlet master $u^B_2=0$ is folded into
# the constraint.

# +
S_B = np.array([4.5, 0.0])  # prescribed S_11 (first entry; u^B_2 = 0 is a constraint), 50 x the linear demo
S_D = np.array([2.5, 0.0])  # prescribed (S_12, S_22)
area = L  # area of a side of the cell, per unit thickness

bc_A_sc, _ = dirichletbc_at_point(V, corner(0, 0))
V1 = V.sub(1).collapse()[0]
bc_B1_sc = fem.dirichletbc(fem.Function(V1), fem.locate_dofs_geometrical((V.sub(1), V1), corner(L, 0)), V.sub(1))
bcs_sc = [bc_A_sc, bc_B1_sc]  # u^A = 0, u^B_2 = 0
mpc_sc = MultiPointConstraint(V, dtype=dtype, bcs=bcs_sc)


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

# ### Residual with the nodal forces
#
# The nodal forces are subtracted from the residual assembled by DOLFINx-MPC, at the dofs of $B$
# and $D$. These are masters of the constraint, so they belong to the reduced system, and they
# are neither Dirichlet dofs nor slaves.

# +
problem_sc, uh_sc = nonlinear_problem(mpc_sc, bcs_sc, "stress_")
dofs_B = [fem.locate_dofs_geometrical((V.sub(c), V.sub(c).collapse()[0]), corner(L, 0))[0] for c in range(gdim)]
dofs_D = [fem.locate_dofs_geometrical((V.sub(c), V.sub(c).collapse()[0]), corner(0, L))[0] for c in range(gdim)]
n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
load_factor = [0.0]


_, (assemble_residual, residual_args, residual_kwargs) = problem_sc.solver.getFunction()


def residual_with_forces(snes, x_vec, F_vec):
    assemble_residual(snes, x_vec, F_vec, *residual_args, **residual_kwargs)  # residual of DOLFINx-MPC
    t = load_factor[0]  # internal minus external forces, on the processes that own B and D
    F_vec.array[dofs_B[0][dofs_B[0] < n_owned]] -= t * area * S_B[0]
    for c in range(gdim):
        F_vec.array[dofs_D[c][dofs_D[c] < n_owned]] -= t * area * S_D[c]


problem_sc.solver.setFunction(residual_with_forces, problem_sc.b)


def set_stress(t: float):
    load_factor[0] = t


# -

# ### Verification
#
# **Homogeneous cell.** The exact solution is affine, with $\bar F_{21}=0$ and $\bar F_{11}$,
# $\bar F_{12}$, $\bar F_{22}$ the solution of $P_{11}(\bar{\mathbf{F}})=\bar S_{11}$,
# $P_{12}(\bar{\mathbf{F}})=\bar S_{12}$, $P_{22}(\bar{\mathbf{F}})=\bar S_{22}$, computed here by
# Newton's method on the three scalar equations. It must be reproduced to round-off.

# +
set_young_modulus(E_uniform)
solve_in_steps(problem_sc, uh_sc, mpc_sc, set_stress, 20, np.zeros((gdim, gdim)))
mu0, lmbda0 = E_uniform / (2 * (1 + nu)), E_uniform * nu / ((1 + nu) * (1 - 2 * nu))


def piola_np(F_: np.ndarray) -> np.ndarray:
    Finv_T = np.linalg.inv(F_).T
    return mu0 * (F_ - Finv_T) + lmbda0 * np.log(np.linalg.det(F_)) * Finv_T


def residual_np(z: np.ndarray) -> np.ndarray:
    P = piola_np(np.array([[z[0], z[1]], [0.0, z[2]]]))
    return np.array([P[0, 0] - S_B[0], P[0, 1] - S_D[0], P[1, 1] - S_D[1]])


z = np.array([1.0, 0.0, 1.0])
for _ in range(50):
    jac = np.column_stack([(residual_np(z + h) - residual_np(z - h)) / 2e-7 for h in 1e-7 * np.eye(3)])
    z -= np.linalg.solve(jac, residual_np(z))
F_exact = np.array([[z[0], z[1]], [0.0, z[2]]])
diff = uh_sc - ufl.dot(ufl.as_tensor(F_exact - np.eye(gdim)), x)
error_sc = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < atol
# -

# **Heterogeneous cell.** $\bar{\mathbf{F}}$ is read from the free masters $B$ and $D$ through
# {eq}`eq:nl-sc-corners`. As a cross-check, the strain-controlled problem of the first part is
# solved with the $\bar{\mathbf{F}}$ obtained here: it gives the same $\bar{\mathbf{S}}$, with the
# prescribed $\bar S_{11}$, $\bar S_{12}$ and $\bar S_{22}$.

# +
set_young_modulus(50.0 * E_uniform)
S_sc, F_sc = solve_in_steps(problem_sc, uh_sc, mpc_sc, set_stress, 20, np.zeros((gdim, gdim)))
_, S_check = homogenized_stress(F_sc, n_steps=20)
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  F_bar = [[{F_sc[0, 0]:.6e}, {F_sc[0, 1]:.6e}], [{F_sc[1, 0]:.6e}, {F_sc[1, 1]:.6e}]]")
    print(f"  S_bar = [[{S_sc[0, 0]:.6e}, {S_sc[0, 1]:.6e}], [{S_sc[1, 0]:.6e}, {S_sc[1, 1]:.6e}]]")
    print(
        f"  strain control with this F_bar: S_bar = [[{S_check[0, 0]:.6e}, {S_check[0, 1]:.6e}],"
        f" [{S_check[1, 0]:.6e}, {S_check[1, 1]:.6e}]]"
    )
# -

# The cell under the prescribed stress.

# + tags=["hide-input"]
stress_load = f"S11 = {S_B[0]:g}, S12 = {S_D[0]:g}, S22 = {S_D[1]:g} prescribed"
plot_cell(
    uh_sc,
    F_sc,
    [f"Deformed cell, stress control\n{stress_load}", "Periodic fluctuation\nu* = u - (F - I) X"],
    "demo_periodic_homogenization_nl_stress_control.png",
)
# -

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homognl-
# ```

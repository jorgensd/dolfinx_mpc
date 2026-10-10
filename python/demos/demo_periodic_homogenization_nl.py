# # Periodic boundary conditions for a representative volume element: finite strain
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT

# +
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot

import dolfinx_mpc
from dolfinx_mpc import MultiPointConstraint, dofs_at_point

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
# Crossed: each square is cut into four triangles, so the mesh has the symmetries of the square
domain = mesh.create_rectangle(
    comm, [[0, 0], [L, L]], [N, N], diagonal=mesh.DiagonalType.crossed, dtype=default_real_type
)
# Largest distance between a node and a point it is located at, from the rounding of the coordinates
geom_tol = 500 * np.finfo(default_real_type).eps * L
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
# \begin{aligned}
# \mathbf{u}(\mathbf{X}) &= (\bar{\mathbf{F}}-\boldsymbol{\delta})\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),\\
# \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) &= \mathbf{u}^*(\mathbf{X}),
# \end{aligned}
# $$
#
# where $\bar{\mathbf{F}}$ is the average deformation gradient (`F_bar`), a full, non-symmetric
# tensor. With the corners $A=(0,0)$, $B=(L,0)$, $C=(L,L)$, $D=(0,L)$ and $\mathbf{u}^A=\mathbf{0}$,
#
# $$
# \begin{aligned}
# \mathbf{u}^A &= \mathbf{0},\\
# \mathbf{u}^B &= (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}L\\0\end{pmatrix},\\
# \mathbf{u}^D &= (\bar{\mathbf{F}}-\boldsymbol{\delta})\begin{pmatrix}0\\L\end{pmatrix},
# \end{aligned}
# $$ (eq:nl-corners)
#
# and periodicity of $\mathbf{u}^*$ reduces every other constraint to
#
# $$
# \begin{aligned}
# \mathbf{u}^{\text{RIGHT}} &= \mathbf{u}^{\text{LEFT}} + \mathbf{u}^B,\\
# \mathbf{u}^{\text{TOP}} &= \mathbf{u}^{\text{BOTTOM}} + \mathbf{u}^D,\\
# \mathbf{u}^C &= \mathbf{u}^B + \mathbf{u}^D.
# \end{aligned}
# $$ (eq:nl-periodic)
#
# ### Dirichlet conditions on the corners
#
# The corners $A$, $B$, $D$ are fixed with {py:class}`dolfinx.fem.DirichletBC`. The values are
# stored in functions, so that they can be updated during the load stepping.

# +


def corner(px: float, py: float):
    """Indicator function for a single point, padded for a 3D coordinate array."""
    return lambda x: np.isclose(x[0], px, atol=geom_tol) & np.isclose(x[1], py, atol=geom_tol)


def dirichletbc_at_point(V: fem.FunctionSpace, indicator) -> tuple[fem.DirichletBC, fem.Function]:
    """A Dirichlet condition on every degree of freedom at one point; its value is the function returned."""
    dofs = fem.locate_dofs_geometrical(V, indicator)
    fn = fem.Function(V, dtype=dtype)
    return fem.dirichletbc(fn, dofs), fn


def set_value(fn: fem.Function, value: np.ndarray):
    fn.interpolate(lambda x: np.tile(np.asarray(value, dtype=default_scalar_type).reshape(-1, 1), x.shape[1]))


bc_A, _ = dirichletbc_at_point(V, corner(0, 0))
bc_B, value_B = dirichletbc_at_point(V, corner(L, 0))
bc_D, value_D = dirichletbc_at_point(V, corner(0, L))
bcs = [bc_A, bc_B, bc_D]
# -

# ### Periodic constraints with corner masters
#
# As in the {doc}`linear demo <demo_periodic_homogenization>`, the relations {eq}`eq:nl-periodic`
# share one form: for a node $\mathbf{X}$ on RIGHT, TOP or at $C$, with $s_i=1$ if $X_i=L$ and
# $s_i=0$ otherwise, and its image $\mathbf{X}^-=\mathbf{X}-L\mathbf{s}$,
# $\mathbf{u}(\mathbf{X}) = \mathbf{u}(\mathbf{X}^-) + s_1\mathbf{u}^B + s_2\mathbf{u}^D$.
# {py:meth}`create_periodic_constraint_geometrical
# <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>` ties each node to its
# image, and {py:meth}`add_master_from_point <dolfinx_mpc.MultiPointConstraint.add_master_from_point>`
# adds the corner of direction $i$ as an extra master of the nodes with $X_i=L$, one call per
# direction. Here the corners carry Dirichlet conditions,
# whose values are folded into the constraint.


# +
def set_macroscopic_deformation(F_case: np.ndarray):
    """Corner values {eq}`eq:nl-corners` for the average deformation F_case."""
    set_value(value_B, (F_case - np.eye(gdim)) @ np.array([L, 0.0]))
    set_value(value_D, (F_case - np.eye(gdim)) @ np.array([0.0, L]))


master_corners = [(L, 0.0), (0.0, L)]  # B and D: the corner one period from A in direction i


def periodic_nodes(x):
    """RIGHT, TOP and C: the nodes with some X_i = L, except the corners B and D."""
    shifted = np.isclose(x[0], L, atol=geom_tol) | np.isclose(x[1], L, atol=geom_tol)
    return shifted & ~np.logical_or.reduce([corner(*p)(x) for p in master_corners])


def to_image(x):
    """X -> X^- = X - L s."""
    out = x.copy()
    out[:gdim][np.isclose(x[:gdim], L, atol=geom_tol)] -= L
    return out


def periodic_cell_constraint(bcs_: list[fem.DirichletBC]) -> MultiPointConstraint:
    """The constraint {eq}`eq:nl-periodic`, finalized, with the conditions `bcs_` folded in."""
    constraint = MultiPointConstraint(V, dtype=dtype, bcs=bcs_)
    constraint.create_periodic_constraint_geometrical(V, periodic_nodes, to_image, bcs_, scale=dtype.type(1.0))
    for i, corner_point in enumerate(master_corners):  # s_i u^{corner} on the nodes with X_i = L

        def on_side(x, i=i):
            return periodic_nodes(x) & np.isclose(x[i], L, atol=geom_tol)

        constraint.add_master_from_point(V, on_side, 1.0, corner_point)
    constraint.finalize()  # collective: every rank must reach this
    return constraint


xdt = domain.geometry.x.dtype
# Tolerance of the checks below, from the precision of the mesh coordinates
atol = 50 * np.sqrt(np.finfo(xdt).resolution)
set_macroscopic_deformation(np.eye(gdim))
mpc = periodic_cell_constraint(bcs)
# -

# ## Hyperelastic material
#
# Both phases are compressible neo-Hookean, in plane strain, with first Piola-Kirchhoff stress
#
# $$
# \begin{aligned}
# \mathbf{P}(\mathbf{F}) &= \mu\left(\mathbf{F}-\mathbf{F}^{-T}\right) + \lambda\ln J\,\mathbf{F}^{-T},\\
# \mathbf{F} &= \boldsymbol{\delta}+\nabla\mathbf{u},\\
# J &= \det\mathbf{F}.
# \end{aligned}
# $$
#
# The macroscopic stress is the volume average
#
# $$
# \bar{\mathbf{S}} = \frac{1}{|\Omega|}\int_\Omega \mathbf{P}(\mathbf{F})~\mathrm{d}x,
# $$
#
# which is not symmetric. The Young modulus is a cellwise constant function, uniform for the first test
# and with the stiff inclusion afterwards.

# +
E_uniform, nu = 10.0, 0.3
Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q, dtype=dtype)
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
# The owned cells of the inclusion, a disc of radius 0.25 at the centre of the cell
cells0 = np.flatnonzero((midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2).astype(np.int32)
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


def set_young_modulus(E_inclusion: float):
    E.interpolate(lambda x: np.full(x.shape[1], E_uniform))
    E.interpolate(lambda x: np.full(x.shape[1], E_inclusion), cells0=cells0)
    E.x.scatter_forward()


def piola(F):
    Finv_T = ufl.inv(F).T
    return mu * (F - Finv_T) + lmbda * ufl.ln(ufl.det(F)) * Finv_T


def assemble_scalar_global(form: fem.Form):
    """The value of a compiled scalar form, summed over all processes."""
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


# The stress is averaged for the solutions of several problems, so its forms are compiled once, for
# a function in V that average_stress fills: the leading entries of a constraint space are those of V
u_avg = fem.Function(V, dtype=dtype)
P_avg = piola(ufl.Identity(gdim) + ufl.grad(u_avg))
P_forms = [[fem.form(P_avg[i, j] * ufl.dx, dtype=dtype) for j in range(gdim)] for i in range(gdim)]


def average_stress(uh: fem.Function) -> np.ndarray:
    u_avg.x.array[:] = uh.x.array[: u_avg.x.array.size]
    return np.array([[assemble_scalar_global(P_ij) for P_ij in row] for row in P_forms]).real / L**2


# The dofs of the corners B and D, and the process owning them, located once
corner_dofs = {point: dofs_at_point(V, point) for point in ((L, 0.0), (0.0, L))}


def value_at(u: fem.Function, point) -> np.ndarray:
    """The value of `u` at the dofs of V at the corner `point`, read by the process owning them and sent to all."""
    dofs, owner = corner_dofs[point]
    value = None
    if comm.rank == owner:
        value = u.x.array[dofs - V.dofmap.index_map.local_range[0] * V.dofmap.index_map_bs].real
    return comm.bcast(value, root=owner)


def average_deformation(u: fem.Function) -> np.ndarray:
    """F_bar from the corner displacements u^B, u^D (eq:nl-corners)."""
    return np.eye(gdim) + np.column_stack([value_at(u, (L, 0.0)), value_at(u, (0.0, L))]) / L


# -

# ## Nonlinear problem and load stepping
#
# The residual is $\int_\Omega\mathbf{P}(\mathbf{F}):\nabla\mathbf{v}~\mathrm{d}x$, with
# the unknown in the space of the constraint and the test and trial functions in `V`, the space of
# the Dirichlet conditions. The macroscopic deformation is applied in load steps; after the
# Dirichlet values change,
# {py:meth}`update_constants <dolfinx_mpc.MultiPointConstraint.update_constants>` must be called
# before solving, so that the constraint uses the new values.

# +
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    # From the precision of the scalar type: about 2e-12 and 2e-11 in double precision
    "snes_rtol": 1e4 * np.finfo(dtype).eps,
    "snes_atol": 1e5 * np.finfo(dtype).eps,
    "snes_max_it": 40,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def nonlinear_problem(constraint: MultiPointConstraint, bcs_: list[fem.DirichletBC], prefix: str, external_work=None):
    """The problem of the cell, with the work `external_work(v)` of external forces, if any,
    subtracted from the residual. It does not depend on the solution, so not the Jacobian."""
    uh = fem.Function(constraint.function_space, dtype=dtype)
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    internal_work = ufl.inner(piola(ufl.Identity(gdim) + ufl.grad(uh)), ufl.grad(v)) * ufl.dx
    residual = internal_work if external_work is None else internal_work - external_work(v)
    problem = dolfinx_mpc.NonlinearProblem(
        residual,
        uh,
        constraint,
        bcs=bcs_,
        J=ufl.derivative(internal_work, uh, du),
        petsc_options=petsc_options,
        petsc_options_prefix=prefix,
    )
    return problem, uh


def set_affine(uh: fem.Function, G: np.ndarray):
    """uh = G X; the leading entries of the constraint space are those of V, the extra ghosts are updated."""
    affine = fem.Function(V, dtype=dtype)
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


def homogenized_stress(problem, uh: fem.Function, F_case: np.ndarray, n_steps: int = 20):
    """Solve the cell for the average deformation F_case and return (a copy of uh, S_bar). The solution
    is a copy: the problem solves into the same function every time."""
    S_bar, _ = solve_in_steps(
        problem,
        uh,
        mpc,
        lambda t: set_macroscopic_deformation(np.eye(gdim) + t * (F_case - np.eye(gdim))),
        n_steps,
        F_case - np.eye(gdim),
    )
    return uh.copy(), S_bar


# -

# ## Verifying the mechanism: a homogeneous unit cell
#
# For a homogeneous material the fluctuation vanishes and the exact solution is the affine field
# $\mathbf{u}=(\bar{\mathbf{F}}-\boldsymbol{\delta})\mathbf{X}$, whatever the material law. It lies
# in the P1 space, so it must be reproduced to round-off.

# +
set_young_modulus(E_uniform)
uh_homogeneous, _ = homogenized_stress(problem, uh, F_bar)
x = ufl.SpatialCoordinate(domain)
diff = uh_homogeneous - ufl.dot(ufl.as_tensor(F_bar - np.eye(gdim)), x)
u_affine = ufl.dot(ufl.as_tensor(F_bar - np.eye(gdim)), x)
error = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_affine = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_affine, u_affine) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < atol * norm_affine
# -

# ## A heterogeneous microstructure
#
# The matrix now contains a stiff circular inclusion, 50 times stiffer. As in the linear demo:
#
# 1. with no macroscopic deformation the average stress vanishes;
# 2. under an isotropic macroscopic stretch the average stress is isotropic, to round-off, by the
#    symmetry of the inclusion and of the crossed mesh;
# 3. under a general $\bar{\mathbf{F}}$ we report $\bar{\mathbf{S}}$.

set_young_modulus(50.0 * E_uniform)

# #### Test case 1: No macroscopic deformation

_, S_zero = homogenized_stress(problem, uh, np.eye(gdim), n_steps=1)
if comm.rank == 0:
    print(f"----No macroscopic deformation----\n  S_bar = {S_zero.tolist()}  (should vanish)")
assert np.abs(S_zero).max() < atol

# #### Test case 2: Isotropic macroscopic stretch

_, S_iso = homogenized_stress(problem, uh, (1.0 + 10 * 0.02) * np.eye(gdim))  # 10 x the linear demo
if comm.rank == 0:
    print(
        f"----Isotropic macroscopic stretch----\n  S11={S_iso[0, 0]:.5f}  S22={S_iso[1, 1]:.5f}  "
        f"S12={S_iso[0, 1]:.2e}  S21={S_iso[1, 0]:.2e}  (should be isotropic: S11≈S22, S12≈S21≈0)"
    )
assert abs(S_iso[0, 0] - S_iso[1, 1]) < atol * abs(S_iso[0, 0])
assert max(abs(S_iso[0, 1]), abs(S_iso[1, 0])) < atol * abs(S_iso[0, 0])

# #### Test case 3: General macroscopic deformation

uh_general, S_general = homogenized_stress(problem, uh, F_bar)
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
    """Owned-cell PyVista grids with a cellwise-constant field, gathered on ``root``, and the range
    of the field over all processes."""
    owned_cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, tdim, owned_cells))
    values = field.x.array.real[: len(owned_cells)]
    grid.cell_data[name] = values
    lo = comm.allreduce(float(values.min()) if values.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(values.max()) if values.size else -np.inf, op=MPI.MAX)
    return comm.gather(grid, root=root), [lo, hi]


def fluctuation(u: fem.Function, F_case: np.ndarray) -> tuple[fem.Function, fem.Function]:
    """The displacement in V and its periodic fluctuation u - (F_case - I) X."""
    u_V, w = fem.Function(V, dtype=dtype), fem.Function(V, dtype=dtype)
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
    material_pieces, material_clim = gather_cell_data(E, "E")
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
            clim=material_clim,
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
# The energy of a neo-Hookean material is objective: a rigid rotation $\mathbf{R}$ of the deformed
# cell turns $\bar{\mathbf{F}}$ into $\mathbf{R}\bar{\mathbf{F}}$, without changing the energy or
# doing work. With $\bar{\mathbf{F}}$ unknown, nothing fixes this rotation. We remove it by
# choosing the $\mathbf{R}$ that turns $\bar{\mathbf{F}}\mathbf{e}_1$ onto $\mathbf{e}_1$, which
# makes $\bar F_{21}=0$, that is $u^B_2=0$. By {eq}`eq:nl-corners`,
#
# $$
# \begin{aligned}
# \mathbf{u}^B &= L\begin{pmatrix}\bar F_{11}-1\\ 0\end{pmatrix},\\
# \mathbf{u}^D &= L\begin{pmatrix}\bar F_{12}\\ \bar F_{22}-1\end{pmatrix}.
# \end{aligned}
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
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}~\mathrm{d}s
# = v^B_i\int_{\text{RIGHT}}T_i~\mathrm{d}s + v^D_i\int_{\text{TOP}}T_i~\mathrm{d}s .
# $$
#
# With $\operatorname{Div}\mathbf{P}=\mathbf{0}$, the divergence theorem gives
# $\int_{\partial\Omega}T_iX_j~\mathrm{d}s=\int_\Omega P_{ij}~\mathrm{d}x=|\Omega|\,\bar S_{ij}$.
# For $j=1$, $X_1=L$ on RIGHT and $X_1=0$ on LEFT, while on TOP and BOTTOM the anti-periodic
# tractions cancel at each $X_1$. Hence $\int_{\text{RIGHT}}T_i~\mathrm{d}s=|\Gamma|\bar S_{i1}$, and
# likewise $\int_{\text{TOP}}T_i~\mathrm{d}s=|\Gamma|\bar S_{i2}$, with $|\Gamma|=|\Omega|/L$ the length of an
# edge. The principle of virtual work, with $\bar{\mathbf{S}}$ prescribed, becomes
#
# $$
# \int_\Omega \mathbf{P}(\mathbf{F}):\nabla\mathbf{v}~\mathrm{d}x
# = |\Gamma|\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i\right)
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=0$. The prescribed stress is
# therefore a set of **nodal forces**:
#
# | component | control |
# |---|---|
# | $u^B_1=L(\bar F_{11}-1)$ | force $|\Gamma|\bar S_{11}$ |
# | $u^B_2=0$ | removes the rigid rotation |
# | $u^D_1=L\bar F_{12}$ | force $|\Gamma|\bar S_{12}$ |
# | $u^D_2=L(\bar F_{22}-1)$ | force $|\Gamma|\bar S_{22}$ |
#
# $\bar{\mathbf{S}}$ is not symmetric, and only three of its components are independent: the
# balance of moments requires $\bar{\mathbf{S}}\bar{\mathbf{F}}^T$ to be symmetric. The reaction at
# $u^B_2$ is $|\Gamma|\bar S_{21}$, which that balance determines from the other three.

# ### Constraints with free masters
#
# The masters $B$ and $D$ are now degrees of freedom, and the constraint is the same as under
# strain control, built by `periodic_cell_constraint`, with the conditions
# $\mathbf{u}^A=\mathbf{0}$ and $u^B_2=0$. The Dirichlet master $u^B_2=0$ is folded into the
# constraint, while $u^B_1$, $u^D_1$ and $u^D_2$ stay masters.

# +
S_B = np.array([4.5, 0.0])  # prescribed S_11 (first entry; u^B_2 = 0 is a constraint), 50 x the linear demo
S_D = np.array([2.5, 0.0])  # prescribed (S_12, S_22)
area = L  # area of a side of the cell, per unit thickness

bc_A_sc, _ = dirichletbc_at_point(V, corner(0, 0))
V1 = V.sub(1).collapse()[0]
bc_B1_sc = fem.dirichletbc(
    fem.Function(V1, dtype=dtype), fem.locate_dofs_geometrical((V.sub(1), V1), corner(L, 0)), V.sub(1)
)
bcs_sc = [bc_A_sc, bc_B1_sc]  # u^A = 0, u^B_2 = 0
mpc_sc = periodic_cell_constraint(bcs_sc)
# -

# ### The nodal forces as vertex integrals
#
# The nodal forces enter the residual as vertex integrals over the corners $B$ and $D$, scaled by
# the load factor $t$ of the load steps:
#
# $$
# \int_\Omega \mathbf{P}(\mathbf{F}):\nabla\mathbf{v}~\mathrm{d}x
# - t\left(\int_{\{B\}} |\Gamma|\bar S_{i1}\,v_i~\mathrm{d}P + \int_{\{D\}} |\Gamma|\bar S_{i2}\,v_i~\mathrm{d}P\right).
# $$
#
# Their degrees of freedom are masters of the constraint, so the forces stay in the reduced
# system. The force on $u^B_2$ is zero, and its Dirichlet condition would remove it anyway.

# +
TAG_B, TAG_D = 1, 2
vertex_map = domain.topology.index_map(0)
vertex_values = np.full(vertex_map.size_local + vertex_map.num_ghosts, -1, dtype=np.int32)
for tag, point_ in ((TAG_B, (L, 0)), (TAG_D, (0, L))):
    vertex_values[mesh.locate_entities_boundary(domain, 0, corner(*point_))] = tag
marked_vertices = np.flatnonzero(vertex_values != -1).astype(np.int32)
corner_tags = mesh.meshtags(domain, 0, marked_vertices, vertex_values[marked_vertices])
dP = ufl.Measure("dP", domain=domain, subdomain_data=corner_tags)
# Column j is the force per area on the corner of direction j: (S_11, 0) on B and (S_12, S_22) on D.
# Not the stress tensor: the entry on u^B_2 is a placeholder, removed by its Dirichlet condition,
# where the reaction is S_21.
corner_loads = fem.Constant(domain, np.column_stack([S_B, S_D]).astype(default_scalar_type))
A_side = fem.Constant(domain, default_scalar_type(area))
load_factor = fem.Constant(domain, default_scalar_type(0.0))


def point_forces(v):
    return (
        load_factor
        * A_side
        * (ufl.inner(corner_loads[:, 0], v) * dP(TAG_B) + ufl.inner(corner_loads[:, 1], v) * dP(TAG_D))
    )


problem_sc, uh_sc = nonlinear_problem(mpc_sc, bcs_sc, "stress_", point_forces)


def set_stress(t: float):
    load_factor.value = t


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
u_exact = ufl.dot(ufl.as_tensor(F_exact - np.eye(gdim)), x)
diff = uh_sc - u_exact
error_sc = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_exact = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < atol * norm_exact
# -

# **Heterogeneous cell.** $\bar{\mathbf{F}}$ is read from the free masters $B$ and $D$ through
# {eq}`eq:nl-sc-corners`. As a cross-check, the strain-controlled problem of the first part is
# solved with the $\bar{\mathbf{F}}$ obtained here: it must give the same $\bar{\mathbf{S}}$, and
# $\bar S_{11}$, $\bar S_{12}$ and $\bar S_{22}$ must be the prescribed ones.

# +
set_young_modulus(50.0 * E_uniform)
S_sc, F_sc = solve_in_steps(problem_sc, uh_sc, mpc_sc, set_stress, 20, np.zeros((gdim, gdim)))
_, S_check = homogenized_stress(problem, uh, F_sc, n_steps=20)
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  F_bar = [[{F_sc[0, 0]:.6e}, {F_sc[0, 1]:.6e}], [{F_sc[1, 0]:.6e}, {F_sc[1, 1]:.6e}]]")
    print(f"  S_bar = [[{S_sc[0, 0]:.6e}, {S_sc[0, 1]:.6e}], [{S_sc[1, 0]:.6e}, {S_sc[1, 1]:.6e}]]")
    print(
        f"  strain control with this F_bar: S_bar = [[{S_check[0, 0]:.6e}, {S_check[0, 1]:.6e}],"
        f" [{S_check[1, 0]:.6e}, {S_check[1, 1]:.6e}]]"
    )
S_scale = np.abs(S_sc).max()
assert np.allclose(S_sc, S_check, rtol=0, atol=atol * S_scale)
assert np.allclose([S_sc[0, 0], S_sc[0, 1], S_sc[1, 1]], [S_B[0], S_D[0], S_D[1]], rtol=0, atol=atol * S_scale)
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

# +
del problem, problem_sc
PETSc.garbage_cleanup(comm)
# -

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homognl-
# ```

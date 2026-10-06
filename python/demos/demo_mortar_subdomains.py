# # Mortar coupling of two subdomains
# **Author** Jørgen S. Dokken
#
# {doc}`demo_mortar` uses a Lagrange multiplier {cite}`mortarsub-Babuska1973` to impose a
# boundary condition. Here the same idea joins two subdomains, each meshed and
# discretised on its own, across the interface between them. The coupling terms are
# integrals over *interior* facets of the parent mesh, and each subdomain's functions
# exist on one side of those facets only.
#
# ## Mathematical formulation
#
# $\Omega=(0,1)^2$ is split along $\Gamma=\{\tfrac{1}{2}\}\times(0,1)$ into
# $\Omega_1=(0,\tfrac{1}{2})\times(0,1)$ and $\Omega_2=(\tfrac{1}{2},1)\times(0,1)$.
# We seek $u_i$ on $\Omega_i$ with
#
# $$
# \begin{align*}
# -\Delta u_i &= f && \text{in } \Omega_i,\\
# u_1 &= g && \text{on } \{0\}\times(0,1),\\
# u_2 &= g && \text{on } \{1\}\times(0,1),\\
# u_1 = u_2,\quad \frac{\partial u_1}{\partial x} &= \frac{\partial u_2}{\partial x}
#   && \text{on } \Gamma,
# \end{align*}
# $$
#
# and both $u_i$ periodic in $y$. The periodicity is a
# {py:class}`dolfinx_mpc.MultiPointConstraint` on each subdomain, and the Dirichlet
# conditions are imposed strongly. The interface conditions are what the multiplier
# handles. Introducing $\lambda\in\Lambda := H^{-1/2}(\Gamma)$ we seek
# $(u_1, u_2, \lambda)\in V_1\times V_2\times\Lambda$ with
#
# $$
# \begin{align*}
# \int_{\Omega_1} \nabla u_1 \cdot \nabla v_1 ~\mathrm{d}x
#   - \int_\Gamma \lambda v_1 ~\mathrm{d}s &= \int_{\Omega_1} f v_1 ~\mathrm{d}x
#   && \forall v_1 \in V_1,\\
# \int_{\Omega_2} \nabla u_2 \cdot \nabla v_2 ~\mathrm{d}x
#   + \int_\Gamma \lambda v_2 ~\mathrm{d}s &= \int_{\Omega_2} f v_2 ~\mathrm{d}x
#   && \forall v_2 \in V_2,\\
# \int_\Gamma \mu (u_2 - u_1) ~\mathrm{d}s &= 0 && \forall \mu \in \Lambda.
# \end{align*}
# $$
#
# The last equation is continuity of $u$ across $\Gamma$. Integrating the first two by
# parts identifies $\lambda = \partial u/\partial x$ on $\Gamma$, the flux that both
# subdomains share, so flux continuity holds without being imposed.

# + tags=["hide-input"]
from __future__ import annotations

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot

import dolfinx_mpc

# -

# ## Subdomain and interface meshes
#
# Each subdomain is a submesh of the parent mesh, and so is $\Gamma$, built from the
# parent's facets on $x=\tfrac{1}{2}$. Their
# {py:class}`entity maps<dolfinx.mesh.EntityMap>` relate them to the parent, which is
# where the coupling terms are integrated. Interior facet integrals need the cells on
# both sides of a facet, so the parent mesh ghosts across shared facets.

# +
N = 32
msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mesh.GhostMode.shared_facet)
tdim = msh.topology.dim
fdim = tdim - 1

left_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
right_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] >= 0.5)
omega_1, entity_map_1 = mesh.create_submesh(msh, tdim, left_cells)[:2]
omega_2, entity_map_2 = mesh.create_submesh(msh, tdim, right_cells)[:2]

msh.topology.create_entities(fdim)
gamma_facets = mesh.locate_entities(msh, fdim, lambda x: np.isclose(x[0], 0.5))
gamma, entity_map_gamma = mesh.create_submesh(msh, fdim, gamma_facets)[:2]
entity_maps = [entity_map_1, entity_map_2, entity_map_gamma]
# -

# ## The interface measure
#
# An interior facet integral has a "+" and a "-" side, and nothing ties a side to a
# subdomain unless we say so. We list the integration entities ourselves, as
# `(cell, local facet)` for each side, with the $\Omega_1$ cell first, so that the
# "+" restriction always means $\Omega_1$ and "-" means $\Omega_2$.
# {py:func}`compute_integration_domains<dolfinx.fem.compute_integration_domains>`
# gives the entities of the facets this process owns, in arbitrary order, and we
# swap the two sides wherever the first cell is not in $\Omega_1$.


# +
def interface_entities(msh, facets, plus_cells):
    """Integration entities of `facets`, with cells in `plus_cells` on the "+" side."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    entities = fem.compute_integration_domains(fem.IntegralType.interior_facet, msh.topology, facets)
    entities = entities.reshape(-1, 4)

    cell_map = msh.topology.index_map(tdim)
    is_plus = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
    is_plus[plus_cells] = True
    swap = ~is_plus[entities[:, 0]]
    entities[swap] = entities[swap][:, [2, 3, 0, 1]]
    return entities.ravel()


interface_tag = 1
dS = ufl.Measure("dS", domain=msh, subdomain_data=[(interface_tag, interface_entities(msh, gamma_facets, left_cells))])
dGamma = dS(interface_tag)
# -

# ## Problem data
#
# We manufacture $u_{ex}=(1+x^2)\cos(2\pi y)$, which is periodic in $y$ and smooth
# across $\Gamma$, with flux $\lambda_{ex} = \cos(2\pi y)$ there. The source is
# derived from it with UFL, on each subdomain's own coordinates.

# +
degree = 2
V_1 = fem.functionspace(omega_1, ("Lagrange", degree))
V_2 = fem.functionspace(omega_2, ("Lagrange", degree))
Q = fem.functionspace(gamma, ("Lagrange", degree))


def u_exact(x):
    return (1 + x[0] ** 2) * ufl.cos(2 * ufl.pi * x[1])


x_1 = ufl.SpatialCoordinate(omega_1)
x_2 = ufl.SpatialCoordinate(omega_2)
f_1 = -ufl.div(ufl.grad(u_exact(x_1)))
f_2 = -ufl.div(ufl.grad(u_exact(x_2)))
# -

# ## The block system
#
# The coupling blocks are integrated over $\Gamma$ on the parent mesh. $v_1$ is
# restricted to the "+" side and $v_2$ to the "-" side, where their cells are; the
# multiplier lives on the facet itself, so either restriction gives the same value.
#
# As in {doc}`demo_mortar`, the $(2,2)$ block is mathematically zero but kept, as a
# {py:class}`ufl.ZeroBaseForm`, because the periodic constraint on $\Lambda$ has
# constrained rows there.

u_1, v_1 = ufl.TrialFunction(V_1), ufl.TestFunction(V_1)
u_2, v_2 = ufl.TrialFunction(V_2), ufl.TestFunction(V_2)
lmbda, mu = ufl.TrialFunction(Q), ufl.TestFunction(Q)
a = [
    [ufl.inner(ufl.grad(u_1), ufl.grad(v_1)) * ufl.dx, None, -ufl.inner(lmbda("+"), v_1("+")) * dGamma],
    [None, ufl.inner(ufl.grad(u_2), ufl.grad(v_2)) * ufl.dx, ufl.inner(lmbda("+"), v_2("-")) * dGamma],
    [-ufl.inner(u_1("+"), mu("+")) * dGamma, ufl.inner(u_2("-"), mu("+")) * dGamma, ufl.ZeroBaseForm((lmbda, mu))],
]
L = [ufl.inner(f_1, v_1) * ufl.dx, ufl.inner(f_2, v_2) * ufl.dx, ufl.ZeroBaseForm((mu,))]

# ## Boundary conditions and constraints
#
# The Dirichlet conditions live on the outer edges of each subdomain. Every space is
# periodic in $y$, the multiplier included, for the reason given in
# {doc}`demo_mortar`. The point $(\tfrac{1}{2}, 1)$ is a slave of all three spaces at
# once, and lies on facets where the two subdomains meet.

# +
g_1, g_2 = fem.Function(V_1), fem.Function(V_2)
g_1.interpolate(fem.Expression(u_exact(x_1), V_1.element.interpolation_points))
g_2.interpolate(fem.Expression(u_exact(x_2), V_2.element.interpolation_points))
bc_1 = fem.dirichletbc(g_1, fem.locate_dofs_geometrical(V_1, lambda x: np.isclose(x[0], 0.0)))
bc_2 = fem.dirichletbc(g_2, fem.locate_dofs_geometrical(V_2, lambda x: np.isclose(x[0], 1.0)))


def on_top(coords):
    return np.isclose(coords[1], 1.0)


def to_bottom(coords):
    return np.vstack([coords[0], 1.0 - coords[1], coords[2]])


constraints = []
for space, bcs in ((V_1, [bc_1]), (V_2, [bc_2]), (Q, [])):
    constraint = dolfinx_mpc.MultiPointConstraint(space)
    constraint.create_periodic_constraint_geometrical(space, on_top, to_bottom, bcs, default_scalar_type(1.0))
    constraint.finalize()
    constraints.append(constraint)
# -

# ## Solving
#
# The saddle point system is solved with MINRES and a block-diagonal preconditioner:
# the two subdomain Laplacians, which the Dirichlet conditions make non-singular, and
# the multiplier mass matrix in place of the zero block.

# +
P = [
    [ufl.inner(ufl.grad(u_1), ufl.grad(v_1)) * ufl.dx, None, None],
    [None, ufl.inner(ufl.grad(u_2), ufl.grad(v_2)) * ufl.dx, None],
    [None, None, ufl.inner(lmbda, mu) * ufl.dx],
]
block_options = {
    f"fieldsplit_{i}_{option}": value
    for i in range(3)
    for option, value in (("ksp_type", "preonly"), ("pc_type", "lu"))
}
problem = dolfinx_mpc.LinearProblem(
    a,
    L,
    constraints,
    bcs=[bc_1, bc_2],
    P=P,
    entity_maps=entity_maps,
    petsc_options_prefix="demo_mortar_subdomains_",
    petsc_options={
        "ksp_type": "minres",
        "ksp_rtol": 1e-10,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        **block_options,
    },
)
uh_1, uh_2, lh = problem.solve()
# -

# ## Verification
#
# $u$ against the manufactured solution on both subdomains, and $\lambda$ against the
# exact flux.


# +
def l2_error(uh, u_ex):
    error = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    return np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error), op=MPI.SUM).real)


error_u = np.hypot(l2_error(uh_1, u_exact(x_1)), l2_error(uh_2, u_exact(x_2)))
error_l = l2_error(lh, ufl.cos(2 * ufl.pi * ufl.SpatialCoordinate(gamma)[1]))
if msh.comm.rank == 0:
    print(
        f"N={N}, P{degree}:  {problem.solver.getIterationNumber()} MINRES iterations,"
        f"  |u-u_ex|_L2 = {error_u:.3e}   |lambda-lambda_ex|_L2 = {error_l:.3e}"
    )
assert error_u < 1e-3
# -

# The solutions on the two subdomains, drawn apart to show that each has its own mesh, and
# the multiplier $\lambda$ on the interface between them. The two solutions meet without a
# jump. Each process draws the cells it owns, which are gathered on the first process.

# +
pyvista.global_theme.allow_empty_mesh = True


def gathered_grid(u, V, name):
    """The cells this process owns as a PyVista grid with the values of `u`, gathered on the first process.

    `u` may live in the extended space of a constraint, whose leading entries are those of `V`.
    """
    tdim = V.mesh.topology.dim
    owned = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned))
    grid.point_data[name] = u.x.array.real[: grid.n_points]
    pieces = V.mesh.comm.gather(grid, root=0)
    # A process may own no cells, as of an interface, and its grid then holds no values
    return None if pieces is None else [piece for piece in pieces if piece.n_points > 0]


def value_range(pieces, name):
    return [min(p[name].min(initial=np.inf) for p in pieces), max(p[name].max(initial=-np.inf) for p in pieces)]


pieces_1 = gathered_grid(uh_1, V_1, "u")
pieces_2 = gathered_grid(uh_2, V_2, "u")
lambda_pieces = gathered_grid(lh, Q, "lambda")
if pieces_1 is not None:  # only the first process received the grids
    u_range = value_range(pieces_1 + pieces_2, "u")
    plotter = pyvista.Plotter(window_size=[800, 600])
    for piece, shift in [(p, -0.04) for p in pieces_1] + [(p, 0.04) for p in pieces_2]:
        plotter.add_mesh(piece.translate((shift, 0.0, 0.0)), scalars="u", cmap="viridis", clim=u_range)
    for piece in lambda_pieces:
        plotter.add_mesh(
            piece,
            scalars="lambda",
            cmap="coolwarm",
            clim=value_range(lambda_pieces, "lambda"),
            line_width=10,
            render_lines_as_tubes=True,
            scalar_bar_args={"position_y": 0.88},
        )
    plotter.view_xy()
    # The figure is named after the demo, as the gallery of the documentation expects
    figure = Path("demo_mortar_subdomains.py")
    if pyvista.OFF_SCREEN:
        plotter.screenshot(figure.with_suffix(".png"))
    else:
        # The interactive scene, for the gallery
        plotter.export_html(figure.with_suffix(".html"))
        plotter.show(screenshot=figure.with_suffix(".png"))
# -

# ## Convergence
#
# A repeat of the above, parametrised by resolution and degree and solved directly
# with MUMPS. $u$ converges at the optimal rate, $\lambda$ at second order.

# + tags=["hide-input"]


def solve_mortar(N: int, degree: int) -> tuple[float, float]:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    tdim = msh.topology.dim
    fdim = tdim - 1
    left_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    right_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] >= 0.5)
    omega_1, entity_map_1 = mesh.create_submesh(msh, tdim, left_cells)[:2]
    omega_2, entity_map_2 = mesh.create_submesh(msh, tdim, right_cells)[:2]
    msh.topology.create_entities(fdim)
    gamma_facets = mesh.locate_entities(msh, fdim, lambda x: np.isclose(x[0], 0.5))
    gamma, entity_map_gamma = mesh.create_submesh(msh, fdim, gamma_facets)[:2]
    dS = ufl.Measure("dS", domain=msh, subdomain_data=[(1, interface_entities(msh, gamma_facets, left_cells))])

    V_1 = fem.functionspace(omega_1, ("Lagrange", degree))
    V_2 = fem.functionspace(omega_2, ("Lagrange", degree))
    Q = fem.functionspace(gamma, ("Lagrange", degree))
    x_1, x_2 = ufl.SpatialCoordinate(omega_1), ufl.SpatialCoordinate(omega_2)
    u_1, v_1 = ufl.TrialFunction(V_1), ufl.TestFunction(V_1)
    u_2, v_2 = ufl.TrialFunction(V_2), ufl.TestFunction(V_2)
    lmbda, mu = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    a = [
        [ufl.inner(ufl.grad(u_1), ufl.grad(v_1)) * ufl.dx, None, -ufl.inner(lmbda("+"), v_1("+")) * dS(1)],
        [None, ufl.inner(ufl.grad(u_2), ufl.grad(v_2)) * ufl.dx, ufl.inner(lmbda("+"), v_2("-")) * dS(1)],
        [-ufl.inner(u_1("+"), mu("+")) * dS(1), ufl.inner(u_2("-"), mu("+")) * dS(1), ufl.ZeroBaseForm((lmbda, mu))],
    ]
    L = [
        -ufl.inner(ufl.div(ufl.grad(u_exact(x_1))), v_1) * ufl.dx,
        -ufl.inner(ufl.div(ufl.grad(u_exact(x_2))), v_2) * ufl.dx,
        ufl.ZeroBaseForm((mu,)),
    ]

    g_1, g_2 = fem.Function(V_1), fem.Function(V_2)
    g_1.interpolate(fem.Expression(u_exact(x_1), V_1.element.interpolation_points))
    g_2.interpolate(fem.Expression(u_exact(x_2), V_2.element.interpolation_points))
    bc_1 = fem.dirichletbc(g_1, fem.locate_dofs_geometrical(V_1, lambda x: np.isclose(x[0], 0.0)))
    bc_2 = fem.dirichletbc(g_2, fem.locate_dofs_geometrical(V_2, lambda x: np.isclose(x[0], 1.0)))
    constraints = []
    for space, bcs in ((V_1, [bc_1]), (V_2, [bc_2]), (Q, [])):
        constraint = dolfinx_mpc.MultiPointConstraint(space)
        constraint.create_periodic_constraint_geometrical(space, on_top, to_bottom, bcs, default_scalar_type(1.0))
        constraint.finalize()
        constraints.append(constraint)

    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        constraints,
        bcs=[bc_1, bc_2],
        entity_maps=[entity_map_1, entity_map_2, entity_map_gamma],
        petsc_options_prefix=f"demo_mortar_subdomains_p{degree}_n{N}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    uh_1, uh_2, lh = problem.solve()

    error_u = np.hypot(l2_error(uh_1, u_exact(x_1)), l2_error(uh_2, u_exact(x_2)))
    return error_u, l2_error(lh, ufl.cos(2 * ufl.pi * ufl.SpatialCoordinate(gamma)[1]))


for degree in (1, 2):
    previous = None
    for resolution in (4, 8, 16):
        err_u, err_l = solve_mortar(resolution, degree)
        rate = "" if previous is None else f"   rate {np.log2(previous / err_u):.2f}"
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"P{degree}  N={resolution:3d}   |u-u_ex|_L2 = {err_u:.3e}{rate}   |lambda-lambda_ex|_L2 = {err_l:.3e}"
            )
        last_rate: None | float = None if previous is None else np.log2(previous / err_u)
        previous = err_u
    assert last_rate is not None
    assert last_rate > degree + 0.9
# -

# The PETSc objects of the problem are freed, and those the garbage collector
# has released are cleaned up on every process together.

del problem
PETSc.garbage_cleanup(MPI.COMM_WORLD)

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: mortarsub-
# ```

# # Prescribing a flow rate with an affine multi-point constraint
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# {doc}`demo_boundary_average_constraint` enforces a scalar boundary integral
# $\int_\Gamma u~\mathrm{d}s=\gamma$ with an affine multi-point constraint. This
# demo applies the same construction to a **vector field on a blocked space**,
# which gives a *defective* outlet condition for Stokes flow: the volumetric flow
# rate is prescribed, but not the profile that carries it.
#
# In a channel $\Omega=(0,L)\times(0,H)$ with
# $\sigma = \nu\nabla\mathbf{u} - p\mathbb{I}$ we solve
#
# $$
# \begin{align*}
# -\nabla\cdot\sigma &= 0, \quad \nabla\cdot\mathbf{u} = 0 &&\text{in }\Omega,\\
# \mathbf{u} &= \mathbf{0} &&\text{on the walls},\\
# \sigma\cdot\mathbf{n} &= \mathbf{0} &&\text{on } \Gamma_{in},\\
# \int_{\Gamma_{out}} \mathbf{u}\cdot\mathbf{n}~\mathrm{d}s &= Q.
# \end{align*}
# $$
#
# As in the scalar case the constraint produces its conjugate condition on
# $\Gamma_{out}$, here $\sigma\cdot\mathbf{n} = \mu\mathbf{n}$: a constant normal
# traction with no tangential component. The solution is Poiseuille flow,
# $\mathbf{u} = (6Qy(H-y)/H^3, 0)$ and $p = -12\nu Qx/H^3$, both of which lie in
# the Taylor-Hood space, so the discrete solution is exact.
#
# Two things differ from the scalar case in {doc}`demo_boundary_average_constraint`.
# The velocity space is blocked, and the corner nodes of the outlet carry
# the no-slip condition. Those may not be chosen as the slave,
# but they are perfectly good *masters*: passing `bcs` to
# {py:class}`dolfinx_mpc.MultiPointConstraint` eliminates them from the relation
# and folds their contribution into the constraint offset.

# We start by importing the relevant modules.

# + tags=["hide-input"]
from __future__ import annotations

import time

from mpi4py import MPI

import basix.ufl
import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot

from dolfinx_mpc import LinearProblem, MultiPointConstraint

# -

# ## Marking the boundary

# +

WALL, INLET, OUTLET = 1, 2, 3


def stokes_markers(domain, length):
    """Tag walls, inlet and outlet of a channel of length ``length``."""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    exterior = mesh.exterior_facet_indices(domain.topology)
    inlet = mesh.locate_entities_boundary(domain, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    outlet = mesh.locate_entities_boundary(domain, tdim - 1, lambda x: np.isclose(x[0], length))
    values = np.full(len(exterior), WALL, dtype=np.int32)
    values[np.isin(exterior, inlet)] = INLET
    values[np.isin(exterior, outlet)] = OUTLET
    return mesh.meshtags(domain, tdim - 1, exterior, values)


# -

# ## Building the constraint
#
# The same builder as in {doc}`demo_boundary_average_constraint`. The functional
# is now $\mathbf{v}\mapsto\int_{\Gamma_{out}} \mathbf{v}\cdot\mathbf{n}~\mathrm{d}s$,
# so only the components along the outlet normal carry a weight and everything
# else is discarded by the filter.


def build_constraint(V, weight_form, value, bcs=()):
    """Constrain ``L(u) = value`` and report how many masters survived."""
    mpc = MultiPointConstraint(V, bcs=list(bcs))
    mpc.add_integral_constraint(weight_form, value, bcs=list(bcs))
    mpc.finalize()
    kept = sum(len(mpc.masters.links(s)) for s in mpc.slaves[: mpc.num_local_slaves])
    return mpc, V.mesh.comm.allreduce(kept, op=MPI.SUM)


# ## Variational form


def stokes_forms(V, P, nu):
    """Stokes forms with the stress tensor nu*grad(u) - p*I."""
    domain = V.mesh
    W = ufl.MixedFunctionSpace(V, P)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(ufl.div(u), q) * ufl.dx
    )
    zero_v = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=default_scalar_type))
    L = ufl.inner(zero_v, v) * ufl.dx + ufl.inner(fem.Constant(domain, default_scalar_type(0.0)), q) * ufl.dx
    return a, L


# ## Setting up the problem

# +

nx, ny = 32, 16
nu, flow_rate, length, height = 1.0, 1.0, 2.0, 1.0
comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([length, height])], [nx, ny])
tdim = domain.topology.dim
mt = stokes_markers(domain, length)
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
n = ufl.FacetNormal(domain)

V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(tdim,)))
P = fem.functionspace(domain, ("Lagrange", 1))
a, L = stokes_forms(V, P, nu)

u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0
bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, tdim - 1, mt.find(WALL)))

# -

# Only the components along the outlet normal carry a weight; the rest are
# filtered out. The no-slip corner dofs stay as masters and are folded into the
# offset by passing bcs to the constraint.

weight_form = ufl.dot(ufl.TestFunction(V), n) * ds(OUTLET)
comm.Barrier()
_t0 = time.perf_counter()
mpc_u, num_masters = build_constraint(V, weight_form, flow_rate, bcs=[bc])
mpc_p = MultiPointConstraint(P)
mpc_p.finalize()
comm.Barrier()
_t1 = time.perf_counter()
t_constraint = _t1 - _t0

# ## Solving

petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = LinearProblem(
    ufl.extract_blocks(a), ufl.extract_blocks(L), [mpc_u, mpc_p], bcs=[bc], petsc_options=petsc_options
)
uh, ph = problem.solve()
comm.Barrier()
t_mpc = time.perf_counter() - _t1

# ## Verification
#
# The flow rate is checked against its target, and the Poiseuille profile is
# recovered.

# +
x = ufl.SpatialCoordinate(domain)
u_ex = ufl.as_vector((6 * flow_rate * x[1] * (height - x[1]) / height**3, 0.0))
p_ex = -12 * nu * flow_rate * x[0] / height**3
area = comm.allreduce(
    fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds(OUTLET))), op=MPI.SUM
)
# The manufactured profile must itself carry the prescribed flow rate
exact_flux = comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(u_ex, n) * ds(OUTLET))), op=MPI.SUM)
exact_flux_error = abs(exact_flux - flow_rate)

flux = comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(OUTLET))), op=MPI.SUM)
error_u = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)), op=MPI.SUM))
error_p = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((ph - p_ex) ** 2 * ufl.dx)), op=MPI.SUM))
# -

# + tags=["hide-input"]
if comm.rank == 0:
    print("----Verification----")
    print(f"  block size                {V.dofmap.index_map_bs}")
    print(f"  masters                   {num_masters}")
    print(f"  int_out u.n ds            {flux:.15f} (target {flow_rate})")
    print(f"  L2(u_h - u_ex)            {error_u:.3e}")
    print(f"  L2(p_h - p_ex)            {error_p:.3e}")
    print(f"  build constraint [s]      {t_constraint:.3e}")
    print(f"  mpc solve [s]             {t_mpc:.3e}")
# -

assert exact_flux_error < 1e-12
assert abs(flux - flow_rate) < 1e-12
assert error_u < 1e-11
assert error_p < 1e-10

# Only 30 masters are kept: the outlet has 33 velocity nodes, one becomes the
# slave, and the two no-slip corners are eliminated by the Dirichlet conditions
# passed to the constraint and folded into its offset.

# ### Relation to a real space, on a submesh of the outlet
#
# The multiplier conjugate to a boundary functional lives on the outlet, so the
# reference puts the real space on a **submesh of the marked facets**, giving a
# three block system. See {doc}`demo_boundary_average_constraint` for the details
# of the mixed dimensional assembly, including the warning about which facets the
# submesh must cover.


def solve_stokes_real_space(domain, mt, nu, flow_rate, kind="mpi"):
    """Reference Stokes solve with the flow rate imposed by a real space.

    ``kind`` selects the PETSc matrix format (``"mpi"`` for a monolithic AIJ
    matrix, ``"nest"`` for the block format `dolfinx_mpc.LinearProblem` uses);
    it changes nothing about the discrete problem, only how it is assembled.
    """
    tdim = domain.topology.dim
    submesh, entity_map = mesh.create_submesh(domain, tdim - 1, mt.find(OUTLET))[:2]
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(tdim,)))
    P = fem.functionspace(domain, ("Lagrange", 1))
    R = fem.functionspace(submesh, basix.ufl.real_element(submesh.basix_cell(), dtype=submesh.geometry.x.dtype))
    W = ufl.MixedFunctionSpace(V, P, R)
    u, p, lam = ufl.TrialFunctions(W)
    v, q, mu = ufl.TestFunctions(W)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    n = ufl.FacetNormal(domain)
    area = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds(OUTLET))), op=MPI.SUM
    )

    zero_v = fem.Constant(domain, np.zeros(tdim, dtype=default_scalar_type))
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(ufl.div(u), q) * ufl.dx
        + ufl.inner(lam, ufl.dot(v, n)) * ds(OUTLET)
        + ufl.inner(ufl.dot(u, n), mu) * ds(OUTLET)
    )
    L = (
        ufl.inner(zero_v, v) * ufl.dx
        + ufl.inner(fem.Constant(domain, default_scalar_type(0.0)), q) * ufl.dx
        + ufl.inner(fem.Constant(domain, default_scalar_type(flow_rate / area)), mu) * ds(OUTLET)
    )

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, tdim - 1, mt.find(WALL)))
    problem = fem.petsc.LinearProblem(
        ufl.extract_blocks(a),
        ufl.extract_blocks(L),
        bcs=[bc],
        kind=kind,
        entity_maps=[entity_map],
        petsc_options_prefix=f"demo_stokes_real_{kind}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    uh, ph, lamh = problem.solve()
    lam_value = lamh.x.array[0] if lamh.x.array.size else 0.0
    return uh, ph, float(np.real(lam_value)), problem


# On a fully developed outlet sigma.n reduces to -p, and the reference form
# carries +lam, so the multiplier is the mean exact pressure there.

# +
lambda_exact = comm.allreduce(fem.assemble_scalar(fem.form(p_ex * ds(OUTLET))), op=MPI.SUM) / area

_t2 = time.perf_counter()
u_real, p_real, lam_real, real_problem = solve_stokes_real_space(domain, mt, nu, flow_rate, kind="mpi")
comm.Barrier()
t_real_mpi = time.perf_counter() - _t2

# Same problem, assembled as a `nest` instead of a monolithic matrix, timed only
# to isolate the format's own cost from the discrete problem it is solving.
_t3 = time.perf_counter()
solve_stokes_real_space(domain, mt, nu, flow_rate, kind="nest")
comm.Barrier()
t_real_nest = time.perf_counter() - _t3

num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
_diff = np.max(np.abs(uh.x.array[:num_owned] - u_real.x.array[:num_owned])) if num_owned else 0.0
mpc_vs_real = comm.allreduce(_diff, op=MPI.MAX)
# -

# + tags=["hide-input"]
if comm.rank == 0:
    print(f"  real space multiplier     {lam_real:.9f} (exact {lambda_exact:.9f})")
    print(f"  max|u_mpc - u_real|       {mpc_vs_real:.3e}")
    print(f"  real space solve, mpi  [s] {t_real_mpi:.3e}")
    print(f"  real space solve, nest [s] {t_real_nest:.3e}")
# -

assert abs(lam_real - lambda_exact) < 1e-8
assert mpc_vs_real < 1e-11

# No nnz comparison here: the constrained block system is assembled as a PETSc
# `nest`, which has no MatGetInfo, and the reference is monolithic. The fill and
# conditioning story is measured in {doc}`demo_boundary_average_constraint`.
#
# The printed solve times are not evidence for that story either. `MultiPointConstraint`
# forces a `nest` matrix for this blocked velocity/pressure system, and the
# `real space solve, nest [s]` line above solves the *same* reference problem as
# `real space solve, mpi [s]`, only in that format, to isolate its cost. At the
# small size of these demo problems, a `nest` matrix's one-time setup cost for
# direct factorization is what the gap between those two lines, and between the
# MPC solve and the `mpi` reference, is actually measuring.

# ## Visualization
#
# Each process builds a PyVista grid over the cells it *owns*, so a shared cell is
# not drawn twice, and the grids are gathered onto one process and drawn into a
# single figure with common colour limits.

# +

pyvista.global_theme.allow_empty_mesh = True


def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str, root: int = 0):
    """Owned-cell PyVista grids with ``u`` attached, gathered on ``root``.

    Vector fields are padded to three components, as PyVista expects. Returns the
    grids on ``root`` (``None`` elsewhere) and the global range of the magnitude,
    so every piece can be drawn with the same colour limits.
    """
    comm = V.mesh.comm
    bs = V.dofmap.index_map_bs
    tdim = V.mesh.topology.dim
    owned_cells = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned_cells))
    # vtk_mesh emits one point per dof block, in local numbering
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
    # gather returns the list on `root` and None everywhere else, so the caller
    # can test the result instead of comparing ranks itself
    return comm.gather(grid, root=root), [lo, hi]


# -

# `uh` lives in the constraint's extended space, which carries the master dofs as
# extra ghosts, so its array is longer than the original space in parallel. The
# extended index map keeps the original dofs first, so the leading entries are
# exactly the values of the original space. The constraint fixes a single number,
# the flow rate, and the solve produces the whole profile that carries it.

# +
V_plot = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(2,)))
u_plot = fem.Function(V_plot)
u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
pieces, clim = gather_grids(u_plot, V_plot, "u")

if pieces is not None:  # only the root process received the grids
    plotter = pyvista.Plotter(window_size=[700, 450])
    plotter.add_text(f"flow rate = {flux:.4f}", font_size=10)
    for piece in pieces:
        plotter.add_mesh(
            piece.glyph(orient="u", scale="|u|", factor=0.10),
            scalars="|u|",
            cmap="viridis",
            clim=clim,
            scalar_bar_args={"vertical": True},
        )
    plotter.view_xy()
    plotter.camera.tight(padding=0.6, view="xy", adjust_render_window=False)
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_flow_rate_constraint.png")
    else:
        plotter.show()
# -

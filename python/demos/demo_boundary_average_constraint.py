# # Prescribing a boundary average with an affine multi-point constraint
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# The companion demo {doc}`demo_mean_value_constraint` enforces
# $\int_\Omega u~\mathrm{d}x=\gamma$ with an affine multi-point constraint, and
# shows that a dense integral condition of that kind is more costly to enforce
# with a multi-point constraint than with a Lagrange multiplier. This demo
# applies the same construction to a functional supported on a **facet**,
#
# $$
# \int_\Gamma u ~\mathrm{d}s = \gamma,
# $$
#
# and highlights the opposite conclusion: with the master set restricted to the
# degrees of freedom on $\Gamma$, both the extra fill and the loss of
# conditioning scale with that (small) set rather than with all of $\Omega$, so
# for an integral condition with small support the multi-point constraint is the
# better tool.
#
# On $\Omega=(0,1)^2$ with $\Gamma=\{x=0\}$ we solve
#
# $$
# \begin{align*}
# -\Delta u &= f &&\text{in } \Omega,\\
# \frac{\partial u}{\partial n} &= h &&\text{on } \partial\Omega\setminus\Gamma,\\
# \int_\Gamma u ~\mathrm{d}s &= \gamma.
# \end{align*}
# $$
#
# Nothing is said about $u$ pointwise on $\Gamma$, so what happens there is worth
# spelling out. The reduced system $K^TAK\hat u = K^T(b-Ag)$ makes the residual
# orthogonal to $\mathrm{range}(K)$, which is the set of test functions with
# $\int_\Gamma v~\mathrm{d}s=0$. Hence $Au-b$ is parallel to the weight vector of
# the functional, which is the weak statement
#
# $$
# \frac{\partial u}{\partial n} = \mu \quad\text{on } \Gamma,
# $$
#
# for an *unknown constant* $\mu$. This is the classical "defective boundary
# condition" of {cite}`FormaggiaGerbeauNobileQuarteroni2002`: an averaged datum is
# prescribed, and the constant flux conjugate to it is produced by the solve. The
# demo recovers $\mu$ from the discrete solution and compares it with the
# multiplier of the equivalent real space formulation.
#
# We use the manufactured solution $u_{ex}=x^2/2 + x + C$. As in
# {doc}`demo_mean_value_constraint`, the source $f=-\Delta u_{ex}$, the boundary
# flux $h=\nabla u_{ex}\cdot n$, the target $\gamma=\int_\Gamma
# u_{ex}~\mathrm{d}s$ and the expected multiplier $\mu$ are all derived from it
# with UFL rather than worked out by hand. The one thing this solution has to
# satisfy for the problem to be well posed is that $\nabla u_{ex}\cdot n$ really
# is constant on $\Gamma$, which the demo checks.
#
# A second demo, {doc}`demo_flow_rate_constraint`, applies the same construction to
# a vector field on a blocked space.

# +
from __future__ import annotations

import time

from mpi4py import MPI

import basix.ufl
import numpy as np
import pandas
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint

# -

# ## Marking the boundary
#
# The whole exterior boundary is tagged, $\Gamma$ with `1` and the rest with `2`.
# This matters: with `subdomain_data` supplied, *untagged* exterior facets are not
# part of any `ds(...)` subdomain, so integrating the natural data over an
# unmarked remainder would silently drop it and solve a different problem.

# +

GAMMA, REST = 1, 2
C = 3.0


def boundary_average(domain, expr, ds_measure, tag):
    """Average of ``expr`` over the facets marked with ``tag``."""
    comm = domain.comm
    integral = comm.allreduce(fem.assemble_scalar(fem.form(expr * ds_measure(tag))), op=MPI.SUM)
    length = comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds_measure(tag))),
        op=MPI.SUM,
    )
    return integral, length


def mark_boundary(domain, indicator, tag, other_tag):
    """Tag every exterior facet, those satisfying ``indicator`` with ``tag``."""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    exterior = mesh.exterior_facet_indices(domain.topology)
    selected = mesh.locate_entities_boundary(domain, tdim - 1, indicator)
    values = np.full(len(exterior), other_tag, dtype=np.int32)
    values[np.isin(exterior, selected)] = tag
    return mesh.meshtags(domain, tdim - 1, exterior, values)


# -

# ## Building the constraint
#
# Identical to the cell integral case, only the functional changes: the weights
# now come from a facet integral, so every degree of freedom away from $\Gamma$
# has weight exactly zero and is discarded by the filter.


def build_constraint(V, weight_form, value, bcs=()):
    """Constrain ``L(u) = value`` and report how many masters survived."""
    mpc = MultiPointConstraint(V, bcs=list(bcs))
    mpc.add_integral_constraint(weight_form, value, bcs=list(bcs))
    mpc.finalize()
    kept = sum(len(mpc.masters.links(s)) for s in mpc.slaves[: mpc.num_local_slaves])
    return mpc, V.mesh.comm.allreduce(kept, op=MPI.SUM)


# ## Variational problem
#
# The forms are the standard Poisson ones, with natural data on the
# unconstrained part of the boundary only; nothing in them knows about the
# constraint.


def build_poisson_forms(V, u_ex, ds, natural_tag):
    """Bilinear and linear form of the Poisson problem, Neumann on ``natural_tag``."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(V.mesh)
    f = -ufl.div(ufl.grad(u_ex))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ds(natural_tag)
    return a, L


# ## Setting up the problem

# +
N = 24
degree = 2
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, N, N)
mt = mark_boundary(domain, lambda x: np.isclose(x[0], 0.0), GAMMA, REST)
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

V = fem.functionspace(domain, ("Lagrange", degree))
x = ufl.SpatialCoordinate(domain)
u_ex = x[0] ** 2 / 2 + x[0] + C
n = ufl.FacetNormal(domain)
a, L = build_poisson_forms(V, u_ex, ds, REST)
# -

# The target and the expected multiplier both come from the manufactured
# solution. `mu` is the average of the exact flux over `GAMMA`; the demo also
# checks that the flux really is constant there, which is what makes the
# defective condition well posed.

gamma_value, length = boundary_average(domain, u_ex, ds, GAMMA)
flux = ufl.dot(ufl.grad(u_ex), n)
mu_exact = boundary_average(domain, flux, ds, GAMMA)[0] / length
flux_variation = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((flux - mu_exact) ** 2 * ds(GAMMA))), op=MPI.SUM))

# Time the two paths against each other, as in {doc}`demo_mean_value_constraint`

comm.Barrier()
_t0 = time.perf_counter()
mpc, num_masters = build_constraint(V, ufl.conj(ufl.TestFunction(V)) * ds(GAMMA), gamma_value)
comm.Barrier()
_t1 = time.perf_counter()
t_constraint = _t1 - _t0

# ## Solving

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(a, L, mpc, bcs=[], petsc_options=petsc_options)
uh = problem.solve()
comm.Barrier()
t_mpc = time.perf_counter() - _t1


def operator_stats(A, comm, singular=False, root=0):
    """Global nnz and 2-norm condition number of an assembled operator.

    The condition number is computed from a dense SVD on ``root``, so this is a
    diagnostic for demo sized problems only. Used just below to make the
    verification tolerance conditioning-aware; see "Cost and conditioning"
    further down for why $K^TAK$ needs this at all.
    """
    # petsc4py defaults to MatInfoType.GLOBAL_SUM, so this is already reduced
    nnz = int(A.getInfo()["nz_used"])
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    cond = None
    if comm.rank == root:
        sv = np.linalg.svd(A_csr.toarray(), compute_uv=False)
        # A is singular for the bare Poisson problem (pure Neumann away from
        # Gamma), so compare against the smallest *nonzero* singular value.
        cond = sv[0] / (sv[-2] if singular else sv[-1])
    return nnz, comm.bcast(cond, root=root)


# ## Verification
#
# The constraint is checked against its target, the manufactured solution is
# recovered, and the recovered flux is compared with the exact one.

# +
integral = comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(GAMMA))), op=MPI.SUM)
error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM))
mu = comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(uh), n) * ds(GAMMA))), op=MPI.SUM) / length
_, cond_mpc = operator_stats(problem.A, comm)

if comm.rank == 0:
    print("----Verification----")
    print(f"  dofs                      {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"  masters                   {num_masters}")
    print(f"  int_Gamma u_h ds          {integral:.15f} (target {gamma_value:.15f})")
    print(f"  |int_Gamma u_h ds - g|    {abs(integral - gamma_value):.3e}")
    print(f"  L2(u_h - u_ex)            {error:.3e}")
    print(f"  recovered flux mu         {mu:.12f} (exact {mu_exact:.12f})")
    print(f"  cond(K^TAK)               {cond_mpc:.3e}")
tol = 1e3 * np.finfo(default_scalar_type()).eps
assert abs(integral - gamma_value) < tol
# error, mu and (below) mpc_vs_real all depend on how accurately the solve
# resolved the ill-conditioned K^TAK system, so their bound scales with the
# measured condition number rather than a flat constant that only happens to
# work at float64 -- see {doc}`demo_mean_value_constraint` for the same fix.
assert error < 10 * cond_mpc * tol
# The exact flux must be constant on Gamma, or the defective condition would not
# be the problem this demo claims to solve -- a property of u_ex alone, so this
# one stays flat.
assert flux_variation < tol
assert abs(mu - mu_exact) < 10 * cond_mpc * tol
# -

# ### Relation to a real space, on a submesh of $\Gamma$
#
# The multiplier $\lambda$ conjugate to a boundary functional lives on $\Gamma$,
# not on $\Omega$, so the reference puts the real space on a **submesh of the
# marked facets** rather than using a domain-global real element. The form is
# still integrated on the parent mesh, as mixed dimensional forms always use the
# higher dimensional domain as the integration domain and the
# {py:class}`EntityMap<dolfinx.mesh.EntityMap>` returned by
# {py:func}`dolfinx.mesh.create_submesh` relates the two.
#
# ```{warning}
# :class: dropdown
# The facets given to {py:func}`create_submesh<dolfinx.mesh.create_submesh>`
# must cover everything the measure integrates over.
# Facets outside the submesh map to `-1`, which is not checked,
# and for a real element silently resolves to its single degree of freedom, so the
# form is integrated over the wrong domain without any error. Here the submesh is
# exactly `mt.find(GAMMA)` and the measure is `ds(GAMMA)`.
# ```
#
# Since the real basis function is identically one, `inner(u, mu) * ds(GAMMA)`
# assembles the weight vector $w$ directly, and the constraint row of the right
# hand side must equal $\gamma$, so we integrate $\gamma/|\Gamma|$ against it.


def solve_poisson_real_space(domain, mt, degree, u_ex, value):
    """Reference solve with a Lagrange multiplier in a real space on Gamma."""
    submesh, entity_map = mesh.create_submesh(domain, domain.topology.dim - 1, mt.find(GAMMA))[:2]
    V = fem.functionspace(domain, ("Lagrange", degree))
    R = fem.functionspace(submesh, basix.ufl.real_element(submesh.basix_cell(), dtype=submesh.geometry.x.dtype))
    W = ufl.MixedFunctionSpace(V, R)
    u, lam = ufl.TrialFunctions(W)
    v, mu = ufl.TestFunctions(W)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    n = ufl.FacetNormal(domain)
    f = -ufl.div(ufl.grad(u_ex))
    length = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds(GAMMA))), op=MPI.SUM
    )

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(lam, v) * ds(GAMMA) + ufl.inner(u, mu) * ds(GAMMA)
    L = (
        ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ds(REST)
        + ufl.inner(fem.Constant(domain, default_scalar_type(value / length)), mu) * ds(GAMMA)
    )
    problem = fem.petsc.LinearProblem(
        ufl.extract_blocks(a),
        ufl.extract_blocks(L),
        bcs=[],
        kind="mpi",
        entity_maps=[entity_map],
        petsc_options_prefix="demo_facet_real_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    uh, lamh = problem.solve()
    # The single real dof is ghosted on every rank, so it must not be summed
    lam_value = lamh.x.array[0] if lamh.x.array.size else 0.0
    return uh, float(np.real(lam_value)), problem


_t2 = time.perf_counter()
u_real, lam_real, real_problem = solve_poisson_real_space(domain, mt, degree, u_ex, gamma_value)
comm.Barrier()
t_real = time.perf_counter() - _t2

num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
assert isinstance(uh, fem.Function)
_diff = np.max(np.abs(uh.x.array[:num_owned] - u_real.x.array[:num_owned])) if num_owned else 0.0
mpc_vs_real = comm.allreduce(_diff, op=MPI.MAX)
if comm.rank == 0:
    print(f"  real space multiplier     {lam_real:.12f} (equals -mu)")
    print(f"  max|u_mpc - u_real|       {mpc_vs_real:.3e}")

# The multiplier enters the reference form as +lam, so it is -mu
assert abs(lam_real + mu) < 10 * cond_mpc * tol
# The real-space (saddle point) solve keeps A's own, much better conditioning,
# so this difference is dominated by uh's error and needs the same
# conditioning-aware bound as the checks above.
assert mpc_vs_real < 10 * cond_mpc * tol

# ### Cost and conditioning
#
# {doc}`demo_mean_value_constraint` derives
# $K^TAK = A_{mm} + A_{ms}c^T + cA_{sm} + \left(cc^T\right)A_{ss}$
# for the same elimination and shows that the rank-one
# term $\left(cc^T\right)A_{ss}$ is dense over every surviving master. The
# difference here is only the size of that master set: it is just the boundary,
# so it grows like $\sqrt{N}$ instead of $N$, and the penalty stays mild. The
# reference pays less here too: its real space lives on a submesh of $\Gamma$, so
# the coupling blocks only reach the cells meeting $\Gamma$ rather than adding a
# row and column over the whole mesh. `operator_stats`, defined and used above
# to make the verification tolerance conditioning-aware, measures both effects;
# the sweep below repeats it over a refinement range.

# The same table as in {doc}`demo_mean_value_constraint`, and the comparison is the
# point of this demo: the master set is the boundary rather than the whole mesh,
# so it grows like the square root of the number of degrees of freedom, and both
# the extra fill and the loss of conditioning follow it.


# + tags=["hide-input"]
def measure(N: int) -> dict:
    """Solve at resolution ``N`` and collect cost, conditioning and timings."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N)
    mt = mark_boundary(domain, lambda x: np.isclose(x[0], 0.0), GAMMA, REST)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    u_ex = x[0] ** 2 / 2 + x[0] + C
    a, L = build_poisson_forms(V, u_ex, ds, REST)
    gamma_value, _ = boundary_average(domain, u_ex, ds, GAMMA)

    comm.Barrier()
    t0 = time.perf_counter()
    mpc, num_masters = build_constraint(V, ufl.TestFunction(V) * ds(GAMMA), gamma_value)
    comm.Barrier()
    t1 = time.perf_counter()
    problem = LinearProblem(a, L, mpc, bcs=[], petsc_options=petsc_options)
    problem.solve()
    comm.Barrier()
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    _, _, real_problem = solve_poisson_real_space(domain, mt, degree, u_ex, gamma_value)
    comm.Barrier()
    t_real = time.perf_counter() - t3

    A_plain = fem.petsc.assemble_matrix(fem.form(a), bcs=[])
    A_plain.assemble()
    nnz_A, cond_A = operator_stats(A_plain, comm, singular=True)
    nnz_mpc, cond_mpc = operator_stats(problem.A, comm)
    nnz_real, cond_real = operator_stats(real_problem.A, comm)
    A_plain.destroy()
    return {
        "N": V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        "M": num_masters,
        "norm_c": float(np.linalg.norm(mpc.coefficients()[0])),
        "nnz_A": nnz_A,
        "nnz_mpc": nnz_mpc,
        "nnz_real": nnz_real,
        "cond_A": cond_A,
        "cond_mpc": cond_mpc,
        "cond_real": cond_real,
        "t_constraint": t1 - t0,
        "t_mpc": t2 - t1,
        "t_real": t_real,
    }


rows = [measure(N) for N in (8, 12, 16)]

COLUMNS = {
    "N": "dofs",
    "M": "masters",
    "norm_c": "||c||",
    "nnz_A": "nnz(A)",
    "nnz_mpc": "nnz(KtAK)",
    "nnz_real": "nnz(saddle)",
    "cond_A": "cond(A)",
    "cond_mpc": "cond(KtAK)",
    "cond_real": "cond(saddle)",
    "t_constraint": "build [s]",
    "t_mpc": "mpc [s]",
    "t_real": "real [s]",
    "mpc/real": "mpc/real",
}
FORMATS = {
    "||c||": "{:.2f}",
    "cond(A)": "{:.3e}",
    "cond(KtAK)": "{:.3e}",
    "cond(saddle)": "{:.3e}",
    "build [s]": "{:.4f}",
    "mpc [s]": "{:.4f}",
    "real [s]": "{:.4f}",
    "mpc/real": "{:.2f}",
}

table = pandas.DataFrame(rows)
table["mpc/real"] = table["t_mpc"] / table["t_real"]
table = table[list(COLUMNS)].rename(columns=COLUMNS).set_index("dofs")
table.style.format(FORMATS)

# -

# Compare with {doc}`demo_mean_value_constraint`, where the functional is supported
# on the whole mesh: there $M$ is the number of degrees of freedom, `nnz(KtAK)`
# grows like $M^2$, the condition number is inflated several hundred fold, and the
# solve ends up an order of magnitude *slower* than the real space one. Here $M$
# is only the number of dofs on $\Gamma$, so it grows like $\sqrt{N}$, the reduced
# operator stays sparse and the conditioning penalty is mild. The last column
# reverses accordingly: eliminating the constraint is about twice as fast as
# solving the saddle point system, because it produces a smaller, positive
# definite operator instead of an indefinite one.
#
# That contrast is the practical summary. An affine multi-point constraint can
# stand in for a real space in either case, but it is the right tool when the
# functional has *small support*, and a real space is the right tool when it does
# not.

# ## Visualization
#
# Each process builds a PyVista grid over the cells it *owns*, so a shared cell is
# not drawn twice, and the grids are gathered onto one process and drawn into a
# single figure with common colour limits.

# + tags=["hide-input"]

pyvista.global_theme.allow_empty_mesh = True


def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str, root: int = 0):
    """Owned-cell PyVista grids with ``u`` attached, gathered on ``root``.

    Returns the grids on ``root`` (``None`` elsewhere) and the global range of
    the values, so that every piece can be drawn with the same colour limits.
    """
    comm = V.mesh.comm
    bs = V.dofmap.index_map_bs
    tdim = V.mesh.topology.dim
    owned_cells = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned_cells))
    # vtk_mesh emits one point per dof block, in local numbering
    values = u.x.array.real[: grid.n_points * bs]
    grid.point_data[name] = values if bs == 1 else values.reshape(-1, bs)
    local = np.linalg.norm(values.reshape(-1, bs), axis=1) if bs > 1 else values
    lo = comm.allreduce(float(local.min()) if local.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(local.max()) if local.size else -np.inf, op=MPI.MAX)
    # gather returns the list on `root` and None everywhere else, so the caller
    # can test the result instead of comparing ranks itself
    return comm.gather(grid, root=root), [lo, hi]


# -

# `uh` lives in the constraint's extended space, which carries the master dofs as
# extra ghosts, so its array is longer than the original space in parallel. The
# extended index map keeps the original dofs first, so the leading entries are
# exactly the values of the original space.

# + tags =["hide-input"]

u_plot = fem.Function(V)
u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
pieces, clim = gather_grids(u_plot, V, "u")

if pieces is not None:  # only the root process received the grids
    plotter = pyvista.Plotter(window_size=[700, 500])
    plotter.add_text(f"boundary average = {integral:.4f}, recovered flux = {mu:.4f}", font_size=10)
    for piece in pieces:
        plotter.add_mesh(
            piece.warp_by_scalar("u", factor=0.15),
            scalars="u",
            cmap="viridis",
            clim=clim,
            show_edges=False,
            scalar_bar_args={"vertical": True},
        )
    plotter.view_isometric()  # type: ignore[call-arg]
    plotter.camera.zoom(1.4)
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_boundary_average_constraint.png")
    else:
        plotter.show()
# -
# ```{bibliography}
#    :filter: cited and ({"python/demos/demo_boundary_average_constraint"} >= docnames)
# ```

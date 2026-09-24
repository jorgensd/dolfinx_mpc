# # Fixing a mean value with an affine multi-point constraint
# **Author** Jørgen S. Dokken
#
# This demo shows how an *affine* multi-point constraint can enforce a
# constraint on an integral of the solution,
#
# $$
# \int_\Omega u ~\mathrm{d}x = \gamma,
# $$
#
# and how that relates to the more familiar way of imposing such a condition,
# a Lagrange multiplier in a *real space*
# ({py:func}`basix.ufl.real_element`), as seen in
# [SciFEM real space demo](https://scientificcomputing.github.io/scifem/examples/real_function_space.html).
# The two are algebraically equivalent; they differ sharply in cost, and this demo measures the difference.
#
# ## Mathematical formulation
#
# We solve the pure Neumann Poisson problem
#
# $$
# \begin{align*}
# -\Delta u &= f &&\text{in } \Omega,\\
# \frac{\partial u}{\partial n} &= h &&\text{on } \partial\Omega,
# \end{align*}
# $$
#
# on $\Omega=(0,1)^2$. The problem is singular: constants lie in the kernel, so
# $u$ is determined only up to an additive constant, and the data must satisfy
# the compatibility condition $\int_\Omega f~\mathrm{d}x + \int_{\partial\Omega}
# h~\mathrm{d}s = 0$. Prescribing the mean value $\int_\Omega u~\mathrm{d}x =
# \gamma$ selects the unique solution.
#

# We start by import the required modules:

# + tags=["hide-input"]
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

# ### Problem data
#
# We use the manufactured solution $u_{ex} = x^2 - x + C$. Everything else is
# derived from it with UFL rather than worked out by hand: the source is
# $f=-\Delta u_{ex}$, the boundary flux is $h=\nabla u_{ex}\cdot n$, and the
# target $\gamma$ is the integral of $u_{ex}$ itself. Deriving $f$ and $h$ from
# the same expression makes the compatibility condition hold by construction,
# since $\int_\Omega -\Delta u_{ex}~\mathrm{d}x + \int_{\partial\Omega} \nabla
# u_{ex}\cdot n~\mathrm{d}s = 0$ is the divergence theorem. Since $u_{ex}$ is a
# second order polynomial it lies in the second order Lagrange space, so the
# discrete solution should reproduce it up to solver accuracy.

C = 3.0
degree = 2

# ## Transforming an integral condition into a multi-point constraint
#
# Expanding $u_h=\sum_i u_i\phi_i$ turns the integral condition into a single
# linear equation between *all* the coefficients,
#
# $$
# \int_\Omega u_h~\mathrm{d}x = \sum_i u_i \int_\Omega \phi_i ~\mathrm{d}x
# = \sum_i w_i u_i = \gamma,
# \qquad w_i = \int_\Omega \phi_i~\mathrm{d}x.
# $$
#
# The weights $w$ are just the assembled vector of the linear form
# $v\mapsto\int_\Omega v~\mathrm{d}x$. Picking one degree of freedom $s$ with
# $w_s\neq 0$ and solving for it gives exactly the affine multi-point
# constraint $u = K\hat{u} + g$ of {py:class}`dolfinx_mpc.MultiPointConstraint`,
#
# $$
# u_s = \sum_{i\neq s} \underbrace{\left(-\frac{w_i}{w_s}\right)}_{c_{si}} u_i
#       + \underbrace{\frac{\gamma}{w_s}}_{g_s}.
# $$
#
# So $s$ is the single *slave*, every other degree of freedom in the support of
# the functional is a *master*, the coefficients are $-w_i/w_s$.
#
# There are two new concepts in this demo compared to the others:
# 1. The inhomogeneity $g_s=\frac{\gamma}{w_s}$ is computed and assigned to the constraint.
# 2. The constraint involves an integral, so the weights $w_i$ are computed by assembling a
#    linear form rather than reading them off a mesh entity.
# We create a convenience function `build_constraint` that takes in:
# 1. the function space `V`
# that our {py:class}`dolfinx_mpc.MultiPointConstraint` will be built on
# 2. the {py:class}`linear form <ufl.Form>` `weight_form` that defines the weights
# 3. The target value `value`, which is $\gamma$
# 4. Any {py:class}`boundary conditions<dolfinx.fem.DirichletBC>` `bcs` that should be applied to the constraint.
# The function returns the finalized MPC as well as the number of masters in the MPC constraint.


def build_constraint(V, weight_form, value, bcs=()):
    """Constrain ``L(u) = value`` and report how many masters survived."""
    mpc = MultiPointConstraint(V, bcs=list(bcs))
    mpc.add_integral_constraint(weight_form, value, bcs=list(bcs))
    mpc.finalize()
    kept = sum(len(mpc.masters.links(s)) for s in mpc.slaves[: mpc.num_local_slaves])
    return mpc, V.mesh.comm.allreduce(kept, op=MPI.SUM)


# ## Variational problem
#
# The forms are the standard pure Neumann Poisson ones; nothing in them knows
# about the constraint.

# +


def build_poisson_forms(V: fem.FunctionSpace, u_ex: ufl.core.expr.Expr) -> tuple[ufl.Form, ufl.Form]:
    """Bilinear and linear form of the pure Neumann Poisson problem.

    The source and the boundary flux are both differentiated out of ``u_ex``, so
    the data is compatible by construction.
    """
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(V.mesh)
    f = -ufl.div(ufl.grad(u_ex))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ufl.ds
    return a, L


def exact_mean(domain: mesh.Mesh, u_ex):
    """The target value, integrated from the manufactured solution."""
    compiled_mean = fem.form(u_ex * ufl.dx(domain=domain))
    return compiled_mean.mesh.comm.allreduce(fem.assemble_scalar(compiled_mean), op=MPI.SUM)


# -


# ## Setting up the problem
#
# The reduced operator of a cell integral constraint is dense, so the mesh is
# deliberately coarse; the cost table at the end of the demo shows why.

N = 16
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
comm = domain.comm
V = fem.functionspace(domain, ("Lagrange", degree))
x = ufl.SpatialCoordinate(domain)
u_ex = x[0] ** 2 - x[0] + C
a, L = build_poisson_forms(V, u_ex)
gamma = exact_mean(domain, u_ex)

# We use the form `ufl.TestFunction(V) * ufl.dx` to compute the weights $w_i$
# of the integral constraint.


comm.Barrier()
_t0 = time.perf_counter()
mpc, num_masters = build_constraint(V, ufl.conj(ufl.TestFunction(V)) * ufl.dx, gamma)
comm.Barrier()
_t1 = time.perf_counter()
t_constraint = _t1 - _t0

# ## Solving
#
# {py:class}`dolfinx_mpc.LinearProblem` handles the affine constraint for us: it
# refreshes the offset with
# {py:meth}`update_constants<dolfinx_mpc.MultiPointConstraint.update_constants>`
# and applies {py:func}`dolfinx_mpc.apply_mpc_lifting`, which contributes the
# $-K^TAg$ term. Note that no null space has to be attached even though $A$ is
# singular: the constraint removes the constant kernel, because
# $w\cdot\mathbf{1} = |\Omega| \neq 0$.

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
    # over the communicator and must not be summed again.
    nnz = int(A.getInfo()["nz_used"])
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    cond = None
    if comm.rank == root:
        sv = np.linalg.svd(A_csr.toarray(), compute_uv=False)
        # A is singular for the pure Neumann problem, so compare against the
        # smallest *nonzero* singular value.
        cond = sv[0] / (sv[-2] if singular else sv[-1])
    return nnz, comm.bcast(cond, root=root)


# ## Verification
#
# Three things are checked: that the constraint is satisfied and that the
# manufactured solution is recovered.

mean_value = comm.allreduce(fem.assemble_scalar(fem.form(uh * ufl.dx)), op=MPI.SUM)
error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM))
_, cond_mpc = operator_stats(problem.A, comm)

# + tags=["hide-input"]
if comm.rank == 0:
    print("----Verification----")
    print(f"  dofs                  {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"  masters               {num_masters}")
    print(f"  mean(u_h)             {mean_value:.15f} (target {gamma:.15f})")
    print(f"  |mean(u_h) - gamma|   {abs(mean_value - gamma):.3e}")
    print(f"  L2(u_h - u_ex)        {error:.3e}")
    print(f"  cond(K^TAK)           {cond_mpc:.3e}")

# - tags=["hide-input"]

tol = 100 * np.finfo(default_scalar_type()).eps
assert abs(mean_value - gamma) < tol

# u_ex lies in the discrete space, so the discretization is exact. The
# remaining error is the conditioning of the dense reduced operator: a direct
# solve loses roughly $\log_{10}\kappa$ digits, so the bound below scales with
# the *measured* $\kappa(K^TAK)$ rather than a fixed constant -- a flat bound
# tight enough for `float64` would be far too strict for a lower-precision
# build, where `tol` itself is orders of magnitude larger to begin with.

assert error < 10 * cond_mpc * tol


# ### Relation to a real space
#
# The constrained problem has the saddle point form
#
# $$
# \begin{pmatrix} A & w \\ w^T & 0\end{pmatrix}
# \begin{pmatrix} u \\ \lambda \end{pmatrix}
# = \begin{pmatrix} b \\ \gamma \end{pmatrix},
# $$
#
# where $\lambda$ is the scalar Lagrange multiplier carried by a real space.
# Since $\mathrm{range}(K)=\{v: w^Tv=0\}$, the first block row says $b-Au \in
# \mathrm{span}(w)$, i.e. $K^T(b-Au)=0$; together with $w^Tu=\gamma$ that *is*
# the reduced system $K^TAK\hat{u}=K^T(b-Ag)$.
#
# The same problem, solved as a saddle point system with a scalar multiplier in
# a real space. Because the real basis function is identically one,
# `inner(lam, v) * dx` assembles the dense row $w$ directly. The right hand side
# entry of the constraint row must be $\gamma$ itself, so the constant we
# integrate against the real test function is $\gamma/|\Omega|$.


def solve_real_space(domain, degree, u_ex, value):
    """Reference solve using a Lagrange multiplier in a real space."""
    V = fem.functionspace(domain, ("Lagrange", degree))
    R = fem.functionspace(domain, basix.ufl.real_element(domain.basix_cell(), dtype=domain.geometry.x.dtype))
    W = ufl.MixedFunctionSpace(V, R)
    u, lam = ufl.TrialFunctions(W)
    v, mu = ufl.TestFunctions(W)
    n = ufl.FacetNormal(domain)
    f = -ufl.div(ufl.grad(u_ex))

    volume = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ufl.dx)), op=MPI.SUM
    )
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(lam, v) * ufl.dx + ufl.inner(u, mu) * ufl.dx
    L = (
        ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ufl.ds
        + ufl.inner(fem.Constant(domain, default_scalar_type(value / volume)), mu) * ufl.dx
    )
    problem = fem.petsc.LinearProblem(
        ufl.extract_blocks(a),
        ufl.extract_blocks(L),
        bcs=[],
        kind="mpi",
        petsc_options_prefix="demo_mean_value_real_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    uh, lamh = problem.solve()
    # The problem is returned alongside the solution: it owns the PETSc matrix,
    # which is destroyed with it.
    return uh, lamh, problem


# We compare the solution of the Lagrange multiplier problem to the one with the multi-point constraint.

_t2 = time.perf_counter()
u_real, lam_real, real_problem = solve_real_space(domain, degree, u_ex, gamma)
comm.Barrier()
t_real = time.perf_counter() - _t2

num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
assert isinstance(uh, fem.Function)
_diff = np.max(np.abs(uh.x.array[:num_owned] - u_real.x.array[:num_owned])) if num_owned else 0.0
mpc_vs_real = comm.allreduce(_diff, op=MPI.MAX)
# The real-space (saddle point) solve keeps A's own, much better conditioning,
# so this difference is dominated by uh's error and needs the same
# conditioning-aware bound as the check against u_ex above.
assert mpc_vs_real < 10 * cond_mpc * tol
print(f"  max|u_mpc - u_real|   {mpc_vs_real:.3e}")

# ### Cost and conditioning
#
# Eliminating the slave is not free. Order the degrees of freedom with the masters
# $m$ first and the slave $s$ last, so that the constraint above is
# $u=K\hat{u}+g$ and both $A$ and $K$ split as
#
# $$
# A = \begin{pmatrix} A_{mm} & A_{ms} \\ A_{sm} & A_{ss}\end{pmatrix},
# \qquad
# K = \begin{pmatrix} I \\ c^T \end{pmatrix},
# \qquad c_i = -\frac{w_i}{w_s},
# $$
#
# where $c$ collects the coefficients of the one slave and $A_{ss}$ is a scalar.
# Multiplying out,
#
# $$
# K^TAK = A_{mm} + A_{ms}c^T + cA_{sm} + \left(cc^T\right)A_{ss}.
# $$
#
# The final term is a rank one matrix that is **dense over every pair of
# masters that survive filtering**, since $c$ has no near-zero entries left once
# they are dropped. For a cell integral nearly every degree of freedom is a
# master -- only the P2 vertex dofs are filtered out, since a P2 vertex basis
# function integrates to exactly zero on simplices -- so $K^TAK$ is essentially
# full, and its norm is inflated by $\lVert c\rVert^2$, which costs roughly a
# factor $M$ in the condition number. The saddle point system pays neither
# price: it keeps $A$ intact and
# appends one row and column. That row is itself dense -- every $w_i$ is stored,
# so the system grows by exactly $2N$ entries -- but $2N$ is *linear* in the
# problem size, against the $M^2$ of the elimination. `operator_stats`,
# defined and used above to make the verification tolerance conditioning-aware,
# measures both effects; the sweep below repeats it over a refinement range.

# ## Cost and conditioning, measured
#
# The helper below repeats the solve over a small refinement sweep so that the
# growth is visible rather than asserted.


def measure(N: int) -> dict:
    """Solve at resolution ``N`` and collect cost, conditioning and timings."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    comm = domain.comm
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    u_ex = x[0] ** 2 - x[0] + C
    a, L = build_poisson_forms(V, u_ex)
    gamma = exact_mean(domain, u_ex)

    comm.Barrier()
    t0 = time.perf_counter()
    mpc, num_masters = build_constraint(V, ufl.TestFunction(V) * ufl.dx, gamma)
    comm.Barrier()
    t1 = time.perf_counter()
    problem = LinearProblem(a, L, mpc, bcs=[], petsc_options=petsc_options)
    problem.solve()
    comm.Barrier()
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    _, _, real_problem = solve_real_space(domain, degree, u_ex, gamma)
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


# +

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

# The table shows `nnz(KtAK)` growing like $M^2$ and `cond(KtAK)` like
# $\lVert c\rVert^2\sim M$, because the rank one term $cc^TA_{ss}$ couples every
# master to every other. Here the functional is supported on the whole mesh, so
# $M$ is the number of degrees of freedom (bar the vertex dofs, whose weights
# vanish for P2 on simplices) and the reduced operator is essentially full. A
# real space instead appends a single row and column -- dense, but only $2N$
# entries, so `nnz(saddle)` grows linearly where `nnz(KtAK)` grows quadratically
# -- and leaves the conditioning of $A$ untouched.
#
# The timings say the same in wall clock terms. Building the constraint is cheap
# and mesh independent; what costs is solving with the operator it produces, and
# the ratio to the real space solve grows with the mesh. For a functional
# supported on a facet $M$ is far smaller and both penalties shrink with it, see
# {doc}`demo_boundary_average_constraint`.

# ## Visualization
#
# Each process builds a PyVista grid over the cells it *owns*, so a shared cell is
# not drawn twice, and the grids are gathered onto one process and drawn into a
# single figure with common colour limits.

# +
pyvista.global_theme.allow_empty_mesh = True


def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str, root: int = 0):
    """Owned-cell PyVista grids with ``u`` attached, gathered on ``root``.

    Returns the list of grids on rank 0 (``None`` elsewhere) and the global
    value range, so that every piece can be drawn with the same colour limits.
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


# `uh` lives in the constraint's extended space, which carries the master dofs as
# extra ghosts, so its array is longer than the original space in parallel. The
# extended index map keeps the original dofs first, so the leading entries are
# exactly the values of the original space.

u_plot = fem.Function(V)
u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
pieces, clim = gather_grids(u_plot, V, "u")

# The solution varies only with $x$, and the constraint has picked the vertical
# offset $C-1/6$ out of the one parameter family the pure Neumann problem admits.
# Warping by the value makes that offset visible as the height above zero.

if pieces is not None:  # only the root process received the grids
    plotter = pyvista.Plotter(window_size=[700, 500])
    plotter.add_text(f"mean(u) = {mean_value:.6f}", font_size=10)
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
        plotter.screenshot("demo_mean_value_constraint.png")
    else:
        plotter.show()
# -

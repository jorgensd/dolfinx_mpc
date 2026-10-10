# # Bulk–surface coupling through a submesh
# **Author** Jørgen S. Dokken
#
# Some boundary conditions carry a differential equation of their own, posed on the
# boundary: a Wentzell condition {cite}`bulksurface-Wentzell1959`, a membrane bonded to
# a body, a thin conducting layer. A natural discretisation puts that equation on a
# submesh of the boundary facets, with its own function space, and ties the surface
# field to the trace of the bulk field. This demo makes the tie a {py:class}`dolfinx_mpc.MultiPointConstraint`
# between the two spaces, created with
# {py:meth}`dolfinx_mpc.MultiPointConstraint.create_submesh_constraint`. The relation
# between a facet of the submesh and the cell of the parent mesh holding it comes from
# the entity map of {py:func}`dolfinx.mesh.create_submesh`, so no geometric search is
# needed.
#
# As in {doc}`demo_mean_value_constraint`, the constraint is compared with a Lagrange
# multiplier imposing the same tie weakly. The two give the same solution; they differ
# in the system that has to be solved.
#
# ## Mathematical formulation
#
# Let $\Omega=(0,1)^2$ and $\Gamma=(0,1)\times\{1\}$ its top edge. We seek $u$ with
#
# $$
# \begin{align*}
# -\Delta u + u &= f && \text{in } \Omega,\\
# \frac{\partial u}{\partial n} &= 0 && \text{on } \partial\Omega\setminus\Gamma,\\
# \frac{\partial u}{\partial n} - \Delta_\Gamma u + u &= g && \text{on } \Gamma,\\
# \frac{\partial u}{\partial x} &= 0 && \text{at the ends of } \Gamma,
# \end{align*}
# $$
#
# where $\Delta_\Gamma = \partial^2/\partial x^2$ is the Laplacian along $\Gamma$. The
# weak form seeks $u\in H^1(\Omega)$ with trace $u|_\Gamma\in H^1(\Gamma)$ such that
#
# $$
# \int_\Omega \nabla u\cdot\nabla v + uv ~\mathrm{d}x
# + \int_\Gamma \nabla_\Gamma u\cdot\nabla_\Gamma v + uv ~\mathrm{d}s
# = \int_\Omega f v ~\mathrm{d}x + \int_\Gamma g v ~\mathrm{d}s
# $$
#
# for all such $v$. Writing $u_\Gamma$ for the surface field, the problem becomes one
# for the pair $(u, u_\Gamma)\in V\times V_\Gamma$, with $V$ on $\Omega$ and
# $V_\Gamma$ on $\Gamma$, under the constraint $u_\Gamma = u|_\Gamma$:
#
# $$
# \begin{align*}
# \int_\Omega \nabla u\cdot\nabla v + uv ~\mathrm{d}x
#   &+ \int_\Gamma \nabla u_\Gamma\cdot\nabla v_\Gamma + u_\Gamma v_\Gamma ~\mathrm{d}x
#   = \int_\Omega f v ~\mathrm{d}x + \int_\Gamma g v_\Gamma ~\mathrm{d}x.
# \end{align*}
# $$
#
# The form is block diagonal, each block assembled on its own mesh. The constraint
# makes every degree of freedom of $u_\Gamma$ a slave of the trace of $u$, so the
# reduced operator $K^TAK$ adds the surface operator to the bulk degrees of freedom on
# $\Gamma$.
#
# We use the manufactured solution $u=\cos(\pi x)(1+y^2)$, whose normal derivative
# vanishes on the three other edges and whose tangential derivative vanishes at the
# ends of $\Gamma$. Then $f=-\Delta u+u$ and, on $y=1$,
# $g=\partial_y u-\partial_{xx}u+u=\cos(\pi x)\left(2y+(\pi^2+1)(1+y^2)\right)$.

# + tags=["hide-input"]
from __future__ import annotations

import time
from pathlib import Path

from mpi4py import MPI

import dolfinx.fem.petsc
import numpy as np
import pandas
import pyvista
import ufl
from dolfinx import default_real_type, fem, mesh, plot

import dolfinx_mpc
import dolfinx_mpc.utils

# -


# +
def u_exact(x):
    return ufl.cos(ufl.pi * x[0]) * (1 + x[1] ** 2)


def g_exact(x):
    return ufl.cos(ufl.pi * x[0]) * (2 * x[1] + (ufl.pi**2 + 1) * (1 + x[1] ** 2))


direct = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
# -

# ## The two meshes
#
# The surface mesh is a submesh of the facets on $\Gamma$, created with
# {py:func}`dolfinx.mesh.create_submesh`. Besides the submesh, it returns an
# {py:class}`EntityMap<dolfinx.mesh.EntityMap>` relating each cell of the submesh to its
# facet of the parent mesh, a second one relating their vertices, and the parent's
# geometry node of each node of the submesh. Only the first map is needed here.


# +
def create_meshes(N: int):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, dtype=default_real_type)
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    surface, entity_map, _, _ = mesh.create_submesh(msh, fdim, facets)
    tags = mesh.meshtags(msh, fdim, facets, np.full(len(facets), 1, dtype=np.int32))
    return msh, surface, entity_map, tags


# -

# ## The coupled problem
#
# The constraint is created on the constraint of the surface space, with the bulk space
# as the master space, and the two constraints are finalized together, as their blocks
# depend on each other.


# +
def create_forms(V, V_G):
    """The bulk and surface problems, without the tie between them."""
    msh, surface = V.mesh, V_G.mesh
    W = ufl.MixedFunctionSpace(V, V_G)
    u, u_G = ufl.TrialFunctions(W)
    v, v_G = ufl.TestFunctions(W)
    x, x_G = ufl.SpatialCoordinate(msh), ufl.SpatialCoordinate(surface)
    dx, dx_G = ufl.Measure("dx", domain=msh), ufl.Measure("dx", domain=surface)
    f = -ufl.div(ufl.grad(u_exact(x))) + u_exact(x)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * dx
    a += (ufl.inner(ufl.grad(u_G), ufl.grad(v_G)) + ufl.inner(u_G, v_G)) * dx_G
    L = ufl.inner(f, v) * dx + ufl.inner(g_exact(x_G), v_G) * dx_G
    return ufl.extract_blocks(a), ufl.extract_blocks(L)


def tie(V, V_G, entity_map):
    """Tie the surface field to the trace of the bulk field."""
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc_G = dolfinx_mpc.MultiPointConstraint(V_G)
    mpc_G.create_submesh_constraint(V_G, V, entity_map)
    dolfinx_mpc.finalize_multipointconstraints([mpc, mpc_G])
    return [mpc, mpc_G]


def solve_coupled(V, V_G, constraints, kind):
    a, L = create_forms(V, V_G)
    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        constraints,
        bcs=[],
        kind=kind,
        petsc_options_prefix=f"demo_bulk_surface_{kind}_",
        petsc_options=direct,
    )
    uh, uh_G = problem.solve()
    # The problem owns the matrix, which is destroyed with it
    return uh, uh_G, problem


def create_spaces(msh, surface, degree: int):
    return fem.functionspace(msh, ("Lagrange", degree)), fem.functionspace(surface, ("Lagrange", degree))


# -

# ## The same tie with a Lagrange multiplier
#
# Instead of eliminating $u_\Gamma$, the tie can be imposed weakly with a multiplier
# $\lambda\in\Lambda$ on $\Gamma$, which adds a third field and a saddle point:
#
# $$
# \begin{align*}
# \int_\Omega \nabla u\cdot\nabla v + uv ~\mathrm{d}x - \int_\Gamma \lambda v ~\mathrm{d}s
#   &= \int_\Omega f v ~\mathrm{d}x,\\
# \int_\Gamma \nabla u_\Gamma\cdot\nabla v_\Gamma + u_\Gamma v_\Gamma ~\mathrm{d}x
#   + \int_\Gamma \lambda v_\Gamma ~\mathrm{d}x &= \int_\Gamma g v_\Gamma ~\mathrm{d}x,\\
# \int_\Gamma (u_\Gamma - u)\mu ~\mathrm{d}s &= 0.
# \end{align*}
# $$
#
# The coupling terms are integrated over the facets of $\Gamma$ on the bulk mesh and
# carry arguments from the surface mesh, so they need the entity map. With $\Lambda$
# the same space as $V_\Gamma$, the last equation holds for $\mu = u_\Gamma - u$, so
# $u_\Gamma = u|_\Gamma$ exactly, and the solution is that of the constraint.


# +
def solve_multiplier(V, V_G, entity_map, tags):
    msh, surface = V.mesh, V_G.mesh
    Q = V_G.clone()
    W = ufl.MixedFunctionSpace(V, V_G, Q)
    u, u_G, lmbda = ufl.TrialFunctions(W)
    v, v_G, mu = ufl.TestFunctions(W)
    x, x_G = ufl.SpatialCoordinate(msh), ufl.SpatialCoordinate(surface)
    dx, dx_G = ufl.Measure("dx", domain=msh), ufl.Measure("dx", domain=surface)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags, subdomain_id=1)

    f = -ufl.div(ufl.grad(u_exact(x))) + u_exact(x)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * dx
    a += (ufl.inner(ufl.grad(u_G), ufl.grad(v_G)) + ufl.inner(u_G, v_G)) * dx_G
    a += ufl.inner(lmbda, v_G - v) * ds + ufl.inner(u_G - u, mu) * ds
    # The multiplier has no load, which `ufl.extract_blocks` would drop, so the blocks of the
    # right-hand side are given by hand
    L = [ufl.inner(f, v) * dx, ufl.inner(g_exact(x_G), v_G) * dx_G, ufl.ZeroBaseForm((mu,))]
    problem = dolfinx.fem.petsc.LinearProblem(
        ufl.extract_blocks(a),
        L,
        kind="mpi",
        entity_maps=[entity_map],
        petsc_options_prefix="demo_bulk_surface_multiplier_",
        petsc_options=direct,
    )
    uh, uh_G, _ = problem.solve()
    return uh, uh_G, problem


# -

# ## Verification
#
# Both layouts of the constrained system, nest and monolithic, against the multiplier
# formulation, for the bulk and the surface field.

# +
msh, surface, entity_map, tags = create_meshes(16)
V, V_G = create_spaces(msh, surface, 1)
u_ref, u_ref_G, _ = solve_multiplier(V, V_G, entity_map, tags)
tol = 500 * np.finfo(default_real_type).eps


def max_difference(uh, u_ref):
    n = u_ref.function_space.dofmap.index_map.size_local
    difference = np.max(np.abs(uh.x.array[:n] - u_ref.x.array[:n]), initial=0)
    return msh.comm.allreduce(difference, op=MPI.MAX)


u_max = msh.comm.allreduce(np.max(np.abs(u_ref.x.array), initial=0), op=MPI.MAX)
for kind in ("nest", "mpi"):
    uh, uh_G, _ = solve_coupled(V, V_G, tie(V, V_G, entity_map), kind)
    differences = (max_difference(uh, u_ref), max_difference(uh_G, u_ref_G))
    if msh.comm.rank == 0:
        print(f"kind={kind!s:5}  max |u - u_ref| = {differences[0]:.2e},  max |u_G - u_ref_G| = {differences[1]:.2e}")
    assert max(differences) < tol * u_max
# -

# The bulk solution, raised by its value, and the surface field, drawn as a tube on its
# top edge, where it lies on the trace of the bulk solution. Each process draws the cells
# it owns, which are gathered on the first process.

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


bulk_pieces = gathered_grid(uh, V, "u")
surface_pieces = gathered_grid(uh_G, V_G, "u")
if bulk_pieces is not None:  # only the first process received the grids
    u_range = value_range(bulk_pieces + surface_pieces, "u")
    plotter = pyvista.Plotter(window_size=[800, 600])
    for piece in bulk_pieces:
        warped = piece.warp_by_scalar("u", factor=0.2, normal=(0.0, 0.0, 1.0))
        plotter.add_mesh(
            warped,
            scalars="u",
            cmap="viridis",
            clim=u_range,
            show_edges=True,
            edge_color="gray",
            scalar_bar_args={"vertical": True, "position_x": 0.85, "position_y": 0.2},
        )
    for piece in surface_pieces:
        warped = piece.warp_by_scalar("u", factor=0.2, normal=(0.0, 0.0, 1.0))
        plotter.add_mesh(warped, color="red", line_width=8, render_lines_as_tubes=True)
    plotter.view_isometric()
    # The figure is named after the demo, as the gallery of the documentation expects
    figure = Path("demo_bulk_surface.py")
    if pyvista.OFF_SCREEN:
        plotter.screenshot(figure.with_suffix(".png"))
    else:
        # The interactive scene, for the gallery
        plotter.export_vtksz(figure.with_suffix(".vtksz"))
        plotter.show(screenshot=figure.with_suffix(".png"))
# -

# ## Cost and conditioning
#
# As in {doc}`demo_mean_value_constraint`, both formulations are solved with MUMPS, and
# compared through the number of nonzeros and the 2-norm condition number of the
# operator, against the bulk and surface problems without the tie, $A$.
#
# The constrained operator $K^TAK$ keeps a row for every degree of freedom of both
# spaces, the rows of the slaves holding only their diagonal, and stays symmetric
# positive definite. The multiplier adds a third block of the size of $V_\Gamma$ and
# makes the system a saddle point.


# + tags=["hide-input"]
def operator_stats(A, comm, root=0):
    """Global nnz and 2-norm condition number of an assembled operator.

    The condition number is computed from a dense SVD on ``root``, so this is a
    diagnostic for demo sized problems only.
    """
    # petsc4py defaults to MatInfoType.GLOBAL_SUM, so this is already reduced
    nnz = int(A.getInfo()["nz_used"])
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    cond = None
    if comm.rank == root:
        sv = np.linalg.svd(A_csr.toarray(), compute_uv=False)
        cond = sv[0] / sv[-1]
    return nnz, comm.bcast(cond, root=root)


def measure(N: int) -> dict:
    """Solve at resolution ``N`` and collect cost, conditioning and timings."""
    comm = MPI.COMM_WORLD
    msh, surface, entity_map, tags = create_meshes(N)
    V, V_G = create_spaces(msh, surface, 1)

    comm.Barrier()
    t0 = time.perf_counter()
    constraints = tie(V, V_G, entity_map)
    comm.Barrier()
    t1 = time.perf_counter()
    _, _, constraint_problem = solve_coupled(V, V_G, constraints, "mpi")
    comm.Barrier()
    t2 = time.perf_counter()
    _, _, multiplier_problem = solve_multiplier(V, V_G, entity_map, tags)
    comm.Barrier()
    t3 = time.perf_counter()

    A_plain = dolfinx.fem.petsc.assemble_matrix(fem.form(create_forms(V, V_G)[0]), kind="mpi")
    A_plain.assemble()
    nnz_A, cond_A = operator_stats(A_plain, comm)
    nnz_mpc, cond_mpc = operator_stats(constraint_problem.A, comm)
    nnz_saddle, cond_saddle = operator_stats(multiplier_problem.A, comm)
    A_plain.destroy()
    return {
        "N": V.dofmap.index_map.size_global + V_G.dofmap.index_map.size_global,
        "slaves": V_G.dofmap.index_map.size_global,
        "nnz_A": nnz_A,
        "nnz_mpc": nnz_mpc,
        "nnz_saddle": nnz_saddle,
        "cond_A": cond_A,
        "cond_mpc": cond_mpc,
        "cond_saddle": cond_saddle,
        "t_constraint": t1 - t0,
        "t_mpc": t2 - t1,
        "t_multiplier": t3 - t2,
    }


rows = [measure(N) for N in (8, 12, 16)]

COLUMNS = {
    "N": "dofs",
    "slaves": "slaves",
    "nnz_A": "nnz(A)",
    "nnz_mpc": "nnz(KtAK)",
    "nnz_saddle": "nnz(saddle)",
    "cond_A": "cond(A)",
    "cond_mpc": "cond(KtAK)",
    "cond_saddle": "cond(saddle)",
    "t_constraint": "build [s]",
    "t_mpc": "mpc [s]",
    "t_multiplier": "multiplier [s]",
    "mpc/multiplier": "mpc/multiplier",
}
FORMATS = {
    "cond(A)": "{:.3e}",
    "cond(KtAK)": "{:.3e}",
    "cond(saddle)": "{:.3e}",
    "build [s]": "{:.4f}",
    "mpc [s]": "{:.4f}",
    "multiplier [s]": "{:.4f}",
    "mpc/multiplier": "{:.2f}",
}

table = pandas.DataFrame(rows)
table["mpc/multiplier"] = table["t_mpc"] / table["t_multiplier"]
table = table[list(COLUMNS)].rename(columns=COLUMNS).set_index("dofs")
if MPI.COMM_WORLD.rank == 0:
    print(table.to_string(formatters={k: v.format for k, v in FORMATS.items()}))
table.style.format(FORMATS)
# -

# Each slave has a single master, the bulk degree of freedom at the same point, so the
# reduced operator is barely denser than the problems without the tie: the surface
# operator is added to rows of $A$ that already couple the same degrees of freedom. Its
# condition number, $1.6\times10^3$ to $1.1\times10^4$ on these meshes, is even somewhat
# below that of $A$, while the saddle point is 35 to 41 times worse than $K^TAK$, and the
# gap widens under refinement. Both systems are small enough here that MUMPS solves them in a few
# milliseconds, the constrained one slightly faster; the conditioning is what an
# iterative solver would feel.
#
# Compare with {doc}`demo_mean_value_constraint`, where every degree of freedom is a
# master of the one slave and the reduced operator fills in. A tie between a submesh
# and its parent is the opposite case: one master per slave, and the constraint is the
# cheaper of the two formulations.

# ## Convergence
#
# Both fields converge at the optimal rate, $h^{k+1}$ in $L^2$ for degree $k$.


# +
def l2_error(uh, x):
    error = fem.form(ufl.inner(uh - u_exact(x), uh - u_exact(x)) * ufl.dx)
    return np.sqrt(uh.function_space.mesh.comm.allreduce(fem.assemble_scalar(error), op=MPI.SUM).real)


# P2 stops at $N=8$, as its error at $N=16$ approaches rounding in single precision
for degree, resolutions in ((1, (8, 16, 32)), (2, (4, 8))):
    previous = None
    for N in resolutions:
        msh, surface, entity_map, tags = create_meshes(N)
        V, V_G = create_spaces(msh, surface, degree)
        uh, uh_G, _ = solve_coupled(V, V_G, tie(V, V_G, entity_map), "mpi")
        errors = (l2_error(uh, ufl.SpatialCoordinate(msh)), l2_error(uh_G, ufl.SpatialCoordinate(surface)))
        if previous is not None:
            rates = [np.log2(e0 / e1) for e0, e1 in zip(previous, errors)]
            if msh.comm.rank == 0:
                print(
                    f"P{degree} N={N:3d}  |u-u_ex| = {errors[0]:.2e} (rate {rates[0]:.2f})"
                    f"  |u_G-u_ex| = {errors[1]:.2e} (rate {rates[1]:.2f})"
                )
        previous = errors
    assert min(rates) > degree + 0.8
# -

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: bulksurface-
# ```

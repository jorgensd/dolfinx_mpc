# # Prescribing a boundary average with an affine multi-point constraint
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# The companion demo {doc}`demo_mean_value_constraint` enforces
# $\int_\Omega u~\mathrm{d}x=\gamma$ with an affine multi-point constraint, and
# shows that the price is a reduced operator that is completely full. This demo
# applies the same construction to a functional supported on a **facet**,
#
# $$
# \int_\Gamma u ~\mathrm{d}s = \gamma,
# $$
#
# where the master set is only the degrees of freedom on $\Gamma$. Both the extra
# sparsity and the loss of conditioning scale with that set, so here the
# constraint is genuinely cheap, and this is where it is the better tool.
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
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, la, mesh, plot

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


def integral_constraint(V, weight_form, value, bcs=(), rtol=1e-14):
    """Build an affine MPC enforcing ``L(u) = value`` for a linear functional.

    Args:
        V: The function space the constraint acts on.
        weight_form: A linear form in ``TestFunction(V)`` defining the
            functional, e.g. ``v * ds(tag)`` or ``ufl.dot(v, n) * ds(tag)``.
        value: The prescribed value of the functional.
        bcs: Dirichlet conditions on ``V``. Constrained dofs are excluded from
            being the slave, and are folded into the constraint offset if they
            appear as masters.
        rtol: Passed to `finalize` as the master filter: a master whose
            coefficient is below this fraction of the largest coefficient
            of its slave is discarded.

    Returns:
        The finalized constraint, the ``Function`` holding the inhomogeneity
        (keep it alive to change ``value`` via ``update_constants``), and the
        number of masters.
    """
    comm = V.mesh.comm
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    num_owned = imap.size_local * bs
    dtype = default_scalar_type
    mpi_scalar = MPI._typedict[np.dtype(dtype).char]

    # Ghost contributions are unaccumulated until scatter_reverse
    w = fem.assemble_vector(fem.form(weight_form, dtype=dtype))
    w.scatter_reverse(la.InsertMode.add)
    w_owned = w.array[:num_owned]

    # A Dirichlet dof may be a master, but must not be the slave
    bc_marker = np.zeros(num_owned, dtype=np.int8)
    for bc in bcs:
        dofs, num_owned_bc = bc.dof_indices()
        owned_bc = dofs[:num_owned_bc]
        bc_marker[owned_bc[owned_bc < num_owned]] = 1

    # Every owned dof is offered as a master candidate. The global index of a
    # blocked space is global block index * bs + component.
    local_dofs = np.arange(num_owned, dtype=np.int32)
    global_dofs = (imap.local_to_global(local_dofs // bs) * bs + local_dofs % bs).astype(np.int64)

    counts = np.array(comm.allgather(global_dofs.size), dtype=np.int32)
    displ = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    total = int(counts.sum())
    all_dofs = np.empty(total, dtype=np.int64)
    all_weights = np.empty(total, dtype=dtype)
    all_bc = np.empty(total, dtype=np.int8)
    # mpi4py reads a three-entry tuple as (buffer, counts, datatype), so the
    # datatype must be spelled out whenever displacements are given.
    comm.Allgatherv(global_dofs, (all_dofs, counts, displ, MPI.INT64_T))
    comm.Allgatherv(np.ascontiguousarray(w_owned, dtype=dtype), (all_weights, counts, displ, mpi_scalar))
    comm.Allgatherv(np.ascontiguousarray(bc_marker), (all_bc, counts, displ, MPI.SIGNED_CHAR))
    all_owners = np.repeat(np.arange(comm.size, dtype=np.int32), counts)

    # The largest weight makes the best slave: it bounds every |c_i| by one. The
    # gathered arrays are identical on every rank, so no reduction is needed to
    # find it, and every rank independently picks the same slave.
    candidates = np.where(all_bc == 0, np.abs(all_weights), -1.0)
    slave = int(np.argmax(candidates))
    if candidates[slave] < 0:
        raise RuntimeError("Every dof in the support of the functional is Dirichlet constrained")
    if candidates[slave] == 0:
        # Otherwise the coefficients below would silently be 0/0
        raise RuntimeError("The functional vanishes identically on V")
    w_slave = all_weights[slave]
    slave_global = int(all_dofs[slave])

    masters = np.delete(all_dofs, slave).astype(np.int64)
    coeffs = (-np.delete(all_weights, slave) / w_slave).astype(dtype)
    owners = np.delete(all_owners, slave).astype(np.int32)
    offsets = np.array([0, masters.size], dtype=np.int32)

    g = fem.Function(V, dtype=dtype)
    g.x.array[:] = 0.0
    mpc = MultiPointConstraint(V, dtype=dtype, bcs=list(bcs), rhs_coeffs=g)
    slave_block = int(imap.global_to_local(np.array([slave_global // bs], dtype=np.int64))[0])
    if slave_block != -1:  # the owning rank, and every rank ghosting the slave
        slave_local = np.int32(slave_block * bs + slave_global % bs)
        g.x.array[slave_local] = dtype(value) / w_slave
        mpc.add_constraint(V, np.array([slave_local], dtype=np.int32), masters, coeffs, owners, offsets)
    g.x.scatter_forward()
    # Offering every dof as a master leaves most coefficients at zero, and
    # `filter` discards them: for a facet functional that is nearly the whole
    # mesh, and for a cell integral with P2 it is every vertex dof, since the
    # integral of a P2 vertex basis function vanishes exactly on simplices. A
    # negligible coefficient changes nothing in the constraint but still costs a
    # ghost, a row of the sparsity pattern and an entry in every element matrix
    # modification. For a very large problem it is worth restricting the gather
    # above to the support of the functional too, so that the communication is
    # not O(num_dofs) on every rank.
    mpc.finalize(filter=rtol)  # collective: every rank must reach this

    # Report the masters that survived the filter, not the ones offered
    kept = sum(len(mpc.masters.links(s)) for s in mpc.slaves[: mpc.num_local_slaves])
    return mpc, g, comm.allreduce(kept, op=MPI.SUM)


# ## Cost and conditioning
#
# Eliminating the slave is not free. Writing the masters as $m$ and the slave as
# $s$, the reduced operator is
#
# $$
# K^TAK = A_{mm} + A_{ms}c^T + cA_{sm} + \left(cc^T\right)A_{ss}.
# $$
#
# The final term is rank one and dense over every pair of masters. Here that set
# is only the boundary, so it grows like $\sqrt{N}$ and the penalty stays mild,
# in sharp contrast to the cell integral of {doc}`demo_mean_value_constraint`.


def operator_stats(A, comm, root=0):
    """Global nnz and 2-norm condition number of an assembled operator.

    The condition number needs a dense SVD, so this is a diagnostic for demo
    sized problems only.
    """
    # petsc4py defaults to MatInfoType.GLOBAL_SUM, so this is already reduced
    nnz = int(A.getInfo()["nz_used"])
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    cond = None
    if comm.rank == root:
        cond = float(np.linalg.cond(A_csr.toarray()))
    return nnz, comm.bcast(cond, root=root)


# ## The real space reference, on a submesh of $\Gamma$
#
# The multiplier $\lambda$ conjugate to a boundary functional lives on $\Gamma$,
# not on $\Omega$, so the reference puts the real space on a **submesh of the
# marked facets** rather than using a domain-global real element. The form is
# still integrated on the parent mesh -- mixed dimensional forms always use the
# higher dimensional domain as the integration domain -- and the `EntityMap`
# returned by {py:func}`dolfinx.mesh.create_submesh` relates the two.
#
# ```{warning}
# The facets given to `create_submesh` must cover everything the measure
# integrates over. Facets outside the submesh map to `-1`, which is not checked,
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
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    uh, lamh = problem.solve()
    # The single real dof is ghosted on every rank, so it must not be summed
    lam_value = lamh.x.array[0] if lamh.x.array.size else 0.0
    return uh, float(np.real(lam_value)), problem


# ## Solving
#
# One function does the whole solve, so that the cost table below can repeat it at
# several resolutions.


def solve_boundary_average(N, degree, collect_stats=False):
    """Solve the boundary averaged Poisson problem on an ``N`` by ``N`` square."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N)
    mt = mark_boundary(domain, lambda x: np.isclose(x[0], 0.0), GAMMA, REST)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    V = fem.functionspace(domain, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    u_ex = x[0] ** 2 / 2 + x[0] + C
    n = ufl.FacetNormal(domain)
    f = -ufl.div(ufl.grad(u_ex))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # Natural data on the unconstrained part of the boundary only
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(ufl.dot(ufl.grad(u_ex), n), v) * ds(REST)

    # The target and the expected multiplier both come from the manufactured
    # solution. mu is the average of the exact flux over Gamma; the demo also
    # checks that the flux really is constant there, which is what makes the
    # defective condition well posed.
    gamma_value, length = boundary_average(domain, u_ex, ds, GAMMA)
    flux = ufl.dot(ufl.grad(u_ex), n)
    mu_exact = boundary_average(domain, flux, ds, GAMMA)[0] / length
    flux_variation = np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form((flux - mu_exact) ** 2 * ds(GAMMA))), op=MPI.SUM)
    )

    # Time the two paths against each other, as in demo_mean_value_constraint.py
    comm.Barrier()
    t0 = time.perf_counter()
    mpc, g, num_masters = integral_constraint(V, ufl.TestFunction(V) * ds(GAMMA), gamma_value)
    comm.Barrier()
    t1 = time.perf_counter()
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = LinearProblem(a, L, mpc, bcs=[], petsc_options=petsc_options)
    uh = problem.solve()
    comm.Barrier()
    t2 = time.perf_counter()

    results = {"N": V.dofmap.index_map.size_global * V.dofmap.index_map_bs, "M": num_masters}
    results["gamma"] = gamma_value
    results["mu_exact"] = mu_exact
    results["flux_variation"] = flux_variation
    results["t_constraint"] = t1 - t0
    results["t_mpc"] = t2 - t1
    results["integral"] = comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(GAMMA))), op=MPI.SUM)
    results["error"] = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM))
    # The constant flux conjugate to the constraint, recovered from the solution
    results["mu"] = (
        comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(uh), n) * ds(GAMMA))), op=MPI.SUM) / length
    )

    t3 = time.perf_counter()
    u_real, lam_real, real_problem = solve_poisson_real_space(domain, mt, degree, u_ex, gamma_value)
    comm.Barrier()
    results["t_real"] = time.perf_counter() - t3
    results["lambda"] = lam_real
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    diff = np.max(np.abs(uh.x.array[:num_owned] - u_real.x.array[:num_owned])) if num_owned else 0.0
    results["mpc_vs_real"] = comm.allreduce(diff, op=MPI.MAX)

    if collect_stats:
        A_plain = fem.petsc.assemble_matrix(fem.form(a), bcs=[])
        A_plain.assemble()
        sv_stats = operator_stats(A_plain, comm)
        # A is singular here too (pure Neumann away from Gamma), so report the
        # nnz only and take the conditioning from the constrained operators.
        results["nnz_A"] = sv_stats[0]
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_plain, root=0)
        cond_A = None
        if comm.rank == 0:
            sv = np.linalg.svd(A_csr.toarray(), compute_uv=False)
            cond_A = sv[0] / sv[-2]
        results["cond_A"] = comm.bcast(cond_A, root=0)
        results["nnz_mpc"], results["cond_mpc"] = operator_stats(problem.A, comm)
        results["nnz_real"], results["cond_real"] = operator_stats(real_problem.A, comm)
        results["norm_c"] = float(np.linalg.norm(mpc.coefficients()[0]))
        A_plain.destroy()

    return uh, domain, results


# ## Verification
#
# The constraint is checked against its target, the manufactured solution is
# recovered, the recovered flux is compared with the exact one, and the whole
# solve is compared with the real space formulation.

# +

comm = MPI.COMM_WORLD
degree = 2
uh, domain, res = solve_boundary_average(24, degree, collect_stats=True)

if comm.rank == 0:
    print("\n----Prescribed boundary average----")
    print(f"  dofs                      {res['N']}")
    print(f"  masters                   {res['M']}")
    print(f"  int_Gamma u_h ds          {res['integral']:.15f} (target {res['gamma']:.15f})")
    print(f"  |int_Gamma u_h ds - g|    {abs(res['integral'] - res['gamma']):.3e}")
    print(f"  L2(u_h - u_ex)            {res['error']:.3e}")
    print(f"  recovered flux mu         {res['mu']:.12f} (exact {res['mu_exact']:.12f})")
    print(f"  real space multiplier     {res['lambda']:.12f} (equals -mu)")
    print(f"  max|u_mpc - u_real|       {res['mpc_vs_real']:.3e}")

assert abs(res["integral"] - res["gamma"]) < 1e-12
assert res["error"] < 1e-11
# The exact flux must be constant on Gamma, or the defective condition would not
# be the problem this demo claims to solve
assert res["flux_variation"] < 1e-12
assert abs(res["mu"] - res["mu_exact"]) < 1e-8
# The multiplier enters the reference form as +lam, so it is -mu
assert abs(res["lambda"] + res["mu"]) < 1e-8
assert res["mpc_vs_real"] < 1e-11

# -

# ## Cost and conditioning, measured
#
# The same table as in {doc}`demo_mean_value_constraint`, and the comparison is the
# point of this demo: the master set is the boundary rather than the whole mesh,
# so it grows like the square root of the number of degrees of freedom, and both
# the extra fill and the loss of conditioning follow it.

# +

if comm.rank == 0:
    header = f"{'N':>6} {'M':>6} {'||c||':>8} {'nnz(A)':>9} {'nnz(KtAK)':>11} {'nnz(saddle)':>12}"
    header += f" {'cond(A)':>11} {'cond(KtAK)':>11} {'cond(saddle)':>12}"
    header += f" | {'build[s]':>9} {'mpc[s]':>8} {'real[s]':>8} {'mpc/real':>9}"
    print("\n----Cost and conditioning----")
    print(header)

for N in (8, 12, 16):
    r = solve_boundary_average(N, degree, collect_stats=True)[2]
    if comm.rank == 0:
        print(
            f"{r['N']:6d} {r['M']:6d} {r['norm_c']:8.2f} {r['nnz_A']:9d} {r['nnz_mpc']:11d} "
            f"{r['nnz_real']:12d} {r['cond_A']:11.3e} {r['cond_mpc']:11.3e} {r['cond_real']:12.3e}"
            f" | {r['t_constraint']:9.4f} {r['t_mpc']:8.4f} {r['t_real']:8.4f}"
            f" {r['t_mpc'] / r['t_real']:9.2f}"
        )

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
# The mesh is partitioned in parallel, so each process holds only a piece of the
# field. Each one builds a PyVista grid over the cells it *owns* and the grids are
# gathered onto rank 0 and drawn into a single figure. Two details matter:
# restricting to owned cells, so a shared cell is not drawn twice, and giving
# every piece the same colour limits, so the partitions are comparable.

# +

pyvista.global_theme.allow_empty_mesh = True


def gather_grids(u: fem.Function, V: fem.FunctionSpace, name: str):
    """Owned-cell PyVista grids with ``u`` attached, gathered on rank 0.

    Vector fields are padded to three components, as PyVista expects. Returns the
    grids on rank 0 (``None`` elsewhere) and the global range of the magnitude,
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
    return comm.gather(grid, root=0), [lo, hi]


# -

# `uh` lives in the constraint's extended space, which carries the master dofs as
# extra ghosts, so its array is longer than the original space in parallel. The
# extended index map keeps the original dofs first, so the leading entries are
# exactly the values of the original space.

# +

V_plot = fem.functionspace(domain, ("Lagrange", degree))
u_plot = fem.Function(V_plot)
u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
pieces, clim = gather_grids(u_plot, V_plot, "u")

if comm.rank == 0:
    plotter = pyvista.Plotter(window_size=[700, 500])
    plotter.add_text(
        f"boundary average = {res['integral']:.4f}, recovered flux = {res['mu']:.4f}",
        font_size=10,
    )
    for piece in pieces:
        plotter.add_mesh(
            piece.warp_by_scalar("u", factor=0.15),
            scalars="u",
            cmap="viridis",
            clim=clim,
            scalar_bar_args={"vertical": True},
        )
    plotter.view_isometric()
    plotter.camera.zoom(1.4)
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_boundary_average_constraint.png")
    else:
        plotter.show()

# -

# ```{bibliography}
#    :filter: cited and ({"python/demos/demo_boundary_average_constraint"} >= docnames)
# ```

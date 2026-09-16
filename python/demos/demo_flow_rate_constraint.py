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
# the Taylor-Hood space, so the discrete solution is exact. The value the
# multiplier should take is read off $p$ with UFL rather than derived by hand: for
# a fully developed profile $(\sigma\cdot\mathbf{n})\cdot\mathbf{n}$ reduces to
# $-p$ on the outlet, so it is the mean of $p$ there.
#
# Two things differ from the scalar demo. The velocity space is blocked, so a
# global master index is `index_map.local_to_global(dof // bs) * bs + dof % bs`;
# and the corner nodes of the outlet carry the no-slip condition. Those may not be
# chosen as the slave, but they are perfectly good *masters*: passing `bcs` to
# {py:class}`dolfinx_mpc.MultiPointConstraint` eliminates them from the relation
# and folds their contribution into the constraint offset.

# +
from __future__ import annotations

from mpi4py import MPI

import basix.ufl
import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, la, mesh, plot

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
# The same builder as in the scalar demo. The functional is now
# $\mathbf{v}\mapsto\int_{\Gamma_{out}} \mathbf{v}\cdot\mathbf{n}~\mathrm{d}s$,
# so only the components along the outlet normal carry a weight and everything
# else is discarded by the filter.


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


# ## The real space reference, on a submesh of the outlet
#
# The multiplier conjugate to a boundary functional lives on the outlet, so the
# reference puts the real space on a **submesh of the marked facets**, giving a
# three block system. See {doc}`demo_boundary_average_constraint` for the details
# of the mixed dimensional assembly, including the warning about which facets the
# submesh must cover.


def solve_stokes_real_space(domain, mt, nu, flow_rate):
    """Reference Stokes solve with the flow rate imposed by a real space."""
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
        kind="mpi",
        entity_maps=[entity_map],
        petsc_options_prefix="demo_stokes_real_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    uh, ph, lamh = problem.solve()
    lam_value = lamh.x.array[0] if lamh.x.array.size else 0.0
    return uh, ph, float(np.real(lam_value)), problem


# ## Solving


def solve_flow_rate(nx, ny, nu=1.0, flow_rate=1.0, length=2.0, height=1.0):
    """Stokes flow in a channel with a prescribed outlet flow rate."""
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

    # Only the components along the outlet normal carry a weight; the rest are
    # filtered out. The no-slip corner dofs stay as masters and are folded into
    # the offset by passing bcs to the constraint.
    weight_form = ufl.dot(ufl.TestFunction(V), n) * ds(OUTLET)
    mpc_u, g, num_masters = integral_constraint(V, weight_form, flow_rate, bcs=[bc])
    mpc_p = MultiPointConstraint(P)
    mpc_p.finalize()

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = LinearProblem(
        ufl.extract_blocks(a), ufl.extract_blocks(L), [mpc_u, mpc_p], bcs=[bc], petsc_options=petsc_options
    )
    uh, ph = problem.solve()

    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector((6 * flow_rate * x[1] * (height - x[1]) / height**3, 0.0))
    p_ex = -12 * nu * flow_rate * x[0] / height**3
    area = comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds(OUTLET))),
        op=MPI.SUM,
    )
    # The Poiseuille profile must carry exactly the flow rate we prescribed
    exact_flux = comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(u_ex, n) * ds(OUTLET))), op=MPI.SUM)
    results_flux_check = abs(exact_flux - flow_rate)

    results = {"M": num_masters, "bs": V.dofmap.index_map_bs}
    results["exact_flux_error"] = results_flux_check
    # Outlet traction of Poiseuille flow; the multiplier enters the reference
    # form as +lam, so it carries the opposite sign to the flux mu.
    # The multiplier is the outlet traction. With sigma = nu*grad(u) - p*I and a
    # fully developed profile, (sigma.n).n reduces to -p there, and the reference
    # form carries the opposite sign, so lambda is the mean exact pressure on the
    # outlet. Taken from p_ex with UFL rather than written out by hand.
    results["lambda_exact"] = comm.allreduce(fem.assemble_scalar(fem.form(p_ex * ds(OUTLET))), op=MPI.SUM) / area
    results["flux"] = comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(OUTLET))), op=MPI.SUM)
    results["error_u"] = np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)), op=MPI.SUM)
    )
    results["error_p"] = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((ph - p_ex) ** 2 * ufl.dx)), op=MPI.SUM))
    # No nnz comparison here: the constrained block system is assembled as a
    # PETSc `nest`, which has no MatGetInfo, and the reference is monolithic.
    # The fill and conditioning story is measured in Part 1.
    u_real, p_real, lam_real, real_problem = solve_stokes_real_space(domain, mt, nu, flow_rate)
    results["lambda"] = lam_real
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    diff = np.max(np.abs(uh.x.array[:num_owned] - u_real.x.array[:num_owned])) if num_owned else 0.0
    results["mpc_vs_real"] = comm.allreduce(diff, op=MPI.MAX)

    return uh, ph, domain, results


# ## Verification
#
# The flow rate is checked against its target, the Poiseuille profile is
# recovered, and the multiplier matches the exact outlet traction.

# +

comm = MPI.COMM_WORLD
uh, ph, domain, res = solve_flow_rate(32, 16)

if comm.rank == 0:
    print("\n----Prescribed flow rate----")
    print(f"  block size                {res['bs']}")
    print(f"  masters                   {res['M']}")
    print(f"  int_out u.n ds            {res['flux']:.15f} (target 1.0)")
    print(f"  L2(u_h - u_ex)            {res['error_u']:.3e}")
    print(f"  L2(p_h - p_ex)            {res['error_p']:.3e}")
    print(f"  real space multiplier     {res['lambda']:.9f} (exact {res['lambda_exact']:.9f})")
    print(f"  max|u_mpc - u_real|       {res['mpc_vs_real']:.3e}")

assert abs(res["flux"] - 1.0) < 1e-12
assert res["error_u"] < 1e-11
assert res["error_p"] < 1e-10
assert res["mpc_vs_real"] < 1e-11
assert abs(res["lambda"] - res["lambda_exact"]) < 1e-8
# The manufactured profile must itself carry the prescribed flow rate
assert res["exact_flux_error"] < 1e-12

# -

# Only 30 masters are kept: the outlet has 33 velocity nodes, one becomes the
# slave, and the two no-slip corners are eliminated by the Dirichlet conditions
# passed to the constraint and folded into its offset.

# ## Visualization
#
# The mesh is partitioned in parallel, so each process holds only a piece of the
# field. Each one builds a PyVista grid over the cells it *owns* and the grids are
# gathered onto rank 0 and drawn into a single figure. Two details matter:
# restricting to owned cells, so a shared cell is not drawn twice, and giving
# every piece the same colour limits, so the partitions are comparable.

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
    plotter.add_text(f"flow rate = {res['flux']:.4f}", font_size=10)
    for piece in pieces:
        plotter.add_mesh(
            piece.glyph(orient="u", scale="|u|", factor=0.10),
            scalars="|u|",
            cmap="viridis",
            clim=clim,
            scalar_bar_args={"vertical": True},
        )
    plotter.view_xy()
    plotter.camera.tight(padding=0.25, view="xy", adjust_render_window=False)
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_flow_rate_constraint.png")
    else:
        plotter.show()

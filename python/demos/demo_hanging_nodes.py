# # Hanging nodes
# **Author** Jørgen S. Dokken
#
# Local refinement of a quadrilateral or hexahedral mesh leaves *T-joints*: a coarse facet
# $F$ faces the two (or four) fine facets $F_1, F_2$ that subdivide it, and the vertex in the
# middle of $F$ is a *hanging node*. DOLFINx has no notion of this: the coarse cell and the fine
# cells share only the end points of $F$, so $F$, $F_1$ and $F_2$ are all exterior facets, each
# attached to a single cell. A Lagrange space on such a mesh is continuous at the end points of
# $F$, but not along it.
#
# This demo restores continuity with a {py:class}`dolfinx_mpc.MultiPointConstraint`. The trace of
# a fine cell on $F_i$ must equal the trace of the coarse cell on $F$. The coarse trace is the
# richer of the two sides, as a polynomial of degree $p$ on $F$ restricts to a polynomial of degree
# $p$ on each $F_i$. Every degree of freedom $s$ on $F_1\cup F_2$ is therefore made a slave of the
# coarse cell $K$:
#
# $$
# \begin{align*}
# u_s = \sum_{j} \phi^K_j(x_s)\, u_j,
# \end{align*}
# $$
#
# where $x_s$ is the point of the slave and $\phi^K_j$ the basis functions of $K$. The degrees of
# freedom at the end points of $F$ are shared by both sides, so they are already continuous and are
# not constrained. This is the inelastic contact condition
# {py:meth}`create_contact_inelastic_condition<dolfinx_mpc.MultiPointConstraint.create_contact_inelastic_condition>`,
# with the fine facets as slaves and the coarse facets as masters.
#
# ## Problem
#
# We solve $-\Delta u = f$ in $\Omega = (0, 2)\times(0, 1)$, $u = g$ on $\partial\Omega$, with the
# exact solution
#
# $$
# \begin{align*}
# u_{ex} = x^p + y^p + xy,
# \end{align*}
# $$
#
# which is in $Q_p$, the space of polynomials of degree at most $p$ in each direction.
# The constrained space is conforming and contains $Q_p$, so the finite element solution
# is $u_{ex}$ up to round-off.

# + tags=["hide-input"]
from __future__ import annotations

from mpi4py import MPI

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh

import dolfinx_mpc

# -

# ## Mesh
#
# The mesh is a $3\times3$ grid where the centre cell is replaced by $2\times2$ cells. This gives
# four T-joints with a hanging node each. The mesh is created on rank 0 and distributed.

COARSE, FINE = 2, 3
xs, ys = np.array([0, 2 / 3, 4 / 3, 2]), np.array([0, 1 / 3, 2 / 3, 1])
if MPI.COMM_WORLD.rank == 0:
    grid = np.array([[x, y] for y in ys for x in xs])
    hanging = np.array([[1, 1 / 3], [2 / 3, 1 / 2], [1, 2 / 3], [4 / 3, 1 / 2], [1, 1 / 2]])
    nodes = np.vstack([grid, hanging])
    coarse = [[i + 4 * j, i + 1 + 4 * j, i + 4 * (j + 1), i + 1 + 4 * (j + 1)] for j in range(3) for i in range(3)]
    coarse.pop(4)
    fine = [[5, 16, 17, 20], [16, 6, 20, 19], [17, 20, 9, 18], [20, 19, 18, 10]]
    cells = np.array(coarse + fine, dtype=np.int64)
else:
    nodes = np.empty((0, 2), dtype=np.float64)
    cells = np.empty((0, 4), dtype=np.int64)
domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
msh = mesh.create_mesh(MPI.COMM_WORLD, cells, domain, nodes)

# The interface facets lie on the boundary of the refined patch. Each is attached to one cell,
# which is fine if it is inside the patch and coarse otherwise.

# +
tol = 500 * np.finfo(msh.geometry.x.dtype).eps


def on_interface(x):
    in_x = (x[0] > xs[1] - tol) & (x[0] < xs[2] + tol)
    in_y = (x[1] > ys[1] - tol) & (x[1] < ys[2] + tol)
    on_x = np.isclose(x[0], xs[1], atol=tol) | np.isclose(x[0], xs[2], atol=tol)
    on_y = np.isclose(x[1], ys[1], atol=tol) | np.isclose(x[1], ys[2], atol=tol)
    return (on_x & in_y) | (on_y & in_x)


tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
interface = mesh.locate_entities(msh, tdim - 1, on_interface)
f_to_c = msh.topology.connectivity(tdim - 1, tdim)
interface_cells = np.array([f_to_c.links(f)[0] for f in interface], dtype=np.int32)
midpoints = mesh.compute_midpoints(msh, tdim, interface_cells)
in_patch = (midpoints[:, 0] > xs[1]) & (midpoints[:, 0] < xs[2]) & (midpoints[:, 1] > ys[1]) & (midpoints[:, 1] < ys[2])
ft = mesh.meshtags(msh, tdim - 1, interface, np.where(in_patch, FINE, COARSE).astype(np.int32))
# -

# The interface facets are exterior facets, so
# {py:func}`dolfinx.mesh.locate_entities_boundary` would also return them. The Dirichlet condition
# is instead imposed on the dofs located geometrically on $\partial\Omega$.


def outer_boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 2) | np.isclose(x[1], 0) | np.isclose(x[1], 1)


# ## Solving with hanging-node constraints


def solve(degree: int) -> tuple[int, float]:
    """Solve the Poisson problem in Q_p, returning the number of slaves and the max nodal error."""
    V = fem.functionspace(msh, ("Lagrange", degree))
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_contact_inelastic_condition(ft, FINE, COARSE)
    mpc.finalize()

    x = ufl.SpatialCoordinate(msh)
    u_ex = x[0] ** degree + x[1] ** degree + x[0] * x[1]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = -ufl.div(ufl.grad(u_ex)) * v * ufl.dx

    g = fem.Function(V)
    g.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(g, fem.locate_dofs_geometrical(V, outer_boundary))

    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        mpc,
        bcs=[bc],
        petsc_options_prefix=f"demo_hanging_nodes_{degree}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    uh = problem.solve()
    assert isinstance(uh, fem.Function)
    problem.solver.destroy()

    num_owned = V.dofmap.index_map.size_local
    error = np.max(np.abs(uh.x.array[:num_owned] - g.x.array[:num_owned]), initial=0)
    num_slaves = msh.comm.allreduce(mpc.num_local_slaves, op=MPI.SUM)
    return num_slaves, msh.comm.allreduce(error, op=MPI.MAX)


# For each degree there is one slave per hanging node, and $p-1$ per fine facet for the dofs in
# its interior: $4 + 8(p - 1)$ in total. A constraint on the hanging nodes alone, e.g.
# $u_{16} = \tfrac{1}{2}(u_5 + u_6)$, is correct only for $p=1$.

eps = np.finfo(default_scalar_type).eps
for degree in range(1, 4):
    num_slaves, error = solve(degree)
    if msh.comm.rank == 0:
        print(f"Q{degree}: {num_slaves} slaves, max nodal error {error:.2e}")
    assert num_slaves == 4 + 8 * (degree - 1)
    assert error < 1e3 * eps

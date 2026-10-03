# # Joining two bodies with a rigid spider (RBE2)
#
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# A *spider* joins a single point, its body, to any number of nodes, its feet
# (see [FE Training on spiders](https://fetraining.net/spiders/)). In the rigid
# variant, known as RBE2 in most finite element codes, the feet follow the
# rigid-body motion of the body,
#
# $$
# u(x) = t + \theta \times (x - x_c),
# $$
#
# where $x_c$ is the body, $t$ its translation and $\theta$ its rotation.
#
# This demo joins two separately meshed elastic cubes, one of tetrahedra and one
# of hexahedra, through a single spider between them. The body is a mesh of one
# point, with six unknowns: $t$ and $\theta$. Every node on the facing side of
# each cube is a foot. Nothing else couples the cubes: the first is clamped at its
# far end, the second carries a load at its far end, and the load reaches the
# clamp through the spider only.
#
# Each foot is a slave of the body, so the masters of the constraint on a cube
# live in another block of the system, on another mesh.

# +
from mpi4py import MPI

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, la, mesh

import dolfinx_mpc

# -

# ## The two cubes and the spider
# The tetrahedral cube occupies $[0, 1]^3$, the hexahedral one
# $[1.2, 2.2] \times [0, 1]^2$. The spider sits in the gap between them.

# +
comm = MPI.COMM_WORLD
N = 6
cube_tet = mesh.create_box(comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [N, N, N], mesh.CellType.tetrahedron)
cube_hex = mesh.create_box(comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [N, N, N], mesh.CellType.hexahedron)
V_tet = fem.functionspace(cube_tet, ("Lagrange", 1, (3,)))
V_hex = fem.functionspace(cube_hex, ("Lagrange", 1, (3,)))
# -

# The bodies of all spiders are the points of one mesh, made by
# {py:func}`create_spider_mesh<dolfinx_mpc.create_spider_mesh>`. Each process
# passes the points it owns; here the first process owns the only one. The space
# on it has six components per point: the translation $t$ and the rotation
# $\theta$. A spider is named by the index of its point, which
# {py:func}`locate_spider<dolfinx_mpc.locate_spider>` finds on the process that
# owns it and broadcasts to all.

# +
x_c = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
spiders = dolfinx_mpc.create_spider_mesh(comm, x_c.reshape(1, 3) if comm.rank == 0 else np.zeros((0, 3)))
W = fem.functionspace(spiders, basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type))
spider_index = dolfinx_mpc.locate_spider(spiders, lambda x: np.isclose(x[0], x_c[0]))
# -

# ## The constraints
# The feet are the nodes on the facing sides, $x = 1$ and $x = 1.2$, tied to the
# spider with {py:meth}`add_rbe2_geometrical<dolfinx_mpc.MultiPointConstraint.add_rbe2_geometrical>`
# and {py:meth}`add_rbe2_topological<dolfinx_mpc.MultiPointConstraint.add_rbe2_topological>`.
# The body has no constraint of its own, but is finalized together with the cubes,
# as its space holds their masters.

# +
tol = 1e3 * np.finfo(default_real_type).eps
mpc_tet = dolfinx_mpc.MultiPointConstraint(V_tet)
mpc_tet.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0, atol=tol), W, spider_index)
mpc_hex = dolfinx_mpc.MultiPointConstraint(V_hex)
facets = mesh.locate_entities_boundary(cube_hex, 2, lambda x: np.isclose(x[0], 1.2, atol=tol))
mpc_hex.add_rbe2_topological(2, facets, W, spider_index)
mpc_body = dolfinx_mpc.MultiPointConstraint(W)
mpcs = [mpc_tet, mpc_hex, mpc_body]
dolfinx_mpc.finalize_multipointconstraints(mpcs)
# -

# ## Linear elasticity on each cube
# The forms do not couple the blocks, and the body has no form at all. The
# off-diagonal blocks, and the body's own block, are filled by the constraint.

# +
E, nu = 1.0e3, 0.3
mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def sigma(u):
    return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)


def elasticity(V):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx


# A downward traction on the far end of the hexahedral cube
far_end = mesh.locate_entities_boundary(cube_hex, 2, lambda x: np.isclose(x[0], 2.2, atol=tol))
facet_tags = mesh.meshtags(cube_hex, 2, far_end, np.full(len(far_end), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=cube_hex, subdomain_data=facet_tags)
traction = fem.Constant(cube_hex, np.array([0.0, 0.0, -1.0], dtype=default_scalar_type))

a = [[elasticity(V_tet), None, None], [None, elasticity(V_hex), None], [None, None, None]]
L = [
    ufl.ZeroBaseForm((ufl.TestFunction(V_tet),)),
    ufl.inner(traction, ufl.TestFunction(V_hex)) * ds(1),
    None,
]

# The tetrahedral cube is clamped at x = 0
clamped = fem.locate_dofs_geometrical(V_tet, lambda x: np.isclose(x[0], 0.0, atol=tol))
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V_tet)
# -

# ## Solve
# The blocks are assembled into one matrix and solved directly.

problem = dolfinx_mpc.LinearProblem(
    a,
    L,
    mpcs,
    bcs=[bc],
    kind="mpi",
    petsc_options_prefix="demo_spider_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
)
u_tet, u_hex, spider = problem.solve()

# ## Checks
# The feet move rigidly with the spider. The body's values are held only where
# they are owned or needed by a foot, so
# {py:func}`spider_values<dolfinx_mpc.spider_values>` gathers them on every process.

# +
values = dolfinx_mpc.spider_values(spider, spider_index)
t, theta = values[:3], values[3:6]


def rigid_motion_error(u, V, side):
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], side, atol=tol))
    r = V.tabulate_dof_coordinates()[dofs] - x_c
    expected = t + np.cross(theta, r)
    error = np.abs(u.x.array.reshape(-1, 3)[dofs] - expected).max() if len(dofs) > 0 else 0.0
    return comm.allreduce(error, op=MPI.MAX)


errors = (rigid_motion_error(u_tet, V_tet, 1.0), rigid_motion_error(u_hex, V_hex, 1.2))
# -

# The load reaches the clamp: the reaction there balances the applied traction.
# The reaction is the residual of the tetrahedral cube's equations at the clamped
# dofs, evaluated with the solution.

# +
u_ref = fem.Function(V_tet)
n_owned = V_tet.dofmap.index_map.size_local * 3
u_ref.x.array[:n_owned] = u_tet.x.array[:n_owned]
u_ref.x.scatter_forward()
residual = fem.assemble_vector(fem.form(ufl.action(elasticity(V_tet), u_ref)))
residual.scatter_reverse(la.InsertMode.add)
owned_clamped = clamped[clamped < V_tet.dofmap.index_map.size_local]
reaction = comm.allreduce(residual.array.reshape(-1, 3)[owned_clamped].sum(axis=0), op=MPI.SUM)
load = comm.allreduce(fem.assemble_scalar(fem.form(traction[2] * ds(1))), op=MPI.SUM)

if comm.rank == 0:
    print(f"Spider translation t = {t}, rotation theta = {theta}")
    print(f"Feet off the rigid motion: {errors[0]:.2e} (tetrahedra), {errors[1]:.2e} (hexahedra)")
    print(f"Reaction at the clamp: {reaction}, applied load: {[0.0, 0.0, load]}")

atol = 1e4 * np.finfo(default_real_type).eps
assert max(errors) < atol
assert np.allclose(reaction, [0.0, 0.0, -load], atol=atol * abs(load) * 100)
# -

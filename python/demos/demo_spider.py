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
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, la, mesh, plot

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
# passes the points it owns; here the first process owns the only one.
# Coinciding points are merged. A spider is named by its input index, its
# position among the points of all processes, process 0 first, so this one is
# spider 0. The space on the mesh has six components per point: the translation
# $t$ and the rotation $\theta$; with three, the spider only translates. The
# spider's position $x_c$ is the coordinate of its dofs in this space.

# +
x_c = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
spiders = dolfinx_mpc.create_spider_mesh(comm, x_c.reshape(1, 3) if comm.rank == 0 else np.zeros((0, 3)))
W = fem.functionspace(spiders, basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type))
# -

# ## The constraints
# The feet are the nodes on the facing sides, $x = 1$ and $x = 1.2$, tied to
# spider 0 with {py:meth}`add_rbe2_geometrical<dolfinx_mpc.MultiPointConstraint.add_rbe2_geometrical>`
# and {py:meth}`add_rbe2_topological<dolfinx_mpc.MultiPointConstraint.add_rbe2_topological>`.
# With several spiders, either method takes a list with one entry per spider,
# entry $k$ holding the locator (or the entities) of the feet of spider $k$.
# The body has no constraint of its own, but is finalized together with the cubes,
# as its space holds their masters.
#
# Each process needs the global dofs, owner and position of the spiders its feet
# are tied to. These are found through a post office keyed on the input index
# (`dolfinx_mpc::locate_spiders` in C++), so no process gathers the spider mesh.

# +
tol = 1e3 * np.finfo(default_real_type).eps
mpc_tet = dolfinx_mpc.MultiPointConstraint(V_tet)
mpc_tet.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0, atol=tol), W)
mpc_hex = dolfinx_mpc.MultiPointConstraint(V_hex)
facets = mesh.locate_entities_boundary(cube_hex, 2, lambda x: np.isclose(x[0], 1.2, atol=tol))
mpc_hex.add_rbe2_topological(2, facets, W)
mpc_body = dolfinx_mpc.MultiPointConstraint(W)
mpcs = [mpc_tet, mpc_hex, mpc_body]
dolfinx_mpc.finalize_multipointconstraints(mpcs)
# -

# ## Linear elasticity on each cube
# The forms do not couple the blocks, and the body has no bilinear form. The
# off-diagonal blocks, and the body's own block, are filled by the constraint.
#
# A load can also act on the spider itself. A force $F$ does work on the
# translation $t$ and a moment $M$ on the rotation $\theta$, so the load is the
# linear form
#
# $$
# \ell(w) = F \cdot w_t + M \cdot w_\theta
# $$
#
# on the space of the spider, with $w_t$ and $w_\theta$ the translation and
# rotation components of the test function. On a point mesh, `ufl.dx` evaluates
# its integrand at the point with weight one. The moment is about the spider
# point, in the global axes. The load starts at zero and is set further down.

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

# The force and moment (F, M) on the spider
spider_load = fem.Constant(spiders, np.zeros(6, dtype=default_scalar_type))

a = [[elasticity(V_tet), None, None], [None, elasticity(V_hex), None], [None, None, None]]
L = [
    ufl.ZeroBaseForm((ufl.TestFunction(V_tet),)),
    ufl.inner(traction, ufl.TestFunction(V_hex)) * ds(1),
    ufl.inner(spider_load, ufl.TestFunction(W)) * ufl.dx(spiders),
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
feet_tet = fem.locate_dofs_geometrical(V_tet, lambda x: np.isclose(x[0], 1.0, atol=tol))
feet_hex = fem.locate_dofs_geometrical(V_hex, lambda x: np.isclose(x[0], 1.2, atol=tol))


def rigid_motion_error(u, V, feet, x_c, t, theta):
    """Largest deviation of the feet from t + theta x (x - x_c), over all processes."""
    r = V.tabulate_dof_coordinates()[feet] - x_c
    expected = t + np.cross(theta, r)
    error = np.abs(u.x.array.reshape(-1, 3)[feet] - expected).max() if len(feet) > 0 else 0.0
    return comm.allreduce(error, op=MPI.MAX)


values = dolfinx_mpc.spider_values(spider, 0)
t, theta = values[:3], values[3:6]
errors = (
    rigid_motion_error(u_tet, V_tet, feet_tet, x_c, t, theta),
    rigid_motion_error(u_hex, V_hex, feet_hex, x_c, t, theta),
)
# -

# The load reaches the clamp: the reaction there balances the applied traction.
# The reaction is the residual of the tetrahedral cube's equations at the clamped
# dofs, evaluated with the solution. Its moment is taken about the centre of the
# clamped face.

# +
x_clamp = np.array([0.0, 0.5, 0.5])


def clamp_reaction(u):
    """Force and moment about `x_clamp` that the clamp exerts on the tetrahedral cube."""
    u_ref = fem.Function(V_tet)
    n_owned = V_tet.dofmap.index_map.size_local * 3
    u_ref.x.array[:n_owned] = u.x.array[:n_owned]
    u_ref.x.scatter_forward()
    residual = fem.assemble_vector(fem.form(ufl.action(elasticity(V_tet), u_ref)))
    residual.scatter_reverse(la.InsertMode.add)
    owned_clamped = clamped[clamped < V_tet.dofmap.index_map.size_local]
    f = residual.array.reshape(-1, 3)[owned_clamped]
    r = V_tet.tabulate_dof_coordinates()[owned_clamped] - x_clamp
    force = comm.allreduce(f.sum(axis=0), op=MPI.SUM)
    moment = comm.allreduce(np.cross(r, f).sum(axis=0), op=MPI.SUM)
    return force, moment


reaction, _ = clamp_reaction(u_tet)
load = comm.allreduce(fem.assemble_scalar(fem.form(traction[2] * ds(1))), op=MPI.SUM)

if comm.rank == 0:
    print(f"Spider translation t = {t}, rotation theta = {theta}")
    print(f"Feet off the rigid motion: {errors[0]:.2e} (tetrahedra), {errors[1]:.2e} (hexahedra)")
    print(f"Reaction at the clamp: {reaction}, applied load: {[0.0, 0.0, load]}")

atol = 1e4 * np.finfo(default_real_type).eps
assert max(errors) < atol
assert np.allclose(reaction, [0.0, 0.0, -load], atol=atol * abs(load) * 100)
# -

# ## A force and a moment on the spider
# The traction is now replaced by a force and a moment on the spider. The load
# reaches the clamp through the tetrahedral cube alone, while the hexahedral
# cube, joined to the same spider, moves rigidly with it.

# +
force = np.array([0.0, 0.2, -1.0])
moment = np.array([0.0, 0.3, 0.1])
traction.value[:] = 0.0
spider_load.value[:] = np.concatenate([force, moment])
u_tet_F, u_hex_F, spider_F = problem.solve()
# -

# In equilibrium, the clamp reacts with $-F$, and with the opposite of the load's
# moment about the centre of the clamped face, $M + (x_c - x_0) \times F$.

# +
reaction_force, reaction_moment = clamp_reaction(u_tet_F)
load_moment = moment + np.cross(x_c - x_clamp, force)
values = dolfinx_mpc.spider_values(spider_F, 0)
hex_error = rigid_motion_error(u_hex_F, V_hex, feet_hex, x_c, values[:3], values[3:6])
if comm.rank == 0:
    print(f"Reaction force {reaction_force}, applied {force}")
    print(f"Reaction moment {reaction_moment}, moment of the load {load_moment}")
assert np.allclose(reaction_force, -force, atol=100 * atol)
assert np.allclose(reaction_moment, -load_moment, atol=100 * atol)
assert hex_error < atol
# -

# ## Following the motion of the spider
# The relation $u = t + \theta \times (x - x_c)$ is linear in the rotation, so
# it holds for small rotations about the current position of the spider. Under a
# larger load we apply it in increments, each about the configuration reached by
# the previous ones (an updated Lagrangian approach): solve for the displacement
# increment, move both cubes and the spider by it, and recompute the coefficients
# of the constraint about the new positions with
# {py:meth}`update_rbe2<dolfinx_mpc.MultiPointConstraint.update_rbe2>`. It reads
# the feet's and the spider's current dof coordinates. Every rotation term of the
# constraint is kept, also where its coefficient is zero, so the masters stay the
# same and the matrix layout is reused.
#
# A mesh is moved through the space of its coordinate element, whose dofs per
# cell are the cell's geometry nodes.

# +


def move(domain, du):
    """Add the displacement `du` to the geometry of `domain`."""
    V_x = fem.functionspace(domain, domain.ufl_domain().ufl_coordinate_element())
    du_x = fem.Function(V_x)
    du_x.interpolate(du)
    gdim = domain.geometry.dim
    nodes = domain.geometry.dofmaps[0].reshape(-1)
    domain.geometry.x[nodes, :gdim] += du_x.x.array.reshape(-1, gdim)[V_x.dofmap.list.reshape(-1)]


def move_spiders(W, body):
    """Move each point of the spider mesh by the translation of its spider."""
    num_points = spiders.topology.index_map(0).size_local
    nodes = spiders.geometry.dofmaps[0][:num_points, 0]
    dofs = W.dofmap.list[:num_points, 0]
    bs = W.dofmap.index_map_bs
    t = body.x.array.reshape(-1, bs)[dofs, :3]
    spiders.geometry.x[nodes] += t


num_steps = 10
spider_load.value[:] = 0.0
traction.value[:] = [0.0, 0.0, -2.0]
u_total = [np.zeros((V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, 3)) for V in (V_tet, V_hex)]
x_spider = x_c.copy()
frames = []
# -

# Each process draws the cells it owns and the legs of the spider to the feet it
# owns; the pieces are gathered on one process, see
# [Plotting in parallel](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html).

# +
pyvista.global_theme.allow_empty_mesh = True


def pieces(x_spider):
    """The cubes, coloured by the total displacement, and the legs of the spider."""
    grids = []
    for V, u, feet in ((V_tet, u_total[0], feet_tet), (V_hex, u_total[1], feet_hex)):
        owned = np.arange(V.mesh.topology.index_map(3).size_local, dtype=np.int32)
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned))
        grid["|u|"] = np.linalg.norm(u[: grid.n_points], axis=1)
        grids.append(grid)
    feet = np.vstack(
        [
            V.tabulate_dof_coordinates()[f[f < V.dofmap.index_map.size_local]]
            for V, f in ((V_tet, feet_tet), (V_hex, feet_hex))
        ]
    )
    legs = pyvista.PolyData(np.vstack([x_spider, feet]))
    num_feet = len(feet)
    legs.lines = np.column_stack(
        [np.full(num_feet, 2), np.zeros(num_feet, dtype=np.int64), 1 + np.arange(num_feet)]
    ).reshape(-1)
    return grids, legs


for step in range(num_steps + 1):
    if step > 0:
        du_tet, du_hex, du_spider = problem.solve()
        values = dolfinx_mpc.spider_values(du_spider, 0)
        # The constraint holds about the configuration the increment was computed in
        error = max(
            rigid_motion_error(du_tet, V_tet, feet_tet, x_spider, values[:3], values[3:6]),
            rigid_motion_error(du_hex, V_hex, feet_hex, x_spider, values[:3], values[3:6]),
        )
        assert error < atol, f"Feet off the rigid motion by {error:.2e} at step {step}"
        for domain, du, total in ((cube_tet, du_tet, u_total[0]), (cube_hex, du_hex, u_total[1])):
            move(domain, du)
            total += du.x.array.reshape(-1, 3)[: len(total)]
        move_spiders(W, du_spider)
        x_spider += values[:3]
        mpc_tet.update_rbe2()
        mpc_hex.update_rbe2()
    # Collective: every process gathers, only the root draws
    frame = comm.gather(pieces(x_spider), root=0)
    if comm.rank == 0:
        frames.append(frame)

if comm.rank == 0:
    print(f"Spider after {num_steps} increments: {x_spider}, moved by {x_spider - x_c}")
# -

# The spider has moved down and the hexahedral cube has rotated with it.

# +
assert x_spider[2] < x_c[2] - 0.1

if comm.rank == 0:
    clim = [0.0, max(max(g["|u|"].max(initial=0.0) for g in grids) for grids, _ in frames[-1])]
    plotter = pyvista.Plotter(off_screen=True, window_size=[800, 500])
    plotter.open_gif("demo_spider.gif", fps=3)
    for frame in frames:
        plotter.clear()
        cubes = pyvista.merge([g for grids, _ in frame for g in grids])
        legs = pyvista.merge([legs for _, legs in frame])
        plotter.add_mesh(
            cubes,
            scalars="|u|",
            clim=clim,
            show_edges=True,
            cmap="viridis",
            scalar_bar_args={"title": "Displacement", "vertical": True, "position_x": 0.85, "position_y": 0.2},
        )
        plotter.add_mesh(legs, color="crimson", line_width=2)
        plotter.add_mesh(pyvista.Sphere(radius=0.04, center=legs.points[0]), color="crimson")
        plotter.camera_position = [(1.0, -5.2, 1.4), (1.1, 0.5, 0.0), (0.0, 0.0, 1.0)]
        plotter.write_frame()
    plotter.close()
# -

# <img src="./demo_spider.gif" alt="gif" class="bg-primary mb-1" width="800px">

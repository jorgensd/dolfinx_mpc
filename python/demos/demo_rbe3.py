# # Spreading a load over two bodies with a flexible spider (RBE3)
#
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# A *spider* joins a single point, its body, to any number of nodes, its feet
# (see [FE Training on spiders](https://fetraining.net/spiders/)). In the
# flexible variant, known as RBE3 in most finite element codes, the body follows
# its feet: its motion is the rigid motion that best fits theirs, in the
# weighted least-squares sense,
#
# $$
# \min_{t, \theta} \sum_i w_i \lvert u_i - t - \theta \times (x_i - x_c) \rvert^2,
# $$
#
# where $x_c$ is the body, $t$ its translation, $\theta$ its rotation, and $u_i$
# the displacement of foot $i$ at $x_i$, with weight $w_i$. Writing $B_i$ for the
# map from $(t, \theta)$ to the rigid motion at foot $i$, and
# $A = \sum_i w_i B_i^T B_i$, this gives
#
# $$
# (t, \theta) = A^{-1} \sum_i w_i B_i^T u_i.
# $$
#
# The unknowns of the body are the slaves, and every component of every foot is
# one of their masters. Unlike the rigid spider (RBE2, see {doc}`demo_spider`), an
# RBE3 spider adds no stiffness: the feet move freely, and a load on the body is
# spread over them.
#
# This demo applies a force and a moment to a spider between two separately
# meshed elastic cubes, one of tetrahedra and one of hexahedra, each clamped at
# its far end. The spider's feet are the nodes on the facing sides of both cubes.

# +
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

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
cube_tet = mesh.create_box(
    comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [N, N, N], mesh.CellType.tetrahedron, dtype=default_real_type
)
cube_hex = mesh.create_box(
    comm, [[1.2, 0.0, 0.0], [2.2, 1.0, 1.0]], [N, N, N], mesh.CellType.hexahedron, dtype=default_real_type
)
V_tet = fem.functionspace(cube_tet, ("Lagrange", 1, (3,)))
V_hex = fem.functionspace(cube_hex, ("Lagrange", 1, (3,)))
# -

# The body of the spider is a point of a mesh made by
# {py:func}`create_spider_mesh<dolfinx_mpc.create_spider_mesh>`, as in
# {doc}`demo_spider`. The space on it has six components: the translation $t$ and
# the rotation $\theta$.

# +
x_c = np.array([1.1, 0.5, 0.5], dtype=default_real_type)
spiders = dolfinx_mpc.create_spider_mesh(comm, x_c.reshape(1, 3) if comm.rank == 0 else np.zeros((0, 3)))
W = fem.functionspace(spiders, basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type))
# -

# ## The constraint
# The slaves are the spider's dofs, so the constraint is on the space of the
# spider, and each call names the space of the feet:
# {py:meth}`add_rbe3_geometrical<dolfinx_mpc.MultiPointConstraint.add_rbe3_geometrical>`
# and {py:meth}`add_rbe3_topological<dolfinx_mpc.MultiPointConstraint.add_rbe3_topological>`.
# All feet have weight one here; a weight per foot can be given as a number or a
# function of the coordinates. The constraint is built when it is finalized
# together with the constraints of the cubes, which have none of their own but
# hold its masters.

# +
tol = 1e3 * np.finfo(default_real_type).eps
facets = mesh.locate_entities_boundary(cube_hex, 2, lambda x: np.isclose(x[0], 1.2, atol=tol))
mpc_tet = dolfinx_mpc.MultiPointConstraint(V_tet, dtype=default_scalar_type)
mpc_hex = dolfinx_mpc.MultiPointConstraint(V_hex, dtype=default_scalar_type)
mpc_body = dolfinx_mpc.MultiPointConstraint(W, dtype=default_scalar_type)
mpc_body.add_rbe3_geometrical(V_tet, lambda x: np.isclose(x[0], 1.0, atol=tol))
mpc_body.add_rbe3_topological(V_hex, 2, facets)
mpcs = [mpc_tet, mpc_hex, mpc_body]
dolfinx_mpc.finalize_multipointconstraints(mpcs)
# -

# ## Linear elasticity on each cube, and a load on the spider
# A force $F$ does work on the translation of the spider and a moment $M$ on its
# rotation, so the load is the linear form $F \cdot w_t + M \cdot w_\theta$ on the
# space of the spider. On a point mesh, `ufl.dx` evaluates its integrand at the
# point. Both cubes are clamped at their far ends.

# +
E, nu = 1.0e3, 0.3
mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def sigma(u):
    return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)


def elasticity(V):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx


force = np.array([0.0, 0.2, -1.0])
moment = np.array([0.0, 0.3, 0.1])
spider_load = fem.Constant(spiders, np.concatenate([force, moment]).astype(default_scalar_type))

a = [[elasticity(V_tet), None, None], [None, elasticity(V_hex), None], [None, None, None]]
L = [
    ufl.ZeroBaseForm((ufl.TestFunction(V_tet),)),
    ufl.ZeroBaseForm((ufl.TestFunction(V_hex),)),
    ufl.inner(spider_load, ufl.TestFunction(W)) * ufl.dx(spiders),
]

clamped_tet = fem.locate_dofs_geometrical(V_tet, lambda x: np.isclose(x[0], 0.0, atol=tol))
clamped_hex = fem.locate_dofs_geometrical(V_hex, lambda x: np.isclose(x[0], 2.2, atol=tol))
zero = np.zeros(3, dtype=default_scalar_type)
bcs = [fem.dirichletbc(zero, clamped_tet, V_tet), fem.dirichletbc(zero, clamped_hex, V_hex)]
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
# -

# ## Solve
# The blocks are assembled into one matrix and solved directly.

problem = dolfinx_mpc.LinearProblem(
    a, L, mpcs, bcs=bcs, kind="mpi", petsc_options_prefix="demo_rbe3_", petsc_options=petsc_options
)
u_tet, u_hex, spider = problem.solve()

# ## Checks
# The spider moves with the least-squares rigid fit of its feet. The feet of both
# cubes are gathered to compute the fit independently.

# +
feet_tet = fem.locate_dofs_geometrical(V_tet, lambda x: np.isclose(x[0], 1.0, atol=tol))
feet_hex = fem.locate_dofs_geometrical(V_hex, lambda x: np.isclose(x[0], 1.2, atol=tol))


def rigid_map(r):
    """B with B @ [t, theta] = t + theta x r."""
    return np.hstack([np.eye(3), [[0, r[2], -r[1]], [-r[2], 0, r[0]], [r[1], -r[0], 0]]])


def rigid_fit(u_feet, x_c):
    """Least-squares rigid motion (t, theta) of the feet about x_c, and the largest misfit."""
    pieces = []
    for V, u, feet in zip((V_tet, V_hex), u_feet, (feet_tet, feet_hex)):
        owned = feet[feet < V.dofmap.index_map.size_local]
        pieces.extend(comm.allgather((V.tabulate_dof_coordinates()[owned], u.x.array.reshape(-1, 3)[owned])))
    x = np.vstack([p[0] for p in pieces])
    u = np.vstack([p[1] for p in pieces])
    B = np.vstack([rigid_map(x_i - x_c) for x_i in x])
    fit = np.linalg.lstsq(B, u.reshape(-1), rcond=None)[0]
    return fit, np.abs(B @ fit - u.reshape(-1)).max()


fit, misfit = rigid_fit((u_tet, u_hex), x_c)
values = dolfinx_mpc.spider_values(spider, 0)
atol = 1e4 * np.finfo(default_real_type).eps
assert np.allclose(values, fit, atol=atol)
# -

# The load reaches the clamps: their reactions balance the force, and the moment
# of the load about the spider. A reaction is the residual of a cube's equations
# at its clamped dofs, evaluated with the solution.

# +


def clamp_reaction(V, u, clamped):
    """Force and moment about `x_c` that a clamp exerts on its cube."""
    u_ref = fem.Function(V, dtype=default_scalar_type)
    n_owned = V.dofmap.index_map.size_local * 3
    u_ref.x.array[:n_owned] = u.x.array[:n_owned]
    u_ref.x.scatter_forward()
    residual = fem.assemble_vector(fem.form(ufl.action(elasticity(V), u_ref), dtype=default_scalar_type))
    residual.scatter_reverse(la.InsertMode.add)
    owned = clamped[clamped < V.dofmap.index_map.size_local]
    f = residual.array.reshape(-1, 3)[owned]
    r = V.tabulate_dof_coordinates()[owned] - x_c
    return comm.allreduce(f.sum(axis=0), op=MPI.SUM), comm.allreduce(np.cross(r, f).sum(axis=0), op=MPI.SUM)


reactions = [clamp_reaction(V_tet, u_tet, clamped_tet), clamp_reaction(V_hex, u_hex, clamped_hex)]
reaction_force = sum(r[0] for r in reactions)
reaction_moment = sum(r[1] for r in reactions)
if comm.rank == 0:
    print(f"RBE3 spider: t = {values[:3]}, theta = {values[3:]}")
    print(f"Reaction force {reaction_force} (tetrahedra {reactions[0][0]}, hexahedra {reactions[1][0]})")
    print(f"Reaction moment about the spider {reaction_moment}, applied {moment}")
assert np.allclose(reaction_force, -force, atol=100 * atol)
assert np.allclose(reaction_moment, -moment, atol=100 * atol)
# -

# ## Compared with a rigid spider
# The same load through an RBE2 spider ties the feet to the spider's rigid
# motion, so the facing sides stay plane. With RBE3 they deform: the feet are off
# the rigid fit. Making the sides rigid can only stiffen the structure, so the
# work of the load, $F \cdot t + M \cdot \theta$, is smaller for RBE2.

# +
mpc_tet2 = dolfinx_mpc.MultiPointConstraint(V_tet, dtype=default_scalar_type)
mpc_tet2.add_rbe2_geometrical(lambda x: np.isclose(x[0], 1.0, atol=tol), W)
mpc_hex2 = dolfinx_mpc.MultiPointConstraint(V_hex, dtype=default_scalar_type)
mpc_hex2.add_rbe2_topological(2, facets, W)
mpcs2 = [mpc_tet2, mpc_hex2, dolfinx_mpc.MultiPointConstraint(W, dtype=default_scalar_type)]
dolfinx_mpc.finalize_multipointconstraints(mpcs2)
problem2 = dolfinx_mpc.LinearProblem(
    a, L, mpcs2, bcs=bcs, kind="mpi", petsc_options_prefix="demo_rbe3_rigid_", petsc_options=petsc_options
)
u_tet2, u_hex2, spider2 = problem2.solve()
fit2, misfit2 = rigid_fit((u_tet2, u_hex2), x_c)
values2 = dolfinx_mpc.spider_values(spider2, 0)
load = np.concatenate([force, moment])
if comm.rank == 0:
    print(f"Feet off the rigid fit: {misfit:.2e} (RBE3), {misfit2:.2e} (RBE2)")
    print(f"Work of the load: {load @ values:.4e} (RBE3), {load @ values2:.4e} (RBE2)")
assert misfit2 < atol < misfit
assert load @ values2 < load @ values
# -

# ## Following the motion of the spider
# As for the rigid spider, the relation is linear in the rotation and holds about
# the current positions. Under a larger load we apply it in increments, each about
# the configuration reached by the previous ones: solve for the displacement
# increment, move both cubes and the spider by it, and recompute the coefficients
# about the new positions with
# {py:meth}`update_rbe3<dolfinx_mpc.MultiPointConstraint.update_rbe3>`. The feet
# are masters for every configuration, so the matrix layout is reused.
#
# {py:func}`dolfinx_mpc.spider.move` moves a mesh by a displacement, through the
# space of its coordinate element, whose dofs in each cell are the cell's geometry
# nodes. Given the increment on the spider space, it moves each spider by its
# translation.

# +
num_steps = 10
spider_load.value[:] = np.concatenate([[0.0, 0.0, -8.0], [0.0, -1.0, 0.0]])
u_total = [
    np.zeros((V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, 3), dtype=default_scalar_type)
    for V in (V_tet, V_hex)
]
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
    for V, u in ((V_tet, u_total[0]), (V_hex, u_total[1])):
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
        # The spider follows the fit of its feet about the configuration of the increment
        fit, _ = rigid_fit((du_tet, du_hex), x_spider)
        values = dolfinx_mpc.spider_values(du_spider, 0)
        assert np.allclose(values, fit, atol=atol), f"Spider off the fit of its feet at step {step}"
        for domain, du, total in ((cube_tet, du_tet, u_total[0]), (cube_hex, du_hex, u_total[1])):
            dolfinx_mpc.spider.move(domain, du)
            total += du.x.array.reshape(-1, 3)[: len(total)]
        dolfinx_mpc.spider.move(spiders, du_spider)
        x_spider += values[:3].real
        mpc_body.update_rbe3()
    # Collective: every process gathers, only the root draws
    frame = comm.gather(pieces(x_spider), root=0)
    if comm.rank == 0:
        frames.append(frame)

if comm.rank == 0:
    print(f"Spider after {num_steps} increments: {x_spider}, moved by {x_spider - x_c}")
assert x_spider[2] < x_c[2] - 0.05
# -

# The facing sides bend under the spider, which follows their average motion.

# + tags=["hide-input"]
if comm.rank == 0:
    clim = [0.0, max(max(g["|u|"].max(initial=0.0) for g in grids) for grids, _ in frames[-1])]
    plotter = pyvista.Plotter(off_screen=True, window_size=[800, 500])
    plotter.open_gif(Path("demo_rbe3.py").with_suffix(".gif"), fps=3)
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

# The PETSc objects of the problems are freed, and those the garbage collector
# has released are cleaned up on every process together.

del problem, problem2
PETSc.garbage_cleanup(comm)

# <img src="./demo_rbe3.gif" alt="gif" class="bg-primary mb-1" width="800px">

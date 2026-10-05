# # A double pendulum: two beams on two pins
#
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# Two separately meshed elastic beams hang from each other as a double pendulum. Beam 1
# turns about a fixed pin through a bore at its top. A second pin, through a bore lower
# down in beam 1, carries beam 2, which hangs beside beam 1 and turns about that pin.
# We start by importing the required dependencies:

# + tags =["hide-input"]
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import gmsh
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, plot
from dolfinx.io import VTXWriter
from dolfinx.io import gmsh as gmshio

import dolfinx_mpc

# -


# ## The spiders
#
# The pins are not meshed. The surface of each bore is tied rigidly to a spider on the
# axis of its pin (RBE2, as in {doc}`demo_spider`), so the bores keep their shape. The
# spiders are points:
#
# - $x_P = (0, W_0/2, 0)$ on the fixed pin, at the middle of the upper bore of beam 1;
# - $x_A = (0, W_0/2, -0.8 L)$ and $x_B = (0, W_0 + s + W_1/2, -0.8 L)$ on the lower pin,
#   at the middle of the bores of beam 1, of width $W_0$, and of beam 2, of width $W_1$, a
#   gap $s$ apart, so $d = x_B - x_A = (0, (W_0 + W_1)/2 + s, 0)$ runs along the pin.
#
# The pins run along $y$ through these points, and the bores of the beams are cut around
# them. Each spider has six unknowns, its translation $t$ and its rotation $\theta$.
#
# - **The fixed pin** lets beam 1 turn about it, but not move: the translation of spider P
#   is pinned, $t_P = 0$, by Dirichlet conditions on its translation dofs. Its rotations
#   about $x$ and $z$ are pinned as well, as a pin through a bore holds them too, so only
#   the rotation about the pin, $\theta_P \cdot a$ with $a = (0, 1, 0)$, is free.
# - **The lower pin** lets beam 2 turn about it, relative to beam 1, and holds every other
#   relative motion. Beam 1 and beam 2 each have their own spider on it, A and B, joined
#   by a 0-D spring that is stiff in every direction except rotation about the pin.
#
# Each spider is the only point of its own spider mesh, made by
# {py:func}`create_spider_mesh<dolfinx_mpc.create_spider_mesh>`, and
# {py:func}`create_spider_pair<dolfinx_mpc.create_spider_pair>` relates spider A to
# spider B, for the spring.

# +
comm = MPI.COMM_WORLD
L, W_0, W_1, H, L_2 = 2.0, 0.4, 0.25, 0.3, 1.5
radius, gap = 0.08, 0.05
margin = 2.5 * radius

x_P = np.array([0.0, 0.5 * W_0, 0.0], dtype=default_real_type)
x_A = np.array([0.0, 0.5 * W_0, -0.8 * L], dtype=default_real_type)
x_B = np.array([0.0, W_0 + gap + 0.5 * W_1, -0.8 * L], dtype=default_real_type)
# Each pin, (spider, y0, y1), from y0 to y1 along y, a little beyond the beams it carries
pins = ((x_P, -0.05, W_0 + 0.05), (x_A, -0.05, W_0 + gap + W_1 + 0.05))
spiders_P, spiders_A, spiders_B = (dolfinx_mpc.create_spider_mesh(comm, x.reshape(1, 3)) for x in (x_P, x_A, x_B))
pair = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)
# -

# Next, we create the function spaces on the spider meshes, which should have six components,
# three for translation and three for rotation.

element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type)
W_P, W_A, W_B = (fem.functionspace(spiders, element) for spiders in (spiders_P, spiders_A, spiders_B))

# ## Meshing the two beams with bores
#
# Gravity points along $-z$. Beam 1 hangs from the fixed pin and occupies
# $[-H/2, H/2] \times [0, W_0] \times [-L, e]$, with bores of radius $r$ around the pins
# through spiders P and A. Beam 2, of length $L_2$ below its bore, occupies
# $[W_0 + s, W_0 + s + W_1]$ in $y$, beside beam 1, and hangs from a bore around the lower
# pin through spider B. They are meshed
# separately with second order cells: beam 1 with tetrahedra, beam 2 with hexahedra, by
# extruding a quadrilateral mesh of its side along the pin. The facets of bore $k$ are
# tagged $k + 1$.


# + tags=["hide-input"]
def beam(z0: float, z1: float, y0: float, y1: float, bores: list[float], size: float, name: str, hexahedra=False):
    """The beam [-H/2, H/2] x [y0, y1] x [z0, z1] with a bore along y through (x=0, b) for each b in
    `bores`, meshed with second order cells, and its facet tags."""
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    if comm.rank == 0:
        if hexahedra:
            # The side in the xy-plane, turned into the xz-plane at y = y0: (x, y, 0) -> (x, y0, y)
            side = gmsh.model.occ.addRectangle(-H / 2, z0, 0, H, z1 - z0)
            holes = [(2, gmsh.model.occ.addDisk(0, b, 0, radius, radius)) for b in bores]
            section, _ = gmsh.model.occ.cut([(2, side)], holes)
            gmsh.model.occ.rotate(section, 0, 0, 0, 1, 0, 0, np.pi / 2)
            gmsh.model.occ.translate(section, 0, y0, 0)
            layers = int(np.ceil((y1 - y0) / size))
            extruded = gmsh.model.occ.extrude(section, 0, y1 - y0, 0, numElements=[layers], recombine=True)
            block = [entity for entity in extruded if entity[0] == 3]
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
        else:
            box = gmsh.model.occ.addBox(-H / 2, y0, z0, H, y1 - y0, z1 - z0)
            cylinders = [(3, gmsh.model.occ.addCylinder(0, y0, b, 0, y1 - y0, 0, radius)) for b in bores]
            block, _ = gmsh.model.occ.cut([(3, box)], cylinders)
            gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [block[0][1]], tag=1)
        # A bore's surfaces lie within its circle in x and z
        surfaces: dict[int, list[int]] = {k + 1: [] for k in range(len(bores))}
        for _, surface in gmsh.model.getBoundary(block, oriented=False):
            xmin, _, zmin, xmax, _, zmax = gmsh.model.getBoundingBox(2, surface)
            for k, b in enumerate(bores):
                within = zmin > b - 1.1 * radius and zmax < b + 1.1 * radius
                if within and xmin > -1.1 * radius and xmax < 1.1 * radius:
                    surfaces[k + 1].append(surface)
        for tag, entities in surfaces.items():
            gmsh.model.addPhysicalGroup(2, entities, tag=tag)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
    data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3, dtype=default_real_type)
    gmsh.model.remove()
    return data.mesh, data.facet_tags


# -

# The bores are centred on the spiders' pins

beam_1, tags_1 = beam(-L, x_P[2] + margin, 0.0, W_0, [x_P[2], x_A[2]], 0.08, "beam_1")
beam_2, tags_2 = beam(
    x_B[2] - L_2, x_B[2] + margin, W_0 + gap, W_0 + W_1 + gap, [x_B[2]], 0.05, "beam_2", hexahedra=True
)

# The meshes of the beams, and the spider nodes as spheres, taken from the spider meshes.
# Each process makes a grid of the cells it owns, from the geometry of the mesh alone, and
# the pieces are gathered on one process. The cells are drawn curved, as the mesh's second
# order cells, with the edges of the first order cells through their vertices.
# In the collapsed section below you will find some utlities for plotting the meshes with pyvista.

# + tags = ["hide-input"]
pyvista.global_theme.allow_empty_mesh = True


def pyvista_ugrid(input: dolfinx.mesh.Mesh | dolfinx.fem.FunctionSpace) -> pyvista.UnstructuredGrid:
    """The cells of `mesh` that this process owns, on its geometry nodes."""
    if isinstance(input, dolfinx.mesh.Mesh):
        imap = input.topology.index_map(3)
    elif isinstance(input, dolfinx.fem.FunctionSpace):
        imap = input.mesh.topology.index_map(3)
    else:
        raise TypeError(f"Expected a Mesh or FunctionSpace, got {type(input)}")
    owned = np.arange(imap.size_local, dtype=np.int32)
    if isinstance(input, dolfinx.mesh.Mesh):
        return pyvista.UnstructuredGrid(*plot.vtk_mesh(input, 3, owned))
    else:
        return pyvista.UnstructuredGrid(*plot.vtk_mesh(input, owned))
    return


# The first order cell with the vertices of each Lagrange cell
first_order = {
    pyvista.CellType.LAGRANGE_HEXAHEDRON: (pyvista.CellType.HEXAHEDRON, 8),
    pyvista.CellType.LAGRANGE_TETRAHEDRON: (pyvista.CellType.TETRA, 4),
}


def cell_edges(grid: pyvista.UnstructuredGrid) -> pyvista.UnstructuredGrid:
    """The grid of the first order cells through the vertices of the Lagrange cells of `grid`, on its
    points, for drawing the cells' edges: pyvista draws the edges of a subdivision of a Lagrange cell.
    A VTK Lagrange cell lists its vertices first."""
    if grid.n_cells == 0:
        return grid
    cell_type, num_nodes = first_order[pyvista.CellType(grid.celltypes[0])]
    nodes = grid.cells.reshape(grid.n_cells, -1)[:, 1 : 1 + num_nodes]
    cells = np.hstack([np.full((grid.n_cells, 1), num_nodes), nodes]).ravel()
    return pyvista.UnstructuredGrid(cells, np.full(grid.n_cells, cell_type, dtype=np.uint8), grid.points)


def add_cells(plotter: pyvista.Plotter, grid: pyvista.UnstructuredGrid, **kwargs):
    """Draw the Lagrange cells of `grid`, and their edges."""
    plotter.add_mesh(grid, **kwargs)
    plotter.add_mesh(cell_edges(grid), style="wireframe", color="black", line_width=0.5)


beam_pieces = comm.gather([pyvista_ugrid(mesh) for mesh in (beam_1, beam_2)], root=0)
spider_nodes = comm.gather(
    np.vstack(
        [
            spiders.geometry.x[: spiders.topology.index_map(0).size_local]
            for spiders in (spiders_P, spiders_A, spiders_B)
        ]
    ),
    root=0,
)
if comm.rank == 0:
    plotter = pyvista.Plotter(window_size=(700, 900))
    # Merging the points shared by the processes leaves no face between their pieces
    for b, colour in enumerate(("lightsteelblue", "wheat")):
        add_cells(
            plotter, pyvista.merge([piece[b] for piece in beam_pieces]).clean(tolerance=1e-6), color=colour, opacity=0.6
        )
    for node in np.vstack(spider_nodes):
        plotter.add_mesh(pyvista.Sphere(radius=0.05, center=node), color="red")
    plotter.camera_position = [(4.5, -6.0, -0.4), (0.0, 0.45, -1.45), (0.0, 0.0, 1.0)]
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_spider_hinge_setup.png")
    else:
        plotter.show()
# -

# ## The constraints
#
# The displacement of each beam is in a space of second order vector Lagrange elements.
# The upper bore of beam 1 follows spider P, its lower bore spider A, and the bore of
# beam 2 spider B. Spiders A and B have no constraint of their own, and we therefore create
# empty {py:class}`MultiPointConstraint<dolfinx_mpc.MultiPointConstraint>`s for them.
#
# The pinned dofs of spider P, its translation and its rotations about $x$ and $z$, are
# masters of the upper bore. Their Dirichlet conditions are therefore given to spider P's
# constraint, which removes them from the bore's relation, as well as to the problems
# below, which set them.

# +
V_1 = fem.functionspace(beam_1, ("Lagrange", 2, (3,)))
V_2 = fem.functionspace(beam_2, ("Lagrange", 2, (3,)))

mpc_1 = dolfinx_mpc.MultiPointConstraint(V_1, dtype=default_scalar_type)
mpc_1.add_rbe2_topological(2, tags_1.find(1), W_P)
mpc_1.add_rbe2_topological(2, tags_1.find(2), W_A)

mpc_2 = dolfinx_mpc.MultiPointConstraint(V_2, dtype=default_scalar_type)
mpc_2.add_rbe2_topological(2, tags_2.find(1), W_B)
# The translation (0, 1, 2) and the rotations about x and z (3, 5) of spider P
cells_P = np.arange(spiders_P.topology.index_map(0).size_local, dtype=np.int32)
zero = default_scalar_type(0.0)
pinned = [
    fem.dirichletbc(zero, fem.locate_dofs_topological(W_P.sub(i), 0, cells_P), W_P.sub(i)) for i in (0, 1, 2, 3, 5)
]
mpc_P = dolfinx_mpc.MultiPointConstraint(W_P, dtype=default_scalar_type, bcs=pinned)
mpcs = [mpc_1, mpc_2, mpc_P] + [dolfinx_mpc.MultiPointConstraint(V, dtype=default_scalar_type) for V in (W_A, W_B)]

dolfinx_mpc.finalize_multipointconstraints(mpcs)
# -

# (demo-spider-hinge-spring-coupling)=
# ## The spring coupling
#
# With $w = (t, \theta)$ the translation and rotation of a spider, the two spiders of
# the lower pin, at $x_A$ in beam 1 and $x_B$ in beam 2, are held together if spider B
# moves as a rigid extension of spider A, $t_B = t_A + \theta_A \times d$ with
# $d = x_B - x_A$, and $\theta_B = \theta_A$. The spring stores
#
# $$
# \frac12 \delta \cdot K \delta, \qquad
# \delta = \begin{pmatrix} t_B - t_A - \theta_A \times d \\ \theta_B - \theta_A \end{pmatrix},
# \qquad
# K = \begin{pmatrix} k I & 0 \\ 0 & k (I - a a^T) + k_t a a^T \end{pmatrix},
# $$
#
# with $a$ the direction of the pin, $k$ large, and $k_t$ the stiffness against turning
# about the pin, zero for a free pin. The fixed pin is held by its Dirichlet conditions,
# so its spring only acts against turning, $K_P = k_t\, a a^T$ on the rotation of spider P.
# With $z_P$, $z_A$ and $z_B$ the test functions of the spiders, the springs of both pins
# form the bilinear form
#
# $$
# s(u, v) = \int K \delta(w_A, w_B) \cdot \delta(z_A, z_B) + \int K_P w_P \cdot z_P,
# $$
#
# the first integrated over spider mesh A, reaching spider B through the pair, the second
# over spider mesh P, where $u$ and $v$ hold the unknowns and the test functions of the
# whole system, below. The function `springs` builds it for given stiffnesses. Both
# springs are locked, $k_t = k$, for the static deflection, and free, $k_t = 0$, for the
# swing, the two stages of the solution below.

# +
axis = np.array([0.0, 1.0, 0.0])
d = fem.Constant(spiders_A, (x_B - x_A).astype(default_scalar_type))


def spring_gap(w_A, w_B):
    """The motion of spider B away from the rigid extension of spider A."""
    t_A, theta_A = ufl.as_vector([w_A[i] for i in range(3)]), ufl.as_vector([w_A[i] for i in range(3, 6)])
    t_B, theta_B = ufl.as_vector([w_B[i] for i in range(3)]), ufl.as_vector([w_B[i] for i in range(3, 6)])
    translation = t_B - t_A - ufl.cross(theta_A, d)
    rotation = theta_B - theta_A
    return ufl.as_vector([translation[i] for i in range(3)] + [rotation[i] for i in range(3)])


def spring_stiffness(k_t: float) -> np.ndarray:
    """The 6x6 stiffness of the spring, with `k_t` against turning about the pin."""
    K = np.zeros((6, 6))
    K[:3, :3] = k * np.eye(3)
    K[3:, 3:] = k * (np.eye(3) - np.outer(axis, axis)) + k_t * np.outer(axis, axis)
    return K.astype(default_scalar_type)


def torsion(k_t: float) -> np.ndarray:
    """The 6x6 stiffness of the fixed pin against turning about it, its other dofs being pinned."""
    K_P = np.zeros((6, 6))
    K_P[3:, 3:] = k_t * np.outer(axis, axis)
    return K_P.astype(default_scalar_type)


def springs(K_pins, w, z):
    """The springs of both pins, s(w, z), with the stiffnesses `K_pins`, for the spider parts
    `w = (w_P, w_A, w_B)` and `z` of the unknowns and the test functions."""
    (w_P, w_A, w_B), (z_P, z_A, z_B) = w, z
    lower = ufl.inner(K_pins[0] * spring_gap(w_A, w_B), spring_gap(z_A, z_B)) * ufl.dx(spiders_A)
    return lower + ufl.inner(K_pins[1] * w_P, z_P) * ufl.dx(spiders_P)


# -

# ## The variational problem
#
# On each beam $\Omega_b$, $b = 1, 2$, the displacement $u_b$ satisfies linear
# elastodynamics,
#
# $$
# \rho_b \ddot u_b - \nabla \cdot \sigma_b(u_b) = \rho_b g, \qquad
# \sigma_b(u) = 2 \mu_b \varepsilon(u) + \lambda_b \operatorname{tr} \varepsilon(u) I,
# $$
#
# with the density $\rho_b$ and the Lamé parameters of beam $b$, from its Young's modulus
# $E_b$ and Poisson's ratio $\nu_b$, and gravity $g$. The bores of each beam follow its
# spiders, and the spiders act on each other, and on the ground, only through the
# springs. With $u = (u_1, u_2, w_P, w_A, w_B)$ and test functions
# $v = (v_1, v_2, z_P, z_A, z_B)$, the weak form is
#
# $$
# \begin{aligned}
# M(\ddot u, v) + a(u, v) + s(u, v) &= F(v),\\
# M(u, v) &= \sum_b \int_{\Omega_b} \rho_b u_b \cdot v_b,\\
# a(u, v) &= \sum_b \int_{\Omega_b} \sigma_b(u_b) : \varepsilon(v_b),\\
# F(v) &= \sum_b \int_{\Omega_b} \rho_b g \cdot v_b,
# \end{aligned}
# $$
#
# with $s$ the springs of {ref}`demo-spider-hinge-spring-coupling`, above.
#
# ## Discretization
#
# ### In space
#
# Both beams are discretized in the same way, with second-order vector Lagrange elements
# on their second-order meshes, of tetrahedra and of hexahedra respectively. Each spider
# has six unknowns, $w = (t, \theta)$: three translations $t$ and three rotations
# $\theta$.
#
# #### Pre-stressing the beams
#
# Standard linear elasticity evaluates forces on the undeformed geometry. If we decompose
# the displacement of a beam into a rigid translation and a rotation about its pin,
# $u = t + \theta \times r$ with $r = x - x_P$, the linear strain of the pure rotation
# $\theta \times r$ is exactly zero, i.e., $a(\theta \times r, v) = 0$. This means that a
# rigid rotation does not generate any internal resistance or restoring force to the swing
# motion. Furthermore, because external loads are integrated over the undeformed beam,
# the computed moment of gravity remains constant rather than updating to reflect the
# displaced position $x + u$.
#
# Instead, we rely on stress stiffening. Physically, when a pendulum swings sideways,
# it moves in an arc, meaning its mass must lift slightly upward against the downward
# pull of gravity. This means work is done against the initial tension (prestress $\sigma_0$)
# in the hanging beam.
#
# Mathematically, this is captured by the nonlinear portion of the strain tensor.
# While a pure rotation produces zero linear strain, it produces a nonzero second-order
# strain, $\frac12 \nabla u^T \nabla u$. Rotating the beam against the initial tension
# $\sigma_0$ stores potential energy equal to $\int \sigma_0 : (\frac12 \nabla u^T \nabla u)$.
# Because the spatial gradient $\nabla u$ is nonzero for a rotation, this interaction
# provides the restoring potential of the pendulum to second order in $\theta$. Its
# second variation is the geometric stiffness
#
# $$
# a_{\sigma_0}(u, v) = \sum_b \int_{\Omega_b} \nabla u_b \, \sigma_{0,b} : \nabla v_b.
# $$
#
# By baking this energy into the model, the pendulum can be solved efficiently within
# a linear framework in two stages:
#
# 1. The static deflection $u_0$ under gravity, $a(u_0, v) + s_\text{locked}(u_0, v) = F(v)$,
#    calculates the initial stress $\sigma_0 = \sigma_b(u_{0,b})$ on each beam. Because the
#    beams are symmetric about $x = 0$, gravity exerts no initial moment. This allows us
#    to compute the deflection with the pins safely locked against turning, ensuring the
#    static problem is well-posed.
#
# 2. Small motions $u$ about this hanging state, now with the pins free, satisfy the
#    equation $M(\ddot u, v) + \hat K(u, v) = 0$. The augmented stiffness $\hat K$ includes
#    the geometric stiffness of the prestress:
#
#    $$
#    \hat K(u, v) = a(u, v) + a_{\sigma_0}(u, v) + s_\text{free}(u, v).
#    $$
#
#    Because gravity is already balanced by $\sigma_0$, it does not appear as a body force
#    in this dynamic stage.
#
# ### In time
#
# The whole coupled system, beams and spiders alike, is stepped with one time step
# $\Delta t$. To step it, the second order equation $M \ddot u + \hat K u = 0$, with $M$
# and $\hat K$ the matrices of the forms above acting on the vectors of the unknowns, is
# written as a first order system, with the velocity $\dot u$ as an unknown of its own:
#
# $$
# \frac{\mathrm{d} u}{\mathrm{d} t} = \dot u, \qquad
# M \frac{\mathrm{d} \dot u}{\mathrm{d} t} = -\hat K u.
# $$
#
# The implicit midpoint rule replaces each time derivative by the difference of the values
# at the two ends of a step, $u^n$ at time $n \Delta t$ and $u^{n+1}$ at
# $(n + 1) \Delta t$, divided by $\Delta t$, and evaluates the right-hand side at the middle
# of the step, as the average of its values at the two ends:
#
# $$
# \text{(i)} \quad \frac{u^{n+1} - u^n}{\Delta t} = \frac{\dot u^{n+1} + \dot u^n}{2},
# \qquad
# \text{(ii)} \quad M \frac{\dot u^{n+1} - \dot u^n}{\Delta t} = -\hat K \frac{u^{n+1} + u^n}{2}.
# $$
#
# Equation (i) gives the new velocity from the new displacement,
#
# $$
# \text{(iii)} \quad \dot u^{n+1} = \frac{2}{\Delta t}\left(u^{n+1} - u^n\right) - \dot u^n.
# $$
#
# Inserting (iii) into (ii) removes $\dot u^{n+1}$:
#
# $$
# M \frac{1}{\Delta t}\left(\frac{2}{\Delta t}\left(u^{n+1} - u^n\right) - 2 \dot u^n\right)
# = -\frac12 \hat K \left(u^{n+1} + u^n\right),
# $$
#
# and multiplying by 2 and gathering the unknown $u^{n+1}$ on the left leaves one linear
# system for the new displacement,
#
# $$
# \text{(iv)} \quad \left(\frac{4}{\Delta t^2} M + \hat K\right) u^{n+1}
# = \frac{4}{\Delta t^2} M u^n + \frac{4}{\Delta t} M \dot u^n - \hat K u^n.
# $$
#
# A step is therefore:
#
# 1. assemble the right-hand side of (iv) from $u^n$ and $\dot u^n$;
# 2. solve (iv) for $u^{n+1}$, whose matrix is the same in every step, so it is assembled
#    and factorized once;
# 3. update the velocity with (iii).
#
# The pendulum is released at rest, $\dot u^0 = 0$, from the displacement $u^0$ given
# below.
#
# ## Energy conservation
# The total energy of the system at time step $n$ can be written as the sum of the
# kinetic and stored energy:
#
# $$
# E^n = \frac12 \dot u^n \cdot M \dot u^n + \frac12 u^n \cdot \hat K u^n
# $$
#
# We can derive that the energy is conserved with our discrete time stepping scheme by
# multiply (ii) by $\Delta t$ and take its dot product with the average velocity over the step,
# $\frac12 (\dot u^{n+1} + \dot u^n)$:
#
# $$
# \frac12 \left(\dot u^{n+1} + \dot u^n\right) \cdot M \left(\dot u^{n+1} - \dot u^n\right)
# = -\frac{\Delta t}{2} \left(\dot u^{n+1} + \dot u^n\right) \cdot \frac12 \hat K \left(u^{n+1} + u^n\right).
# $$
#
# On the left, expanding the product gives four terms, and the two mixed ones cancel, as
# $\dot u^{n+1} \cdot M \dot u^n = \dot u^n \cdot M \dot u^{n+1}$ for the symmetric $M$:
#
# $$
# \begin{aligned}
# \frac12 \left(\dot u^{n+1} + \dot u^n\right) \cdot M \left(\dot u^{n+1} - \dot u^n\right)
# &= \frac12 \dot u^{n+1} \cdot M \dot u^{n+1} - \frac12 \dot u^n \cdot M \dot u^n,
# \end{aligned}
# $$
#
# which gives us the change of the kinetic energy over the step of the left side.
# On the right, (i) says that $\frac{\Delta t}{2} (\dot u^{n+1} + \dot u^n) = u^{n+1} - u^n$,
# so the right-hand side is $-\frac12 (u^{n+1} - u^n) \cdot \hat K (u^{n+1} + u^n)$, and the
# same expansion with the symmetric $\hat K$ gives
#
# $$
# \begin{aligned}
# -\frac12 \left(u^{n+1} - u^n\right) \cdot \hat K \left(u^{n+1} + u^n\right)
# &= -\left(\frac12 u^{n+1} \cdot \hat K u^{n+1} - \frac12 u^n \cdot \hat K u^n\right),
# \end{aligned}
# $$
#
# minus the change of the stored energy. The two sides are equal, so
#
# $$
# \frac12 \dot u^{n+1} \cdot M \dot u^{n+1} + \frac12 u^{n+1} \cdot \hat K u^{n+1}
# = \frac12 \dot u^n \cdot M \dot u^n + \frac12 u^n \cdot \hat K u^n,
# $$
#
# that is $E^{n+1} = E^n$: the kinetic energy gained in a step is the stored energy lost,
# for any $\Delta t$. In floating point it holds up to the rounding of the solves, which
# the demo checks below.

# +
materials = {"E": (2.0e4, 0.7e5), "nu": (0.3, 0.25), "rho": (1.0, 1.2)}
beams = (beam_1, beam_2)
g = 9.81
dt, num_steps = 0.02, 150


def sigma(u, b: int):
    """The stress of displacement `u` in the material of beam `b`."""
    E, nu = materials["E"][b], materials["nu"][b]
    mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)


# The stiffness of the springs, large next to the beams', and of the lower pin's spring, on spider
# mesh A, and of the fixed pin's, on spider mesh P, locked for the static deflection and free for the swing
k = 1e3 * max(materials["E"])
K_locked = (fem.Constant(spiders_A, spring_stiffness(k)), fem.Constant(spiders_P, torsion(k)))
K_free = (fem.Constant(spiders_A, spring_stiffness(0.0)), fem.Constant(spiders_P, torsion(0.0)))
# -

# ## The forms
#
# The forms are written over the five spaces of the constraints, in the order of the
# blocks of the system, and split into blocks when compiled, with the names of the text:
# `M`, `a`, `s_locked`, `s_free`, `F`, `a_sigma_0` and `K_hat`. The prestress `sigma_0` is
# written in terms of the functions `u_0` that will hold the static deflection, and the
# right-hand side of a step in terms of those holding $u^n$ and $\dot u^n$, `u_n` and
# `u_dot_n`, all in the constraints' spaces.

# +
mixed = ufl.MixedFunctionSpace(*(mpc.input_space for mpc in mpcs))
trials, tests = ufl.TrialFunctions(mixed), ufl.TestFunctions(mixed)
u_1, u_2, w_P, w_A, w_B = trials
v_1, v_2, z_P, z_A, z_B = tests
dx_1, dx_2 = ufl.dx(beam_1), ufl.dx(beam_2)


def in_constraint_spaces():
    """One function per block, in the space of its constraint."""
    return [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]


a = ufl.inner(sigma(u_1, 0), ufl.sym(ufl.grad(v_1))) * dx_1
a += ufl.inner(sigma(u_2, 1), ufl.sym(ufl.grad(v_2))) * dx_2
M = materials["rho"][0] * ufl.inner(u_1, v_1) * dx_1 + materials["rho"][1] * ufl.inner(u_2, v_2) * dx_2
s_locked = springs(K_locked, trials[2:], tests[2:])
s_free = springs(K_free, trials[2:], tests[2:])

# The static deflection, a(u_0, v) + s_locked(u_0, v) = F(v), with gravity, which has no spider blocks
down = np.array([0.0, 0.0, -1.0])
gravity = [fem.Constant(m, (rho * g * down).astype(default_scalar_type)) for m, rho in zip(beams, materials["rho"])]
F = [
    ufl.inner(gravity[0], v_1) * dx_1,
    ufl.inner(gravity[1], v_2) * dx_2,
    ufl.ZeroBaseForm((z_P,)),
    ufl.ZeroBaseForm((z_A,)),
    ufl.ZeroBaseForm((z_B,)),
]

# The swing: the prestress sigma_0 of the static deflection u_0, and free pins
u_0 = in_constraint_spaces()
sigma_0 = (sigma(u_0[0], 0), sigma(u_0[1], 1))
a_sigma_0 = ufl.inner(ufl.grad(u_1) * sigma_0[0], ufl.grad(v_1)) * dx_1
a_sigma_0 += ufl.inner(ufl.grad(u_2) * sigma_0[1], ufl.grad(v_2)) * dx_2
K_hat = a + a_sigma_0 + s_free
# The displacement u^n, the velocity u_dot^n and the new displacement u^{n+1}
u_n, u_dot_n, u_new = in_constraint_spaces(), in_constraint_spaces(), in_constraint_spaces()


def acting(form, functions):
    """The linear form of `form` with its trial functions replaced by `functions`."""
    return ufl.replace(form, dict(zip(trials, functions)))


step_lhs = 4 / dt**2 * M + K_hat
step_rhs = acting(4 / dt**2 * M, u_n) + acting(4 / dt * M, u_dot_n) - acting(K_hat, u_n)
# -

# ## The hanging pendulum
#
# The static deflection, with the pins locked, into the functions of the prestress.

# +
static = dolfinx_mpc.LinearProblem(
    ufl.extract_blocks(a + s_locked),
    F,
    mpcs,
    bcs=pinned,
    u=u_0,
    kind="mpi",
    entity_maps=[pair],
    petsc_options_prefix="demo_spider_hinge_static_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
)
u_1_0, u_2_0, w_P_0, w_A_0, w_B_0 = static.solve()
# -

# The spring holds beam 2 up, $K \delta$ balancing its weight, and carries no moment
# about the pin. The mass, centre of mass and moment of inertia about its pin of each
# beam, with its bores, are integrated over its mesh, for the checks here and below.

# +


def integrate(form) -> float:
    return comm.allreduce(fem.assemble_scalar(fem.form(form, dtype=default_scalar_type)), op=MPI.SUM).real


mass, centre, inertia = [], [], []
for b, (m, x_pivot) in enumerate(((beam_1, x_P), (beam_2, x_B))):
    x = ufl.SpatialCoordinate(m)
    rho = materials["rho"][b]
    mass.append(integrate(rho * ufl.dx(m)))
    centre.append(np.array([integrate(rho * x[i] * ufl.dx(m)) for i in range(3)]) / mass[-1])
    arm = ufl.as_vector([x[0] - x_pivot[0], x[2] - x_pivot[2]])
    inertia.append(integrate(rho * ufl.inner(arm, arm) * ufl.dx(m)))

values_A, values_B = dolfinx_mpc.spider_values(w_A_0, 0).real, dolfinx_mpc.spider_values(w_B_0, 0).real
delta = np.concatenate([values_B[:3] - values_A[:3] - np.cross(values_A[3:], x_B - x_A), values_B[3:] - values_A[3:]])
spring_force = (spring_stiffness(k).real @ delta)[:3]
spring_moment = np.dot((spring_stiffness(k).real @ delta)[3:], axis)
weight_2 = mass[1] * g * down
lever_2 = mass[1] * g * (x_B[2] - centre[1][2])
tol = max(1e-3, 2e5 * np.finfo(default_real_type).eps)
if comm.rank == 0:
    print(f"Masses {mass[0]:.4f}, {mass[1]:.4f}; centres {centre[0].round(4)}, {centre[1].round(4)}")
    print(f"Spring force {spring_force.round(5)} next to the weight of beam 2 {weight_2.round(5)}")
    print(f"Spring moment about the pin {spring_moment:.2e} next to m_2 g c_2 = {lever_2:.2e}")
assert np.allclose(spring_force, weight_2, atol=tol * np.abs(weight_2).max())
assert abs(spring_moment) < tol * lever_2
# -

# ## The pendulum at rest, seen
#
# The deflection under gravity, magnified, with the pins in translucent red, so that the
# bores around them show.

# +
grids = []
for V, u in ((V_1, u_1_0), (V_2, u_2_0)):
    grid = pyvista_ugrid(V)
    grid["u"] = u.x.array.reshape(-1, 3)[: grid.n_points].real
    grids.append(grid)
gathered = comm.gather(grids, root=0)
if comm.rank == 0:
    plotter = pyvista.Plotter(window_size=(700, 900))
    # One grid per beam, from the pieces of all processes, so that no partition boundary is drawn
    merged = [
        pyvista.merge([piece[b] for piece in gathered if piece[b].n_points > 0]).clean(tolerance=1e-6) for b in range(2)
    ]
    factor = 0.1 / max(np.abs(grid["u"]).max() for grid in merged)
    for grid in merged:
        grid["|u|"] = np.linalg.norm(grid["u"], axis=1)
        add_cells(plotter, grid.warp_by_vector("u", factor=factor), scalars="|u|")
    for x_c, y0, y1 in pins:
        pin = pyvista.Cylinder(center=(x_c[0], 0.5 * (y0 + y1), x_c[2]), direction=axis, radius=radius, height=y1 - y0)
        # Translucent, so that the bore around the pin shows
        plotter.add_mesh(pin, color="red", opacity=0.5)
    plotter.camera_position = [(4.5, -6.0, -0.4), (0.0, 0.45, -1.45), (0.0, 0.0, 1.0)]
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_spider_hinge.png")
    else:
        plotter.show()
# -

# ## Assembling the step
#
# The matrix of a step does not change, so it is assembled and factorized once.

# +
a_step = fem.form(ufl.extract_blocks(step_lhs), entity_maps=[pair], dtype=default_scalar_type)
L_step = fem.form(ufl.extract_blocks(step_rhs), entity_maps=[pair], dtype=default_scalar_type)
A = dolfinx_mpc.assemble_matrix(a_step, mpcs, bcs=pinned, kind="mpi")
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setErrorIfNotConverged(True)
solver.setType("preonly")
solver.getPC().setType("lu")
solver.getPC().setFactorSolverType("mumps")
b = dolfinx_mpc.create_vector(L_step, mpcs, kind="mpi")
x = dolfinx_mpc.create_vector(L_step, mpcs, kind="mpi")
# The pinned dofs, by block, which are zero in every step
pinned_by_block = fem.bcs_by_block([mpc.input_space for mpc in mpcs], pinned)
# -

# The energy, as one form per domain: the two beams, and the springs on spider meshes A and P.
# Spring A reaches spider B through the pair.

# +


def energy_forms(u, u_dot):
    """The forms of the energy of the motion `u`, with velocity `u_dot`, one per domain."""
    forms = []
    for b, dx_b in enumerate((dx_1, dx_2)):
        kinetic = 0.5 * materials["rho"][b] * ufl.inner(u_dot[b], u_dot[b])
        elastic_b = 0.5 * ufl.inner(sigma(u[b], b), ufl.sym(ufl.grad(u[b])))
        geometric_b = 0.5 * ufl.inner(ufl.grad(u[b]) * sigma_0[b], ufl.grad(u[b]))
        forms.append((kinetic + elastic_b + geometric_b) * dx_b)
    forms.append(0.5 * ufl.inner(K_free[0] * spring_gap(u[3], u[4]), spring_gap(u[3], u[4])) * ufl.dx(spiders_A))
    forms.append(0.5 * ufl.inner(K_free[1] * u[2], u[2]) * ufl.dx(spiders_P))
    return [fem.form(form, entity_maps=[pair], dtype=default_scalar_type) for form in forms]


def energy(forms) -> float:
    return sum(comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM).real for form in forms)


energies = energy_forms(u_n, u_dot_n)
# -

# ## Releasing the pendulum
#
# The pendulum starts at rest, beam 1 turned by $\theta_1 = 10°$ about the fixed pin and
# beam 2 by a further $\theta_2 = -10°$ about the lower pin: the rigid motions
# $\theta_1 a \times (x - x_P)$, and $\theta_2 a \times (x - x_B)$ on top of it for beam 2
# and spider B. They satisfy the constraints and leave the springs unstretched, so the
# energy they start with is that of $\sigma_0$ alone, the pendulum's potential.

# +


def turn(theta: float, x_c: np.ndarray):
    """The rigid rotation by `theta` about the pin through `x_c`, at points `x`."""
    return lambda x: theta * np.cross(axis, (x - x_c[:, None]).T).T


def rigid_turns(phi_1: float, phi_2: float, u):
    """Set `u`, on the five spaces, to beam 1 turned by `phi_1` about the fixed pin and beam 2 by a
    further `phi_2` about the lower pin."""
    u[0].interpolate(turn(phi_1, x_P))
    u[1].interpolate(lambda x: turn(phi_1, x_P)(x) + turn(phi_2, x_B)(x))
    spiders = (
        (np.zeros(3), phi_1 * axis),
        (phi_1 * np.cross(axis, x_A - x_P), phi_1 * axis),
        (phi_1 * np.cross(axis, x_B - x_P), (phi_1 + phi_2) * axis),
    )
    for f, (t, theta) in zip(u[2:], spiders):
        f.x.array[:] = np.tile(np.concatenate([t, theta]), f.x.array.size // 6)


theta_1, theta_2 = np.deg2rad(10.0), np.deg2rad(-10.0)
rigid_turns(theta_1, theta_2, u_n)

# -

# ## The pins, for viewing
#
# Both pins are meshed, as two cylinders of the bores' radius in one mesh, with the cells of
# each tagged by its pin, only for viewing. Each moves as one rigid body with its spider,
# $t + \theta \times (x - x_c)$: the fixed pin with spider P, the lower pin with spider A.


# +
def pin_mesh(size: float):
    """The pins as cylinders along y, `(centre, y0, y1)` each, of second order, the cells of pin k
    tagged k + 1."""
    gmsh.model.add("pins")
    if comm.rank == 0:
        for tag, (x_c, y0, y1) in enumerate(pins, start=1):
            cylinder = gmsh.model.occ.addCylinder(x_c[0], y0, x_c[2], 0, y1 - y0, 0, radius)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [cylinder], tag=tag)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
    data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3, dtype=default_real_type)
    gmsh.model.remove()
    return data.mesh, data.cell_tags


pin_domain, pin_tags = pin_mesh(0.05)
V_pins = fem.functionspace(pin_domain, ("Lagrange", 2, (3,)))
u_pins = fem.Function(V_pins, dtype=default_scalar_type, name="u")


def move_pins(u):
    """Move each pin rigidly with its spider in `u`. Collective."""
    for tag, (k, (x_c, _, _)) in enumerate(zip((2, 3), pins), start=1):
        values = dolfinx_mpc.spider_values(u[k], 0)
        t, theta = values[:3], values[3:]
        u_pins.interpolate(
            lambda x: (t[:, None] + np.cross(theta, (x - x_c[:, None]).T).T).astype(default_scalar_type),
            cells0=pin_tags.find(tag),
        )
    u_pins.x.scatter_forward()


for k in range(2):
    u_n[k].name = "u"
writers = [VTXWriter(comm, f"demo_spider_hinge_{k + 1}.bp", [u_n[k]], engine="BP4") for k in range(2)]
writers.append(VTXWriter(comm, "demo_spider_hinge_pins.bp", [u_pins], engine="BP4"))


def spider_turns(u) -> tuple[float, float]:
    """The turns of beam 1, at spider P, and of beam 2, at spider B, about the pins."""
    return tuple(float(np.dot(dolfinx_mpc.spider_values(u[k], 0).real[3:], axis)) for k in (2, 4))


# -

# ## The animation
#
# As in {doc}`demo_linear_wave_problem`, each process builds grids of the cells it owns,
# gathered once onto one process, as the meshes do not move; per frame only the nodal
# values are gathered. The beams and the pins are drawn displaced by the true
# displacement, the pins in translucent red.

# +
pyvista.OFF_SCREEN = True
shown = (u_n[0], u_n[1], u_pins)
local_grids = [pyvista_ugrid(u.function_space) for u in shown]
gathered_grids = comm.gather(local_grids, root=0)
if comm.rank == 0:
    # The pieces of every process, in rank order, in one grid per mesh
    grids = [pyvista.merge([piece[k] for piece in gathered_grids], merge_points=False) for k in range(3)]
    plotter = pyvista.Plotter(off_screen=True, window_size=(450, 600))
    plotter.open_gif("demo_spider_hinge.gif", fps=1 / (2 * dt))
    # Set from the first frame, and kept, so that the colours compare between frames
    clim: list[float] = []


def write_frame():
    """Draw the beams and the pins, displaced. Collective."""
    values = comm.gather([u.x.array.reshape(-1, 3)[: grid.n_points].real.copy() for u, grid in zip(shown, local_grids)])
    if comm.rank != 0:
        return
    plotter.clear()
    for k, grid in enumerate(grids):
        grid["u"] = np.vstack([piece[k] for piece in values])
        grid["|u|"] = np.linalg.norm(grid["u"], axis=1)
    if not clim:
        clim.extend([0.0, 1.2 * max(grid["|u|"].max() for grid in grids[:2])])
    for k, grid in enumerate(grids):
        if k < 2:
            add_cells(
                plotter, grid.warp_by_vector("u"), scalars="|u|", clim=clim, cmap="viridis", show_scalar_bar=False
            )
        else:
            plotter.add_mesh(grid.warp_by_vector("u"), color="red", opacity=0.5)
    plotter.camera_position = [(2.5, -7.0, -1.0), (0.0, 0.45, -1.4), (0.0, 0.0, 1.0)]
    plotter.write_frame()


history = [(0.0, *spider_turns(u_n))]
E_0 = energy(energies)
move_pins(u_n)
write_frame()
drift = 0.0
for writer in writers:
    writer.write(0.0)
for step in range(1, num_steps + 1):
    with b.localForm() as b_local:
        b_local.set(0.0)
    dolfinx_mpc.assemble_vector(L_step, mpcs, b)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, pinned_by_block)
    solver.solve(b, x)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(x, u_new)
    # Every block is homogenized before any is back-substituted, as spider masters are read across blocks
    for mpc, u in zip(mpcs, u_new):
        mpc.homogenize(u)
    for mpc in mpcs:
        mpc.backsubstitution(u_new)
    # (iii): u_dot^{n+1} = 2 / dt (u^{n+1} - u^n) - u_dot^n, then u^n = u^{n+1}
    for u, u_old, u_dot in zip(u_new, u_n, u_dot_n):
        u_dot.x.array[:] = 2 / dt * (u.x.array - u_old.x.array) - u_dot.x.array
        u_old.x.array[:] = u.x.array
    history.append((step * dt, *spider_turns(u_n)))
    move_pins(u_n)
    for writer in writers:
        writer.write(step * dt)
    if step % 2 == 0:
        write_frame()
    if step % 25 == 0:
        drift = max(drift, abs(energy(energies) / E_0 - 1))
        if comm.rank == 0:
            t, turn_1, turn_2 = history[-1]
            print(
                f"t = {t:.2f}: beam 1 at {np.rad2deg(turn_1):6.2f}°, beam 2 at {np.rad2deg(turn_2):6.2f}°,"
                f" |E / E_0 - 1| <= {drift:.1e}"
            )
for writer in writers:
    writer.close()
if comm.rank == 0:
    plotter.close()
# -

# The energy is conserved up to the rounding of the solves, which the stiff springs next to
# the mass make large.

assert drift < min(1e-2, 1e8 * np.finfo(default_real_type).eps)

# ## Against rigid double pendulums
#
# Rigid beams turning by $\varphi_1$ about the fixed pin and $\varphi_2$ about the lower
# pin, for small angles, satisfy $M_\varphi \ddot\varphi + K_\varphi \varphi = 0$. Two
# such models are stepped with the same midpoint rule as the beams, starting from the same
# turns.
#
# The first restricts the finite element model to the two rigid turns $R_1$ and $R_2$,
# $(M_\varphi)_{ij} = M(R_i, R_j)$ and $(K_\varphi)_{ij} = \hat K(R_i, R_j)$. It differs
# from the beams only by their elasticity, which beam 1, the softer, shows the most.
#
# The second is the textbook compound double pendulum,
#
# $$
# M_\varphi = \begin{pmatrix} I_1 + m_2 \ell^2 & m_2 \ell h_2 \\ m_2 \ell h_2 & I_2 \end{pmatrix},
# \qquad
# K_\varphi = g \begin{pmatrix} m_1 h_1 + m_2 \ell & 0 \\ 0 & m_2 h_2 \end{pmatrix},
# $$
#
# with $m_b$ the masses, $I_b$ the moments of inertia about each beam's own pin, $h_b$ the
# depths of the centres of mass below them, and $\ell = 0.8 L$ the distance between the
# pins, integrated over the meshes. Its masses are those of the first model, but not its
# stiffness: in the textbook model a pin pushes on its bore through its axis, while the
# feet of an RBE2 spider are rigidly attached to it, a bore radius $r$ away. Turning a bore
# that carries the weight hanging from it then changes the stiffness by a fraction of the
# order $r / h$, which the prestress $\sigma_0$ around the bore picks up. Over a few
# swings the textbook pendulum drifts out of phase with the beams.

# +
ell = x_P[2] - x_B[2]
h = (x_P[2] - centre[0][2], x_B[2] - centre[1][2])
textbook = (
    np.array([[inertia[0] + mass[1] * ell**2, mass[1] * ell * h[1]], [mass[1] * ell * h[1], inertia[1]]]),
    g * np.diag([mass[0] * h[0] + mass[1] * ell, mass[1] * h[1]]),
)

modes = [[fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs] for _ in range(3)]
# Beam 2 turned by phi_2 in all, so that both models use the absolute turns
rigid_turns(1.0, -1.0, modes[0])
rigid_turns(0.0, 1.0, modes[1])
zero = [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]


def quadratic(i: int, j: int, kinetic: bool) -> float:
    """M(R_i, R_j) if `kinetic`, else K(R_i, R_j), by polarization of the energy."""
    for f, f_i, f_j in zip(modes[2], modes[i], modes[j]):
        f.x.array[:] = f_i.x.array + f_j.x.array

    def twice_energy(u):
        return 2 * energy(energy_forms(zero, u) if kinetic else energy_forms(u, zero))

    return 0.5 * (twice_energy(modes[2]) - twice_energy(modes[i]) - twice_energy(modes[j]))


ritz = tuple(np.array([[quadratic(i, j, kinetic) for j in range(2)] for i in range(2)]) for kinetic in (True, False))


def swing(M_phi: np.ndarray, K_phi: np.ndarray) -> np.ndarray:
    """The turns of a rigid double pendulum, stepped as the beams were."""
    step_matrix = np.linalg.inv(4 / dt**2 * M_phi + K_phi)
    phi, phi_dot = np.array([theta_1, theta_1 + theta_2]), np.zeros(2)
    turns = [phi]
    for _ in range(num_steps):
        phi_new = step_matrix @ (4 / dt**2 * M_phi @ phi + 4 / dt * M_phi @ phi_dot - K_phi @ phi)
        phi_dot = 2 / dt * (phi_new - phi) - phi_dot
        phi = phi_new
        turns.append(phi)
    return np.array(turns)


turns = np.array(history)[:, 1:]
mismatch = {
    name: np.abs(turns - swing(*model)).max() / np.abs(turns).max()
    for name, model in (("ritz", ritz), ("textbook", textbook))
}
mass_difference = np.abs(ritz[0] - textbook[0]).max() / np.abs(textbook[0]).max()
stiffness_difference = np.diag(ritz[1]) / np.diag(textbook[1]) - 1
if comm.rank == 0:
    print(f"Masses, relative to the textbook pendulum's: {mass_difference:.1e}")
    print(f"Stiffness, relative to the textbook pendulum's: {stiffness_difference.round(3)}")
    print(
        f"Largest difference of the turns to the rigid pendulums: {mismatch['ritz']:.1%} (finite elements),"
        f" {mismatch['textbook']:.1%} (textbook)"
    )
# -

# The masses agree to rounding, the stiffnesses to the order of $r / h$, and the beams
# follow the rigid pendulum of the finite element model to a few percent.

assert mass_difference < 1e4 * np.finfo(default_real_type).eps
assert np.all(np.abs(stiffness_difference) < radius / min(h))
assert mismatch["ritz"] < 0.1

# <img src="./demo_spider_hinge.gif" alt="gif" class="bg-primary mb-1" width="600px">

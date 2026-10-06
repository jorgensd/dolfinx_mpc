# # A double pendulum: two beams on two pins
#
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# In this demo we will explore coupling two separately meshes elastic beams that are coupled to
# each other through a stiff spring, forming a double pendulum. Beam 1
# turns about a fixed pin through a bore at its top. A second pin, through a bore lower
# down in beam 1, carries beam 2, which hangs beside beam 1 and turns about that pin.
# All the pins are created by using RBE2 spiders, as in {doc}`demo_spider`,
# and the two spiders of the lower pin are coupled by a stiff spring.
# We start by importing the required dependencies:

# + tags =["hide-input"]
from pathlib import Path

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

# ## The beams
#
# Two beams hang from two pins that run along $y$; gravity points along $-z$. Each beam is a
# box of depth $H$ along $x$, centred on $x = 0$, with a bore of radius $r$ around each pin
# it hangs on or carries:
#
# - **Beam 1**, of width $W_0$ along the pins, $y \in [0, W_0]$, hangs from the fixed pin at
#   height $z = 0$ and reaches down to $z = -L$. A second bore, for the lower pin, is at
#   $z = -0.8 L$.
# - **Beam 2**, of width $W_1$, hangs beside beam 1, a gap $s$ away along the pins,
#   $y \in [W_0 + s, W_0 + s + W_1]$, from a bore on the lower pin, and reaches a length $L_2$
#   below it.
#
# Above its top bore, each beam has a margin $e = 2.5 r$ of material. On the axis of each
# pin, at the middle of each beam's bore, the points $x_P$, $x_A$ and $x_B$ are the spiders
# of the next section.

# +
comm = MPI.COMM_WORLD
L, W_0, H = 2.0, 0.4, 0.3  # Beam 1: length below its top bore, width along the pins, depth
L_2, W_1 = 1.5, 0.25  # Beam 2: length below its bore, width along the pins
radius, gap = 0.08, 0.05  # The bores' radius, and the gap between the beams along the pins
margin = 2.5 * radius  # The material above a beam's top bore
z_top, z_pin = 0.0, -0.8 * L  # The heights of the fixed pin and of the lower pin

# Each beam's extent along the pins (y) and along gravity (z), and the heights of its bores
beam_1_y, beam_1_z, beam_1_bores = (0.0, W_0), (-L, z_top + margin), [z_top, z_pin]
beam_2_y, beam_2_z, beam_2_bores = (W_0 + gap, W_0 + gap + W_1), (z_pin - L_2, z_pin + margin), [z_pin]

# The spiders, on the pins' axes at x = 0, at the middle of each bore along the pin
x_P = np.array([0.0, 0.5 * sum(beam_1_y), z_top], dtype=default_real_type)
x_A = np.array([0.0, 0.5 * sum(beam_1_y), z_pin], dtype=default_real_type)
x_B = np.array([0.0, 0.5 * sum(beam_2_y), z_pin], dtype=default_real_type)
# -


# ## The spiders
#
# The pins are rigid, and they are not part of the simulation mesh: neither the pins nor
# anything inside the bores is meshed. Instead, the surface of each bore is tied rigidly to
# a spider (RBE2, as in {doc}`demo_spider`), so that the bore keeps its shape and moves as
# one rigid body with its spider. The spiders are the points $x_P$, $x_A$ and $x_B$ defined
# above, on the pins' axes at the middle of each bore: $x_P$ in beam 1's upper bore, and
# $x_A$ and $x_B$ in the lower bores of beam 1 and beam 2, so that $d = x_B - x_A$ runs
# along the lower pin. Each spider has six unknowns, its translation $t$ and its rotation
# $\theta$.
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
spiders_P, spiders_A, spiders_B = (dolfinx_mpc.create_spider_mesh(comm, x.reshape(1, 3)) for x in (x_P, x_A, x_B))
pair = dolfinx_mpc.create_spider_pair(spiders_A, spiders_B)
# -

# Next, we create the function spaces on the spider meshes, which should have six components,
# three for translation and three for rotation.

element = basix.ufl.element("DG", "point", 0, shape=(6,), dtype=default_real_type)
W_P, W_A, W_B = (fem.functionspace(spiders, element) for spiders in (spiders_P, spiders_A, spiders_B))

# ## Meshing the two beams with bores
#
# Each beam is meshed separately from its extent and the heights of its bores, above, with
# second order cells: beam 1 with tetrahedra, beam 2 with hexahedra, by extruding a
# quadrilateral mesh of its side along the pin. The facets of bore $k$ are tagged $k + 1$.


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

beam_1, tags_1 = beam(*beam_1_z, *beam_1_y, beam_1_bores, 0.08, "beam_1")
beam_2, tags_2 = beam(*beam_2_z, *beam_2_y, beam_2_bores, 0.05, "beam_2", hexahedra=True)

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
# Both beams are discretized in the same way: the displacement of each is in a space of
# second order vector Lagrange elements on its second order mesh, of tetrahedra and of
# hexahedra respectively.
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
# $d = x_B - x_A$, and $\theta_B = \theta_A$. The spring stores the energy
# $\frac12 \delta \cdot K \delta$, with
#
# $$
# \begin{aligned}
# \delta &= \begin{pmatrix} t_B - t_A - \theta_A \times d \\ \theta_B - \theta_A \end{pmatrix},\\
# K &= \begin{pmatrix} k I & 0 \\ 0 & k (I - a a^T) + k_t a a^T \end{pmatrix},
# \end{aligned}
# $$
#
# with $a$ the direction of the pin, $k$ large, and $k_t$ the stiffness against turning
# about the pin, zero for a free pin. The fixed pin is held by its Dirichlet conditions,
# so its spring only acts against turning, $K_P = k_t\, a a^T$ on the rotation of spider P.
# The springs involve only the spiders. With $w = (w_P, w_A, w_B)$ the unknowns of the
# three spiders and $z = (z_P, z_A, z_B)$ their test functions, the springs of both pins
# form the bilinear form
#
# $$
# s(w, z) = \int K \delta(w_A, w_B) \cdot \delta(z_A, z_B)~\mathrm{d}x + \int K_P w_P \cdot z_P~\mathrm{d}x,
# $$
#
# the first integrated over spider mesh A, reaching spider B through the pair, the second
# over spider mesh P. The function `springs(K_pins, w, z)` builds it for given stiffnesses. Both
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
# The motion of the system follows from its energies. With the unknowns $u = (u_1, u_2, w)$
# of the whole system, the displacements $u_b$ of the beams $\Omega_b$, $b = 1, 2$, and the
# spiders' $w = (w_P, w_A, w_B)$, the kinetic energy is $\frac12 M(\dot u, \dot u)$ and the
# potential energy is
#
# $$
# \begin{aligned}
# \Pi(u) &= \sum_b \int_{\Omega_b} W_b\left(\mathcal{E}(u_b)\right)~\mathrm{d}x - F(u) + \frac12 s(w, w),\\
# W_b(\mathcal{E}) &= \mu_b\, \mathcal{E} : \mathcal{E}
# + \frac{\lambda_b}{2} \left(\operatorname{tr} \mathcal{E}\right)^2,\\
# \mathcal{E}(u) &= \varepsilon(u) + \frac12 \nabla u^T \nabla u,
# \end{aligned}
# $$
#
# with $W_b$ the strain energy density of beam $b$, of Lamé parameters $\mu_b$ and
# $\lambda_b$ from its Young's modulus $E_b$ and Poisson's ratio $\nu_b$, $\mathcal{E}$ the
# Green–Lagrange strain, $F$ the work of gravity $g$, and $s$ the springs of
# {ref}`demo-spider-hinge-spring-coupling`, above. Here
#
# $$
# \begin{aligned}
# M(u, v) &= \sum_b \int_{\Omega_b} \rho_b u_b \cdot v_b~\mathrm{d}x,\\
# F(v) &= \sum_b \int_{\Omega_b} \rho_b g \cdot v_b~\mathrm{d}x,
# \end{aligned}
# $$
#
# with $\rho_b$ the density of beam $b$. By Hamilton's principle, the motion makes
# $\int \left(\frac12 M(\dot u, \dot u) - \Pi(u)\right)~\mathrm{d}t$ stationary.
#
# The meshes are the reference configuration: the beams as built, before gravity acts. We
# decompose the displacement from it into two parts, $u_0 + u$:
#
# - $u_0$, the static deflection under gravity, which brings the beams to the hanging
#   state, at rest;
# - $u$, the small motion about the hanging state: the swing.
#
# Both follow from $\Pi$. As $u_0$ is small, both are computed on the reference
# configuration, and the animation below shows the swing $u$.
#
# $M$, $a$, $a_{\sigma_0}$ and $F$, below, involve only the beams, and $s$ only the spiders.

# ### The hanging state
#
# The deflection $u_0 = (u_{0,1}, u_{0,2}, w_0)$ minimizes $\Pi$. It is small, so
# $\mathcal{E} \approx \varepsilon$, and the minimum satisfies
#
# $$
# \begin{aligned}
# a(u_0, v) + s_\text{locked}(w_0, z) &= F(v),\\
# a(u, v) &= \sum_b \int_{\Omega_b} \sigma_b(u_b) : \varepsilon(v_b)~\mathrm{d}x,
# \end{aligned}
# $$
#
# for all test functions $v = (v_1, v_2, z)$, with the stress
# $\sigma_b(u) = 2 \mu_b \varepsilon(u) + \lambda_b \operatorname{tr} \varepsilon(u) I$.
# Because the beams are symmetric about $x = 0$, gravity has no moment about either pin,
# so the deflection is the same with the pins locked against turning, which makes the
# static problem well posed. It gives the prestress $\sigma_{0,b} = \sigma_b(u_{0,b})$.

# The forms of the hanging state, over the five spaces of the constraints, in the order of
# the blocks of the system: the materials, gravity, and the locked springs, with the
# stiffness $k$ large next to the beams'. The static deflection is solved into functions
# `u_0` in the constraints' spaces.

# +
materials = {"E": (2.0e4, 0.7e5), "nu": (0.3, 0.25), "rho": (1.0, 1.2)}
beams = (beam_1, beam_2)
g = 9.81
down = np.array([0.0, 0.0, -1.0])


def sigma(u, b: int):
    """The stress of displacement `u` in the material of beam `b`."""
    E, nu = materials["E"][b], materials["nu"][b]
    mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)


mixed = ufl.MixedFunctionSpace(*(mpc.input_space for mpc in mpcs))
trials, tests = ufl.TrialFunctions(mixed), ufl.TestFunctions(mixed)
u_1, u_2, w_P, w_A, w_B = trials
v_1, v_2, z_P, z_A, z_B = tests
dx_1, dx_2 = ufl.dx(beam_1), ufl.dx(beam_2)

a = ufl.inner(sigma(u_1, 0), ufl.sym(ufl.grad(v_1))) * dx_1
a += ufl.inner(sigma(u_2, 1), ufl.sym(ufl.grad(v_2))) * dx_2
# F by block: gravity has no spider blocks. One constant serves both beams.
gravity = fem.Constant(beam_1, (g * down).astype(default_scalar_type))
F = [
    ufl.inner(materials["rho"][0] * gravity, v_1) * dx_1,
    ufl.inner(materials["rho"][1] * gravity, v_2) * dx_2,
    ufl.ZeroBaseForm((z_P,)),
    ufl.ZeroBaseForm((z_A,)),
    ufl.ZeroBaseForm((z_B,)),
]

# The stiffness of the springs, large next to that of the beams. In single precision it is
# smaller, so that its rounding, k times the machine epsilon, stays small next to the weights

double_precision = np.finfo(default_real_type).bits == 64
k = (1e3 if double_precision else 1e1) * max(materials["E"])

# The lower pin's spring, on spider mesh A, and the fixed pin's, on spider mesh P, locked
K_locked = (fem.Constant(spiders_A, spring_stiffness(k)), fem.Constant(spiders_P, torsion(k)))
s_locked = springs(K_locked, trials[2:], tests[2:])


def in_constraint_spaces():
    """One function per block, in the space of its constraint."""
    return [fem.Function(mpc.function_space, dtype=default_scalar_type) for mpc in mpcs]


u_0 = in_constraint_spaces()
# -

# The static deflection, with the pins locked:

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
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)
u_1_0, u_2_0, w_P_0, w_A_0, w_B_0 = static.solve()
# -

# ````{admonition} Verification of the hanging state
# :class: dropdown
#
# Beam 2 hangs from the lower spring alone. Ordered as the spiders' unknowns $w = (t, \theta)$,
# translations first, the spring's generalized force in the hanging state splits into its
# force $f$ and its moment $\mu$,
#
# $$
# \begin{pmatrix} f \\ \mu \end{pmatrix} = K \delta(w_{A,0}, w_{B,0}).
# $$
#
# We therefore expect the force to equal the weight of beam 2, and the moment about the
# pin to vanish, as gravity has no moment about the pin:
#
# $$
# \begin{aligned}
# f &= -m_2 g\, e_z,\\
# a \cdot \mu &= 0.
# \end{aligned}
# $$
#
# Here $m_2 = \int_{\Omega_2} \rho_2~\mathrm{d}x$ is the mass of beam 2, with its bore. The
# demo asserts both, to the tolerance `tol`: the force relative to $m_2 g$, and the moment
# relative to $m_2 g L_2$, a bound on the moment of beam 2's weight about the pin.
# ````

# The code for the verification described in the dropdown above can be inspected
# by expanding the cell below.

# + tags=["hide-cell"]


def integrate(form) -> float:
    """The value of a functional, summed over the processes."""
    return comm.allreduce(fem.assemble_scalar(fem.form(form, dtype=default_scalar_type)), op=MPI.SUM).real


mass_2 = integrate(materials["rho"][1] * ufl.dx(beam_2))
values_A, values_B = dolfinx_mpc.spider_values(w_A_0, 0).real, dolfinx_mpc.spider_values(w_B_0, 0).real
delta = np.concatenate([values_B[:3] - values_A[:3] - np.cross(values_A[3:], x_B - x_A), values_B[3:] - values_A[3:]])
spring_force = (spring_stiffness(k).real @ delta)[:3]
spring_moment = np.dot((spring_stiffness(k).real @ delta)[3:], axis)
weight_2 = mass_2 * g * down
lever_2 = mass_2 * g * L_2
tol = max(1e-3, 2e5 * np.finfo(default_real_type).eps)
if comm.rank == 0:
    print(f"Spring force {spring_force.round(5)} next to the weight of beam 2 {weight_2.round(5)}")
    print(f"Spring moment about the pin {spring_moment:.2e} next to m_2 g L_2 = {lever_2:.2e}")
assert np.allclose(spring_force, weight_2, atol=tol * np.abs(weight_2).max())
assert abs(spring_moment) < tol * lever_2
# -

# #### Visualization of the pins
#
# Both pins are meshed, as two cylinders of the bores' radius in one mesh, with the cells of
# each tagged by its pin, only for viewing. Each moves as one rigid body with its spider,
# $t + \theta \times (x - x_c)$: the fixed pin with spider P, the lower pin with spider A.


# + tags=["hide-input"]
# Each pin, (spider, y0, y1), from y0 to y1 along y, a little beyond the beams it carries
pins = ((x_P, beam_1_y[0] - 0.05, beam_1_y[1] + 0.05), (x_A, beam_1_y[0] - 0.05, beam_2_y[1] + 0.05))


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


# -

# #### Visualization of the hanging state
#
# The beams are coloured by their von Mises stress, $\sigma_m = \sqrt{\frac32 s : s}$, with
# $s = \sigma - \frac13 \operatorname{tr}(\sigma) I$ the deviatoric stress. As the
# displacement is of second order on second order cells, the stress varies within a cell;
# it is shown by its value at each cell's midpoint, interpolated into a space of cell-wise
# constants (DG-0), and drawn on the displaced grid.


# + tags=["hide-input"]
def von_mises(u, b: int):
    """The von Mises stress of the displacement `u` of beam `b`."""
    s = sigma(u, b) - ufl.tr(sigma(u, b)) / 3 * ufl.Identity(3)
    return ufl.sqrt(3 / 2 * ufl.inner(s, s))


Q = [fem.functionspace(m, ("DG", 0)) for m in beams]


def von_mises_functions(us):
    """The von Mises stress of the displacements `us` of the beams: the DG-0 functions that hold
    it, and the expressions that fill them."""
    functions = [fem.Function(Q_b, dtype=default_scalar_type) for Q_b in Q]
    expressions = [
        fem.Expression(von_mises(u, b), Q[b].element.interpolation_points, dtype=default_scalar_type)
        for b, u in enumerate(us)
    ]
    return functions, expressions


def owned_cell_values(f: fem.Function) -> np.ndarray:
    """The values of a DG-0 function in the cells this process owns, in the cells' order, which is
    that of the grid of :func:`pyvista_ugrid`."""
    num_owned = f.function_space.mesh.topology.index_map(3).size_local
    return f.x.array[f.function_space.dofmap.list[:num_owned, 0]].real.copy()


# -

# The deflection under gravity, magnified, coloured by the von Mises stress of the hanging
# state, the prestress $\sigma_0$, over the outline of the undeformed beams in grey, with the
# pins in translucent red, so that the bores around them show. The pins are the meshed
# cylinders above, moved with their spiders.

# + tags=["hide-input"]
stress_0, stress_0_expressions = von_mises_functions(u_0[:2])
for f, expression in zip(stress_0, stress_0_expressions):
    f.interpolate(expression)
move_pins(u_0)
grids = []
for V, u in ((V_1, u_1_0), (V_2, u_2_0), (V_pins, u_pins)):
    grid = pyvista_ugrid(V)
    grid["u"] = u.x.array.reshape(-1, 3)[: grid.n_points].real
    grids.append(grid)
for grid, f in zip(grids, stress_0):
    grid.cell_data["von Mises"] = owned_cell_values(f)
gathered = comm.gather(grids, root=0)
if comm.rank == 0:
    plotter = pyvista.Plotter(window_size=(700, 900))
    # One grid per beam and one for the pins, from the pieces of all processes, so that no
    # partition boundary is drawn
    *merged, merged_pins = [
        pyvista.merge([piece[b] for piece in gathered if piece[b].n_points > 0]).clean(tolerance=1e-6) for b in range(3)
    ]
    factor = 0.1 / max(np.abs(grid["u"]).max() for grid in merged)
    clim = [0.0, max(grid.cell_data["von Mises"].max() for grid in merged)]
    for grid in merged:
        add_cells(plotter, grid.warp_by_vector("u", factor=factor), scalars="von Mises", clim=clim, cmap="viridis")
        # The outline of the undeformed beam: the edges of its box and the rims of its bores
        outline = (
            cell_edges(grid)
            .extract_surface()
            .extract_feature_edges(
                boundary_edges=False, manifold_edges=False, non_manifold_edges=False, feature_angle=30
            )
        )
        plotter.add_mesh(outline, color="gray", line_width=2)
    # Translucent, so that the bore around each pin shows
    plotter.add_mesh(merged_pins.warp_by_vector("u", factor=factor), color="red", opacity=0.5)
    plotter.camera_position = [(4.5, -6.0, -0.4), (0.0, 0.45, -1.45), (0.0, 0.0, 1.0)]
    if pyvista.OFF_SCREEN:
        plotter.screenshot("demo_spider_hinge.png")
    else:
        plotter.show()
# -

# ### The swing
#
# The motion $u$ about the hanging state, with the pins free, follows from the expansion of
# the potential energy about its minimum $u_0$, to second order in $u$:
#
# $$
# \Pi(u_0 + u) = \Pi(u_0) + D\Pi(u_0)[u] + \frac12 D^2\Pi(u_0)[u, u] + \mathcal{O}(|u|^3).
# $$
#
# - **Zeroth order:** $\Pi(u_0)$ is a constant, and does not enter the equation of motion.
# - **First order:** $D\Pi(u_0)[v] = 0$ for all $v$, as $u_0$ is the minimum: with
#   $\mathcal{E} \approx \varepsilon$, this is exactly the system of the hanging state solved
#   above. (It was solved with the pins locked, but the pins carry no moment there, so the
#   locked and the free springs agree at $u_0$.) The work of gravity, $F$, is linear in the
#   displacement, so it lies entirely in this term, balanced by the internal forces of the
#   hanging state: gravity does not appear in the swing.
# - **Second order:** the second variation of $\Pi$ at $u_0$. The strain energy contributes
#   through the variation of the strain, $\varepsilon(u)$ to leading order in $u_0$, and
#   through the second variation of the strain itself, $\operatorname{sym}(\nabla u^T \nabla v)$
#   from its term $\frac12 \nabla u^T \nabla u$, paired with the stress of the hanging
#   state, the prestress $\sigma_0$. With the free springs,
#
# $$
# \begin{aligned}
# D^2\Pi(u_0)[u, v] &= \hat K(u, v) = a(u, v) + a_{\sigma_0}(u, v) + s_\text{free}(w, z),\\
# a_{\sigma_0}(u, v) &= \sum_b \int_{\Omega_b} \nabla u_b \, \sigma_{0,b} : \nabla v_b~\mathrm{d}x.
# \end{aligned}
# $$
#
# So the hanging state enters the swing only through $\sigma_0$, in $a_{\sigma_0}$.
#
# ````{admonition} The second variation of $\Pi$
# :class: dropdown
#
# Write the strain energy density with the elasticity tensor $C_b$, and define the stress
# $S_b$ as its derivative with respect to the strain (the second Piola–Kirchhoff stress):
#
# $$
# \begin{aligned}
# W_b(\mathcal{E}) &= \frac12 \mathcal{E} : C_b \mathcal{E},\\
# C_b \mathcal{E} &= 2 \mu_b \mathcal{E} + \lambda_b \operatorname{tr}(\mathcal{E}) I,\\
# S_b(\mathcal{E}) &= \frac{\partial W_b}{\partial \mathcal{E}}(\mathcal{E})
#   = C_b \mathcal{E},\\
# \mathcal{E}(u) &= \operatorname{sym} \nabla u + \frac12 \nabla u^T \nabla u.
# \end{aligned}
# $$
#
# Let $\delta u$ and $\delta v$ be variations of the displacement, with spider components
# $\delta w$ and $\delta z$. The first and second variations of the strain at a
# displacement $\hat u$ are
#
# $$
# \begin{aligned}
# D\mathcal{E}(\hat u)[\delta v] &=
#   \operatorname{sym}\left((I + \nabla \hat u)^T \nabla \delta v\right),\\
# D^2\mathcal{E}[\delta u, \delta v] &=
#   \operatorname{sym}\left(\nabla \delta u^T \nabla \delta v\right),
# \end{aligned}
# $$
#
# the latter the same at any $\hat u$. The chain rule then gives, for each beam,
#
# $$
# \begin{aligned}
# D\Pi(\hat u)[\delta v] &= \sum_b \int_{\Omega_b}
#   S_b\left(\mathcal{E}(\hat u_b)\right) : D\mathcal{E}(\hat u_b)[\delta v_b]~\mathrm{d}x
#   - F(\delta v) + s(\hat w, \delta z),\\
# D^2\Pi(\hat u)[\delta u, \delta v] &= \sum_b \int_{\Omega_b}
#   C_b\, D\mathcal{E}(\hat u_b)[\delta u_b] : D\mathcal{E}(\hat u_b)[\delta v_b]~\mathrm{d}x\\
# &\quad + \sum_b \int_{\Omega_b}
#   S_b\left(\mathcal{E}(\hat u_b)\right) :
#   \operatorname{sym}\left(\nabla \delta u_b^T \nabla \delta v_b\right)~\mathrm{d}x
#   + s(\delta w, \delta z).
# \end{aligned}
# $$
#
# At $\hat u = u_0$, to leading order in $u_0$,
# $D\mathcal{E}(u_0)[\delta u] \approx \varepsilon(\delta u)$ and
# $S_b(\mathcal{E}(u_{0,b})) \approx C_b \varepsilon(u_{0,b}) = \sigma_b(u_{0,b})
# = \sigma_{0,b}$, the prestress of the hanging state. As $\sigma_{0,b}$ is symmetric,
# $\sigma_{0,b} : \operatorname{sym}(\nabla \delta u_b^T \nabla \delta v_b)
# = \nabla \delta u_b\, \sigma_{0,b} : \nabla \delta v_b$. In the directions of the swing,
# $\delta u = u$, and of the test function, $\delta v = v$, with the free springs,
#
# $$
# D^2\Pi(u_0)[u, v] = a(u, v) + a_{\sigma_0}(u, v) + s_\text{free}(w, z) = \hat K(u, v).
# $$
#
# The terms dropped are of two kinds. The cross terms, such as
# $C_b\, \varepsilon(\delta u) : \operatorname{sym}(\nabla u_0^T \nabla \delta v)$, are of
# first order in $u_0$, as $\sigma_0$ is, but each contains $\varepsilon(\delta u)$ or
# $\varepsilon(\delta v)$. The term
# $C_b \operatorname{sym}(\nabla u_0^T \nabla \delta u) :
# \operatorname{sym}(\nabla u_0^T \nabla \delta v)$ is of second order in $u_0$.
#
# The swing $u$ is not assumed rigid, but its pendulum motion is close to a rigid motion of
# the reference configuration, the meshes, for which $\varepsilon(u) = 0$. For that motion,
# $a_{\sigma_0}$ is the only stiffness kept, while for the elastic deformation of the beams,
# $a$ dominates. The exact rigid motions about the hanging state rotate the deformed
# geometry, $u = (R - I)(x + u_0 - c)$. For these, $D\mathcal{E}(u_0)[u]$ vanishes, rather
# than $\varepsilon(u)$; the difference is the dropped term
# $\operatorname{sym}(\nabla u_0^T \nabla u)$, of first order in $u_0$.
#
# This is the usual linearization with initial stress, or geometric stiffness.
# ````
#
# The geometric stiffness $a_{\sigma_0}$ is the term $\frac12 \nabla u^T \nabla u$ of the
# strain, paired with the prestress: it is what linear elasticity, with
# $\mathcal{E} \approx \varepsilon$ throughout, leaves out. A rigid rotation has no
# linear strain, so linear elasticity alone has no restoring torque; but as a pendulum
# swings, its mass rises along an arc against the tension of the hanging beam, and that
# work is $\frac12 a_{\sigma_0}(u, u)$, as checked in
# {ref}`demo-spider-hinge-prestress`, below.
#
# The equation of the swing follows from Hamilton's principle, now with the displacement
# $u_0 + u$. As $u_0$ does not depend on time, $\dot u$ is the velocity of the whole
# motion, and the action
#
# $$
# \mathcal{S}(u) = \int_{t_0}^{t_1} \left(\frac12 M(\dot u, \dot u) - \Pi(u_0 + u)\right)
#   ~\mathrm{d}t
# $$
#
# is stationary: its variation vanishes in every direction $v$ that vanishes at $t_0$ and
# $t_1$. Integrating the kinetic term by parts in time,
#
# $$
# D\mathcal{S}(u)[v] = \int_{t_0}^{t_1} \left(M(\dot u, \dot v) - D\Pi(u_0 + u)[v]\right)
#   ~\mathrm{d}t
#   = -\int_{t_0}^{t_1} \left(M(\ddot u, v) + D\Pi(u_0 + u)[v]\right)~\mathrm{d}t = 0,
# $$
#
# and as $v$ is arbitrary in time, $M(\ddot u, v) + D\Pi(u_0 + u)[v] = 0$ for all $v$, at
# all times. This is the full, nonlinear, equation of motion. Its force $D\Pi$ expands about
# the hanging state as
#
# $$
# D\Pi(u_0 + u)[v] = D\Pi(u_0)[v] + D^2\Pi(u_0)[u, v] + \mathcal{O}(|u|^2),
# $$
#
# where the first term vanishes, as $u_0$ is the minimum, and the second is $\hat K(u, v)$.
# To first order in $u$, the equation of the swing is therefore
#
# $$
# M(\ddot u, v) + \hat K(u, v) = 0 \quad \text{for all } v.
# $$
#
# Hamilton's principle itself takes only first variations, of the action: the second
# variation of $\Pi$ enters as the linearization of the force $D\Pi$ about $u_0$, that is,
# as the stiffness of the swing. Equivalently, the expansion of $\Pi$ to second order,
# above, turns the action into that of the Lagrangian
# $\frac12 M(\dot u, \dot u) - \frac12 \hat K(u, u)$, with the same equation. As a second
# derivative, $\hat K$ is symmetric, and the Lagrangian does not depend on time, so the
# energy $\frac12 M(\dot u, \dot u) + \frac12 \hat K(u, u)$ of the swing is conserved, as
# checked in {ref}`the energy conservation <demo-spider-hinge-energy>`, below.

# The forms of the swing: the mass, the free springs, and the prestress of the static
# deflection `u_0`.

# +
M = materials["rho"][0] * ufl.inner(u_1, v_1) * dx_1 + materials["rho"][1] * ufl.inner(u_2, v_2) * dx_2
K_free = (fem.Constant(spiders_A, spring_stiffness(0.0)), fem.Constant(spiders_P, torsion(0.0)))
s_free = springs(K_free, trials[2:], tests[2:])
sigma_0 = (sigma(u_0[0], 0), sigma(u_0[1], 1))
a_sigma_0 = ufl.inner(ufl.grad(u_1) * sigma_0[0], ufl.grad(v_1)) * dx_1
a_sigma_0 += ufl.inner(ufl.grad(u_2) * sigma_0[1], ufl.grad(v_2)) * dx_2
K_hat = a + a_sigma_0 + s_free
# -

# ## Time stepping
#
# The whole coupled system, beams and spiders alike, is stepped with one time step
# $\Delta t$. To step it, the second order equation $M \ddot u + \hat K u = 0$, with $M$
# and $\hat K$ the matrices of the forms above acting on the vectors of the unknowns, is
# written as a first order system, with the velocity $\dot u$ as an unknown of its own:
#
# $$
# \begin{aligned}
# \frac{\mathrm{d} u}{\mathrm{d} t} &= \dot u,\\
# M \frac{\mathrm{d} \dot u}{\mathrm{d} t} &= -\hat K u.
# \end{aligned}
# $$
#
# The implicit midpoint rule replaces each time derivative by the difference of the values
# at the two ends of a step, $u^n$ at time $n \Delta t$ and $u^{n+1}$ at
# $(n + 1) \Delta t$, divided by $\Delta t$, and evaluates the right-hand side at the middle
# of the step, as the average of its values at the two ends:
#
# $$
# \begin{aligned}
# \text{(i)} \quad \frac{u^{n+1} - u^n}{\Delta t} &= \frac{\dot u^{n+1} + \dot u^n}{2},\\
# \text{(ii)} \quad M \frac{\dot u^{n+1} - \dot u^n}{\Delta t} &= -\hat K \frac{u^{n+1} + u^n}{2}.
# \end{aligned}
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

# The left- and right-hand sides of (iv), with the right-hand side in terms of the functions
# holding $u^n$ and $\dot u^n$, `u_n` and `u_dot_n`.

# +
dt, num_steps = 0.02, 150
# The displacement u^n, the velocity u_dot^n and the new displacement u^{n+1}
u_n, u_dot_n, u_new = in_constraint_spaces(), in_constraint_spaces(), in_constraint_spaces()


def acting(form, functions):
    """The linear form of `form` with its trial functions replaced by `functions`."""
    return ufl.replace(form, dict(zip(trials, functions)))


step_lhs = 4 / dt**2 * M + K_hat
step_rhs = acting(4 / dt**2 * M, u_n) + acting(4 / dt * M, u_dot_n) - acting(K_hat, u_n)
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
xh = dolfinx_mpc.create_vector(L_step, mpcs, kind="mpi")
# The pinned dofs, by block, which are zero in every step
pinned_by_block = fem.bcs_by_block([mpc.input_space for mpc in mpcs], pinned)
# -

# (demo-spider-hinge-energy)=
# ````{admonition} Energy conservation
# :class: dropdown
#
# The total energy of the system at time step $n$ can be written as the sum of the
# kinetic and stored energy:
#
# $$
# E^n = \frac12 \dot u^n \cdot M \dot u^n + \frac12 u^n \cdot \hat K u^n
# $$
#
# where the stored energy $\frac12 u^n \cdot \hat K u^n$ holds the elastic energy of the
# beams, the pendulum's potential through $a_{\sigma_0}$, and the springs' energy, as
# checked in {ref}`demo-spider-hinge-prestress`, at the end.
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
# the demo checks as it steps.
#
# The energy, as one form per domain: the two beams, and the springs on spider meshes A and P.
# Spring A reaches spider B through the pair.
# ````

# + tags=["hide-cell"]


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
# The pendulum starts at rest, beam 1 turned by $\theta_1 = 8°$ about the fixed pin and
# beam 2 by a further $\theta_2 = -8°$ about the lower pin: the rigid motions
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


theta_1, theta_2 = np.deg2rad(8.0), np.deg2rad(-8.0)
rigid_turns(theta_1, theta_2, u_n)

for k in range(2):
    u_n[k].name = "u"
writers = [VTXWriter(comm, f"demo_spider_hinge_{k + 1}.bp", [u_n[k]], engine="BP4") for k in range(2)]
writers.append(VTXWriter(comm, "demo_spider_hinge_pins.bp", [u_pins], engine="BP4"))


def spider_turns(u) -> tuple[float, ...]:
    """The turns of beam 1, at spider P, and of beam 2, at spider B, about the pins."""
    return tuple(float(np.dot(dolfinx_mpc.spider_values(u[k], 0).real[3:], axis)) for k in (2, 4))


# -

# ## Visualization
#
# As in {doc}`demo_linear_wave_problem`, each process builds grids of the cells it owns,
# gathered once onto one process, as the meshes do not move; per frame only the nodal
# values and the stresses of the cells are gathered. The beams and the pins are drawn
# displaced by the swing $u$ at true scale, the pins in translucent red, and the beams are
# coloured by the von Mises stress of the total stress, $\sigma(u_0 + u)$: the prestress and
# the swing's.

# + tags=["hide-input"]
pyvista.OFF_SCREEN = True
shown = (u_n[0], u_n[1], u_pins)
stress, stress_expressions = von_mises_functions([u_0[b] + u_n[b] for b in range(2)])
local_grids = [pyvista_ugrid(u.function_space) for u in shown]
gathered_grids = comm.gather(local_grids, root=0)
if comm.rank == 0:
    # The pieces of every process, in rank order, in one grid per mesh
    grids = [pyvista.merge([piece[k] for piece in gathered_grids], merge_points=False) for k in range(3)]
    plotter = pyvista.Plotter(off_screen=True, window_size=(450, 600))
    plotter.open_gif(Path("demo_spider_hinge.py").with_suffix(".gif"), fps=1 / (2 * dt))


def write_frame(t: float):
    """Draw the beams at time `t`, coloured by their von Mises stress, and the pins, displaced.
    Collective."""
    for f, expression in zip(stress, stress_expressions):
        f.interpolate(expression)
    values = comm.gather(
        (
            [u.x.array.reshape(-1, 3)[: grid.n_points].real.copy() for u, grid in zip(shown, local_grids)],
            [owned_cell_values(f) for f in stress],
        )
    )
    if comm.rank != 0:
        return
    plotter.clear()
    for k, grid in enumerate(grids):
        grid["u"] = np.vstack([piece[0][k] for piece in values])
    for k, grid in enumerate(grids[:2]):
        grid.cell_data["von Mises"] = np.concatenate([piece[1][k] for piece in values])
    if not clim:
        clim.extend([0.0, 1.5 * max(grid.cell_data["von Mises"].max() for grid in grids[:2])])
    for k, grid in enumerate(grids):
        if k < 2:
            add_cells(
                plotter, grid.warp_by_vector("u"), scalars="von Mises", clim=clim, cmap="viridis", show_scalar_bar=False
            )
        else:
            plotter.add_mesh(grid.warp_by_vector("u"), color="red", opacity=0.5)
    plotter.add_text(f"t = {t:.2f}", position="upper_left", font_size=12, color="black")
    plotter.camera_position = [(2.5, -7.0, -1.0), (0.0, 0.45, -1.4), (0.0, 0.0, 1.0)]
    plotter.write_frame()


# -

# We can now perform the time stepping, writing a frame every two steps, and checking every
# 25 steps that the energy is conserved, as shown in
# {ref}`the energy conservation <demo-spider-hinge-energy>`.

# +
history = [(0.0, *spider_turns(u_n))]
E_0 = energy(energies)
move_pins(u_n)
write_frame(0.0)
drift = 0.0
for writer in writers:
    writer.write(0.0)
for step in range(1, num_steps + 1):
    with b.localForm() as b_local:
        b_local.set(0.0)
    dolfinx_mpc.assemble_vector(L_step, mpcs, b)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, pinned_by_block)
    solver.solve(b, xh)
    xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(xh, u_new)
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
        write_frame(step * dt)
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

# The PETSc objects of the step are destroyed explicitly, and those that Python's garbage
# collector has freed are cleaned up on every process together, so that no process is left
# waiting for another when PETSc finalizes
solver.destroy()
A.destroy()
b.destroy()
xh.destroy()
PETSc.garbage_cleanup(comm)
# -

# The swing, released from rest:
#
# <img src="./demo_spider_hinge.gif" alt="gif" class="bg-primary mb-1" width="600px">

# The energy is conserved up to the rounding of the solves, which the stiff springs next to
# the mass make large.
#
# ```{admonition} Verification invalid in single precision
# :class: dropdown
# The checks of the swing, here and against the rigid pendulums below, are only made in
# double precision. In single precision (`float32` and `complex64`) the swing cannot be
# resolved. Its displacement is close to a large rigid turn, whose strain at a quadrature
# point is the sum of terms of size $|u| / h$ that nearly cancel. Their rounding leaves a
# strain of order $10^{-4}$, and so an elastic stress $E \cdot 10^{-4}$ as large as the
# prestress $\sigma_0$ that drives the swing. The demo still runs, but the swing it computes
# is not accurate.
# ```

if double_precision:
    assert drift < min(1e-2, 1e8 * np.finfo(default_real_type).eps)

# (demo-spider-hinge-rigid-pendulums)=
# ## Verification against rigid double pendulums
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
# \begin{aligned}
# M_\varphi &= \begin{pmatrix} I_1 + m_2 \ell^2 & m_2 \ell h_2 \\ m_2 \ell h_2 & I_2 \end{pmatrix},\\
# K_\varphi &= g \begin{pmatrix} m_1 h_1 + m_2 \ell & 0 \\ 0 & m_2 h_2 \end{pmatrix},
# \end{aligned}
# $$
#
# with $\ell = 0.8 L$ the distance between the pins, and for each beam its mass, its centre
# of mass, the depth of that below the beam's pin $x_b$ ($x_P$ for beam 1, $x_B$ for
# beam 2), and its moment of inertia about the pin, integrated over its mesh, with its
# bores,
#
# $$
# \begin{aligned}
# m_b &= \int_{\Omega_b} \rho_b~\mathrm{d}x,\\
# c_b &= \frac{1}{m_b} \int_{\Omega_b} \rho_b\, x~\mathrm{d}x,\\
# h_b &= (x_b - c_b) \cdot e_z,\\
# I_b &= \int_{\Omega_b} \rho_b \left|(I - a a^T)(x - x_b)\right|^2~\mathrm{d}x.
# \end{aligned}
# $$
#
# Its masses are those of the first model, but not quite its stiffness, as the energy of a
# rigid swing shows.
#
# (demo-spider-hinge-prestress)=
# ### Check: the energy of a rigid swing
#
# To see that the geometric stiffness is the pendulum's potential, turn one
# beam rigidly by an angle $\theta$ about its pin, through $x_c$ along $a = (0, 1, 0)$:
# $u = \theta\, a \times \xi$ with $\xi = x - x_c$. The two parts of the stored energy are then
#
# - **elastic:** $\varepsilon(u) = \operatorname{sym} \nabla u = 0$, as $\nabla u$ is the
#   skew-symmetric matrix of $\theta a \times$, so $\frac12 a(u, u) = 0$;
# - **geometric:** $\nabla u^T \nabla u = \theta^2 (I - a a^T)$, the projection onto the
#   plane of the swing, so
#
#   $$
#   \frac12 a_{\sigma_0}(u, u) = \frac{\theta^2}{2} \int_{\Omega_b} \sigma_0 : (I - a a^T)~\mathrm{d}x
#   = \frac{\theta^2}{2} \int_{\Omega_b} \left(\sigma_{0,xx} + \sigma_{0,zz}\right)~\mathrm{d}x.
#   $$
#
# The integral of the prestress follows from the equilibrium of the hanging beam,
# $-\nabla \cdot \sigma_0 = \rho g$ in $\Omega_b$, with the traction $\sigma_0 n = \tau$ that
# the spiders exert on the bores and no traction elsewhere. Multiplying by $\xi$ and
# integrating by parts gives
#
# $$
# \int_{\Omega_b} \sigma_0~\mathrm{d}x
# = \int_{\partial \Omega_b} \tau \otimes \xi~\mathrm{d}s + \int_{\Omega_b} \rho g \otimes \xi~\mathrm{d}x,
# $$
#
# and taking the trace in the plane of the swing,
#
# $$
# \int_{\Omega_b} \left(\sigma_{0,xx} + \sigma_{0,zz}\right)~\mathrm{d}x
# = \underbrace{\int_{\Omega_b} \rho g \cdot \xi~\mathrm{d}x}_{m_b g h_b}
# + \int_{\text{bores}} \tau \cdot (I - a a^T)\, \xi~\mathrm{d}s,
# $$
#
# with $m_b$ the mass of the beam and $h_b$ the depth of its centre of mass below the pin,
# as $g = (0, 0, -g)$ is perpendicular to $a$. The first term gives
# $\frac12 m_b g h_b \theta^2$, the exact potential of a rigid pendulum,
# $m_b g h_b (1 - \cos\theta)$, to second order in $\theta$. The bore terms add the work of
# the loads the beam carries:
#
# - for beam 1, the weight of beam 2 hangs from its lower bore, a distance $\ell = 0.8 L$
#   below the pin, and adds $\frac12 m_2 g \ell\, \theta^2$, as in $K_\varphi$ of the
#   textbook double pendulum above;
# - at the bore a beam hangs from, the pin's force acts a bore radius $r$ off the axis and
#   adds a term of relative size $r / h_b$. The textbook pendulum leaves this term out, so
#   over a few swings it drifts out of phase with the beams.
#
# So, for a swing, the stored energy $\frac12 u \cdot \hat K u$ is the elastic energy of the
# beams' deformation, $\frac12 a(u, u)$, plus the pendulum's potential,
# $\frac12 a_{\sigma_0}(u, u)$, plus the energy of the springs, $\frac12 s_\text{free}(w, w)$,
# which stays small as long as the pins hold the spiders together.

# +
mass, centre, inertia = [], [], []
for b, (m, x_pivot) in enumerate(((beam_1, x_P), (beam_2, x_B))):
    x = ufl.SpatialCoordinate(m)
    rho = materials["rho"][b]
    mass.append(integrate(rho * ufl.dx(m)))
    centre.append(np.array([integrate(rho * x[i] * ufl.dx(m)) for i in range(3)]) / mass[-1])
    # The distance from the pin's axis, along y: its x and z components
    arm = ufl.as_vector([x[0] - x_pivot[0], x[2] - x_pivot[2]])
    inertia.append(integrate(rho * ufl.inner(arm, arm) * ufl.dx(m)))
if comm.rank == 0:
    print(f"Masses {mass[0]:.4f}, {mass[1]:.4f}; centres {centre[0].round(4)}, {centre[1].round(4)}")

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
# follow the rigid pendulum of the finite element model to a few percent. As noted above,
# the last two are only checked in double precision.

assert mass_difference < 1e4 * np.finfo(default_real_type).eps
if double_precision:
    assert np.all(np.abs(stiffness_difference) < radius / min(h))
    assert mismatch["ritz"] < 0.1

# Finally, the static problem's PETSc objects are released and cleaned up in the same way.

del static
PETSc.garbage_cleanup(comm)

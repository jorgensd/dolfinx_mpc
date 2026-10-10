# # Periodicity in one direction: a layer with an affine periodic part
#
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT
#
# This demo complements the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`,
# in which the unit cell is periodic in both directions and the whole periodic jump of the
# displacement is prescribed by the macroscopic strain. Here the cell is periodic in $x$
# **only**: it is one period of an infinite layer $\mathbb{R}\times(0,L)$. Three load cases are considered:
#
# * **Horizontal tension** (`--load tension`). The layer is stretched along its length by a
#   prescribed periodic jump, and its faces are traction-free: the thickness change is computed.
# * **Horizontal simple shear** (`--load shear`). The layer is sheared between its faces, as
#   between two parallel plates: BOTTOM is fixed and TOP is moved horizontally by $\gamma L$. The
#   periodic jump is zero.
# * **Horizontal tension under stress control** (`--load tension-stress`). The force transmitted
#   by the layer is prescribed instead of its stretch. The horizontal jump is then an *unknown
#   master* of the multi-point constraint, loaded by a nodal force.
#
# In all cases the periodic constraints carry an affine part, as in the periodic homogenization
# demo, but only along $x$. The faces are not periodic: they are either free or prescribed.
#
# The cell, the material and the microstructure are those of the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`:
# a structured mesh of the square cell and a centred circular inclusion 50 times stiffer than
# the matrix. An inclined elliptical inclusion (`--inclusion ellipse`) gives a cell without
# symmetry.
#
# The constraints restrict to one direction the corner-node formulation of the periodic boundary
# conditions of
# {cite}`layer-Danas2017` (Appendix B), first used in
# {cite}`layer-LopezPamiesGoudarziDanas2013`.

#
# ```bash
# python3 demo_periodic_layer.py --load tension
# python3 demo_periodic_layer.py --load shear
# python3 demo_periodic_layer.py --load tension-stress
# python3 demo_periodic_layer.py --load tension --inclusion ellipse
# mpirun -n 4 python3 demo_periodic_layer.py --load shear
# ```

# +
from __future__ import annotations

import argparse

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot

import dolfinx_mpc
from dolfinx_mpc import MultiPointConstraint

parser = argparse.ArgumentParser(description="Layer periodic in x with an affine periodic part")
parser.add_argument(
    "--load",
    choices=["tension", "shear", "tension-stress"],
    default="tension",
    help="tension: horizontal tension, free faces; shear: horizontal simple shear; "
    "tension-stress: horizontal tension under prescribed average stress",
)
parser.add_argument(
    "--inclusion",
    choices=["circle", "ellipse"],
    default="circle",
    help="circle: as in the periodic homogenization demo; ellipse: inclined, no symmetry",
)
parser.add_argument("--strain", type=float, default=1e-2, help="magnitude of the macroscopic strain")
parser.add_argument(
    "--stress",
    type=float,
    default=None,
    help="prescribed average stress S_xx (tension-stress); default: the stress of `tension`",
)
args, _ = parser.parse_known_args()  # parse_known_args: also runs inside Jupyter
if args.stress is None:  # the average stress computed in `tension`, for the default --strain
    args.stress = {"circle": 0.15109879, "ellipse": 0.1563171}[args.inclusion]

comm = MPI.COMM_WORLD
dtype = np.dtype(default_scalar_type)
L = 1.0  # side of the square cell
# -

# ## Geometry, mesh and material
#
# The cell is the square $\Omega=(0,L)^2$ with corners
#
# $$
# \begin{aligned}
# A &= (0,0), & B &= (L,0), & C &= (L,L), & D &= (0,L).
# \end{aligned}
# $$
#
# As in the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`,
# the mesh is structured, so that every node of RIGHT has
# a partner on LEFT at the same height. Both phases are linear, isotropic and elastic, in plane
# strain; the Young modulus is a cellwise constant function.

# +
N = 32
# Crossed: each square is cut into four triangles, so the mesh has the symmetries of the square
domain = mesh.create_rectangle(
    comm, [[0, 0], [L, L]], [N, N], diagonal=mesh.DiagonalType.crossed, dtype=default_real_type
)
gdim = tdim = domain.geometry.dim

E_uniform, nu = 10.0, 0.3
E = fem.Function(fem.functionspace(domain, ("Discontinuous Lagrange", 0)), dtype=dtype)
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
if args.inclusion == "circle":
    inclusion = (midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2
else:  # ellipse with semi-axes 0.38 and 0.15, inclined at 30 degrees
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    xr = c30 * (midpoints[:, 0] - 0.5) + s30 * (midpoints[:, 1] - 0.5)
    yr = -s30 * (midpoints[:, 0] - 0.5) + c30 * (midpoints[:, 1] - 0.5)
    inclusion = (xr / 0.38) ** 2 + (yr / 0.15) ** 2 < 1.0
cells0 = np.flatnonzero(inclusion).astype(np.int32)  # the owned cells of the inclusion


def set_young_modulus(E_inclusion: float):
    E.interpolate(lambda x: np.full(x.shape[1], E_uniform))
    E.interpolate(lambda x: np.full(x.shape[1], E_inclusion), cells0=cells0)
    E.x.scatter_forward()


mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return 2 * mu * epsilon(w) + lmbda * ufl.tr(epsilon(w)) * ufl.Identity(gdim)


V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(sigma(u_), epsilon(v_)) * ufl.dx
rhs = ufl.inner(fem.Constant(domain, np.zeros(gdim, dtype=dtype)), v_) * ufl.dx
# -

# ## Periodicity in one direction
#
# Periodicity in $x$ states that the displacement of each point of RIGHT differs from that of its
# partner on LEFT by one and the same vector, the jump of the layer over one period:
# $\mathbf{u}(\mathbf{X}+L\mathbf{e}_x)=\mathbf{u}(\mathbf{X})+\mathbf{u}^B-\mathbf{u}^A$.
# Fixing $\mathbf{u}^A=\mathbf{0}$ to remove the rigid translation, the constraints are
#
# $$
# \begin{aligned}
# \mathbf{u}^A &= \mathbf{0},\\
# \mathbf{u}^{\text{RIGHT}} &= \mathbf{u}^{\text{LEFT}} + \mathbf{u}^B,\\
# \mathbf{u}^C &= \mathbf{u}^D + \mathbf{u}^B,
# \end{aligned}
# $$ (eq:1d-periodic)
#
# where RIGHT and LEFT are the edges without their corners. The corner $C$ needs its own
# equation, linking it to $D$; without it the solution is not periodic at the top corners.
#
# Compared with full periodicity, BOTTOM and TOP are not periodic, and $\mathbf{u}^B$ is a genuine
# degree of freedom of the layer. The prescribed components of the macroscopic strain
# $\bar{\mathbf{E}}$ (a symmetric tensor) and the load case decide what is prescribed:
#
# | `--load` | $\bar{\mathbf{E}}$ | Dirichlet conditions | jump $\mathbf{u}^B$ |
# |---|---|---|---|
# | `tension` | $\bar E_{xx}=\bar\varepsilon$ | on $A$ and $B$; free faces | $(\bar\varepsilon L,0)$ |
# | `shear` | $\bar E_{xy}=\gamma/2$ | $\mathbf{u}=(\gamma Y,0)$ on BOTTOM and TOP | $\mathbf{0}$ |
# | `tension-stress` | unknown | $\mathbf{u}^A=\mathbf{0}$, $u^B_y=0$; free faces | $(u^B_x,0)$, force $F_x$ |
#
# In `tension`, prescribing the whole jump also removes the rigid rotation that the free faces
# would otherwise allow. In `tension-stress` the rotation is removed by $u^B_y=0$.
#
# ### Stress control: the jump as a free master
#
# Let $\mathbf{T}=\boldsymbol{\sigma}\mathbf{n}$. The tractions are anti-periodic,
# $\mathbf{T}(\mathbf{X}+L\mathbf{e}_x)=-\mathbf{T}(\mathbf{X})$, so for a field $\mathbf{v}$ that
# satisfies {eq}`eq:1d-periodic` the work of the tractions on LEFT $\cup$ RIGHT reduces to
#
# $$
# \begin{aligned}
# \int_{\text{LEFT}\cup\text{RIGHT}}\mathbf{T}\cdot\mathbf{v}~\mathrm{d}s
# &= \int_{\text{RIGHT}}\mathbf{T}(\mathbf{X})\cdot
#   \bigl(\mathbf{v}(\mathbf{X})-\mathbf{v}(\mathbf{X}-L\mathbf{e}_x)\bigr)~\mathrm{d}s
# = \int_{\text{RIGHT}}\mathbf{T}\cdot\mathbf{v}^B~\mathrm{d}s
# = \mathbf{F}\cdot\mathbf{v}^B,\\
# \mathbf{F} &= \int_{\text{RIGHT}}\mathbf{T}~\mathrm{d}s .
# \end{aligned}
# $$ (eq:1d-work)
#
# BOTTOM and TOP are traction-free, so the principle of virtual work reads
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})~\mathrm{d}x
# = \mathbf{F}\cdot\mathbf{v}^B
# $$
#
# for all $\mathbf{v}$ satisfying {eq}`eq:1d-periodic` and vanishing where $\mathbf{u}$ is
# prescribed. The right-hand side is the virtual work of a point force $\mathbf{F}$ applied at
# $B$: a prescribed $F_i$ is a **nodal force** on $u^B_i$, which becomes an unknown. In the code it
# is a vertex integral over $B$, $\int_{\{B\}}\mathbf{F}\cdot\mathbf{v}~\mathrm{d}P$, added to the
# linear form with the measure `ufl.dP`. With
# $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, the divergence theorem gives
#
# $$
# \int_\Omega \sigma_{i1}~\mathrm{d}x = L\,F_i + \int_{\text{BOTTOM}\cup\text{TOP}} T_i X_1~\mathrm{d}s .
# $$
#
# With traction-free faces $F_i=L\,\bar\sigma_{i1}$: in `tension-stress` we prescribe
# $F_x=L\,\bar S_{xx}$, i.e. a uniaxial macroscopic stress $\bar S_{xx}$.

# +
# prescribed affine displacement u = G X (on B in tension, on the faces in shear)
if args.load == "tension":
    G = np.array([[args.strain, 0.0], [0.0, 0.0]])  # E_xx = strain
elif args.load == "shear":
    G = np.array([[0.0, args.strain], [0.0, 0.0]])  # u = (gamma Y, 0): E_xy = gamma / 2
else:
    G = np.zeros((2, 2))  # only u^A = 0 and u^B_y = 0 are prescribed
stress_control = args.load == "tension-stress"
F_B = np.array([L * args.stress, 0.0]) if stress_control else np.zeros(2)  # nodal force on B
if comm.rank == 0:
    print(f"load = {args.load}, G = {G.tolist()}, F_B = {F_B.tolist()}")

# Largest distance between a node and a point it is located at, from the rounding of the coordinates
tol = 500 * np.finfo(domain.geometry.x.dtype).eps * L


def near(a, b):
    return np.abs(a - b) < tol


def at_point(px, py):
    return lambda x: near(x[0], px) & near(x[1], py)


def right_open(x):
    return near(x[0], L) & ~(near(x[1], 0.0) | near(x[1], L))


A, B, C, D = (0.0, 0.0), (L, 0.0), (L, L), (0.0, L)


def affine_bc(component: int, marker):
    """DirichletBC u_c = (G X)_c on the dofs selected by `marker`."""
    Vc, _ = V.sub(component).collapse()
    dofs = fem.locate_dofs_geometrical((V.sub(component), Vc), marker)
    g = fem.Function(Vc, dtype=dtype)
    g.interpolate(lambda x: G[component, 0] * x[0] + G[component, 1] * x[1])
    return fem.dirichletbc(g, dofs, V.sub(component))


bcs = [affine_bc(c, at_point(*A)) for c in range(gdim)]
if args.load == "tension":
    bcs += [affine_bc(c, at_point(*B)) for c in range(gdim)]
elif stress_control:
    bcs += [affine_bc(1, at_point(*B))]  # u^B_y = 0; u^B_x is free
else:
    for face in (lambda x: near(x[1], 0.0), lambda x: near(x[1], L)):
        bcs += [affine_bc(c, face) for c in range(gdim)]
# -

# ## Multi-point constraints
#
# Every constraint of {eq}`eq:1d-periodic` has the same form: a node on RIGHT, or the corner $C$,
# equals its image one period to the left, plus $\mathbf{u}^B$. It is built in two steps, for
# every load case:
#
# 1. {py:meth}`create_periodic_constraint_geometrical
#    <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>` ties each such node
#    to its image, $\mathbf{u}(\mathbf{X})=\mathbf{u}(\mathbf{X}-L\mathbf{e}_x)$: RIGHT to LEFT, and
#    $C$ to $D$.
# 2. {py:meth}`add_master_from_point <dolfinx_mpc.MultiPointConstraint.add_master_from_point>` adds
#    the dofs at $B$, with coefficient $1$, to the right-hand side of each of these constraints,
#    component by component.
#
# What differs between the load cases are the Dirichlet conditions, passed as `bcs` to the
# constraint:
#
# * In `tension` the jump $\mathbf{u}^B=(\bar\varepsilon L,0)$ is prescribed by a Dirichlet
#   condition on $B$, and DOLFINx-MPC folds the value of this master into the constraint.
# * In `shear` $B$ lies on BOTTOM and $C$ on TOP, both prescribed. The jump $\mathbf{u}^B=\mathbf{0}$
#   is folded in as above, and $C$ is not a slave: a dof cannot be both a slave and a Dirichlet dof,
#   and its Dirichlet value already satisfies its equation.
# * In `tension-stress` only $u^B_y=0$ is prescribed and folded in. $u^B_x$ stays a master, a
#   degree of freedom of the reduced system, on which the nodal force $F_x$ acts.


# +
def slave_nodes(x):
    """RIGHT, and the corner C unless it lies on a prescribed face."""
    on_C = at_point(*C)(x) if args.load != "shear" else np.zeros(x.shape[1], dtype=bool)
    return right_open(x) | on_C


mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs)
mpc.create_periodic_constraint_geometrical(
    V, slave_nodes, lambda x: np.vstack([x[0] - L, x[1], x[2]]), bcs, scale=dtype.type(1.0)
)
mpc.add_master_from_point(V, slave_nodes, 1.0, B)
mpc.finalize()  # collective: every rank must reach this
# -

# ## Solution and post-processing
#
# We report the macroscopic stress, the volume average of $\boldsymbol{\sigma}$ over the whole
# cell,
#
# $$
# \bar{\boldsymbol{\sigma}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\sigma}~\mathrm{d}x ,
# $$
#
# and the macroscopic strain, the volume average of $\boldsymbol{\epsilon}$,
#
# $$
# \bar{\mathbf{E}}^{\text{eff}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\epsilon}(\mathbf{u})~\mathrm{d}x ,
# $$
#
# which is the strain that pairs with $\bar{\boldsymbol{\sigma}}$. Its component
# $\bar E^{\text{eff}}_{xx}=u^B_x/L$ is the jump divided by $L$: prescribed in `tension`, computed
# in `tension-stress`. It gives the apparent response of the layer:
#
# * in `tension` and `tension-stress`, the free thickness strain $\bar E^{\text{eff}}_{yy}$ gives
#   the contraction ratio $-\bar E^{\text{eff}}_{yy}/\bar E^{\text{eff}}_{xx}$ and the
#   uniaxial-stress modulus $\bar\sigma_{xx}/\bar E^{\text{eff}}_{xx}$. For a homogeneous
#   plane-strain layer these are $\nu/(1-\nu)$ and $E/(1-\nu^2)$;
# * in `shear`, the apparent shear modulus $\bar\sigma_{xy}/(2\bar E^{\text{eff}}_{xy})$, which is
#   $\mu$ for a homogeneous layer.


# +
def assemble_scalar_global(form: fem.Form):
    """The value of a compiled scalar form, summed over all processes."""
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


# The nodal force F_B of eq:1d-work on the corner B, as a vertex integral in the linear form. Under
# `tension` and `shear` it is zero; under `tension-stress` its x component acts on the free master
# u^B_x, and its y component on u^B_y = 0 is removed by the Dirichlet condition.
TAG_B = 1
vertices_B = mesh.locate_entities_boundary(domain, 0, at_point(*B))
corner_tag = mesh.meshtags(domain, 0, vertices_B, np.full(len(vertices_B), TAG_B, dtype=np.int32))
dP = ufl.Measure("dP", domain=domain, subdomain_data=corner_tag)
nodal_force = fem.Constant(domain, F_B.astype(dtype))
problem = dolfinx_mpc.LinearProblem(
    a,
    rhs + ufl.inner(nodal_force, v_) * dP(TAG_B),
    mpc,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)


# The averages are taken for the solutions of several problems, so their forms are compiled once,
# for a function in V that `average` fills: the leading entries of a constraint space are those of V
u_avg = fem.Function(V, dtype=dtype)


def tensor_forms(T) -> list[list[fem.Form]]:
    return [[fem.form(T[i, j] * ufl.dx, dtype=dtype) for j in range(gdim)] for i in range(gdim)]


stress_forms, gradient_forms = tensor_forms(sigma(u_avg)), tensor_forms(ufl.grad(u_avg))


def average(forms: list[list[fem.Form]], u: fem.Function) -> np.ndarray:
    """The volume average of the tensor of `forms` for the solution `u`."""
    u_avg.x.array[:] = u.x.array[: u_avg.x.array.size]
    return np.array([[assemble_scalar_global(f) for f in row] for row in forms]).real / L**2


def average_stress(u: fem.Function) -> np.ndarray:
    return average(stress_forms, u)


def average_strain(u: fem.Function) -> np.ndarray:
    """The symmetric part of the average gradient."""
    G_avg = average(gradient_forms, u)
    return 0.5 * (G_avg + G_avg.T)


def average_gradient(u: fem.Function) -> np.ndarray:
    return average(gradient_forms, u)


mu_uniform = E_uniform / (2 * (1 + nu))
# -

# ### Verification: a homogeneous cell
#
# With the same stiffness in both phases, the exact solution is affine:
#
# * in `tension` the layer is in uniaxial stress, so
#   $u_x=\bar\varepsilon X$, $u_y=-k\,\bar\varepsilon Y$ with $k=\lambda/(\lambda+2\mu)=\nu/(1-\nu)$;
# * in `tension-stress` the same holds with the unknown strain
#   $\bar\varepsilon=\bar S_{xx}(1-\nu^2)/E$, and the jump $u^B_x=\bar\varepsilon L$ must be found
#   by the solver;
# * in `shear`, $\mathbf{u}=(\gamma Y, 0)$.
#
# The exact solution lies in the finite element space, so it must be reproduced to round-off
# whatever the mesh. A larger error would reveal a wrong constraint.

# +
set_young_modulus(E_uniform)
uh = problem.solve()
assert isinstance(uh, fem.Function)
k = nu / (1 - nu)
u_exact = fem.Function(V, dtype=dtype)


def exact(x):
    values = G @ x[:gdim]
    if args.load in ("tension", "tension-stress"):
        eps_xx = args.strain if args.load == "tension" else args.stress * (1 - nu**2) / E_uniform
        values[0] = eps_xx * x[0]
        values[1] = -k * eps_xx * x[1]
    return values


u_exact.interpolate(exact)
error = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx, dtype=dtype))))
norm_exact = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"---- Homogeneous cell ----\n  L2(u_h - u_exact) = {error:.3e}  (should be round-off)")
# Relative to the size of the solution, with a bound from the precision of the mesh coordinates
assert error < 50 * np.sqrt(np.finfo(domain.geometry.x.dtype).resolution) * norm_exact
# -

# ### Rotation constraint and a cell without symmetry
#
# The layer can rotate rigidly, and $u^B_y=0$ removes this rotation: it fixes the axis of the
# layer along $x$. It must remove *only* the rotation. With traction-free faces the balance of
# moments requires $\bar\sigma_{yx}=0$, so the reaction of this constraint, $F_y=L\bar\sigma_{yx}$, must
# vanish, and TOP must remain free to slide with respect to BOTTOM ($\bar E^{\text{eff}}_{xy}\neq0$). With the
# circular inclusion the symmetry of the cell gives $\bar E^{\text{eff}}_{xy}=0$ anyway, so a constraint that
# also blocked the sliding would go unnoticed. With `--inclusion ellipse`, an inclined elliptical
# inclusion, the layer shears under tension, and both properties are visible.

# ### The heterogeneous cell

# +
set_young_modulus(50.0 * E_uniform)
uh = problem.solve()
assert isinstance(uh, fem.Function)
sigma_bar, E_eff = average_stress(uh), average_strain(uh)
if comm.rank == 0:
    print("---- Heterogeneous cell ----")
    print(
        f"  sigma_bar = [[{sigma_bar[0, 0]:.6e}, {sigma_bar[0, 1]:.6e}],"
        f" [{sigma_bar[1, 0]:.6e}, {sigma_bar[1, 1]:.6e}]]"
    )
    print(f"  E_eff     = [[{E_eff[0, 0]:.6e}, {E_eff[0, 1]:.6e}], [{E_eff[1, 0]:.6e}, {E_eff[1, 1]:.6e}]]")
    if stress_control:
        print(f"  prescribed S_xx = {args.stress:.6e}, computed jump u^B_x / L = {E_eff[0, 0]:.6e}")
    if args.load in ("tension", "tension-stress"):
        print(
            f"  contraction ratio -E_yy/E_xx      = {-E_eff[1, 1] / E_eff[0, 0]:.6f}"
            f"  (homogeneous matrix: nu/(1-nu) = {nu / (1 - nu):.6f})"
        )
        print(
            f"  uniaxial-stress modulus s_xx/E_xx = {sigma_bar[0, 0] / E_eff[0, 0]:.6f}"
            f"  (homogeneous matrix: E/(1-nu^2) = {E_uniform / (1 - nu**2):.6f})"
        )
        print(
            f"  sliding of TOP: E_xy = {E_eff[0, 1]:.6e};"
            f" reaction of u^B_y = 0: s_yx = {sigma_bar[1, 0]:.3e}  (balance of moments: 0)"
        )
    else:
        print(
            f"  apparent shear modulus s_xy/(2 E_xy) = {sigma_bar[0, 1] / (2 * E_eff[0, 1]):.6f}"
            f"  (homogeneous matrix: mu = {mu_uniform:.6f})"
        )
# -

# ## Visualization
#
# As in the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`:
# the microstructure, and the deformed cell coloured by
# $|\mathbf{u}|$, drawn over the outline of the undeformed cell. The third panel is the
# fluctuation $\mathbf{w}=\mathbf{u}-\langle\nabla\mathbf{u}\rangle\mathbf{X}$, periodic in $x$.
# The solution lives in the extended space of the constraint; its leading entries are the values
# of the original space.

# + tags=["hide-input"]
try:
    import pyvista
except ModuleNotFoundError:
    pyvista = None

if pyvista is not None:
    owned = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    u_plot = fem.Function(V, dtype=dtype)
    u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
    w_plot = fem.Function(V, dtype=dtype)
    H_eff = average_gradient(uh)
    w_plot.interpolate(lambda x: H_eff @ x[:gdim])
    w_plot.x.array[:] = u_plot.x.array - w_plot.x.array

    def to_grid(f: fem.Function, name: str):
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned))
        values = np.zeros((grid.n_points, 3))
        values[:, :gdim] = f.x.array.real[: grid.n_points * gdim].reshape(-1, gdim)
        grid.point_data[name] = values
        grid.point_data[f"|{name}|"] = np.linalg.norm(values, axis=1)
        vmax = comm.allreduce(float(np.abs(values).max()) if values.size else 0.0, op=MPI.MAX)
        return comm.gather(grid, root=0), vmax

    grids_u, umax = to_grid(u_plot, "u")
    grids_w, wmax = to_grid(w_plot, "w")
    phase = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, tdim, owned))
    phase.cell_data["E"] = E.x.array.real[: owned.size]
    phases = comm.gather(phase, root=0)
    suffix = "" if args.inclusion == "circle" else "_ellipse"
    load_title = {
        "tension": f"Horizontal tension: E_xx = {args.strain:g} prescribed, free faces",
        "shear": f"Simple shear: BOTTOM u = 0, TOP u = (gamma L, 0), gamma = {args.strain:g}",
        "tension-stress": f"Horizontal tension under stress control: S_xx = {args.stress:g}, free faces",
    }[args.load]
    if comm.rank == 0:
        outline = pyvista.Rectangle([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0)])
        plotter = pyvista.Plotter(shape=(1, 3), window_size=[1500, 520])
        plotter.subplot(0, 0)
        plotter.add_text(f"Microstructure\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)", font_size=10)
        for p in phases:
            plotter.add_mesh(
                p,
                scalars="E",
                cmap="viridis",
                show_edges=False,
                scalar_bar_args={"n_labels": 2, "fmt": "%.0f", "position_x": 0.2, "width": 0.6},
            )
        plotter.view_xy()
        for col, (grids, vmax, name, title) in enumerate(
            [
                (grids_u, umax, "u", f"Deformed cell\n{load_title}"),
                (grids_w, wmax, "w", "Fluctuation\nw = u - <grad u> X"),
            ],
            start=1,
        ):
            factor = 0.1 * L / vmax  # largest displacement drawn as 10 % of the cell size
            plotter.subplot(0, col)
            plotter.add_text(f"{title}\n(amplified x{factor:.0f})", font_size=10)
            for g_ in grids:
                plotter.add_mesh(
                    g_.warp_by_vector(name, factor=factor),
                    scalars=f"|{name}|",
                    cmap="viridis",
                    scalar_bar_args={
                        "fmt": "%.1e",
                        "n_labels": 3,
                        "position_x": 0.2,
                        "width": 0.6,
                        "title": f"|{name}|",
                    },
                )
            plotter.add_mesh(outline, style="wireframe", color="black", line_width=2)
            plotter.view_xy()
        if pyvista.OFF_SCREEN:
            plotter.screenshot(f"demo_periodic_layer_{args.load}{suffix}.png")
        else:
            plotter.show()
# -

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: layer-
# ```

# +
del problem
PETSc.garbage_cleanup(comm)
# -

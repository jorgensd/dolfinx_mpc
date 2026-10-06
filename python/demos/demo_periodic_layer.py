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
from dolfinx import default_scalar_type, fem, mesh, plot
from dolfinx.fem import petsc as fem_petsc

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
    args.stress = {"circle": 0.15371273, "ellipse": 0.1569839}[args.inclusion]

comm = MPI.COMM_WORLD
dtype = np.dtype(default_scalar_type)
L = 1.0  # side of the square cell
# -

# ## Geometry, mesh and material
#
# The cell is the square $\Omega=(0,L)^2$ with corners
#
# $$
# A=(0,0),\qquad B=(L,0),\qquad C=(L,L),\qquad D=(0,L).
# $$
#
# As in the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>`,
# the mesh is structured, so that every node of RIGHT has
# a partner on LEFT at the same height. Both phases are linear, isotropic and elastic, in plane
# strain; the Young modulus is a cellwise constant function.

# +
N = 32
domain = mesh.create_rectangle(comm, [[0, 0], [L, L]], [N, N])
gdim = tdim = domain.geometry.dim

E_uniform, nu = 10.0, 0.3
E = fem.Function(fem.functionspace(domain, ("Discontinuous Lagrange", 0)))
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32))
if args.inclusion == "circle":
    inclusion = (midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2 < 0.25**2
else:  # ellipse with semi-axes 0.38 and 0.15, inclined at 30 degrees
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    xr = c30 * (midpoints[:, 0] - 0.5) + s30 * (midpoints[:, 1] - 0.5)
    yr = -s30 * (midpoints[:, 0] - 0.5) + c30 * (midpoints[:, 1] - 0.5)
    inclusion = (xr / 0.38) ** 2 + (yr / 0.15) ** 2 < 1.0


def set_young_modulus(E_inclusion: float):
    E.x.array[: len(inclusion)] = np.where(inclusion, E_inclusion, E_uniform)
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
# \mathbf{u}^A=\mathbf{0},\qquad
# \mathbf{u}^{\text{RIGHT}} = \mathbf{u}^{\text{LEFT}} + \mathbf{u}^B,\qquad
# \mathbf{u}^C = \mathbf{u}^D + \mathbf{u}^B,
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
# \int_{\text{LEFT}\cup\text{RIGHT}}\mathbf{T}\cdot\mathbf{v}\,\mathrm{d}s
# = \int_{\text{RIGHT}}\mathbf{T}(\mathbf{X})\cdot
#   \bigl(\mathbf{v}(\mathbf{X})-\mathbf{v}(\mathbf{X}-L\mathbf{e}_x)\bigr)\,\mathrm{d}s
# = \int_{\text{RIGHT}}\mathbf{T}\cdot\mathbf{v}^B\,\mathrm{d}s
# = \mathbf{F}\cdot\mathbf{v}^B,
# \qquad \mathbf{F}=\int_{\text{RIGHT}}\mathbf{T}\,\mathrm{d}s .
# $$ (eq:1d-work)
#
# BOTTOM and TOP are traction-free, so the principle of virtual work reads
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})\,\mathrm{d}\Omega
# = \mathbf{F}\cdot\mathbf{v}^B
# $$
#
# for all $\mathbf{v}$ satisfying {eq}`eq:1d-periodic` and vanishing where $\mathbf{u}$ is
# prescribed. The right-hand side is the virtual work of a point force $\mathbf{F}$ applied at
# $B$: a prescribed $F_i$ is a **nodal force** on $u^B_i$, which becomes an unknown. With
# $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, the divergence theorem gives
#
# $$
# \int_\Omega \sigma_{i1}\,\mathrm{d}\Omega = L\,F_i + \int_{\text{BOTTOM}\cup\text{TOP}} T_i X_1\,\mathrm{d}s .
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

tol = 1e-10 * L


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
    g = fem.Function(Vc)
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
# ### Known jump: `tension` and `shear`
#
# The jump is known: $\mathbf{u}^B=(\bar\varepsilon L,0)$ in `tension`, $\mathbf{u}^B=\mathbf{0}$ in
# `shear`. The constraint on
# RIGHT is therefore a periodic constraint with a constant term,
# $\mathbf{u}^{\text{RIGHT}}=\mathbf{u}^{\text{LEFT}}+\mathbf{g}$ with $\mathbf{g}=\mathbf{u}^B$. As in
# the {doc}`periodic homogenization demo <demo_periodic_homogenization>`, it is built with
# {py:meth}`create_periodic_constraint_geometrical
# <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>`. The constant term is
# supplied as a function `g` to the constructor of
# {py:class}`MultiPointConstraint <dolfinx_mpc.MultiPointConstraint>`: it equals $\mathbf{u}^B$ on
# RIGHT and zero elsewhere.
#
# The corner equation $\mathbf{u}^C=\mathbf{u}^D+\mathbf{u}^B$ has two masters, the free corner
# $D$ and the prescribed corner $B$. It is built with
# {py:meth}`create_general_constraint <dolfinx_mpc.MultiPointConstraint.create_general_constraint>`,
# which takes the slave and its masters by coordinates. DOLFINx-MPC removes the Dirichlet master
# $B$ from the list and folds its value into the constraint. In `shear` the corner $C$ lies on the
# prescribed face TOP, so its Dirichlet value already satisfies the equation. A dof cannot be both
# a slave and a Dirichlet dof, so the corner constraint is only added in `tension`.
#
# ### Unknown jump: `tension-stress`
#
# Now $u^B_x$ is unknown, so the constraint $u_x^{\text{RIGHT}}=u_x^{\text{LEFT}}+u_x^B$ has *two*
# masters: the periodic partner and $B$. Both components of RIGHT, and the corner $C$, are
# therefore built with `create_general_constraint`. For $u_y$ the master $B$ carries the
# Dirichlet condition $u^B_y=0$, and DOLFINx-MPC folds it into the constraint. For $u_x$ it stays
# a master, i.e. a degree of freedom of the reduced system. Every rank needs the full list of
# slaves, so their coordinates are gathered from all processes.


# +
def key(point) -> bytes:
    return np.array([point[0], point[1], 0.0], dtype=domain.geometry.x.dtype).tobytes()


uB = G @ np.array([L, 0.0])


def offset(x):
    values = np.zeros((gdim, x.shape[1]), dtype=dtype)
    values[:, right_open(x)] = uB.reshape(-1, 1)
    return values


x_dofs = V.tabulate_dof_coordinates()
if stress_control:
    mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs)
    slaves = np.unique(np.round(np.vstack(comm.allgather(x_dofs[right_open(x_dofs.T)])), 14), axis=0)
    for c in range(gdim):
        slave_master = {key(xs): {key((0.0, xs[1])): 1.0, key(B): 1.0} for xs in slaves}
        slave_master[key(C)] = {key(D): 1.0, key(B): 1.0}
        mpc.create_general_constraint(slave_master, subspace_slave=c, subspace_master=c)
else:
    g = fem.Function(V, dtype=dtype)
    g.interpolate(offset)
    g.x.scatter_forward()
    mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs, rhs_coeffs=g)
    mpc.create_periodic_constraint_geometrical(
        V, right_open, lambda x: np.vstack([x[0] - L, x[1], x[2]]), bcs, scale=dtype.type(1.0)
    )
    if args.load == "tension":
        for c in range(gdim):
            mpc.create_general_constraint({key(C): {key(D): 1.0, key(B): 1.0}}, subspace_slave=c, subspace_master=c)
mpc.finalize()  # collective: every rank must reach this
# -

# ## Solution and post-processing
#
# We report the macroscopic stress, the volume average of $\boldsymbol{\sigma}$ over the whole
# cell,
#
# $$
# \bar{\boldsymbol{\sigma}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\sigma}\,\mathrm{d}\Omega ,
# $$
#
# and the macroscopic strain, the volume average of $\boldsymbol{\epsilon}$,
#
# $$
# \bar{\mathbf{E}}^{\text{eff}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\epsilon}(\mathbf{u})\,\mathrm{d}\Omega ,
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
def assemble(form) -> float:
    return comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


a_form, rhs_form = fem.form(a), fem.form(rhs)
dof_Bx = fem.locate_dofs_geometrical((V.sub(0), V.sub(0).collapse()[0]), at_point(*B))[0]
n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs


def solve() -> fem.Function:
    """Assemble and solve as dolfinx_mpc.LinearProblem does, adding the nodal force F_B
    of eq:1d-work on the free master dof u^B_x (zero unless the jump is stress-controlled)."""
    mpc.update_constants()
    Amat = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    Amat.assemble()
    b = dolfinx_mpc.assemble_vector(rhs_form, mpc)
    dolfinx_mpc.apply_lifting(b, [a_form], bcs=[bcs], constraint=mpc)
    dolfinx_mpc.apply_mpc_lifting(b, [a_form], constraint=mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)
    b.array[dof_Bx[dof_Bx < n_owned]] += F_B[0]  # on the process that owns B
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(Amat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setErrorIfNotConverged(True)
    uh = fem.Function(mpc.function_space)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    mpc.homogenize(uh)
    mpc.backsubstitution(uh)
    ksp.destroy(), Amat.destroy(), b.destroy()
    return uh


def average_stress(u: fem.Function) -> np.ndarray:
    s = sigma(u)
    return np.array([[assemble(s[i, j] * ufl.dx) for j in range(gdim)] for i in range(gdim)]) / L**2


def average_strain(u: fem.Function) -> np.ndarray:
    e = epsilon(u)
    return np.array([[assemble(e[i, j] * ufl.dx) for j in range(gdim)] for i in range(gdim)]) / L**2


def average_gradient(u: fem.Function) -> np.ndarray:
    G_u = ufl.grad(u)
    return np.array([[assemble(G_u[i, j] * ufl.dx) for j in range(gdim)] for i in range(gdim)]) / L**2


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
uh = solve()
k = nu / (1 - nu)
u_exact = fem.Function(V)


def exact(x):
    values = G @ x[:gdim]
    if args.load in ("tension", "tension-stress"):
        eps_xx = args.strain if args.load == "tension" else args.stress * (1 - nu**2) / E_uniform
        values[0] = eps_xx * x[0]
        values[1] = -k * eps_xx * x[1]
    return values


u_exact.interpolate(exact)
error = np.sqrt(assemble(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
if comm.rank == 0:
    print(f"---- Homogeneous cell ----\n  L2(u_h - u_exact) = {error:.3e}  (should be round-off)")
assert error < 1e-10
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
uh = solve()
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
    u_plot = fem.Function(V)
    u_plot.x.array[:] = uh.x.array[: u_plot.x.array.size]
    w_plot = fem.Function(V)
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
    phase.cell_data["E"] = E.x.array[: owned.size]
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

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: layer-
# ```

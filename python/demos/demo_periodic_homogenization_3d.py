# # Periodic boundary conditions for a three-dimensional representative volume element
# **Authors** Jørgen S. Dokken, Maria Bruno
#
# **License** MIT

# +
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot

import dolfinx_mpc
from dolfinx_mpc import MultiPointConstraint, dofs_at_point

# -

# This demo extends the two-dimensional periodic unit cell of the
# {doc}`periodic homogenization demo <demo_periodic_homogenization>` to a cube
# $\Omega=(0,L)^3$ in small-strain linear elasticity. The cell is loaded first by
# a prescribed macroscopic strain, then by a prescribed macroscopic stress. The constraints extend
# to three dimensions the corner-node formulation of the periodic boundary conditions of
# {cite}`homog3d-Danas2017` (Appendix B); periodic conditions on cubic cells were first used in
# {cite}`homog3d-LopezPamiesGoudarziDanas2013`.

# +
comm = MPI.COMM_WORLD
N = 16
L = 1.0
dtype = np.dtype(default_scalar_type)
domain = mesh.create_box(comm, [[0, 0, 0], [L, L, L]], [N, N, N], mesh.CellType.hexahedron, dtype=default_real_type)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
# -

# ## The periodicity condition
#
# The displacement is split into an affine part carrying the macroscopic strain and a
# periodic fluctuation,
#
# $$
# \begin{aligned}
# \mathbf{u}(\mathbf{X}) &= \bar{\mathbf{H}}\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),\\
# \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) &= \mathbf{u}^*(\mathbf{X}),\quad i=1,2,3,
# \end{aligned}
# $$
#
# where $\bar{\mathbf{H}}$ is the average displacement gradient. Only its symmetric part, the
# macroscopic strain $\bar{\mathbf{E}}=\operatorname{sym}\bar{\mathbf{H}}$ of components
# $\bar E_{ij}$, produces stress; the skew part
# $\bar{\boldsymbol{\omega}}=\tfrac12(\bar{\mathbf{H}}-\bar{\mathbf{H}}^T)$ is a rigid rotation, which
# does no work. Any skew part gives the same stress, and we choose
# $\bar\omega_{ij}=\bar E_{ij}$ above the diagonal, $\bar\omega_{ij}=-\bar E_{ij}$ below it. Then
# $\bar{\mathbf{H}}=\bar{\mathbf{E}}+\bar{\boldsymbol{\omega}}$ is upper triangular,
#
# $$
# \bar{\mathbf{H}}=\begin{pmatrix}\bar E_{11}&2\bar E_{12}&2\bar E_{13}\\
# 0&\bar E_{22}&2\bar E_{23}\\ 0&0&\bar E_{33}\end{pmatrix}.
# $$ (eq:3d-H)
#
# Four corners carry the macroscopic strain: $A=(0,0,0)$, $B=(L,0,0)$, $D=(0,L,0)$ and
# $E=(0,0,L)$. Fixing $\mathbf{u}^A=\mathbf{0}$ to remove the rigid translation,
#
# $$
# \begin{aligned}
# \mathbf{u}^B &= \bar{\mathbf{H}}L\mathbf{e}_1=L\begin{pmatrix}\bar E_{11}\\0\\0\end{pmatrix},\\
# \mathbf{u}^D &= \bar{\mathbf{H}}L\mathbf{e}_2=L\begin{pmatrix}2\bar E_{12}\\ \bar E_{22}\\0\end{pmatrix},\\
# \mathbf{u}^E &= \bar{\mathbf{H}}L\mathbf{e}_3=L\begin{pmatrix}2\bar E_{13}\\2\bar E_{23}\\ \bar E_{33}\end{pmatrix}.
# \end{aligned}
# $$ (eq:3d-corners)
#
# ### Relations between nodes
#
# Periodicity of $\mathbf{u}^*$ ties every node on the faces $X_1=L$, $X_2=L$, $X_3=L$ to its
# image on the opposite faces. For a node $\mathbf{X}$ on one of these faces, let
# $s_i=1$ if $X_i=L$ and $s_i=0$ otherwise, and let $\mathbf{X}^-=\mathbf{X}-L\,\mathbf{s}$ be its
# image. Then
#
# $$
# \mathbf{u}(\mathbf{X}) = \mathbf{u}(\mathbf{X}^-) + s_1\,\mathbf{u}^B + s_2\,\mathbf{u}^D + s_3\,\mathbf{u}^E .
# $$ (eq:3d-periodic)
#
# Written out for the faces, edges and corners of the cube:
#
# | slave | master image $\mathbf{X}^-$ | relation |
# |---|---|---|
# | face $X_1=L$ | $(0,X_2,X_3)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^B$ |
# | face $X_2=L$ | $(X_1,0,X_3)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^D$ |
# | face $X_3=L$ | $(X_1,X_2,0)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^E$ |
# | edge $X_1=X_2=L$ | $(0,0,X_3)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^B+\mathbf{u}^D$ |
# | edge $X_1=X_3=L$ | $(0,X_2,0)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^B+\mathbf{u}^E$ |
# | edge $X_2=X_3=L$ | $(X_1,0,0)$ | $\mathbf{u}=\mathbf{u}(\mathbf{X}^-)+\mathbf{u}^D+\mathbf{u}^E$ |
# | corner $(L,L,0)$ | $A$ | $\mathbf{u}=\mathbf{u}^B+\mathbf{u}^D$ |
# | corner $(L,0,L)$ | $A$ | $\mathbf{u}=\mathbf{u}^B+\mathbf{u}^E$ |
# | corner $(0,L,L)$ | $A$ | $\mathbf{u}=\mathbf{u}^D+\mathbf{u}^E$ |
# | corner $(L,L,L)$ | $A$ | $\mathbf{u}=\mathbf{u}^B+\mathbf{u}^D+\mathbf{u}^E$ |
#
# The corners $B$, $D$, $E$ themselves are masters, not slaves: for them
# {eq}`eq:3d-periodic` reduces to an identity.
#
# ### Building the constraint
#
# The relations are imposed with the same DOLFINx-MPC tools as in two dimensions, in two steps:
#
# 1. {py:meth}`create_periodic_constraint_geometrical
#    <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>` ties every node with
#    some $X_i=L$, except $B$, $D$, $E$, to its image, $\mathbf{u}(\mathbf{X})=\mathbf{u}(\mathbf{X}^-)$,
#    with the map $\mathbf{X}\mapsto\mathbf{X}^-$.
# 2. {py:meth}`add_master_from_point <dolfinx_mpc.MultiPointConstraint.add_master_from_point>`
#    adds the dofs at a point to the right-hand side of the constraints of the slaves it marks.
#    One call per direction adds the corner of that direction to the nodes with $X_i=L$, which
#    gives $s_1\mathbf{u}^B+s_2\mathbf{u}^D+s_3\mathbf{u}^E$.
#
# The same constraint serves both controls. Under **strain control** $\mathbf{u}^B$,
# $\mathbf{u}^D$, $\mathbf{u}^E$ are known and carried by Dirichlet conditions on the corners,
# whose values are folded into the constraint. Under **stress control** they are unknowns, and
# stay masters.

# +
xdt = domain.geometry.x.dtype
# Tolerance of the checks below, from the precision of the mesh coordinates
atol = 50 * np.sqrt(np.finfo(xdt).resolution)
# Largest distance between a node and a point it is located at, from the rounding of the coordinates
geom_tol = 500 * np.finfo(xdt).eps * L
A_pt, B_pt, D_pt, E_pt = (np.array(p, dtype=xdt) for p in ([0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]))
corner_of = {0: B_pt, 1: D_pt, 2: E_pt}  # corner carrying the jump in direction i


def point(p):
    return lambda x: np.all(np.isclose(x, np.reshape(p, (3, 1)), atol=geom_tol), axis=0)


def is_master_corner(x):
    return np.logical_or.reduce([point(p)(x) for p in corner_of.values()])


def periodic_nodes(x):
    """The slaves: the nodes with some X_i = L, except the corners B, D and E."""
    return np.any(np.isclose(x, L, atol=geom_tol), axis=0) & ~is_master_corner(x)


def to_image(x):
    """X -> X^- = X - L s."""
    out = x.copy()
    out[np.isclose(x, L, atol=geom_tol)] = 0.0
    return out


def periodic_cell_constraint(bcs_: list[fem.DirichletBC]) -> MultiPointConstraint:
    """The constraint {eq}`eq:3d-periodic`, finalized, with the conditions `bcs_` folded in."""
    constraint = MultiPointConstraint(V, dtype=dtype, bcs=bcs_)
    constraint.create_periodic_constraint_geometrical(V, periodic_nodes, to_image, bcs_, scale=dtype.type(1.0))
    for i, corner_point in corner_of.items():  # s_i u^{corner} on the nodes with X_i = L

        def on_side(x, i=i):
            return periodic_nodes(x) & np.isclose(x[i], L, atol=geom_tol)

        constraint.add_master_from_point(V, on_side, 1.0, corner_point)
    constraint.finalize()  # collective: every rank must reach this
    return constraint


# -

# ## Solver
#
# The constrained stiffness matrix does not depend on the values of the Dirichlet conditions, so
# it is assembled and factorized once per material and constraint. `ConstrainedSolver` is a
# {py:class}`dolfinx_mpc.LinearProblem` that assembles its matrix when it is created, and whose
# `solve` only updates the constraint offsets with
# {py:meth}`update_constants <dolfinx_mpc.MultiPointConstraint.update_constants>` and assembles
# the right-hand side. The nodal forces of stress control are part of the linear form, as vertex
# integrals.


# +
class ConstrainedSolver(dolfinx_mpc.LinearProblem):
    """A {py:class}`dolfinx_mpc.LinearProblem` whose matrix is assembled, and factorized, once:
    each solve assembles the right-hand side only."""

    def __init__(self, a_ufl, L_ufl, mpc: MultiPointConstraint, bcs: list[fem.DirichletBC]):
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }
        super().__init__(a_ufl, L_ufl, mpc, bcs=bcs, petsc_options=petsc_options)
        self.constraint = mpc
        dolfinx_mpc.assemble_matrix(self.a, mpc, bcs=bcs, A=self.A, bc_data=self._bc_data)
        self.A.assemble()

    def solve(self) -> fem.Function:
        self.constraint.update_constants()
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx_mpc.assemble_vector(self.L, self.constraint, self.b)
        dolfinx_mpc.apply_lifting(self.b, [self.a], bcs=[self.bcs], constraint=self.constraint, bc_data=self._bc_data)
        dolfinx_mpc.apply_mpc_lifting(self.b, [self.a], constraint=self.constraint)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self.b, self.bcs)
        self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        dolfinx.fem.petsc.assign(self.x, self.u)
        self.constraint.homogenize(self.u)
        self.constraint.backsubstitution(self.u)
        return self.u


def sigma(w, mu_, lmbda_):
    eps = ufl.sym(ufl.grad(w))
    return 2 * mu_ * eps + lmbda_ * ufl.tr(eps) * ufl.Identity(gdim)


def elasticity_forms(mu_, lmbda_):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_ = ufl.inner(sigma(u, mu_, lmbda_), ufl.sym(ufl.grad(v))) * ufl.dx
    L_ = ufl.inner(fem.Constant(domain, np.zeros(gdim, dtype=dtype)), v) * ufl.dx
    return a_, L_


def assemble_scalar_global(form: fem.Form):
    """The value of a compiled scalar form, summed over all processes."""
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


def fmt(m: np.ndarray) -> str:
    return "[" + ", ".join("[" + ", ".join(f"{v:.6e}" for v in row) + "]" for row in m) + "]"


E_uniform, nu = 10.0, 0.3
mu_uniform = fem.Constant(domain, default_scalar_type(E_uniform / (2 * (1 + nu))))
lmbda_uniform = fem.Constant(domain, default_scalar_type(E_uniform * nu / ((1 + nu) * (1 - 2 * nu))))
a, Lform = elasticity_forms(mu_uniform, lmbda_uniform)

Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q, dtype=dtype)
tdim = domain.topology.dim
n_cells = domain.topology.index_map(tdim).size_local
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(n_cells, dtype=np.int32))
# The owned cells of the inclusion, a sphere of radius L/4 at the centre of the cell
cells0 = np.flatnonzero(np.sum((midpoints - 0.5 * L) ** 2, axis=1) < (0.25 * L) ** 2).astype(np.int32)
E.interpolate(lambda x_: np.full(x_.shape[1], E_uniform))
E.interpolate(lambda x_: np.full(x_.shape[1], 50.0 * E_uniform), cells0=cells0)
E.x.scatter_forward()
mu_field = E / (2 * (1 + nu))
lmbda_field = E * nu / ((1 + nu) * (1 - 2 * nu))
a_het, L_het = elasticity_forms(mu_field, lmbda_field)

# The stress of the heterogeneous cell is averaged for the solutions of several problems, so its
# forms are compiled once, for a function in V that average_stress fills: the leading entries of
# a constraint space are those of V
u_avg = fem.Function(V, dtype=dtype)
sigma_avg_ufl = sigma(u_avg, mu_field, lmbda_field)
sigma_forms = [[fem.form(sigma_avg_ufl[i, j] * ufl.dx, dtype=dtype) for j in range(gdim)] for i in range(gdim)]


def average_stress(uh_: fem.Function) -> np.ndarray:
    """The volume average of the stress of the heterogeneous cell for the solution `uh_`."""
    u_avg.x.array[:] = uh_.x.array[: u_avg.x.array.size]
    return np.array([[assemble_scalar_global(s_ij) for s_ij in row] for row in sigma_forms]).real / L**3


x = ufl.SpatialCoordinate(domain)
# -

# ## Strain control
#
# The corners $A$, $B$, $D$, $E$ carry the Dirichlet conditions {eq}`eq:3d-corners`; their values
# are set by `set_strain`.


# +
def point_bc(p) -> tuple[fem.DirichletBC, fem.Function]:
    fn = fem.Function(V, dtype=dtype)
    return fem.dirichletbc(fn, fem.locate_dofs_geometrical(V, point(p))), fn


def upper_triangular(E_case: np.ndarray) -> np.ndarray:
    """H_bar of eq:3d-H for a symmetric macroscopic strain."""
    return np.triu(2 * E_case) - np.diag(np.diag(E_case))


corners = (A_pt, B_pt, D_pt, E_pt)
corner_bcs = [point_bc(p) for p in corners]
bcs = [bc for bc, _ in corner_bcs]
bc_values = [value for _, value in corner_bcs]
mpc = periodic_cell_constraint(bcs)


def set_strain(E_case: np.ndarray):
    """Dirichlet values eq:3d-corners; the solver calls update_constants to fold them in."""
    H = upper_triangular(E_case)
    for fn, p in zip(bc_values, corners):
        fn.interpolate(lambda xx: np.tile((H @ p).reshape(-1, 1), xx.shape[1]))


# -

# ### Homogeneous cell
#
# For a homogeneous material the fluctuation vanishes and the exact solution is the affine field
# $\bar{\mathbf{H}}\mathbf{X}$, which lies in the space of trilinear elements. It must be reproduced
# to round-off.

# +
E_bar = np.array([[0.02, 0.005, 0.0], [0.005, -0.015, 0.004], [0.0, 0.004, 0.01]])
set_strain(E_bar)
uh = ConstrainedSolver(a, Lform, mpc, bcs).solve()
u_affine = ufl.dot(ufl.as_tensor(upper_triangular(E_bar)), x)
diff = uh - u_affine
error = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_affine = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_affine, u_affine) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < atol * norm_affine
# -

# ### Heterogeneous cell
#
# The matrix contains a spherical inclusion of radius $L/4$, fifty times stiffer. With no
# macroscopic strain the cell carries no load, so the average stress must vanish. The cell has
# cubic symmetry, so an isotropic macroscopic strain must give an isotropic average stress. Both
# are checked to round-off.

# +
solver_het = ConstrainedSolver(a_het, L_het, mpc, bcs)


def homogenized_stress(solver: ConstrainedSolver, E_case: np.ndarray) -> tuple[fem.Function, np.ndarray]:
    set_strain(E_case)
    # The solver returns its own solution, which the next solve overwrites
    uh_case = solver.solve().copy()
    return uh_case, average_stress(uh_case)


_, sigma_zero = homogenized_stress(solver_het, np.zeros((gdim, gdim)))
_, sigma_iso = homogenized_stress(solver_het, 0.02 * np.eye(gdim))
uh_general, sigma_general = homogenized_stress(solver_het, E_bar)
if comm.rank == 0:
    print(f"----No macroscopic strain----\n  sigma_bar = {fmt(sigma_zero)}  (should vanish)")
    print(f"----Isotropic macroscopic strain----\n  sigma_bar = {fmt(sigma_iso)}")
    print(f"----General macroscopic strain----\n  sigma_bar = {fmt(sigma_general)}")
sigma_scale = np.abs(sigma_iso).max()
assert np.abs(sigma_zero).max() < atol * sigma_scale
off_diagonal = sigma_iso - np.diag(np.diag(sigma_iso))
assert np.abs(off_diagonal).max() < atol * sigma_scale
assert np.ptp(np.diag(sigma_iso)) < atol * sigma_scale
# -

# ## Stress control
#
# So far the strain was prescribed and the average stress computed from the solution,
#
# $$
# \bar{\boldsymbol{\sigma}} = \frac{1}{|\Omega|}\int_\Omega \boldsymbol{\sigma}(\mathbf{u})~\mathrm{d}x .
# $$
#
# We now prescribe its value instead, $\bar{\boldsymbol\sigma}=\bar{\mathbf{S}}$. The displacements
# of $B$, $D$, $E$ become unknowns, except the three components that vanish in
# {eq}`eq:3d-corners`: $u^B_2=u^B_3=u^D_3=0$. These remove the three rigid rotations, which the
# periodicity no longer excludes once $\bar{\mathbf{H}}$ is unknown.
#
# This changes how the corners enter the problem. The constraint is the same, with the corners as
# masters of the periodic constraints. Under strain control $\mathbf{u}^B$, $\mathbf{u}^D$,
# $\mathbf{u}^E$ are known: they are Dirichlet values on the corners, folded into the constraint.
# Under stress control they are unknowns, free masters, and the prescribed stress enters the
# right-hand side of the equations as point forces on their degrees of freedom, derived below.
# Each corner degree of freedom gets either a prescribed displacement or a prescribed force, never
# both.
#
# | | strain control | stress control |
# |---|---|---|
# | corner displacements | prescribed, Dirichlet conditions | unknowns, masters of the constraints |
# | periodic constraints | constant from the corner displacements | no constant |
# | right-hand side | no load | point forces $|\Gamma|\bar S_{ij}$ on the corners |
#
# ### The prescribed stress as nodal forces
#
# The stress is periodic, while the outward normals of opposite faces are opposite, so the
# tractions $\mathbf{T}=\boldsymbol{\sigma}\mathbf{n}$ are anti-periodic on opposite faces. For a
# field $\mathbf{v}$ satisfying {eq}`eq:3d-periodic`, the contributions of $\mathbf{v}(\mathbf{X}^-)$
# on opposite faces cancel and the work of the tractions reduces to
#
# $$
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}~\mathrm{d}s
# = v^B_i\int_{X_1=L}T_i~\mathrm{d}s + v^D_i\int_{X_2=L}T_i~\mathrm{d}s
# + v^E_i\int_{X_3=L}T_i~\mathrm{d}s .
# $$
#
# With $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, the divergence theorem gives
# $\int_{\partial\Omega}T_iX_j~\mathrm{d}s=\int_\Omega\sigma_{ij}~\mathrm{d}x=|\Omega|\,\bar\sigma_{ij}$.
# $X_j=L$ on the face $X_j=L$ and $X_j=0$ on its opposite face, while on the other four faces the
# anti-periodic tractions cancel at each $X_j$. So the integral over the face $X_j=L$ is
# $|\Gamma|\bar\sigma_{ij}$, with $|\Gamma|=|\Omega|/L=L^2$ the area of a face. The principle of virtual work,
# with $\bar\sigma_{ij}=\bar S_{ij}$, becomes
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})~\mathrm{d}x
# = |\Gamma|\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i + \bar S_{i3}\,v^E_i\right),
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=v^B_3=v^D_3=0$. The prescribed stress
# is a set of nodal forces. The reactions of the three rotation constraints are
# $|\Gamma|\bar S_{21}$, $|\Gamma|\bar S_{31}$ and $|\Gamma|\bar S_{32}$, which equal
# $|\Gamma|\bar S_{12}$, $|\Gamma|\bar S_{13}$ and $|\Gamma|\bar S_{23}$ as the stress is symmetric.
#
# | component | control |
# |---|---|
# | $u^B_1=L\bar E_{11}$ | force $|\Gamma|\bar S_{11}$ |
# | $u^B_2=u^B_3=0$ | remove two rotations |
# | $u^D_1=2L\bar E_{12}$, $u^D_2=L\bar E_{22}$ | forces $|\Gamma|\bar S_{12}$, $|\Gamma|\bar S_{22}$ |
# | $u^D_3=0$ | removes the third rotation |
# | $u^E_i=L(2\bar E_{13},2\bar E_{23},\bar E_{33})_i$ | forces $|\Gamma|\bar S_{i3}$, $i=1,2,3$ |
#
# As in two dimensions, the cell is loaded by a
# combination of tension and shear, $\bar S_{11}$ and $\bar S_{12}$, with the other components zero.

# +
S_bar = np.array([[0.09, 0.05, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
area = L**2


def component_bc(p, c) -> fem.DirichletBC:
    Vc = V.sub(c).collapse()[0]
    dofs = fem.locate_dofs_geometrical((V.sub(c), Vc), point(p))
    return fem.dirichletbc(fem.Function(Vc, dtype=dtype), dofs, V.sub(c))


bcs_sc = [point_bc(A_pt)[0], component_bc(B_pt, 1), component_bc(B_pt, 2), component_bc(D_pt, 2)]
mpc_sc = periodic_cell_constraint(bcs_sc)

# The nodal forces as vertex integrals: the force on the corner of direction j, B, D or E, is
# |Gamma| S_bar[:, j]. Its components on the rotation constraints are removed by their Dirichlet
# conditions.
vertex_map = domain.topology.index_map(0)
vertex_values = np.full(vertex_map.size_local + vertex_map.num_ghosts, -1, dtype=np.int32)
for j, point_ in enumerate((B_pt, D_pt, E_pt)):
    vertex_values[mesh.locate_entities_boundary(domain, 0, point(point_))] = j
marked_vertices = np.flatnonzero(vertex_values != -1).astype(np.int32)
corner_tags = mesh.meshtags(domain, 0, marked_vertices, vertex_values[marked_vertices])
dP = ufl.Measure("dP", domain=domain, subdomain_data=corner_tags)
# Column j is the force per area on the corner of direction j. The entries on the rotation
# constraints are placeholders, removed by their Dirichlet conditions, where the reactions are.
corner_loads = fem.Constant(domain, np.asarray(S_bar, dtype=dtype))
A_face = fem.Constant(domain, dtype.type(area))
w = ufl.TestFunction(V)
point_forces = A_face * sum(ufl.inner(corner_loads[:, j], w) * dP(j) for j in range(gdim))


# The dofs of the corners B, D and E, and the process owning them, located once
corner_dofs = {tuple(p): dofs_at_point(V, p) for p in (B_pt, D_pt, E_pt)}


def value_at(u: fem.Function, point) -> np.ndarray:
    """The value of `u` at the dofs of V at the corner `point`, read by the process owning them and sent to all."""
    dofs, owner = corner_dofs[tuple(point)]
    value = None
    if comm.rank == owner:
        value = u.x.array[dofs - V.dofmap.index_map.local_range[0] * V.dofmap.index_map_bs].real
    return comm.bcast(value, root=owner)


def strain_from_corners(uh_) -> np.ndarray:
    """E_bar from the free corners through eq:3d-corners."""
    H = np.column_stack([value_at(uh_, p) for p in (B_pt, D_pt, E_pt)]) / L
    return 0.5 * (H + H.T)


# -

# ### Homogeneous cell
#
# The exact solution is affine, with $\bar{\mathbf{H}}$ given by {eq}`eq:3d-H` and
# $\bar{\mathbf{E}}=\frac{1}{2\mu}\left(\bar{\mathbf{S}}-\frac{\lambda}{3\lambda+2\mu}
# \operatorname{tr}\bar{\mathbf{S}}\,\boldsymbol{\delta}\right)$. It must be reproduced to round-off.

# +
uh_sc = ConstrainedSolver(a, Lform + point_forces, mpc_sc, bcs_sc).solve()
lam0, mu0 = float(lmbda_uniform.value.real), float(mu_uniform.value.real)
E_exact = (S_bar - lam0 / (3 * lam0 + 2 * mu0) * np.trace(S_bar) * np.eye(gdim)) / (2 * mu0)
u_exact = ufl.dot(ufl.as_tensor(upper_triangular(E_exact)), x)
diff = uh_sc - u_exact
error_sc = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype))))
norm_exact = np.sqrt(abs(assemble_scalar_global(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx, dtype=dtype))))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < atol * norm_exact
# -

# ### Heterogeneous cell
#
# The strain is read from the free corners through {eq}`eq:3d-corners`. In linear elasticity the
# same state follows by superposition of six strain-controlled solutions, for the unit strains
# $\bar E_{11}$, $\bar E_{22}$, $\bar E_{33}$, $\bar E_{23}=\bar E_{32}$, $\bar E_{13}=\bar E_{31}$,
# $\bar E_{12}=\bar E_{21}$: they give the homogenized stiffness, from which $\bar{\mathbf{E}}$ is
# solved with $\bar{\boldsymbol\sigma}=\bar{\mathbf{S}}$. The two routes must agree, and the average
# stress must be the prescribed one.

# +
uh_sc = ConstrainedSolver(a_het, L_het + point_forces, mpc_sc, bcs_sc).solve()
sigma_sc = average_stress(uh_sc)
E_bar_sc = strain_from_corners(uh_sc)

voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
columns = []
for i, j in voigt:
    E_unit = np.zeros((gdim, gdim))
    E_unit[i, j] = E_unit[j, i] = 1.0
    s_unit = homogenized_stress(solver_het, 1e-2 * E_unit)[1] / 1e-2
    columns.append([s_unit[k, m] for k, m in voigt])
e_sup = np.linalg.solve(np.column_stack(columns), [S_bar[i, j] for i, j in voigt])
E_sup = np.zeros((gdim, gdim), dtype=e_sup.dtype)
for (i, j), value in zip(voigt, e_sup):
    E_sup[i, j] = E_sup[j, i] = value
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  sigma_bar = {fmt(sigma_sc)}")
    print(f"  E_bar     = {fmt(E_bar_sc)}")
    print(f"  superposition of strain-controlled solutions: E_bar = {fmt(E_sup)}")
assert np.allclose(sigma_sc, S_bar, rtol=0, atol=atol * np.abs(S_bar).max())
assert np.allclose(E_bar_sc, E_sup, rtol=0, atol=atol * np.abs(E_sup).max())
# -

# ## Visualization
#
# Each process builds a PyVista grid over the cells it owns; the grids are gathered on one process.
# The microstructure and the fluctuation are shown with the octant $X_i>L/2$ removed, to expose
# the inclusion.


# + tags=["hide-input"]
def gather_grids(u: fem.Function, name: str, root: int = 0):
    owned = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V, entities=owned))
    grid.point_data[name] = u.x.array.real[: grid.n_points * gdim].reshape(-1, gdim)
    magnitude = np.linalg.norm(grid.point_data[name], axis=1)
    grid.point_data[f"|{name}|"] = magnitude
    hi = comm.allreduce(float(magnitude.max()) if magnitude.size else 0.0, op=MPI.MAX)
    return comm.gather(grid, root=root), [0.0, hi]


def gather_cell_data(field: fem.Function, name: str, root: int = 0):
    owned = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, tdim, owned))
    grid.cell_data[name] = field.x.array.real[: len(owned)]
    return comm.gather(grid, root=root)


def plot_cell(uh_: fem.Function, H: np.ndarray, title: str, filename: str):
    u_ = fem.Function(V, dtype=dtype)
    u_.x.array[:] = uh_.x.array[: u_.x.array.size]  # leading entries: the original space
    w_ = fem.Function(V, dtype=dtype)
    w_.interpolate(lambda xx: H @ xx)
    w_.x.array[:] = u_.x.array - w_.x.array  # periodic fluctuation u* = u - H X
    u_pieces, u_clim = gather_grids(u_, "u")
    w_pieces, w_clim = gather_grids(w_, "w")
    e_pieces = gather_cell_data(E, "E")
    if comm.rank != 0:
        return
    merge = lambda pieces: pieces[0].merge(pieces[1:]) if len(pieces) > 1 else pieces[0]  # noqa: E731

    def cut_open(grid):  # remove the cells of the octant X_i > L/2
        centers = grid.cell_centers().points
        return grid.extract_cells(np.flatnonzero(~np.all(centers > 0.5 * L, axis=1)))

    outline = pyvista.Box(bounds=(0, L, 0, L, 0, L))
    bar = {"fmt": "%.1e", "n_labels": 3, "position_x": 0.2, "width": 0.6}
    plotter = pyvista.Plotter(shape=(1, 3), window_size=[1500, 520])
    plotter.subplot(0, 0)
    plotter.add_text(
        f"Microstructure (cut open)\nE = {E_uniform:g} (matrix), {50 * E_uniform:g} (inclusion)", font_size=10
    )
    plotter.add_mesh(
        cut_open(merge(e_pieces)),
        scalars="E",
        cmap="viridis",
        clim=[E_uniform, 50 * E_uniform],
        scalar_bar_args={"n_labels": 2, "fmt": "%.0f", "position_x": 0.2, "width": 0.6},
    )
    plotter.add_mesh(outline, style="wireframe", color="black", line_width=2)
    for col, (grid, clim, name, text, clip) in enumerate(
        [
            (merge(u_pieces), u_clim, "u", f"Deformed cell\n{title}", False),
            (merge(w_pieces), w_clim, "w", "Periodic fluctuation (cut open)\nu* = u - H X", True),
        ],
        start=1,
    ):
        factor = 0.1 * L / clim[1]  # largest displacement drawn as 10 % of the cell size
        shown = cut_open(grid) if clip else grid
        plotter.subplot(0, col)
        plotter.add_text(f"{text}\n(amplified x{factor:.0f})", font_size=10)
        plotter.add_mesh(
            shown.warp_by_vector(name, factor=factor),
            scalars=f"|{name}|",
            cmap="viridis",
            clim=clim,
            scalar_bar_args={**bar, "title": f"|{name}|"},
        )
        plotter.add_mesh(outline, style="wireframe", color="black", line_width=2)
    for col in range(3):
        plotter.subplot(0, col)
        plotter.view_isometric()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(filename)
    else:
        plotter.show()


strain_load = "prescribed E = [" + ", ".join("[" + ", ".join(f"{v:g}" for v in row) + "]" for row in E_bar) + "]"
plot_cell(uh_general, upper_triangular(E_bar), strain_load, "demo_periodic_homogenization_3d.png")
plot_cell(
    uh_sc,
    upper_triangular(E_bar_sc),
    f"stress control: S11 = {S_bar[0, 0]:g}, S12 = {S_bar[0, 1]:g}, other S_ij = 0",
    "demo_periodic_homogenization_3d_stress_control.png",
)
# -

# The PETSc objects of the solver are freed, and those the garbage collector has released are
# cleaned up on every process together.

del solver_het
PETSc.garbage_cleanup(comm)

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homog3d-
# ```

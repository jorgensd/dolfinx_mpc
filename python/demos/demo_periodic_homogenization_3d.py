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
from dolfinx import default_scalar_type, fem, mesh, plot

import dolfinx_mpc
from dolfinx_mpc import MultiPointConstraint

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
domain = mesh.create_box(comm, [[0, 0, 0], [L, L, L]], [N, N, N], mesh.CellType.hexahedron)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
# -

# ## The periodicity condition
#
# The displacement is split into an affine part carrying the macroscopic strain and a
# periodic fluctuation,
#
# $$
# \mathbf{u}(\mathbf{X}) = \bar{\mathbf{H}}\,\mathbf{X} + \mathbf{u}^*(\mathbf{X}),
# \qquad \mathbf{u}^*(\mathbf{X}+L\mathbf{e}_i) = \mathbf{u}^*(\mathbf{X}),\quad i=1,2,3,
# $$
#
# where $\bar{\mathbf{H}}$ is the average displacement gradient. Only its symmetric part, the
# macroscopic strain $\bar{\mathbf{E}}=\operatorname{sym}\bar{\mathbf{H}}$, produces stress; the
# skew part is a rigid rotation. We remove it by taking $\bar{\mathbf{H}}$ upper triangular,
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
# \mathbf{u}^B=\bar{\mathbf{H}}L\mathbf{e}_1=L\begin{pmatrix}\bar E_{11}\\0\\0\end{pmatrix},\qquad
# \mathbf{u}^D=\bar{\mathbf{H}}L\mathbf{e}_2=L\begin{pmatrix}2\bar E_{12}\\ \bar E_{22}\\0\end{pmatrix},\qquad
# \mathbf{u}^E=\bar{\mathbf{H}}L\mathbf{e}_3=L\begin{pmatrix}2\bar E_{13}\\2\bar E_{23}\\ \bar E_{33}\end{pmatrix}.
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
# The relations are imposed with the same DOLFINx-MPC tools as in two dimensions.
#
# * **Strain control.** $\mathbf{u}^B$, $\mathbf{u}^D$, $\mathbf{u}^E$ are known and carried by
#   Dirichlet conditions on the corners. On faces and edges, {eq}`eq:3d-periodic` is a periodic
#   constraint with a constant term: one call to
#   {py:meth}`create_periodic_constraint_geometrical
#   <dolfinx_mpc.MultiPointConstraint.create_periodic_constraint_geometrical>` with the map
#   $\mathbf{X}\mapsto\mathbf{X}^-$, the constant term being a function `g` passed to the
#   constructor. The four slave corners are tied to $B$, $D$, $E$ with
#   {py:meth}`create_general_constraint <dolfinx_mpc.MultiPointConstraint.create_general_constraint>`,
#   which folds the Dirichlet masters into the constraint.
# * **Stress control.** $\mathbf{u}^B$, $\mathbf{u}^D$, $\mathbf{u}^E$ are unknowns, so every slave
#   has up to four masters: $\mathbf{X}^-$ and the corners. All relations are built with
#   `create_general_constraint`, one call per component.

# +
tol = 1e-10 * L
xdt = domain.geometry.x.dtype
A_pt, B_pt, D_pt, E_pt = (np.array(p, dtype=xdt) for p in ([0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]))
corner_of = {0: B_pt, 1: D_pt, 2: E_pt}  # corner carrying the jump in direction i


def point(p):
    return lambda x: np.isclose(x[0], p[0]) & np.isclose(x[1], p[1]) & np.isclose(x[2], p[2])


def key(p) -> bytes:
    return np.asarray(p, dtype=xdt).tobytes()


def is_vertex(x):
    return np.all(np.isclose(x, 0.0) | np.isclose(x, L), axis=0)


def faces_and_edges(x):
    """Slaves of the geometrical constraint: some X_i = L, vertices excluded."""
    return np.any(np.isclose(x, L), axis=0) & ~is_vertex(x)


def to_image(x):
    """X -> X^- = X - L s."""
    out = x.copy()
    out[np.isclose(x, L)] = 0.0
    return out


slave_vertices = [np.array(p, dtype=xdt) for p in ([L, L, 0], [L, 0, L], [0, L, L], [L, L, L])]


def masters_of(p) -> dict[bytes, float]:
    """Masters of eq:3d-periodic for a slave at p: X^- and the corners of the directions with X_i = L.
    A is omitted: u^A = 0."""
    s = np.isclose(p, L)
    image = np.where(s, 0.0, p)
    masters = {} if np.allclose(image, A_pt) else {key(image): 1.0}
    masters.update({key(corner_of[i]): 1.0 for i in range(gdim) if s[i]})
    return masters


# -

# ## Solver
#
# The constrained stiffness matrix does not depend on the values of the Dirichlet conditions, so
# it is assembled and factorized once per material. Each solve updates the constraint offsets
# with {py:meth}`update_constants <dolfinx_mpc.MultiPointConstraint.update_constants>`, assembles
# the right-hand side as {py:class}`dolfinx_mpc.LinearProblem` does, and adds the nodal forces of
# stress control, if any, at the dofs of the process that owns them.


# +
class ConstrainedSolver:
    def __init__(self, a_ufl, L_ufl, mpc: MultiPointConstraint, bcs: list[fem.DirichletBC], forces=()):
        self.a, self.L = fem.form(a_ufl), fem.form(L_ufl)
        self.mpc, self.bcs, self.forces = mpc, bcs, forces
        self.A = dolfinx_mpc.assemble_matrix(self.a, mpc, bcs=bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(self.L, mpc)
        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.ksp.setErrorIfNotConverged(True)
        self.n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    def solve(self) -> fem.Function:
        self.mpc.update_constants()
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx_mpc.assemble_vector(self.L, self.mpc, self.b)
        dolfinx_mpc.apply_lifting(self.b, [self.a], bcs=[self.bcs], constraint=self.mpc)
        dolfinx_mpc.apply_mpc_lifting(self.b, [self.a], constraint=self.mpc)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self.b, self.bcs)
        for dofs, f in self.forces:
            self.b.array[dofs[dofs < self.n_owned]] += f
        self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        uh_ = fem.Function(self.mpc.function_space)
        self.ksp.solve(self.b, uh_.x.petsc_vec)
        uh_.x.scatter_forward()
        self.mpc.homogenize(uh_)
        self.mpc.backsubstitution(uh_)
        return uh_


def sigma(w, mu_, lmbda_):
    eps = ufl.sym(ufl.grad(w))
    return 2 * mu_ * eps + lmbda_ * ufl.tr(eps) * ufl.Identity(gdim)


def elasticity_forms(mu_, lmbda_):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_ = ufl.inner(sigma(u, mu_, lmbda_), ufl.sym(ufl.grad(v))) * ufl.dx
    L_ = ufl.inner(fem.Constant(domain, np.zeros(gdim, dtype=dtype)), v) * ufl.dx
    return a_, L_


def average_stress(uh_, mu_, lmbda_) -> np.ndarray:
    s_ = sigma(uh_, mu_, lmbda_)
    return (
        np.array(
            [
                [comm.allreduce(fem.assemble_scalar(fem.form(s_[i, j] * ufl.dx)), op=MPI.SUM) for j in range(gdim)]
                for i in range(gdim)
            ]
        )
        / L**3
    )


def fmt(m: np.ndarray) -> str:
    return "[" + ", ".join("[" + ", ".join(f"{v:.6e}" for v in row) + "]" for row in m) + "]"


E_uniform, nu = 10.0, 0.3
mu_uniform = fem.Constant(domain, default_scalar_type(E_uniform / (2 * (1 + nu))))
lmbda_uniform = fem.Constant(domain, default_scalar_type(E_uniform * nu / ((1 + nu) * (1 - 2 * nu))))
a, Lform = elasticity_forms(mu_uniform, lmbda_uniform)

Q = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
E = fem.Function(Q)
tdim = domain.topology.dim
n_cells = domain.topology.index_map(tdim).size_local
midpoints = mesh.compute_midpoints(domain, tdim, np.arange(n_cells, dtype=np.int32))
inclusion = np.sum((midpoints - 0.5 * L) ** 2, axis=1) < (0.25 * L) ** 2  # sphere of radius L/4
E.x.array[:n_cells] = np.where(inclusion, 50.0 * E_uniform, E_uniform)
E.x.scatter_forward()
mu_field = E / (2 * (1 + nu))
lmbda_field = E * nu / ((1 + nu) * (1 - 2 * nu))
a_het, L_het = elasticity_forms(mu_field, lmbda_field)
x = ufl.SpatialCoordinate(domain)
# -

# ## Strain control
#
# The corners $A$, $B$, $D$, $E$ carry the Dirichlet conditions {eq}`eq:3d-corners`; their values
# are set by `set_strain`.


# +
def point_bc(p) -> tuple[fem.DirichletBC, fem.Function]:
    fn = fem.Function(V)
    return fem.dirichletbc(fn, fem.locate_dofs_geometrical(V, point(p))), fn


def upper_triangular(E_case: np.ndarray) -> np.ndarray:
    """H_bar of eq:3d-H for a symmetric macroscopic strain."""
    return np.triu(2 * E_case) - np.diag(np.diag(E_case))


corners = (A_pt, B_pt, D_pt, E_pt)
corner_bcs = [point_bc(p) for p in corners]
bcs = [bc for bc, _ in corner_bcs]
bc_values = [value for _, value in corner_bcs]
g = fem.Function(V, dtype=dtype)  # constant term s1 u^B + s2 u^D + s3 u^E on faces and edges
mpc = MultiPointConstraint(V, dtype=dtype, bcs=bcs, rhs_coeffs=g)
mpc.create_periodic_constraint_geometrical(V, faces_and_edges, to_image, bcs, scale=dtype.type(1.0))
for c in range(gdim):
    mpc.create_general_constraint({key(p): masters_of(p) for p in slave_vertices}, subspace_slave=c, subspace_master=c)
mpc.finalize()  # collective: every rank must reach this


def set_strain(E_case: np.ndarray):
    """Dirichlet values eq:3d-corners and constant term g; the solver calls update_constants."""
    H = upper_triangular(E_case)
    for fn, p in zip(bc_values, corners):
        fn.interpolate(lambda xx: np.tile((H @ p).reshape(-1, 1), xx.shape[1]))

    def offset(xx):
        s = np.isclose(xx, L) & faces_and_edges(xx)
        return H @ (L * s)  # sum_i s_i u^i, with u^i = H L e_i

    g.interpolate(offset)
    g.x.scatter_forward()


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
diff = uh - ufl.dot(ufl.as_tensor(upper_triangular(E_bar)), x)
error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Homogeneous unit cell----\n  L2(u_h - affine) = {error:.3e}  (fluctuation should vanish)")
assert error < 1e-10
# -

# ### Heterogeneous cell
#
# The matrix contains a spherical inclusion of radius $L/4$, fifty times stiffer. With no
# macroscopic strain the cell carries no load, so the average stress must vanish. The cell has
# cubic symmetry, so an isotropic macroscopic strain must give an isotropic average stress.

# +
solver_het = ConstrainedSolver(a_het, L_het, mpc, bcs)


def homogenized_stress(E_case: np.ndarray) -> tuple[fem.Function, np.ndarray]:
    set_strain(E_case)
    uh_case = solver_het.solve()
    return uh_case, average_stress(uh_case, mu_field, lmbda_field)


_, sigma_zero = homogenized_stress(np.zeros((gdim, gdim)))
_, sigma_iso = homogenized_stress(0.02 * np.eye(gdim))
uh_general, sigma_general = homogenized_stress(E_bar)
if comm.rank == 0:
    print(f"----No macroscopic strain----\n  sigma_bar = {fmt(sigma_zero)}  (should vanish)")
    print(f"----Isotropic macroscopic strain----\n  sigma_bar = {fmt(sigma_iso)}")
    print(f"----General macroscopic strain----\n  sigma_bar = {fmt(sigma_general)}")
# -

# ## Stress control
#
# We now prescribe the average stress $\bar{\boldsymbol\sigma}=\bar{\mathbf{S}}$. The displacements
# of $B$, $D$, $E$ become unknowns, except the three components that vanish in
# {eq}`eq:3d-corners`: $u^B_2=u^B_3=u^D_3=0$. These remove the three rigid rotations, which the
# periodicity no longer excludes once $\bar{\mathbf{H}}$ is unknown.
#
# ### The prescribed stress as nodal forces
#
# The tractions $\mathbf{T}=\boldsymbol{\sigma}\mathbf{n}$ are anti-periodic on opposite faces. For a
# field $\mathbf{v}$ satisfying {eq}`eq:3d-periodic`, the contributions of $\mathbf{v}(\mathbf{X}^-)$
# cancel and the work of the tractions reduces to
#
# $$
# \int_{\partial\Omega}\mathbf{T}\cdot\mathbf{v}\,\mathrm{d}s
# = v^B_i\int_{X_1=L}T_i\,\mathrm{d}s + v^D_i\int_{X_2=L}T_i\,\mathrm{d}s
# + v^E_i\int_{X_3=L}T_i\,\mathrm{d}s .
# $$
#
# With $\operatorname{div}\boldsymbol{\sigma}=\mathbf{0}$, $\int_{\partial\Omega}T_iX_j\,\mathrm{d}s
# =|\Omega|\,\bar\sigma_{ij}$, so the integral over the face $X_j=L$ is $A\bar\sigma_{ij}$, with
# $A=L^2$ the area of a face. The principle of virtual work becomes
#
# $$
# \int_\Omega \boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\epsilon}(\mathbf{v})\,\mathrm{d}\Omega
# = A\left(\bar S_{i1}\,v^B_i + \bar S_{i2}\,v^D_i + \bar S_{i3}\,v^E_i\right),
# $$
#
# for all $\mathbf{v}$ satisfying the constraints and $v^B_2=v^B_3=v^D_3=0$. The prescribed stress
# is a set of nodal forces; $\bar S_{21}$, $\bar S_{31}$, $\bar S_{32}$ are the reactions of the
# three rotation constraints.
#
# | component | control |
# |---|---|
# | $u^B_1=L\bar E_{11}$ | force $A\bar S_{11}$ |
# | $u^B_2=u^B_3=0$ | remove two rotations |
# | $u^D_1=2L\bar E_{12}$, $u^D_2=L\bar E_{22}$ | forces $A\bar S_{12}$, $A\bar S_{22}$ |
# | $u^D_3=0$ | removes the third rotation |
# | $u^E_i=L(2\bar E_{13},2\bar E_{23},\bar E_{33})_i$ | forces $A\bar S_{i3}$, $i=1,2,3$ |
#
# As in two dimensions, the cell is loaded by a
# combination of tension and shear, $\bar S_{11}$ and $\bar S_{12}$, with the other components zero.

# +
S_bar = np.array([[0.09, 0.05, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
area = L**2


def component_bc(p, c) -> fem.DirichletBC:
    Vc = V.sub(c).collapse()[0]
    return fem.dirichletbc(fem.Function(Vc), fem.locate_dofs_geometrical((V.sub(c), Vc), point(p)), V.sub(c))


def component_dofs(p, c) -> np.ndarray:
    return fem.locate_dofs_geometrical((V.sub(c), V.sub(c).collapse()[0]), point(p))[0]


bcs_sc = [point_bc(A_pt)[0], component_bc(B_pt, 1), component_bc(B_pt, 2), component_bc(D_pt, 2)]
mpc_sc = MultiPointConstraint(V, dtype=dtype, bcs=bcs_sc)
x_dofs = V.tabulate_dof_coordinates()
is_master_corner = np.isclose(x_dofs[:, None, :], [B_pt, D_pt, E_pt]).all(axis=2).any(axis=1)
slaves_local = x_dofs[np.isclose(x_dofs, L).any(axis=1) & ~is_master_corner]
# every rank needs all slaves; the coordinates are rounded because the same node may differ by round-off
# between processes, and would otherwise appear twice
slave_points = np.unique(np.round(np.vstack(comm.allgather(slaves_local)), 14), axis=0)
for c in range(gdim):
    mpc_sc.create_general_constraint({key(p): masters_of(p) for p in slave_points}, subspace_slave=c, subspace_master=c)
mpc_sc.finalize()  # collective: every rank must reach this

forces = [(component_dofs(B_pt, 0), area * S_bar[0, 0])]  # A S_11 on u^B_1
forces += [(component_dofs(D_pt, c), area * S_bar[c, 1]) for c in (0, 1)]  # A S_12, A S_22 on u^D_1, u^D_2
forces += [(component_dofs(E_pt, c), area * S_bar[c, 2]) for c in (0, 1, 2)]  # A S_i3 on u^E_i


def value_at(uh_, p) -> np.ndarray:
    nloc = V.dofmap.index_map.size_local
    i = np.flatnonzero(np.linalg.norm(x_dofs[:nloc] - p, axis=1) < tol)
    local = uh_.x.array[gdim * i[0] : gdim * i[0] + gdim].copy() if len(i) else None
    return next(v for v in comm.allgather(local) if v is not None)


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
uh_sc = ConstrainedSolver(a, Lform, mpc_sc, bcs_sc, forces).solve()
lam0, mu0 = float(lmbda_uniform.value), float(mu_uniform.value)
E_exact = (S_bar - lam0 / (3 * lam0 + 2 * mu0) * np.trace(S_bar) * np.eye(gdim)) / (2 * mu0)
diff = uh_sc - ufl.dot(ufl.as_tensor(upper_triangular(E_exact)), x)
error_sc = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
if comm.rank == 0:
    print(f"----Stress control, homogeneous cell----\n  L2(u_h - u_exact) = {error_sc:.3e}  (should be round-off)")
assert error_sc < 1e-10
# -

# ### Heterogeneous cell
#
# The strain is read from the free corners through {eq}`eq:3d-corners`. In linear elasticity the
# same state follows by superposition of six strain-controlled solutions, for the unit strains
# $\bar E_{11}$, $\bar E_{22}$, $\bar E_{33}$, $\bar E_{23}=\bar E_{32}$, $\bar E_{13}=\bar E_{31}$,
# $\bar E_{12}=\bar E_{21}$: they give the homogenized stiffness, from which $\bar{\mathbf{E}}$ is
# solved with $\bar{\boldsymbol\sigma}=\bar{\mathbf{S}}$.

# +
uh_sc = ConstrainedSolver(a_het, L_het, mpc_sc, bcs_sc, forces).solve()
sigma_sc = average_stress(uh_sc, mu_field, lmbda_field)
E_bar_sc = strain_from_corners(uh_sc)

voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
columns = []
for i, j in voigt:
    E_unit = np.zeros((gdim, gdim))
    E_unit[i, j] = E_unit[j, i] = 1.0
    s_unit = homogenized_stress(1e-2 * E_unit)[1] / 1e-2
    columns.append([s_unit[k, m] for k, m in voigt])
e_sup = np.linalg.solve(np.column_stack(columns), [S_bar[i, j] for i, j in voigt])
E_sup = np.zeros((gdim, gdim))
for (i, j), value in zip(voigt, e_sup):
    E_sup[i, j] = E_sup[j, i] = value
if comm.rank == 0:
    print("----Stress control, heterogeneous cell----")
    print(f"  sigma_bar = {fmt(sigma_sc)}")
    print(f"  E_bar     = {fmt(E_bar_sc)}")
    print(f"  superposition of strain-controlled solutions: E_bar = {fmt(E_sup)}")
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
    u_ = fem.Function(V)
    u_.x.array[:] = uh_.x.array[: u_.x.array.size]  # leading entries: the original space
    w_ = fem.Function(V)
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

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: homog3d-
# ```

# # Mortar coupling: a Lagrange multiplier on a submesh
# **Author** Jørgen S. Dokken
#
# The Lagrange multiplier method of {cite}`mortar-Babuska1973` imposes a Dirichlet
# condition *weakly*, through a multiplier $\lambda$ that lives only on the
# boundary where the condition applies. In DOLFINx that multiplier is a function on a
# **submesh**, so the coupling term $\int_\Gamma \lambda v~\mathrm{d}s$ is a
# bilinear form whose test and trial functions live on *different meshes*.
#
# ## Mathematical formulation
#
# On $\Omega=(0,1)^2$ with $\Gamma=\{0\}\times(0,1)$ we solve
#
# $$
# \begin{align*}
# -\Delta u &= f && \text{in } \Omega,\\
# u &= g && \text{on } \Gamma,\\
# \frac{\partial u}{\partial n} &= 0 && \text{on } \{1\}\times(0,1),\\
# u(x, 0) &= u(x, 1),\quad
#   \frac{\partial u}{\partial n}\Big|_{y=0}
#   + \frac{\partial u}{\partial n}\Big|_{y=1} = 0 && \text{(periodic)}.
# \end{align*}
# $$
#
# The periodicity is a {py:class}`dolfinx_mpc.MultiPointConstraint`; the condition
# on $\Gamma$ is the one imposed weakly. Only the first of the two periodic
# relations is stated to the constraint. The second comes for free: a multi-point
# constraint applies to the test function as well as the trial function, so the
# boundary terms at $y=0$ and $y=1$ are summed into one equation and the natural
# condition there becomes flux continuity rather than zero flux on each side.
#
# Introducing $\lambda \in \Lambda := H^{-1/2}(\Gamma)$ we seek
# $(u, \lambda) \in V \times \Lambda$ with
#
# $$
# \begin{align*}
# \int_\Omega \nabla u \cdot \nabla v ~\mathrm{d}x
#   + \int_\Gamma \lambda v ~\mathrm{d}s &= \int_\Omega f v ~\mathrm{d}x
#   && \forall v \in V,\\
# \int_\Gamma \mu u ~\mathrm{d}s &= \int_\Gamma \mu g ~\mathrm{d}s
#   && \forall \mu \in \Lambda.
# \end{align*}
# $$
#
# Integrating the first equation by parts identifies the multiplier as the
# (negated) normal flux, $\lambda = -\partial u/\partial n$ on $\Gamma$, which
# gives us a second quantity to verify against.

# + tags=["hide-input"]
from __future__ import annotations

from mpi4py import MPI

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh

import dolfinx_mpc

# -

# ## The multiplier submesh
#
# $\Gamma$ becomes a mesh in its own right, built from the facets at $x=0$. The
# {py:class}`entity_map<dolfinx.mesh.EntityMap>` returned alongside it relates
# its cells to the facets of {py:class}`msh<dolfinx.mesh.Mesh>`, and is what
# lets a single form mix functions from the two meshes.

# +

N = 32
msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, tdim)

gamma_facets = np.sort(mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)))
gamma, entity_map, _, _ = mesh.create_submesh(msh, fdim, gamma_facets)
gamma_tag = 7
ft = mesh.meshtags(msh, fdim, gamma_facets, np.full(len(gamma_facets), gamma_tag, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
dGamma = ds(subdomain_id=gamma_tag)
# -

# ## Problem data
#
# We manufacture $u_{ex} = (x-1)^2\cos(2\pi y)$. It is periodic in $y$ and has
# $\partial u/\partial y = 0$ on $y\in\{0,1\}$ and $\partial u/\partial x = 0$ at
# $x=1$, so the natural condition holds on the three sides we do not constrain.
# Everything else is derived from it with UFL rather than by hand,
# as explained in [DOLFINx tutorial: Manufactured
# solutions](https://jsdokken.com/dolfinx-tutorial/chapter2/nonlinpoisson_code.html#test-problem).

# +
degree = 2
V = fem.functionspace(msh, ("Lagrange", degree))
Q = fem.functionspace(gamma, ("Lagrange", degree))

x = ufl.SpatialCoordinate(msh)
u_ex = (x[0] - 1) ** 2 * ufl.cos(2 * ufl.pi * x[1])
f = -ufl.div(ufl.grad(u_ex))
# -

# ## The block system
#
# The off-diagonal blocks are integrated over `msh` but carry an argument from
# `gamma`, so they need the entity map. The forms stay as plain UFL:
# {py:class}`LinearProblem<dolfinx_mpc.LinearProblem>` takes `entity_maps` and
# passes it to the form compiler.
#
# The $(1,1)$ block is mathematically zero, which DOLFINx would normally
# optimise away. We still need the matrix: the periodic condition on $\Lambda$
# puts its constrained rows there. {py:class}`ufl.ZeroBaseForm` keeps the block
# without assembling anything into it.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
lmbda, mu = ufl.TrialFunction(Q), ufl.TestFunction(Q)
a = [
    [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(lmbda, v) * dGamma],
    [ufl.inner(u, mu) * dGamma, ufl.ZeroBaseForm((lmbda, mu))],
]
L = [ufl.inner(f, v) * ufl.dx, ufl.inner(u_ex, mu) * dGamma]

# ## The constraints
#
# Periodicity in $y$ goes on the bulk space, as well as an the multiplier space.
# This second periodicity requirement is strictly required.
# The trace of a $y$-periodic $u$ on $\Gamma$ has one fewer independent value than
# an unconstrained $\Lambda$ has degrees of freedom, so the second equation
# becomes redundant and $\lambda$ is left undetermined.
# The primal solution still converges, but the computed
# multiplier grows without bound under refinement.
#
# The corner at $(0,1)$ is where this is visible: that degree of freedom is
# simultaneously a periodic slave, tied to $(0,0)$, and a dof on $\Gamma$
# carrying a multiplier term.


# +
def on_top(coords):
    return np.isclose(coords[1], 1.0)


def to_bottom(coords):
    return np.vstack([coords[0], 1.0 - coords[1], coords[2]])


scale = default_scalar_type(1.0)
bcs: list[fem.DirichletBC] = []
mpc_u = dolfinx_mpc.MultiPointConstraint(V)
mpc_u.create_periodic_constraint_geometrical(V, on_top, to_bottom, bcs, scale)
mpc_u.finalize()

mpc_l = dolfinx_mpc.MultiPointConstraint(Q)
mpc_l.create_periodic_constraint_geometrical(Q, on_top, to_bottom, bcs, scale)
mpc_l.finalize()
# -

# ## Solving
#
# A sequence of constraints makes {py:class}`dolfinx_mpc.LinearProblem` assemble a `nest` system, one
# block per constraint.
#
# The system is a saddle point, solved here with MINRES and a block-diagonal
# preconditioner supplied through `P`. Neither block can be taken from the
# operator as it stands:
#
# - the $(1,1)$ block is zero, and no sub-solver can invert that. The standard
#   replacement is the multiplier mass matrix, the same choice
#   {doc}`demo_stokes_nest` makes for the pressure block;
# - the $(0,0)$ block is a *pure Neumann* Laplacian — every Dirichlet condition
#   in this problem is carried by $\lambda$ — so constants lie in its kernel and
#   it is singular. MINRES needs a positive definite preconditioner, and a
#   singular one makes it stop early on a wrong answer
#   (`KSP_DIVERGED_INDEFINITE_PC`). Shifting by a mass term fixes it.
#
# Each block is then inverted exactly with LU, so the iteration count measures
# only how well the block-diagonal approximation captures the coupling: 7
# iterations at $P_1$ and 10 at $P_2$, independent of the mesh size.

# +
P = [
    [(ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * ufl.dx, None],
    [None, ufl.inner(lmbda, mu) * ufl.dx],
]

problem = dolfinx_mpc.LinearProblem(
    a,
    L,
    [mpc_u, mpc_l],
    bcs=[],
    P=P,
    entity_maps=[entity_map],
    petsc_options_prefix="demo_mortar_",
    petsc_options={
        "ksp_type": "minres",
        "ksp_rtol": 1e-10,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "lu",
    },
)
uh, lh = problem.solve()
# -

# ## Verification
#
# $u$ against the manufactured solution, and $\lambda$ against the exact normal
# flux $-\partial u_{ex}/\partial n = \partial u_{ex}/\partial x$ at $x=0$, which
# is $-2\cos(2\pi y)$.

y_gamma = ufl.SpatialCoordinate(gamma)[1]
lmbda_ex = -2 * ufl.cos(2 * ufl.pi * y_gamma)
error_u = np.sqrt(
    msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)), op=MPI.SUM)
)
error_l = np.sqrt(
    msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(lh - lmbda_ex, lh - lmbda_ex) * ufl.dx)), op=MPI.SUM)
)
if msh.comm.rank == 0:
    print(
        f"N={N}, P{degree}:  {problem.solver.getIterationNumber()} MINRES iterations,"
        f"  |u-u_ex|_L2 = {error_u:.3e}   |lambda-lambda_ex|_L2 = {error_l:.3e}"
    )
assert error_u < 1e-3

# ## Convergence
#
# The function below repeats everything above verbatim, parametrised by
# resolution and element degree, so that the rates can be measured. The only
# change is the solver: MUMPS factorises the nest directly, which removes the
# preconditioner from the picture and leaves only the discretisation error. $u$
# converges at the optimal rate, and so does $\lambda$ measured in $L^2$.

# + tags=["hide-input"]


def solve_mortar(N: int, degree: int) -> tuple[float, float]:
    """Repeat of the demo above, returning the $L^2$ errors of $u$ and $\\lambda$."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)

    gamma_facets = np.sort(mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)))
    gamma, entity_map, _, _ = mesh.create_submesh(msh, fdim, gamma_facets)
    ft = mesh.meshtags(msh, fdim, gamma_facets, np.full(len(gamma_facets), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    V = fem.functionspace(msh, ("Lagrange", degree))
    Q = fem.functionspace(gamma, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    u_ex = (x[0] - 1) ** 2 * ufl.cos(2 * ufl.pi * x[1])
    f = -ufl.div(ufl.grad(u_ex))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    lmbda, mu = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    a = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(lmbda, v) * ds(1)],
        [ufl.inner(u, mu) * ds(1), ufl.ZeroBaseForm((lmbda, mu))],
    ]
    L = [ufl.inner(f, v) * ufl.dx, ufl.inner(u_ex, mu) * ds(1)]

    mpc_u = dolfinx_mpc.MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, on_top, to_bottom, [], default_scalar_type(1.0))
    mpc_u.finalize()
    mpc_l = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_l.create_periodic_constraint_geometrical(Q, on_top, to_bottom, [], default_scalar_type(1.0))
    mpc_l.finalize()

    problem = dolfinx_mpc.LinearProblem(
        a,
        L,
        [mpc_u, mpc_l],
        bcs=[],
        entity_maps=[entity_map],
        petsc_options_prefix=f"demo_mortar_p{degree}_n{N}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh, lh = problem.solve()

    y_gamma = ufl.SpatialCoordinate(gamma)[1]
    lmbda_ex = -2 * ufl.cos(2 * ufl.pi * y_gamma)
    comm = msh.comm
    return (
        np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)), op=MPI.SUM)),
        np.sqrt(
            comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(lh - lmbda_ex, lh - lmbda_ex) * ufl.dx)), op=MPI.SUM)
        ),
    )


for degree in (1, 2):
    previous = None
    for resolution in (8, 16, 32):
        err_u, err_l = solve_mortar(resolution, degree)
        rate = "" if previous is None else f"   rate {np.log2(previous / err_u):.2f}"
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"P{degree}  N={resolution:3d}   |u-u_ex|_L2 = {err_u:.3e}{rate}   |lambda-lambda_ex|_L2 = {err_l:.3e}"
            )
        previous = err_u
    assert err_u < 1e-3
# -

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: mortar-
# ```

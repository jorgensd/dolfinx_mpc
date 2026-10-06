# # Changing the coefficients of a multi-point constraint
# **Author** Jørgen S. Dokken
#
# A multi-point constraint relates each slave degree of freedom $u_s$ to a set
# of masters,
#
# $$
# u_s = \sum_j c_{sj} u_{m_j} + g_s.
# $$
#
# Creating and finalizing a constraint is expensive: the masters are located,
# communicated and added as ghosts. When only the coefficients $c_{sj}$ change
# between solves, for instance in a Floquet-Bloch condition
# $u(x + L) = e^{i k L} u(x)$ swept over the wave number $k$, the constraint can be
# kept and its coefficients modified instead. This demo sets up a periodic
# constraint and changes its coefficients in three ways, inspecting the
# coefficients after each step. Nothing is solved.

# + tags=["hide-input"]
from mpi4py import MPI

import numpy as np
import ufl
from dolfinx import default_real_type, fem
from dolfinx.mesh import create_unit_square

import dolfinx_mpc

# -

# A Bloch phase is complex, so we use the complex scalar type matching the mesh
# precision, whether or not DOLFINx was built with complex PETSc.

dtype = np.result_type(default_real_type, np.complex64).type
mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(mesh, ("Lagrange", 1))

# ## Setting up the constraint
#
# We make the right boundary $x=1$ periodic with the left one, $u(1, y) = u(0, y)$.
# To also show masters that are eliminated by a Dirichlet condition, we fix
# $u(0, y) = 1 + y$ on the left boundary only. The slaves on the right boundary
# then have masters with known values, and these are moved into the offset
# $g_s$ of the constraint.

# +
tol = 500 * np.finfo(default_real_type).eps


def left(x):
    return np.isclose(x[0], 0, atol=tol)


def right(x):
    return np.isclose(x[0], 1, atol=tol)


def periodic_relation(x):
    out = x.copy()
    out[0] = x[0] - 1
    return out


g = fem.Function(V, dtype=dtype)
g.interpolate(lambda x: 1 + x[1])
bc = fem.dirichletbc(g, fem.locate_dofs_geometrical(V, left))
# -

# The masters of a constraint are fixed when it is created. By default, a master
# whose coefficient is below `coefficient_tol` times the largest of its slave is
# dropped, and could then never be given a coefficient later. Passing
# `coefficient_tol=0` keeps every master. It comes at a cost,
# as each master is a ghost and an entry of the sparsity pattern:

# +
mpc = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype, bcs=[bc])
mpc.create_periodic_constraint_geometrical(V, right, periodic_relation, [bc], scale=dtype(1), coefficient_tol=0)
mpc.finalize()

mpc_cut = dolfinx_mpc.MultiPointConstraint(V, dtype=dtype, bcs=[bc])
mpc_cut.create_periodic_constraint_geometrical(V, right, periodic_relation, [bc], scale=dtype(1))
mpc_cut.finalize()

num_masters = mesh.comm.allreduce(len(mpc.all_masters()), op=MPI.SUM)
num_masters_cut = mesh.comm.allreduce(len(mpc_cut.all_masters()), op=MPI.SUM)
if mesh.comm.rank == 0:
    print(f"Masters with coefficient_tol=0: {num_masters}, with the default: {num_masters_cut}")
# -

# For a P1 space every slave gets the three vertices of the cell its periodic
# image lies in, two of them with coefficient zero.
#
# ## Inspecting the coefficients
#
# {py:meth}`all_coefficients<dolfinx_mpc.MultiPointConstraint.all_coefficients>`
# returns the coefficients of all masters per local degree of freedom, and
# {py:meth}`all_masters<dolfinx_mpc.MultiPointConstraint.all_masters>` the
# corresponding masters. Unlike
# {py:meth}`coefficients<dolfinx_mpc.MultiPointConstraint.coefficients>`, these
# include the masters eliminated by the Dirichlet condition. We mark those by
# checking which masters are absent from
# {py:attr}`masters<dolfinx_mpc.MultiPointConstraint.masters>`.
# The offset $g_s$ is given by
# {py:attr}`constants<dolfinx_mpc.MultiPointConstraint.constants>`.

# +
x_dofs = V.tabulate_dof_coordinates()


def print_constraint(mpc, header, num_slaves=3):
    """Print the non-zero terms of the constraint of the first owned slaves of this process."""
    coeffs, offsets = mpc.all_coefficients()
    masters = mpc.all_masters()
    lines = [header]
    for s in mpc.slaves[: mpc.num_local_slaves][:num_slaves]:
        kept = mpc.masters.links(s)
        terms = []
        for m, c in zip(masters[offsets[s] : offsets[s + 1]], coeffs[offsets[s] : offsets[s + 1]]):
            if abs(c) > tol:
                terms.append(f"({c:.3f}) u_{m}" + ("" if m in kept else " [Dirichlet]"))
        lines.append(f"  y={x_dofs[s, 1]:.2f}: u_{s} = " + " + ".join(terms) + f"   g = {mpc.constants[s]:.3f}")
    print(f"Rank {mesh.comm.rank}: " + "\n".join(lines), flush=True)


print_constraint(mpc, "Created with scale 1")
# -

# Each slave has a single master with a non-zero coefficient, the dof at the same
# height on the left boundary. As that one is fixed by the Dirichlet condition,
# the constraint reduces to $u_s = c\, g(0, y) = 1 + y$, which is the offset $g$.
#
# ## Scaling by a constant
#
# {py:meth}`scale_coefficients<dolfinx_mpc.MultiPointConstraint.scale_coefficients>`
# multiplies all coefficients of slave $s$ by a factor $f_s$, and recomputes the
# offset. Masters eliminated by the Dirichlet condition are scaled too, so the
# offset becomes $f_s c\, g$.

mpc.scale_coefficients(dtype(0.5))
print_constraint(mpc, "Scaled by 0.5")

# ## Scaling by an expression
#
# The factor can also vary in space and be given as a UFL expression. It is
# interpolated into a function in the space of the constraint, and $f_s$ is its
# value at the slave. For the Bloch phase $f = e^{ikL}$ we use a
# {py:class}`Constant<dolfinx.fem.Constant>` for $k$, and compile the expression
# once, so that it is not recompiled when $k$ changes.
# Repeated calls compound, so the coefficients are now $0.5 e^{ikL}$.

# +
k = fem.Constant(mesh, dtype(0))
L = 1.0
phase = fem.Expression(ufl.exp(1j * k * L), V.element.interpolation_points, dtype=dtype)

k.value = np.pi / 2
mpc.scale_coefficients(phase)
print_constraint(mpc, "Scaled by exp(i pi/2)")
# -

# ## Replacing the coefficients
#
# {py:meth}`update_coefficients<dolfinx_mpc.MultiPointConstraint.update_coefficients>`
# replaces all coefficients, in the layout of `all_coefficients`. Here we replace
# the coefficient $c_1$ of the (non-zero) Dirichlet master $u_k$ of each slave by
# $c_3 = 2$. The slave becomes $u_s = c_3 u_k = c_3 g_k$, and the offset is
# recomputed accordingly. The zero coefficients of the other masters are left
# as they are.

# +
coeffs, offsets = mpc.all_coefficients()
masters = mpc.all_masters()
new_coeffs = coeffs.copy()
for s in mpc.slaves:
    kept = mpc.masters.links(s)
    for j in range(offsets[s], offsets[s + 1]):
        if masters[j] not in kept and abs(coeffs[j]) > tol:
            new_coeffs[j] = 2
mpc.update_coefficients(new_coeffs)
print_constraint(mpc, "Dirichlet masters set to 2")
# -

# Note that `mpc.slaves` includes the ghosted slaves: every process must hold the
# same coefficients for a slave, so the ghosts are updated as well.
#
# ## Sweeping a parameter
#
# As scaling compounds, a sweep over $k$ stores the original coefficients once,
# restores them with `update_coefficients`, and scales with the current phase.
# Only the coefficients change; the masters, index maps and sparsity pattern are
# reused for every $k$.

base_coeffs = mpc_cut.all_coefficients()[0].copy()
for k_value in [0, np.pi / 4, np.pi]:
    mpc_cut.update_coefficients(base_coeffs)
    k.value = k_value
    mpc_cut.scale_coefficients(phase)
    print_constraint(mpc_cut, f"k = {k_value:.3f}", num_slaves=1)

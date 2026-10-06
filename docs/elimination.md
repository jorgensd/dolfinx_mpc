# Eliminating multi-point constraints

This section explains how DOLFINx_MPC imposes a multi-point constraint on a linear system: which
system is solved, how the assembly builds it, and what happens in parallel and across the blocks
of a system. {doc}`nonlinear_mpc` builds on it for nonlinear problems.

## The constraint

A multi-point constraint relates each *slave* degree of freedom $s$ to a set of *master* degrees
of freedom $m_j$,

$$
u_s = \sum_{j} c_{sj}\, u_{m_j} + g_s .
$$

Collecting the degrees of freedom that are not slaves into the reduced vector
$\hat{u}\in\mathbb{R}^{m}$, the constraint is the affine map

$$
\begin{aligned}
u &= K\hat{u} + g,\\
K &\in \mathbb{R}^{n\times m},\\
g &\in \mathbb{R}^{n},
\end{aligned}
$$

where $K$ has a unit row for every degree of freedom that is not a slave and the coefficients
$c_{sj}$ in the row of each slave, and $g$ is zero except on the slaves. A linear constraint is the
special case $g = 0$. For complex scalars, $K$ and $g$ are complex.

Two sources contribute to $g$:

1. an inhomogeneity given by the user, as `rhs_coeffs` of
   {py:class}`MultiPointConstraint<dolfinx_mpc.MultiPointConstraint>`;
2. masters with a Dirichlet condition. If master $m_j$ of slave $s$ has the prescribed value
   $g_{m_j}$, it is *eliminated* from the relation and its contribution folded into the offset,

$$
u_s = \sum_{j \notin \mathcal{D}} c_{sj} u_{m_j}
      + \underbrace{\sum_{j \in \mathcal{D}} c_{sj}\, g_{m_j}}_{\text{folded into } g_s} ,
$$

where $\mathcal{D}$ holds the masters with a Dirichlet condition. Because $g_{m_j}$ is read from the
{py:class}`dolfinx.fem.DirichletBC` each time
{py:meth}`update_constants<dolfinx_mpc.MultiPointConstraint.update_constants>` is called,
time-dependent boundary data is supported.

A Dirichlet condition can itself be phrased as a constraint, with an *empty* list of masters and
$g_s$ the prescribed value, so that $u_s = g_s$. Its row of $K$ is zero.

## The reduced system

For $A u = b$, substituting $u = K\hat{u} + g$ and testing the equations with $K^{H}$, the Hermitian
transpose (the transpose for real scalars), gives

$$
K^{H} A K\, \hat{u} = K^{H}\left(b - A g\right).
$$

$K^{H} A K$ is symmetric if $A$ is, Hermitian if $A$ is, and positive definite if $A$ is, so the
solvers suited to $A$ suit the reduced system as well. The term $-K^{H} A g$ is computed by
{py:func}`apply_mpc_lifting<dolfinx_mpc.apply_mpc_lifting>`. It is the same kind of term as the
lifting of a Dirichlet condition, and {py:class}`LinearProblem<dolfinx_mpc.LinearProblem>` applies
it. After the solve, the slaves are recovered by *backsubstitution*,
{py:meth}`backsubstitution<dolfinx_mpc.MultiPointConstraint.backsubstitution>`,

$$
u = K\hat{u} + g .
$$

## Assembly

$K$ is never formed. The reduced system is assembled cell by cell, as DOLFINx assembles $A$ and $b$,
and the constraint is applied to each element tensor before it is added to the global one:

- in the element matrix, the row of each slave is added to the rows of its masters, multiplied by
  the conjugated coefficients $\bar{c}_{sj}$, and its column to their columns, multiplied by
  $c_{sj}$. The row and column of the slave are then zeroed;
- in the element vector, the entry of each slave is added to those of its masters, multiplied by
  $\bar{c}_{sj}$, and then zeroed.

The masters need not be degrees of freedom of the cell, so the sparsity pattern of the matrix is
extended with the couplings the constraint creates. After assembly, the row of each slave holds only
a value on the diagonal, `diagval`, so that the matrix is invertible, and the slave entries of the
right-hand side are zero. The solve therefore gives zero on the slaves, and backsubstitution sets
them.

The matrix and vector are those of the degrees of freedom of the space the constraint was created
with, its {py:attr}`input_space<dolfinx_mpc.MultiPointConstraint.input_space>`, together with the
masters owned by other processes. The constraint therefore has its own space,
{py:attr}`function_space<dolfinx_mpc.MultiPointConstraint.function_space>`, the input space with
those masters added as ghosts, and the solution lives there. Forms and Dirichlet conditions are
stated on the input space.

## Parallel

Communication happens when a constraint is created and finalized: the masters of each slave are
located, possibly on other processes, and the constraint of a slave is sent to every process that
holds it as a ghost. Assembly is then local to each process. Each cell is modified with the
constraints the process holds, and the only communication is that of ordinary DOLFINx assembly, the
accumulation of the ghost entries.

## Dirichlet conditions

A Dirichlet condition on a degree of freedom that is not a master is applied as in DOLFINx: its row
and column are eliminated, with `diagval` on the diagonal, and its value is lifted into the
right-hand side.

A Dirichlet condition on a master must be given to the constraint, as `bcs` of
{py:class}`MultiPointConstraint<dolfinx_mpc.MultiPointConstraint>`, so that the master is
eliminated and its value folded into $g$, as above. The assembly adds the rows of slaves to those of
their masters after the rows of the Dirichlet conditions are eliminated, so a condition on a master
given only to the assembly would leave the row of the master wrong. The assembly checks this, and
raises an error on every process.

## Block systems

A system of several fields, such as a velocity and a pressure, or fields on different meshes, has
one constraint per block, finalized together by
{py:func}`finalize_multipointconstraints<dolfinx_mpc.finalize_multipointconstraints>`. A master may
be in another block than its slave, for instance a spider tying the dofs of a body to a point
({py:meth}`add_rbe2_geometrical<dolfinx_mpc.MultiPointConstraint.add_rbe2_geometrical>`) or a field
on a submesh tied to the trace of its parent
({py:meth}`create_submesh_constraint<dolfinx_mpc.MultiPointConstraint.create_submesh_constraint>`).
$K$ then couples the blocks: the reduced matrix has a block wherever a slave of one block has a
master in another, even if the forms of the system have no such block. The system can be assembled
as a nest matrix, one matrix per block, or as a single matrix.

## Changing the coefficients

The masters of a constraint, and with them the sparsity pattern and the extended space, are fixed
when it is finalized. Its coefficients are not:
{py:meth}`scale_coefficients<dolfinx_mpc.MultiPointConstraint.scale_coefficients>` and
{py:meth}`update_coefficients<dolfinx_mpc.MultiPointConstraint.update_coefficients>` change them
without building the constraint again, for instance to sweep the phase of a periodic condition, as in
{doc}`../python/demos/demo_scaling_coefficients`.

## Implementations

The assembly is written in C++, with a Python interface. The optional {py:mod}`dolfinx_mpc.numba`
module has assemblers written in Python and compiled with Numba, which support linear
constraints ($g = 0$).

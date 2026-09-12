# Multi-point constraints for nonlinear problems

This note derives how an (affine) multi-point constraint is applied to a
nonlinear problem in DOLFINx_MPC, and explains why the treatment of the
inhomogeneity $g$ differs between the linear and the nonlinear solver paths.

## The constraint

A multi-point constraint relates each *slave* degree of freedom $s$ to a set of
*master* degrees of freedom $m_j$,

$$
u_s \;=\; \sum_{j} c_{sj}\, u_{m_j} \;+\; g_s .
$$

Collecting the unconstrained degrees of freedom into the reduced vector
$\hat{u}\in\mathbb{R}^{m}$, this is the affine map

$$
u \;=\; K\hat{u} + g, \qquad K \in \mathbb{R}^{n\times m},\quad g\in\mathbb{R}^{n},
$$

where $K$ has a unit row for every unconstrained dof, the coefficients
$c_{sj}$ in the row of each slave, and $g$ is supported on the slave dofs only.
A purely linear constraint is the special case $g = 0$.

Two sources contribute to $g$:

1. a user-supplied inhomogeneity, passed as `rhs_coeffs`;
2. Dirichlet-constrained masters. If master $m_j$ of slave $s$ carries a
   Dirichlet condition with value $g_{m_j}$, it is *eliminated* from the
   relation and its contribution folded into the offset,

$$
u_s \;=\; \sum_{j \notin \mathcal{D}} c_{sj} u_{m_j}
        \;+\; \underbrace{\sum_{j \in \mathcal{D}} c_{sj}\, g_{m_j}}_{\text{folded into } g_s} .
$$

Because $g_{m_j}$ is read from the `DirichletBC` each time
`MultiPointConstraint.update_constants` is called, time-dependent boundary data
is supported.

A **Dirichlet condition is itself the degenerate case** of this relation: a dof
with an *empty* master list and $g_s$ equal to the prescribed value, so that
$u_s = g_s$. Its row of $K$ is identically zero.

## The linear problem

For $A u = b$, substituting $u = K\hat{u} + g$ and projecting the equations with
$K^{H}$ (the Hermitian transpose; the plain transpose for real scalars) gives

$$
K^{H} A K\, \hat{u} \;=\; K^{H}\!\left(b - A g\right).
$$

The term $-K^{H} A g$ is what `dolfinx_mpc.apply_mpc_lifting` computes. It is
structurally identical to Dirichlet lifting, and `LinearProblem` applies it
automatically. After solving, the full vector is recovered by *backsubstitution*,

$$
u \;=\; K\hat{u} + g .
$$

## The nonlinear problem

Let the residual be $F:\mathbb{R}^{n}\to\mathbb{R}^{n}$, with $F(u)=0$ to be
solved subject to the constraint. Substituting the constraint and projecting,
the reduced problem is

$$
r(\hat{u}) \;:=\; K^{H} F\!\left(K\hat{u} + g\right) \;=\; 0 .
$$

Differentiating with respect to $\hat{u}$ and writing $J(u) = \partial F/\partial u$
for the Jacobian of the full residual,

$$
\frac{\partial r}{\partial \hat{u}}
   \;=\; K^{H} J\!\left(K\hat{u}+g\right) K ,
$$

since $g$ is constant in $\hat{u}$. One Newton step is therefore

$$
K^{H} J(u)\, K \,\delta\hat{u} \;=\; -\,K^{H} F(u),
\qquad
\hat{u} \leftarrow \hat{u} + \delta\hat{u},
\qquad
u = K\hat{u} + g .
$$

### The increment is homogeneous, the iterate is affine

Equivalently, in the full space the update is

$$
\delta u \;=\; K\,\delta\hat{u},
\qquad\text{so}\qquad
u^{(k+1)} = u^{(k)} + \delta u .
$$

Note the absence of $g$ in the increment: two iterates both satisfying the
affine constraint differ by a vector satisfying the **homogeneous** one,

$$
u^{(k+1)} - u^{(k)} = K\!\left(\hat{u}^{(k+1)} - \hat{u}^{(k)}\right).
$$

This is the reason the code keeps two distinct operations:

| operation | applied to | formula |
|---|---|---|
| `backsubstitution` | the **iterate** | $u_s = \sum_j c_{sj} u_{m_j} + g_s$ |
| `homogenize` | an **increment** | $u_s = 0$ |

### Why no lifting of $g$ appears in the residual

This is the key difference from the linear path, and it is easy to get wrong.

In the linear path the unknown solved for is $\hat{u}$ itself, and the
right-hand side $b$ is assembled from the linear form alone — nothing in it
knows about $g$. The term $-K^{H} A g$ must therefore be added explicitly.

In the nonlinear path the residual is assembled **at the current iterate**, and
`assemble_residual_mpc` enforces $u = K\hat{u} + g$ by calling
`backsubstitution` *before* assembling. Hence $F$ is evaluated at a point that
already contains $g$, and the offset enters the reduced residual through $F$
itself. Adding `apply_mpc_lifting` here would count it twice.

The consistency of the two views is easiest to see in the linear case
$F(u) = Au - b$:

$$
r(\hat{u}) \;=\; K^{H}\!\left(A(K\hat{u}+g) - b\right)
         \;=\; \underbrace{K^{H} A K\,\hat{u}}_{\text{reduced operator}}
             \;+\; \underbrace{K^{H} A g \;-\; K^{H} b}_{\text{reduced load}} ,
$$

and setting $r=0$ recovers exactly the linear system of the previous section.

### Dirichlet conditions passed to the solver

Dirichlet conditions that are *not* folded into the constraint are handled by
the usual mechanism, phrased on the increment rather than the iterate. Writing
$x$ for the current iterate and $g_{\mathrm{bc}}$ for the prescribed values,
`assemble_residual_mpc` first calls `apply_lifting` with `scale=-1` and `x0=x`,

$$
F \;\leftarrow\; F \;+\; K^{H} J \left(g_{\mathrm{bc}} - x\right),
$$

and then `set_bc` with `alpha=-1` and `x0=x`, which overwrites the constrained
entries with

$$
F_{\mathrm{bc}} \;=\; -\left(g_{\mathrm{bc}} - x_{\mathrm{bc}}\right)
\;=\; x_{\mathrm{bc}} - g_{\mathrm{bc}} .
$$

Newton drives this residual to zero, giving $x_{\mathrm{bc}} = g_{\mathrm{bc}}$;
equivalently, the increment vanishes on those dofs once the iterate satisfies
the condition.

## Implementation summary

Per Newton iteration, `dolfinx_mpc` performs:

1. `homogenize(u)` then `backsubstitution(u)` — enforce $u = K\hat{u} + g$ on
   the incoming iterate.
2. Assemble $J$ with the constraint, giving $K^{H} J K$. Slave rows and columns
   are eliminated in place, and a value `diagval` is written on the diagonal of
   each slave row so the reduced matrix stays invertible.
3. Assemble $F$ with the constraint, giving $K^{H} F(u)$. The slave entries of
   the residual are zeroed.
4. Apply Dirichlet lifting for any `bcs` supplied to the solver.
5. Solve for $\delta\hat{u}$ and update. Slave entries of the increment are
   zero by construction, and step 1 restores the constraint on the next
   evaluation.

### Consequence for convergence

Because the reduced residual $K^{H}F$ is what must vanish, the slave entries of
the assembled residual must be *exactly* zero — otherwise SNES measures a
residual norm that its Newton step cannot reduce, and the solve stagnates at a
finite norm even though the computed solution is correct. This is why the slave
entry is zeroed for every slave, including one whose master list is empty (a
Dirichlet condition expressed as a constraint).

## Worked example

Expressing a Dirichlet condition entirely as a constraint, with no `DirichletBC`
reaching the assembler:

```python
slaves = np.sort(bc_dofs).astype(np.int32)   # owned *and* ghost dofs

g = dolfinx.fem.Function(V)
g.x.array[:] = 0.0
g.x.array[slaves] = u_bc.x.array[slaves]
g.x.scatter_forward()

mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
mpc.add_constraint(
    V,
    slaves,
    np.array([], dtype=np.int64),          # no masters
    np.array([], dtype=default_scalar_type),
    np.array([], dtype=np.int32),
    np.zeros(len(slaves) + 1, dtype=np.int32),
)
mpc.finalize()

problem = dolfinx_mpc.NonlinearProblem(F, uh, mpc=mpc, bcs=[], J=J)
problem.solve()
```

Ghost slaves must be declared as well: a process that only ghosts a constrained
dof still assembles cells touching it, and would otherwise fail to eliminate it.

`python/tests/test_affine_constraint.py` verifies that this reproduces an
ordinary `DirichletBC` solve, for both the linear and the nonlinear problem.

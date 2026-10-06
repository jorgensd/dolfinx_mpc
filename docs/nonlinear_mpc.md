# Multi-point constraints for nonlinear problems

This section explains how an affine multi-point constraint, $u = K\hat{u} + g$, is applied to a
nonlinear problem, and why the inhomogeneity $g$ is treated differently than in the linear problem.
The constraint, the reduced linear system and its assembly are described in {doc}`elimination`.

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
\begin{aligned}
K^{H} J(u)\, K \,\delta\hat{u} &= -\,K^{H} F(u),\\
\hat{u} &\leftarrow \hat{u} + \delta\hat{u},\\
u &= K\hat{u} + g .
\end{aligned}
$$

### The increment is homogeneous, the iterate is affine

Equivalently, in the full space the update is

$$
\begin{aligned}
\delta u &= K\,\delta\hat{u},\\
u^{(k+1)} &= u^{(k)} + \delta u .
\end{aligned}
$$

Note the absence of $g$ in the increment: two iterates both satisfying the
affine constraint differ by a vector satisfying the **homogeneous** one,

$$
u^{(k+1)} - u^{(k)} = K\!\left(\hat{u}^{(k+1)} - \hat{u}^{(k)}\right).
$$

This is the reason the code keeps two distinct operations:

| operation | applied to | formula |
|---|---|---|
| {py:meth}`backsubstitution<dolfinx_mpc.MultiPointConstraint.backsubstitution>` | the **iterate** | $u_s = \sum_j c_{sj} u_{m_j} + g_s$ |
| {py:meth}`homogenize<dolfinx_mpc.MultiPointConstraint.homogenize>` | an **increment** | $u_s = 0$ |

### Residual evaluation without lifting of $g$

In the linear path the unknown solved for is $\hat{u}$ itself, and the
right-hand side $b$ is assembled from the linear form alone — nothing in it
knows about $g$. The term $-K^{H} A g$ must therefore be added explicitly.

In the nonlinear path the residual is assembled **at the current iterate**, and
{py:func}`dolfinx_mpc.assemble_residual_mpc` enforces $u = K\hat{u} + g$ by calling
{py:meth}`dolfinx_mpc.MultiPointConstraint.backsubstitution` *before* assembling. Hence $F$ is evaluated at a point that
already contains $g$, and the offset enters the reduced residual through $F$
itself. Adding {py:func}`dolfinx_mpc.apply_mpc_lifting` here would count it twice.

The consistency of the two views is easiest to see in the linear case
$F(u) = Au - b$:

$$
r(\hat{u}) \;=\; K^{H}\!\left(A(K\hat{u}+g) - b\right)
         \;=\; \underbrace{K^{H} A K\,\hat{u}}_{\text{reduced operator}}
             \;+\; \underbrace{K^{H} A g \;-\; K^{H} b}_{\text{reduced load}} ,
$$

and setting $r=0$ recovers exactly the linear system of {doc}`elimination`.

### Handling of non-MPC Dirichlet conditions

Dirichlet conditions that are *not* folded into the constraint are handled by
the usual mechanism, phrased on the increment rather than the iterate. Writing
$x$ for the current iterate and $g_{\mathrm{bc}}$ for the prescribed values,
{py:func}`dolfinx_mpc.assemble_residual_mpc` first calls {py:func}`dolfinx_mpc.apply_lifting` with `scale=-1` and `x0=x`,

$$
F \;\leftarrow\; F \;+\; K^{H} J \left(g_{\mathrm{bc}} - x\right),
$$

and then {py:func}`set_bc<dolfinx.fem.petsc.set_bc>` with `alpha=-1` and `x0=x`, which overwrites the constrained
entries with

$$
F_{\mathrm{bc}} \;=\; -\left(g_{\mathrm{bc}} - x_{\mathrm{bc}}\right)
\;=\; x_{\mathrm{bc}} - g_{\mathrm{bc}} .
$$

Newton drives this residual to zero, giving $x_{\mathrm{bc}} = g_{\mathrm{bc}}$;
equivalently, the increment vanishes on those dofs once the iterate satisfies
the condition.

## Implementation summary

Per Newton iteration, {py:mod}`dolfinx_mpc` performs:

1. {py:meth}`homogenize(u)<dolfinx_mpc.MultiPointConstraint.homogenize>` then {py:meth}`backsubstitution(u)<dolfinx_mpc.MultiPointConstraint.backsubstitution>` — enforce $u = K\hat{u} + g$ on
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

Expressing a Dirichlet condition entirely as a constraint, with no {py:class}`dolfinx.fem.DirichletBC`
reaching the assembler:

```python
slaves = np.sort(bc_dofs).astype(np.int32)  # owned *and* ghost dofs

g = dolfinx.fem.Function(V)
g.x.array[:] = 0.0
g.x.array[slaves] = u_bc.x.array[slaves]
g.x.scatter_forward()

mpc = dolfinx_mpc.MultiPointConstraint(V, rhs_coeffs=g)
mpc.add_constraint(
    V,
    slaves,
    np.array([], dtype=np.int64),  # no masters
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

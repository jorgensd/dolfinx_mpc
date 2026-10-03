"""Forms of the cross-block demo: two reaction-diffusion problems on P1 triangles.

The field `u` has a constant diffusion coefficient `kappa`, the field `p` has none. The forms do not
couple the two fields.
"""

from basix.ufl import element
from ufl import (
    Constant,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    pi,
    sin,
)

e = element("Lagrange", "triangle", 1)
mesh = Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = FunctionSpace(mesh, e)
Q = FunctionSpace(mesh, e)
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
x = SpatialCoordinate(mesh)
kappa = Constant(mesh)

a_u = kappa * inner(grad(u), grad(v)) * dx + inner(u, v) * dx
a_p = inner(grad(p), grad(q)) * dx + inner(p, q) * dx
L_u = inner(sin(pi * x[1]), v) * dx
L_p = inner(x[0] * x[1], q) * dx

# Forms other than a, L, M, F and J are only compiled when listed
forms = [a_u, a_p, L_u, L_p]

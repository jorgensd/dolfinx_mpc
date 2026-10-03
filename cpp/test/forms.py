"""Forms of the C++ tests: a reaction-diffusion operator on P1 triangles."""

from basix.ufl import element
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dx, grad, inner

e = element("Lagrange", "triangle", 1)
mesh = Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = FunctionSpace(mesh, e)
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx

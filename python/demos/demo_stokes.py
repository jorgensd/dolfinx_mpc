# # Stokes flow with slip conditions
# **Author** Jørgen S. Dokken
#
# **License** MIT
#
# This demo illustrates how to apply a slip condition on an
# interface not aligned with the coordiante axis.

# We start by the various modules required for this demo

# +
from __future__ import annotations

from pathlib import Path
from typing import Union

from mpi4py import MPI

import basix.ufl
import gmsh
import numpy as np
from dolfinx import common, default_real_type, default_scalar_type, fem, io
from dolfinx.io import gmshio
from numpy.typing import NDArray
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    div,
    dot,
    dx,
    extract_blocks,
    grad,
    inner,
    outer,
    sym,
)
from ufl.core.expr import Expr

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint

# -
# ## Mesh generation
#
# Next we create the computational domain, a channel titled with respect to the coordinate axis.
# We use GMSH and the DOLFINx GMSH-IO to convert the GMSH model into a DOLFINx mesh

# +


def create_mesh_gmsh(
    L: int = 2,
    H: int = 1,
    res: float = 0.1,
    theta: float = np.pi / 5,
    wall_marker: int = 1,
    outlet_marker: int = 2,
    inlet_marker: int = 3,
):
    """
    Create a channel of length L, height H, rotated theta degrees
    around origin, with facet markers for inlet, outlet and walls.


    Parameters
    ----------
    L
        The length of the channel
    H
        Width of the channel
    res
        Mesh resolution (uniform)
    theta
        Rotation angle
    wall_marker
        Integer used to mark the walls of the channel
    outlet_marker
        Integer used to mark the outlet of the channel
    inlet_marker
        Integer used to mark the inlet of the channel
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Square duct")

        # Create rectangular channel
        channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        gmsh.model.occ.synchronize()

        # Find entity markers before rotation
        surfaces = gmsh.model.occ.getEntities(dim=1)
        walls = []
        inlets = []
        outlets = []
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [0, H / 2, 0]):
                inlets.append(surface[1])
            elif np.allclose(com, [L, H / 2, 0]):
                outlets.append(surface[1])
            elif np.isclose(com[1], 0) or np.isclose(com[1], H):
                walls.append(surface[1])
        # Rotate channel theta degrees in the xy-plane
        gmsh.model.occ.rotate([(2, channel)], 0, 0, 0, 0, 0, 1, theta)
        gmsh.model.occ.synchronize()

        # Add physical markers
        gmsh.model.addPhysicalGroup(2, [channel], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid volume")
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inlets, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Fluid inlet")
        gmsh.model.addPhysicalGroup(1, outlets, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Fluid outlet")

        # Set uniform mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

        # Generate mesh
        gmsh.model.mesh.generate(2)
    # Convert gmsh model to DOLFINx Mesh and meshtags
    mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    assert mesh_data.facet_tags is not None
    return mesh_data.mesh, mesh_data.facet_tags


mesh, mt = create_mesh_gmsh(res=0.1)
fdim = mesh.topology.dim - 1

# The next step is the create the function spaces for the fluid velocit and pressure.
# We will use a mixed-formulation, and we use `basix.ufl` to create the Taylor-Hood finite element pair

P2 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
P1 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, dtype=default_real_type)

V = fem.functionspace(mesh, P2)
Q = fem.functionspace(mesh, P1)
W = MixedFunctionSpace(V, Q)
# -

# ## Boundary conditions
# Next we want to prescribe an inflow condition on a set of facets, those marked by 3 in GMSH
# To do so, we start by creating a function in the **collapsed** subspace of V, and create the
# expression of choice for the boundary condition.
# We use a Python-function for this, that takes in an array of shape `(num_points, 3)` and
# returns the function values as an `(num_points, 2)` array.
# Note that we therefore can use vectorized Numpy operations to compute the inlet values for a sequence of points
# with a single function call

inlet_velocity = fem.Function(V)


def inlet_velocity_expression(
    x: NDArray[Union[np.float32, np.float64]],
) -> NDArray[Union[np.float32, np.float64, np.complex64, np.complex128]]:
    return np.stack(
        (
            np.sin(np.pi * np.sqrt(x[0] ** 2 + x[1] ** 2)),
            5 * x[1] * np.sin(np.pi * np.sqrt(x[0] ** 2 + x[1] ** 2)),
        )
    ).astype(default_scalar_type)


inlet_velocity.interpolate(inlet_velocity_expression)
inlet_velocity.x.scatter_forward()

# Next, we have to create the Dirichlet boundary condition. As the function is in the collapsed space,
# we send in the function space `V` of the velocity

# +
dofs = fem.locate_dofs_topological(V, 1, mt.find(3))
bc1 = fem.dirichletbc(inlet_velocity, dofs)
bcs = [bc1]
# -

# Next, we want to create the slip conditions at the side walls of the channel.
# To do so, we compute an approximation of the normal at all the degrees of freedom that
# are associated with the facets marked with `1`, either by being on one of the vertices of a
# facet marked with `1`, or on the facet itself (for instance at a midpoint).
# If a dof is associated with a vertex that is connected to mutliple facets marked with `1`,
# we define the normal as the average between the normals at the vertex viewed from each facet.

# +
n = dolfinx_mpc.utils.create_normal_approximation(V, mt, 1)
# -

# Next, we can create the multipoint-constraint, enforcing that $\mathbf{u}\cdot\mathbf{n}=0$,
# except where we have already enforced a Dirichlet boundary condition

# +
with common.Timer("~Stokes: Create slip constraint"):
    mpc_u = MultiPointConstraint(V)
    mpc_u.create_slip_constraint(V, (mt, 1), n, bcs=bcs)
mpc_u.finalize()

# -

# +
# As we will not have an multipoint constraint for the pressure, we create an empty constraint.
mpc_p = MultiPointConstraint(Q)
mpc_p.finalize()

# ## Variational formulation
# We start by creating some convenience functions, including the tangential projection of a vector,
# the symmetric gradient and traction.

# +


def tangential_proj(u: Expr, n: Expr):
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (Identity(u.ufl_shape[0]) - outer(n, n)) * u


def sym_grad(u: Expr):
    return sym(grad(u))


def T(u: Expr, p: Expr, mu: Expr):
    return 2 * mu * sym_grad(u) - p * Identity(u.ufl_shape[0])


# -


# Next, we define the classical terms of the bilinear form `a` and linear form `L`.
# We note that we use the symmetric formulation after integration by parts.

# +
mu = fem.Constant(mesh, default_scalar_type(1.0))
f = fem.Constant(mesh, default_scalar_type((0, 0)))
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
a = (2 * mu * inner(sym_grad(u), sym_grad(v)) - inner(p, div(v)) - inner(div(u), q)) * dx
L = inner(f, v) * dx + inner(fem.Constant(mesh, default_scalar_type(0.0)), q) * dx
# -

# We could prescibe some shear stress at the slip boundaries. However, in this demo, we set it to `0`,
# but include it in the variational formulation. We add the appropriate terms due to the slip condition,
# as explained in https://arxiv.org/pdf/2001.10639.pdf

# +
n = FacetNormal(mesh)
g_tau = tangential_proj(fem.Constant(mesh, default_scalar_type(((0, 0), (0, 0)))) * n, n)
ds = Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

a -= inner(outer(n, n) * dot(T(u, p, mu), n), v) * ds
L += inner(g_tau, v) * ds
# -

# ## Solve the variational problem
# We use the MUMPS solver provided through the DOLFINX_MPC PETSc interface to solve the variational problem

# +
tol = np.finfo(mesh.geometry.x.dtype).eps * 1000
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
}
problem = LinearProblem(extract_blocks(a), extract_blocks(L), [mpc_u, mpc_p], bcs=bcs, petsc_options=petsc_options)
uh, ph = problem.solve()
# -

# ## Visualization
# We store the solution to the `VTX` file format, which can be opend with the
# `ADIOS2VTXReader` in Paraview

# +

uh.name = "u"
ph.name = "p"

outdir = Path("results").absolute()
outdir.mkdir(exist_ok=True, parents=True)

with io.VTXWriter(mesh.comm, outdir / "demo_stokes_u.bp", uh, engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(mesh.comm, outdir / "demo_stokes_p.bp", ph, engine="BP4") as vtx:
    vtx.write(0.0)
# -

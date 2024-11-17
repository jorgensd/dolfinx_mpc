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
from petsc4py import PETSc

import basix.ufl
import gmsh
import numpy as np
import scipy.sparse.linalg
from dolfinx import common, default_real_type, default_scalar_type, fem, io
from dolfinx.io import gmshio
from numpy.typing import NDArray
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    TestFunctions,
    TrialFunctions,
    div,
    dot,
    dx,
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

        # Set number of threads used for mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)

        # Set uniform mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

        # Generate mesh
        gmsh.model.mesh.generate(2)
    # Convert gmsh model to DOLFINx Mesh and meshtags
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return mesh, ft


mesh, mt = create_mesh_gmsh(res=0.1)
fdim = mesh.topology.dim - 1

# The next step is the create the function spaces for the fluid velocit and pressure.
# We will use a mixed-formulation, and we use `basix.ufl` to create the Taylor-Hood finite element pair

P2 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
P1 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, dtype=default_real_type)

TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(mesh, TH)
# -

# ## Boundary conditions
# Next we want to prescribe an inflow condition on a set of facets, those marked by 3 in GMSH
# To do so, we start by creating a function in the **collapsed** subspace of V, and create the
# expression of choice for the boundary condition.
# We use a Python-function for this, that takes in an array of shape `(num_points, 3)` and
# returns the function values as an `(num_points, 2)` array.
# Note that we therefore can use vectorized Numpy operations to compute the inlet values for a sequence of points
# with a single function call

V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
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
# we send in the un-collapsed space and the collapsed space to locate dofs topological to find the
# appropriate dofs and the mapping from `V` to `W0`.

# +
W0 = W.sub(0)
dofs = fem.locate_dofs_topological((W0, V), 1, mt.find(3))
bc1 = fem.dirichletbc(inlet_velocity, dofs, W0)
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
    mpc = MultiPointConstraint(W)
    mpc.create_slip_constraint(W.sub(0), (mt, 1), n, bcs=bcs)
mpc.finalize()
# -

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
L = inner(f, v) * dx
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
petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = LinearProblem(a, L, mpc, bcs=bcs, petsc_options=petsc_options)
U = problem.solve()
# -

# ## Visualization
# We store the solution to the `VTX` file format, which can be opend with the
# `ADIOS2VTXReader` in Paraview

# +
u = U.sub(0).collapse()
p = U.sub(1).collapse()
u.name = "u"
p.name = "p"

outdir = Path("results").absolute()
outdir.mkdir(exist_ok=True, parents=True)

with io.VTXWriter(mesh.comm, outdir / "demo_stokes_u.bp", u, engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(mesh.comm, outdir / "demo_stokes_p.bp", p, engine="BP4") as vtx:
    vtx.write(0.0)
# -

# ## Verification
# We verify that the MPC implementation is correct by creating the global reduction matrix
# and performing global matrix-matrix-matrix multiplication using numpy

# +
with common.Timer("~Stokes: Verification of problem by global matrix reduction"):
    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    bilinear_form = fem.form(a)
    A_org = fem.petsc.assemble_matrix(bilinear_form, bcs)
    A_org.assemble()
    linear_form = fem.form(L)
    L_org = fem.petsc.assemble_vector(linear_form)

    fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    fem.petsc.set_bc(L_org, bcs)
    root = 0
    dolfinx_mpc.utils.compare_mpc_lhs(A_org, problem.A, mpc, root=root)
    dolfinx_mpc.utils.compare_mpc_rhs(L_org, problem.b, mpc, root=root)

    # Gather LHS, RHS and solution on one process
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
    u_mpc = dolfinx_mpc.utils.gather_PETScVector(U.x.petsc_vec, root=root)
    is_complex = np.issubdtype(default_scalar_type, np.complexfloating)  # type: ignore
    scipy_dtype = np.complex128 if is_complex else np.float64
    if MPI.COMM_WORLD.rank == root:
        KTAK = K.T.astype(scipy_dtype) * A_csr.astype(scipy_dtype) * K.astype(scipy_dtype)
        reduced_L = K.T.astype(scipy_dtype) @ L_np.astype(scipy_dtype)
        # Solve linear system
        d = scipy.sparse.linalg.spsolve(KTAK.astype(scipy_dtype), reduced_L.astype(scipy_dtype))
        # Back substitution to full solution vector
        uh_numpy = K.astype(scipy_dtype) @ d.astype(scipy_dtype)
        assert np.allclose(uh_numpy.astype(u_mpc.dtype), u_mpc, atol=float(5e2 * np.finfo(u_mpc.dtype).resolution))
# -

# ## Timings
# Finally, we list the execution time of various operations used in the demo

common.list_timings(MPI.COMM_WORLD)

# Copyright (C) 2022 Nathan Sime
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# This demo illustrates how to apply a slip condition on an
# interface not aligned with the coordiante axis.
# The demos solves the Stokes problem using the nest functionality to
# avoid using mixed function spaces. The demo also illustrates how to use
#  block preconditioners with PETSc
from __future__ import annotations

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx.io
import gmsh
import numpy as np
import scipy.sparse.linalg
import ufl
from dolfinx import default_real_type, default_scalar_type
from dolfinx.io import gmshio
from ufl.core.expr import Expr

import dolfinx_mpc
import dolfinx_mpc.utils


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
    assert mesh_data.cell_tags is not None
    ft = mesh_data.facet_tags
    gmsh.finalize()
    return mesh_data.mesh, ft


# ------------------- Mesh and function space creation ------------------------
mesh, mt = create_mesh_gmsh(res=0.1)

fdim = mesh.topology.dim - 1

# Create the function space
cellname = mesh.ufl_cell().cellname()
Ve = basix.ufl.element(basix.ElementFamily.P, cellname, 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
Qe = basix.ufl.element(basix.ElementFamily.P, cellname, 1, dtype=default_real_type)

V = dolfinx.fem.functionspace(mesh, Ve)
Q = dolfinx.fem.functionspace(mesh, Qe)


def inlet_velocity_expression(x):
    return np.stack(
        (
            np.sin(np.pi * np.sqrt(x[0] ** 2 + x[1] ** 2)),
            5 * x[1] * np.sin(np.pi * np.sqrt(x[0] ** 2 + x[1] ** 2)),
        )
    )


# ----------------------Defining boundary conditions----------------------
# Inlet velocity Dirichlet BC
inlet_velocity = dolfinx.fem.Function(V)
inlet_velocity.interpolate(inlet_velocity_expression)
inlet_velocity.x.scatter_forward()
dofs = dolfinx.fem.locate_dofs_topological(V, 1, mt.find(3))
bc1 = dolfinx.fem.dirichletbc(inlet_velocity, dofs)

# Collect Dirichlet boundary conditions
bcs: list[dolfinx.fem.DirichletBC] = [bc1]

# Slip conditions for walls
n = dolfinx_mpc.utils.create_normal_approximation(V, mt, 1)
with dolfinx.common.Timer("~Stokes: Create slip constraint"):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_slip_constraint(V, (mt, 1), n, bcs=bcs)
mpc.finalize()

mpc_q = dolfinx_mpc.MultiPointConstraint(Q)
mpc_q.finalize()


def tangential_proj(u: Expr, n: Expr):
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def sym_grad(u: Expr):
    return ufl.sym(ufl.grad(u))


def T(u: Expr, p: Expr, mu: Expr):
    return 2 * mu * sym_grad(u) - p * ufl.Identity(u.ufl_shape[0])


# --------------------------Variational problem---------------------------
# Traditional terms
mu = 1
f = dolfinx.fem.Constant(mesh, default_scalar_type((0, 0)))
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
a00 = 2 * mu * ufl.inner(sym_grad(u), sym_grad(v)) * ufl.dx
a01 = -ufl.inner(p, ufl.div(v)) * ufl.dx
a10 = -ufl.inner(ufl.div(u), q) * ufl.dx
a11 = None

L0 = ufl.inner(f, v) * ufl.dx
L1 = ufl.inner(dolfinx.fem.Constant(mesh, default_scalar_type(0.0)), q) * ufl.dx

# No prescribed shear stress
n = ufl.FacetNormal(mesh)
g_tau = tangential_proj(dolfinx.fem.Constant(mesh, default_scalar_type(((0, 0), (0, 0)))) * n, n)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

# Terms due to slip condition
# Explained in for instance: https://arxiv.org/pdf/2001.10639.pdf
a00 = 2 * mu * ufl.inner(sym_grad(u), sym_grad(v)) * ufl.dx

a00 -= ufl.inner(ufl.outer(n, n) * ufl.dot(2 * mu * sym_grad(u), n), v) * ds
a01 -= ufl.inner(ufl.outer(n, n) * ufl.dot(-p * ufl.Identity(u.ufl_shape[0]), n), v) * ds
L0 += ufl.inner(g_tau, v) * ds

a: list[list[dolfinx.fem.Form]] = dolfinx.fem.form([
    [a00, a01],
    [a10, a11],
])
L: list[dolfinx.fem.Form] = dolfinx.fem.form([L0, L1])

# Assemble LHS matrix and RHS vector
with dolfinx.common.Timer("~Stokes: Assemble LHS and RHS"):
    A = dolfinx_mpc.create_matrix_nest(a, [mpc, mpc_q])
    dolfinx_mpc.assemble_matrix_nest(A, a, [mpc, mpc_q], bcs)
    A.assemble()

    b = dolfinx_mpc.create_vector_nest(L, [mpc, mpc_q])
    dolfinx_mpc.assemble_vector_nest(b, L, [mpc, mpc_q])

# Set Dirichlet boundary condition values in the RHS
bcs1 = dolfinx.fem.bcs_by_block(
                    dolfinx.fem.extract_function_spaces(a, 1),
                    bcs
                )
dolfinx_mpc.apply_lifting(b, a, bcs1, constraint=[mpc, mpc_q])
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(L), bcs)
dolfinx.fem.petsc.set_bc(b, bcs0)

# Preconditioner
P11 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(ufl.inner(p, q) * ufl.dx))
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])  # type: ignore
P.assemble()

# ---------------------- Solve variational problem -----------------------
ksp = PETSc.KSP().create(mesh.comm)  # type: ignore
ksp.setOperators(A, P)
ksp.setMonitor(
    lambda ctx, it, r: PETSc.Sys.Print(  # type: ignore
        f"Iteration: {it:>4d}, |r| = {r:.3e}"
    )
)
ksp.setType("minres")
tol = 10 * np.finfo(mesh.geometry.x.dtype).eps
ksp.setTolerances(rtol=tol, atol=tol)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)  # type: ignore

nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.setTolerances(rtol=tol, atol=tol)
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")
ksp_p.setTolerances(rtol=tol, atol=tol)
ksp.setFromOptions()

Uh = b.copy()
ksp.solve(b, Uh)

for Uh_sub in Uh.getNestSubVecs():
    Uh_sub.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,  # type: ignore
        mode=PETSc.ScatterMode.FORWARD,  # type: ignore
    )  # type: ignore
# ----------------------------- Put NestVec into DOLFINx Function - ---------
uh = dolfinx.fem.Function(mpc.function_space)
uh.x.petsc_vec.setArray(Uh.getNestSubVecs()[0].array)

ph = dolfinx.fem.Function(mpc_q.function_space)
ph.x.petsc_vec.setArray(Uh.getNestSubVecs()[1].array)

uh.x.scatter_forward()
ph.x.scatter_forward()

# Backsubstitute to update slave dofs in solution vector
mpc.backsubstitution(uh)
mpc_q.backsubstitution(ph)

# ------------------------------ Output ----------------------------------

uh.name = "u"
ph.name = "p"
outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)

with dolfinx.io.XDMFFile(mesh.comm, outdir / "demo_stokes_nest.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_meshtags(mt, mesh.geometry)
    outfile.write_function(ph)

with dolfinx.io.VTXWriter(mesh.comm, outdir / "stokes_nest_uh.bp", uh, engine="BP4") as vtx:
    vtx.write(0.0)
# -------------------- Verification --------------------------------
# Transfer data from the MPC problem to numpy arrays for comparison
with dolfinx.common.Timer("~Stokes: Verification of problem by global matrix reduction"):
    W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([Ve, Qe]))
    V, V_to_W = W.sub(0).collapse()
    _, Q_to_W = W.sub(1).collapse()

    # Inlet velocity Dirichlet BC
    inlet_velocity = dolfinx.fem.Function(V)
    inlet_velocity.interpolate(inlet_velocity_expression)
    inlet_velocity.x.scatter_forward()
    W0 = W.sub(0)
    dofs = dolfinx.fem.locate_dofs_topological((W0, V), 1, mt.find(3))
    bc1 = dolfinx.fem.dirichletbc(inlet_velocity, dofs, W0)

    # Collect Dirichlet boundary conditions
    bcs = [bc1]

    # Slip conditions for walls
    n = dolfinx_mpc.utils.create_normal_approximation(V, mt, 1)
    with dolfinx.common.Timer("~Stokes: Create slip constraint"):
        mpc = dolfinx_mpc.MultiPointConstraint(W)
        mpc.create_slip_constraint(W.sub(0), (mt, 1), n, bcs=bcs)
    mpc.finalize()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (2 * mu * ufl.inner(sym_grad(u), sym_grad(v)) - ufl.inner(p, ufl.div(v)) - ufl.inner(ufl.div(u), q)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Terms due to slip condition
    # Explained in for instance: https://arxiv.org/pdf/2001.10639.pdf
    a -= ufl.inner(ufl.outer(n, n) * ufl.dot(T(u, p, mu), n), v) * ds
    L += ufl.inner(g_tau, v) * ds

    af = dolfinx.fem.form(a)
    Lf = dolfinx.fem.form(L)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.petsc.assemble_matrix(af, bcs)
    A_org.assemble()
    L_org = dolfinx.fem.petsc.assemble_vector(Lf)

    dolfinx.fem.petsc.apply_lifting(L_org, [af], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    dolfinx.fem.petsc.set_bc(L_org, bcs)
    root = 0

    # Gather LHS, RHS and solution on one process
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)

    u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh.x.petsc_vec, root=root)
    p_mpc = dolfinx_mpc.utils.gather_PETScVector(ph.x.petsc_vec, root=root)
    up_mpc = np.hstack([u_mpc, p_mpc])
    if MPI.COMM_WORLD.rank == root:
        KTAK = K.T * A_csr * K
        reduced_L = K.T @ L_np
        # Solve linear system
        d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
        # Back substitution to full solution vector
        uh_numpy = K @ d
        np.testing.assert_allclose(np.linalg.norm(uh_numpy, 2), np.linalg.norm(up_mpc, 2), atol=tol, rtol=100 * tol)


A.destroy()
b.destroy()
for Uh_sub in Uh.getNestSubVecs():
    Uh_sub.destroy()
Uh.destroy()
ksp.destroy()
# -------------------- List timings --------------------------
dolfinx.common.list_timings(MPI.COMM_WORLD)

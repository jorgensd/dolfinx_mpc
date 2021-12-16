import dolfinx.fem as fem
import dolfinx_mpc.utils
import gmsh
import numpy as np
import scipy.sparse.linalg

from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.io import XDMFFile
from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)

from mpi4py import MPI
from petsc4py import PETSc

from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunctions,
                 TrialFunctions, VectorElement, div, dot, dx, grad, inner,
                 outer, sym)


def create_mesh_gmsh(L: int = 2, H: int = 1, res: np.float64 = 0.1, theta: np.float64 = np.pi / 5,
                     wall_marker: int = 1, outlet_marker: int = 2, inlet_marker: int = 3):
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
        gmsh.model.occ.rotate([(2, channel)], 0, 0, 0,
                              0, 0, 1, theta)
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
    # Convert gmsh model to DOLFINx Mesh and MeshTags
    mesh, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, facet_data=True, gdim=2)
    gmsh.finalize()
    return mesh, ft


# ------------------- Mesh and function space creation ------------------------
mesh, mt = create_mesh_gmsh(res=0.1)

fdim = mesh.topology.dim - 1
# Create the function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = fem.FunctionSpace(mesh, TH)
V, V_to_W = W.sub(0).collapse(True)
Q = W.sub(1).collapse()


def inlet_velocity_expression(x):
    return np.stack((np.sin(np.pi * np.sqrt(x[0]**2 + x[1]**2)),
                     5 * x[1] * np.sin(np.pi * np.sqrt(x[0]**2 + x[1]**2))))


# ----------------------Defining boundary conditions----------------------
# Inlet velocity Dirichlet BC
inlet_facets = mt.indices[mt.values == 3]
inlet_velocity = fem.Function(V)
inlet_velocity.interpolate(inlet_velocity_expression)
inlet_velocity.x.scatter_forward()
W0 = W.sub(0)
dofs = fem.locate_dofs_topological((W0, V), 1, inlet_facets)
bc1 = fem.DirichletBC(inlet_velocity, dofs, W0)

# Collect Dirichlet boundary conditions
bcs = [bc1]
# Slip conditions for walls
n = dolfinx_mpc.utils.create_normal_approximation(V, mt, 1)
with Timer("~Stokes: Create slip constraint"):
    mpc = MultiPointConstraint(W)
    mpc.create_slip_constraint((mt, 1), n, sub_space=W.sub(0), sub_map=V_to_W, bcs=bcs)
mpc.finalize()


def tangential_proj(u, n):
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (Identity(u.ufl_shape[0]) - outer(n, n)) * u


def sym_grad(u):
    return sym(grad(u))


def T(u, p, mu):
    return 2 * mu * sym_grad(u) - p * Identity(u.ufl_shape[0])


# --------------------------Variational problem---------------------------
# Traditional terms
mu = 1
f = fem.Constant(mesh, PETSc.ScalarType(((0, 0))))
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
a = (2 * mu * inner(sym_grad(u), sym_grad(v))
     - inner(p, div(v))
     - inner(div(u), q)) * dx
L = inner(f, v) * dx

# No prescribed shear stress
n = FacetNormal(mesh)
g_tau = tangential_proj(fem.Constant(mesh, PETSc.ScalarType(((0, 0), (0, 0)))) * n, n)
ds = Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

# Terms due to slip condition
# Explained in for instance: https://arxiv.org/pdf/2001.10639.pdf
a -= inner(outer(n, n) * dot(T(u, p, mu), n), v) * ds
L += inner(g_tau, v) * ds

# Assemble LHS matrix and RHS vector
with Timer("~Stokes: Assemble LHS and RHS"):
    A = assemble_matrix(a, mpc, bcs)
    A.assemble()
    b = assemble_vector(L, mpc)

# Set Dirichlet boundary condition values in the RHS
apply_lifting(b, [a], [bcs], mpc)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.assemble.set_bc(b, bcs)

# ---------------------- Solve variational problem -----------------------
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
uh = b.copy()
ksp.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Backsubstitute to update slave dofs in solution vector
mpc.backsubstitution(uh)

# ------------------------------ Output ----------------------------------
U = fem.Function(mpc.function_space)
U.vector.setArray(uh.array)
u = U.sub(0).collapse()
p = U.sub(1).collapse()
u.name = "u"
p.name = "p"
with XDMFFile(MPI.COMM_WORLD, "results/demo_stokes.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_meshtags(mt)
    outfile.write_function(u)
    outfile.write_function(p)

# -------------------- Verification --------------------------------
# Transfer data from the MPC problem to numpy arrays for comparison
with Timer("~Stokes: Verification of problem by global matrix reduction"):

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    A_org = fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = fem.assemble_vector(L)

    fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L_org, bcs)
    root = 0
    dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)
    dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

    # Gather LHS, RHS and solution on one process
    A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
    K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
    L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
    u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh, root=root)

    if MPI.COMM_WORLD.rank == root:
        KTAK = K.T * A_csr * K
        reduced_L = K.T @ L_np
        # Solve linear system
        d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
        # Back substitution to full solution vector
        uh_numpy = K @ d
        assert np.allclose(uh_numpy, u_mpc)

# -------------------- List timings --------------------------
list_timings(MPI.COMM_WORLD, [TimingType.wall])

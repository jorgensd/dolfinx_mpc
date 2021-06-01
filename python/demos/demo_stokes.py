import dolfinx.cpp
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import gmsh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Length, width and rotation of channel
L = 2
H = 1
theta = np.pi / 5


def create_mesh_gmsh(res=0.1):
    """
    Create a channel of length 2, height one, rotated theta degrees
    around origin, and corresponding facet markers:
    Walls: 1
    Outlet: 2
    Inlet: 3
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Square duct")

        channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [channel], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid volume")

        import numpy as np
        surfaces = gmsh.model.occ.getEntities(dim=1)
        inlet_marker, outlet_marker, wall_marker = 3, 2, 1
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

        gmsh.model.occ.rotate([(2, channel)], 0, 0, 0,
                              0, 0, 1, theta)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inlets, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Fluid inlet")
        gmsh.model.addPhysicalGroup(1, outlets, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Fluid outlet")
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.mesh.generate(2)

    mesh, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model,
                                                    facet_data=True, gdim=2)
    gmsh.finalize()
    return mesh, ft


# ------------------- Mesh and function space creation ------------------------
mesh, mt = create_mesh_gmsh(0.1)

fdim = mesh.topology.dim - 1

# Create the function space
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = dolfinx.FunctionSpace(mesh, TH)
V, V_to_W = W.sub(0).collapse(True)
Q = W.sub(1).collapse()


def inlet_velocity_expression(x):
    return np.stack((np.sin(np.pi * np.sqrt(x[0]**2 + x[1]**2)),
                     5 * x[1] * np.sin(np.pi * np.sqrt(x[0]**2 + x[1]**2))))


# ----------------------Defining boundary conditions----------------------
# Inlet velocity Dirichlet BC
inlet_facets = mt.indices[mt.values == 3]
inlet_velocity = dolfinx.Function(V)
inlet_velocity.interpolate(inlet_velocity_expression)
W0 = W.sub(0)
dofs = dolfinx.fem.locate_dofs_topological((W0, V), 1, inlet_facets)
bc1 = dolfinx.DirichletBC(inlet_velocity, dofs, W0)

# Collect Dirichlet boundary conditions
bcs = [bc1]

# Slip conditions for walls
n = dolfinx_mpc.utils.create_normal_approximation(V, mt.indices[mt.values == 1])
with dolfinx.common.Timer("~Stokes: Create slip constraint"):
    mpc = dolfinx_mpc.MultiPointConstraint(W)
    mpc.create_slip_constraint((mt, 1), n, sub_space=W.sub(0), sub_map=V_to_W, bcs=bcs)
mpc.finalize()


def tangential_proj(u, n):
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def sym_grad(u):
    return ufl.sym(ufl.grad(u))


def T(u, p, mu):
    return 2 * mu * sym_grad(u) - p * ufl.Identity(u.ufl_shape[0])


# --------------------------Variational problem---------------------------
# Traditional terms
mu = 1
f = dolfinx.Constant(mesh, ((0, 0)))
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
a = (2 * mu * ufl.inner(sym_grad(u), sym_grad(v))
     - ufl.inner(p, ufl.div(v))
     - ufl.inner(ufl.div(u), q)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# No prescribed shear stress
n = ufl.FacetNormal(mesh)
g_tau = tangential_proj(dolfinx.Constant(mesh, ((0, 0), (0, 0))) * n, n)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

# Terms due to slip condition
# Explained in for instance: https://arxiv.org/pdf/2001.10639.pdf
a -= ufl.inner(ufl.outer(n, n) * ufl.dot(T(u, p, mu), n), v) * ds
L += ufl.inner(g_tau, v) * ds

# Assemble LHS matrix and RHS vector
with dolfinx.common.Timer("~Stokes: Assemble LHS and RHS"):
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs)
    A.assemble()
    b = dolfinx_mpc.assemble_vector(L, mpc)

# Set Dirichlet boundary condition values in the RHS
dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.assemble.set_bc(b, bcs)

# ---------------------- Solve variational problem -----------------------
ksp = PETSc.KSP().create(mesh.mpi_comm())
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
U = dolfinx.Function(mpc.function_space())
U.vector.setArray(uh.array)
u = U.sub(0).collapse()
p = U.sub(1).collapse()
u.name = "u"
p.name = "p"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/demo_stokes.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_meshtags(mt)
    outfile.write_function(u)
    outfile.write_function(p)

# -------------------- Verification --------------------------------
# Transfer data from the MPC problem to numpy arrays for comparison
with dolfinx.common.Timer("~Stokes: Verification of problem by global matrix reduction"):

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(L)

    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    solver.setOperators(A_org)

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(W, mpc)
    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
    dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

# -------------------- List timings --------------------------
dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

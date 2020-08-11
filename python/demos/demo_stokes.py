import dolfinx.cpp
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import meshio
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from mpi4py import MPI

# Length, width and rotation of channel
L = 2
H = 1
theta = np.pi/5


def create_mesh_gmsh():
    """
    Create a channel of length 2, height one, rotated pi/6 degrees
    around origin, and corresponding facet markers:
    Walls: 1
    Outlet: 2
    Inlet: 3
    """
    lcar = 0.1
    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(0.0, L, 0.0, H, 0.0, lcar=lcar)
    geom.rotate(rect, [0, 0, 0], theta, [0, 0, 1])

    geom.add_physical([rect.line_loop.lines[0], rect.line_loop.lines[2]], 1)
    geom.add_physical([rect.line_loop.lines[1]], 2)
    geom.add_physical([rect.line_loop.lines[3]], 3)
    geom.add_physical([rect.surface], 4)

    # Generate mesh
    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == "triangle"]))

    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                      if cells.type == "line"]))

    facet_data = mesh.cell_data_dict["gmsh:physical"]["line"]
    cell_data = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    triangle_mesh = meshio.Mesh(points=mesh.points,
                                cells=[("triangle", cells)],
                                cell_data={"name_to_read": [cell_data]})

    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": [facet_data]})
    # Write mesh
    meshio.xdmf.write("meshes/mesh.xdmf", triangle_mesh)
    meshio.xdmf.write("meshes/facet_mesh.xdmf", facet_mesh)


# Create mesh
if MPI.COMM_WORLD.size == 1:
    create_mesh_gmsh()
# Load mesh and corresponding facet markers
with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                         "meshes/mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                         "meshes/facet_mesh.xdmf", "r") as xdmf:
    mt = xdmf.read_meshtags(mesh, name="Grid")

outfile = dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, "results/demo_stokes.xdmf", "w")
outfile.write_mesh(mesh)
fdim = mesh.topology.dim - 1

# Create the function space
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = dolfinx.FunctionSpace(mesh, TH)
V, V_to_W = W.sub(0).collapse(True)
Q = W.sub(1).collapse()


def inlet_velocity_expression(x):
    return np.stack((np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2)),
                     5*x[1]*np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2))))


# Inlet velocity Dirichlet BC
inlet_facets = mt.indices[mt.values == 3]
inlet_velocity = dolfinx.Function(V)
inlet_velocity.interpolate(inlet_velocity_expression)
W0 = W.sub(0)

dofs = dolfinx.fem.locate_dofs_topological((W0, V), 1, inlet_facets)
bc1 = dolfinx.DirichletBC(inlet_velocity, dofs, W0)

# Since for this problem the pressure is only determined up to a constant,
# we pin the pressure at the point (0, 0)
zero = dolfinx.Function(Q)
with zero.vector.localForm() as zero_local:
    zero_local.set(0.0)
W1 = W.sub(1)
dofs = dolfinx.fem.locate_dofs_geometrical((W1, Q),
                                           lambda x: np.isclose(x.T, [0, 0, 0])
                                           .all(axis=1))
bc2 = dolfinx.DirichletBC(zero, dofs, W1)

# Collect Dirichlet boundary conditions
bcs = [bc1, bc2]

# Create slip condition
n = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 1)
mpc = dolfinx_mpc.MultiPointConstraint(W)
mpc.create_slip_constraint(W.sub(0), n, np.array(V_to_W), (mt, 1), bcs=bcs)
mpc.finalize()
# Write cell partitioning to file
tdim = mesh.topology.dim
cell_map = mesh.topology.index_map(tdim)
num_cells_local = cell_map.size_local
indices = np.arange(num_cells_local)
values = MPI.COMM_WORLD.rank*np.ones(num_cells_local, np.int32)
ct = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, values)
ct.name = "cells"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "cf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)


# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = dolfinx.Function(V)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(p, ufl.div(v))
     + ufl.inner(ufl.div(u), q)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Assemble LHS matrix and RHS vector
with dolfinx.common.Timer("~Stokes: Assemble LHS and RHS"):
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs)
    A.assemble()
    b = dolfinx_mpc.assemble_vector(L, mpc)

# Set Dirichlet boundary condition values in the RHS
dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.assemble.set_bc(b, bcs)

# Solve problem
ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
uh = b.copy()
ksp.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)
mpc.backsubstitution(uh)

# Write solution to file
U = dolfinx.Function(mpc.function_space())
U.vector.setArray(uh.array)

u = U.sub(0).collapse()
p = U.sub(1).collapse()
u.name = "u"
p.name = "p"
outfile.write_function(u)
# outfile.write_function(p)
outfile.close()

# Transfer data from the MPC problem to numpy arrays for comparison
A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

# Solve the MPC problem using a global transformation matrix
# and numpy solvers to get reference values
# Generate reference matrices and unconstrained solution
with dolfinx.common.Timer("~Stokes: Unconstrained assembly"):
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()

    L_org = dolfinx.fem.assemble_vector(L)

dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
solver.setOperators(A_org)

# Create global transformation matrix
with dolfinx.common.Timer("~Stokes: Solve using global matrix"):
    K = dolfinx_mpc.utils.create_transformation_matrix(W, mpc)
    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)

with dolfinx.common.Timer("~Stokes: Compare global and local solution"):
    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
    dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])
dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

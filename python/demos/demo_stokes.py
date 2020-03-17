import dolfinx.cpp
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import meshio
import numpy as np
import pygmsh
import time
import ufl
from petsc4py import PETSc


# Length, width and rotation of channel
L = 2
H = 1
theta = np.pi/8


def create_mesh_gmsh(dim=2, shape=dolfinx.cpp.mesh.CellType.triangle, order=1):
    """
    Create a channel of length 2, height one, rotated pi/6 degrees
    around origo, and corresponding facet markers:
    Walls: 1
    Outlet: 2
    Inlet: 3
    """
    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(0.0, L, 0.0, H, 0.0, lcar=0.2)
    geom.rotate(rect, [0, 0, 0], theta, [0, 0, 1])

    geom.add_physical([rect.line_loop.lines[0], rect.line_loop.lines[2]], 1)
    geom.add_physical([rect.line_loop.lines[1]], 2)
    geom.add_physical([rect.line_loop.lines[3]], 3)
    geom.add_physical([rect.surface], 4)

    # Generate mesh
    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == "triangle"]))
    triangle_mesh = meshio.Mesh(points=mesh.points,
                                cells=[("triangle", cells)])

    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                      if cells.type == "line"]))
    facet_data = mesh.cell_data_dict["gmsh:physical"]["line"]

    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": [facet_data]})

    # Write mesh
    meshio.xdmf.write("mesh.xdmf", triangle_mesh)
    meshio.xdmf.write("facet_mesh.xdmf", facet_mesh)


# Create cache for numba functions
dolfinx_mpc.utils.cache_numba(matrix=True, vector=True, backsubstitution=True)

# Create mesh
# geom.rotate only in pygmsh master, not stable release yet.
# See: https://github.com/nschloe/pygmsh/commit/d978fa18
if dolfinx.MPI.size(dolfinx.MPI.comm_world) == 1:
    create_mesh_gmsh()


# parallel_read_fix()

# Load mesh and corresponding facet markers
with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
    mesh = xdmf.read_mesh()


def find_line_function(p0, p1):
    """
    Find line y=ax+b for each of the lines in the mesh
    https://mathworld.wolfram.com/Two-PointForm.html
    """
    return lambda x: np.isclose(x[1],
                                p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0]))


points = np.array([[0, 0, 0], [L, 0, 0], [L, H, 0], [0, H, 0]])
r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
rotated_points = np.dot(r_matrix, points.T)
slip_0 = find_line_function(rotated_points[:, 0], rotated_points[:, 1])
outlet = find_line_function(rotated_points[:, 1], rotated_points[:, 2])
slip_1 = find_line_function(rotated_points[:, 2], rotated_points[:, 3])
inlet = find_line_function(rotated_points[:, 3], rotated_points[:, 0])

mf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim-1, 0)
mf.mark(slip_0, 1)
mf.mark(slip_1, 1)
mf.mark(outlet, 2)
mf.mark(inlet, 3)

# Create the function space
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = dolfinx.FunctionSpace(mesh, TH)
V = dolfinx.FunctionSpace(mesh, P2)
Q = dolfinx.FunctionSpace(mesh, P1)


# Inlet velocity Dirichlet BC
def inlet_velocity_expression(x):
    return np.stack((np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2)),
                     5*x[1]*np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2))))


inlet_velocity = dolfinx.Function(V)
inlet_velocity.interpolate(inlet_velocity_expression)
inlet_facets = np.where(mf.values == 3)[0]
dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, inlet_facets)
bc1 = dolfinx.DirichletBC(inlet_velocity, dofs, W.sub(0))

# Since for this problem the pressure is only determined up to a constant,
# we pin the pressure at the point (0, 0)
zero = dolfinx.Function(Q)
with zero.vector.localForm() as zero_local:
    zero_local.set(0.0)
dofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Q),
                                           lambda x: np.isclose(x.T, [0, 0, 0])
                                           .all(axis=1))
bc2 = dolfinx.DirichletBC(zero, dofs, W.sub(1))

# Collect Dirichlet boundary conditions
bcs = [bc1, bc2]


# Find all dofs that corresponding to places where we require MPC constraints.
# x and y components are found separately.

def set_master_slave_slip_relationship(W, V, mf, value, bcs):
    """
    Set a slip condition for all dofs in W (where V is the collapsed
    0th subspace of W) that corresponds to the facets in mf marked with value
    """
    x = W.tabulate_dof_coordinates()
    global_indices = W.dofmap.index_map.global_indices(False)

    wall_facets = np.where(mf.values == value)[0]
    bc_dofs = []
    for bc in bcs:
        bc_g = [global_indices[bdof] for bdof in bc.dof_indices[:, 0]]
        bc_dofs.append(np.hstack(dolfinx.MPI.comm_world.allgather(bc_g)))
    bc_dofs = np.hstack(bc_dofs)

    dofx = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(0),
                                                V.sub(0).collapse()),
                                               1, wall_facets)
    dofy = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(1),
                                                V.sub(1).collapse()),
                                               1, wall_facets)

    slaves = []
    masters = []
    coeffs = []
    offsets = np.linspace(0, len(dofx), len(dofx)+1, dtype=np.int)

    nh = dolfinx_mpc.facet_normal_approximation(V, mf, 1)
    nhx, nhy = nh.sub(0).collapse(), nh.sub(1).collapse()
    nh_out = dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "nh.xdmf")
    nh_out.write(nh)
    nh_out.close()

    # Find index of each pair of x and y components.
    for d_x in dofx:
        for d_y in dofy:
            if np.allclose(x[d_x[0]], x[d_y[0]]):
                slave_dof = np.vstack(dolfinx.MPI.comm_world.allgather(
                    global_indices[d_x[0]]))[0, 0]
                master_dof = np.vstack(dolfinx.MPI.comm_world.allgather(
                    global_indices[d_y[0]]))[0, 0]
                if master_dof not in bc_dofs:
                    slaves.append(slave_dof)
                    masters.append(master_dof)
                    coeffs.append(-nhy.vector[d_y[1]]/nhx.vector[d_x[1]])

    return (np.array(masters), np.array(slaves),
            np.array(coeffs), np.array(offsets))


start = time.time()
(masters, slaves,
 coeffs, offsets) = set_master_slave_slip_relationship(W, V, mf, 1, bcs)
end = time.time()
print("Setup master slave relationship: {0:.2e}".format(end-start))

mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(W._cpp_object, slaves,
                                               masters, coeffs, offsets)
# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = dolfinx.Function(V)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(p, ufl.div(v))
     + ufl.inner(ufl.div(u), q)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Assemble LHS matrix and RHS vector
start = time.time()
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs)
end = time.time()
print("Matrix assembly time: {0:.2e} ".format(end-start))
A.assemble()
start = time.time()
b = dolfinx_mpc.assemble_vector(L, mpc)
end = time.time()
print("Vector assembly time: {0:.2e} ".format(end-start))

dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
dolfinx.fem.assemble.set_bc(b, bcs)

# Create and configure solver
ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# Compute the solution
uh = b.copy()
start = time.time()
ksp.solve(b, uh)
end = time.time()
print("Solve time {0:.2e}".format(end-start))

# Back substitute to slave dofs
dolfinx_mpc.backsubstitution(mpc, uh, W.dofmap)

Wmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, W.element,
                                              mpc.mpc_dofmap())
Wmpc = dolfinx.FunctionSpace(None, W.ufl_element(), Wmpc_cpp)

# Write solution to file
U = dolfinx.Function(Wmpc)
U.vector.setArray(uh.array)

# Split the mixed solution and collapse
u = U.sub(0).collapse()
p = U.sub(1).collapse()

u_out = dolfinx.io.XDMFFile(dolfinx.cpp.MPI.comm_world, "u.xdmf")
u_out.write(u)
u_out.close()
p_out = dolfinx.io.XDMFFile(dolfinx.cpp.MPI.comm_world, "p.xdmf")
p_out.write(p)
p_out.close()


# Transfer data from the MPC problem to numpy arrays for comparison
A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

# Solve the MPC problem using a global transformation matrix
# and numpy solvers to get reference values

# Generate reference matrices and unconstrained solution
print("---Reference timings---")
start = time.time()
A_org = dolfinx.fem.assemble_matrix(a, bcs)
end = time.time()
print("Normal matrix assembly {0:.2e}".format(end-start))
A_org.assemble()

start = time.time()
L_org = dolfinx.fem.assemble_vector(L)
end = time.time()
print("Normal vector assembly {0:.2e}".format(end-start))

dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
solver.setOperators(A_org)

# Create global transformation matrix
K = dolfinx_mpc.utils.create_transformation_matrix(W.dim(), slaves,
                                                   masters, coeffs,
                                                   offsets)
# Create reduced A
A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
start = time.time()
reduced_A = np.matmul(np.matmul(K.T, A_global), K)
end = time.time()
print("Numpy matrix reduction: {0:.2e}".format(end-start))
# Created reduced L
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)
# Solve linear system
start = time.time()
d = np.linalg.solve(reduced_A, reduced_L)
end = time.time()
print("Numpy solve {0:.2e}".format(end-start))
# Back substitution to full solution vector
uh_numpy = np.dot(K, d)

# Compare LHS, RHS and solution with reference values
dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

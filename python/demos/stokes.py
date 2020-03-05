import dolfinx.cpp
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import meshio
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from IPython import embed


def create_mesh_gmsh(dim=2, shape=dolfinx.cpp.mesh.CellType.triangle, order=1):
    """Compute cell topology and geometric points for a range of cells types
    and geometric orders
    """
    import pygmsh
    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(0.0, 2.0, 0.0, 1.0, 0.0, lcar=0.1)
    geom.rotate(rect, [0,0,0], np.pi/4, [0,0,1])

    geom.add_physical([rect.line_loop.lines[0], rect.line_loop.lines[2]], 1)  # Walls
    geom.add_physical([rect.line_loop.lines[1]], 2)  # Outlet
    geom.add_physical([rect.line_loop.lines[3]], 3)  # Inlet
    geom.add_physical([rect.surface], 4)  # Fluid domain

    # Generate mesh
    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
    cells = np.vstack(np.array([cells.data for cells in mesh.cells if cells.type == "triangle"]))
    triangle_mesh = meshio.Mesh(points=mesh.points,
                            cells=[("triangle", cells)])

    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells if cells.type == "line"]))
    facet_data = np.array([mesh.cell_data_dict["gmsh:physical"][key]
                           for key in mesh.cell_data_dict["gmsh:physical"].keys() if key=="line"])[0]

    facet_mesh = meshio.Mesh(points=mesh.points,
                               cells=[("line", facet_cells)],
                            cell_data={"name_to_read":[facet_data]})
    # Write mesh
    meshio.xdmf.write("mesh.xdmf", triangle_mesh, data_format="XML")
    meshio.xdmf.write("facet_mesh.xdmf", facet_mesh, data_format="XML")

create_mesh_gmsh()


# Lid velocity
def inlet_velocity_expression(x):
    return np.stack((np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2)), x[1]*np.sin(np.pi*np.sqrt(x[0]**2+x[1]**2))))


with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
    mesh = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none)

with dolfinx.io.XDMFFile(mesh.mpi_comm(), "facet_mesh.xdmf") as xdmf:
    mvc = xdmf.read_mvc_size_t(mesh, "name_to_read")
mf = dolfinx.MeshFunction("size_t", mesh, mvc, 0)


cmap = dolfinx.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap


# Create the function space
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = dolfinx.FunctionSpace(mesh, TH)
V = dolfinx.FunctionSpace(mesh, P2)
Q = dolfinx.FunctionSpace(mesh, P1)

# No slip boundary condition
noslip = dolfinx.Function(V)
wall_facets = np.where(mf.values == 1)[0]
dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, wall_facets)
bc0 = dolfinx.DirichletBC(noslip, dofs, W.sub(0))


x = W.tabulate_dof_coordinates()
dofx = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(0), V.sub(0).collapse()), 1, wall_facets)[:,0]
dofy = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(1), V.sub(1).collapse()), 1, wall_facets)[:,0]

slaves = []
masters = []
offsets = np.linspace(0, len(dofx), len(dofx)+1, dtype=np.int)
coeffs = np.ones(len(dofx))
global_indices = W.dofmap.index_map.global_indices(False)

for d_x in dofx:
    for d_y in dofy:
        if np.allclose(x[d_x], x[d_y]):
            slaves.append(np.vstack(dolfinx.MPI.comm_world.allgather(global_indices[d_x]))[0,0])
            masters.append(np.vstack(dolfinx.MPI.comm_world.allgather(global_indices[d_y]))[0,0])


masters = np.array(masters)
slaves = np.array(slaves)
print(len(masters))
print(len(slaves))

mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(W._cpp_object, slaves,
                                               masters, coeffs, offsets)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
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
                                       lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
bc2 = dolfinx.DirichletBC(zero, dofs, W.sub(1))

# Collect Dirichlet boundary conditions
# bcs = [bc0, bc1, bc2]
bcs = [bc1, bc2]

# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = dolfinx.Function(V)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(p, ufl.div(v)) + ufl.inner(ufl.div(u), q)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Assemble LHS matrix and RHS vector
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs)
A.assemble()
b = dolfinx_mpc.assemble_vector(L, mpc)

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
ksp.solve(b, uh)

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

u_out = dolfinx.io.XDMFFile(dolfinx.cpp.MPI.comm_world,"u.xdmf")
u_out.write_checkpoint(u, "u",  0.0)
u_out.close()
p_out = dolfinx.io.XDMFFile(dolfinx.cpp.MPI.comm_world,"p.xdmf")
p_out.write_checkpoint(p, "p",  0.0)
p_out.close()


# Transfer data from the MPC problem to numpy arrays for comparison
A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

# Solve the MPC problem using a global transformation matrix
# and numpy solvers to get reference values

# Generate reference matrices and unconstrained solution
A_org = dolfinx.fem.assemble_matrix(a, bcs)

A_org.assemble()
L_org = dolfinx.fem.assemble_vector(L)
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
reduced_A = np.matmul(np.matmul(K.T, A_global), K)
# Created reduced L
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)
# Solve linear system
d = np.linalg.solve(reduced_A, reduced_L)
# Back substitution to full solution vector
uh_numpy = np.dot(K, d)

# Compare LHS, RHS and solution with reference values
from IPython import embed;embed()
dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)

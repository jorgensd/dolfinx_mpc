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


def create_mesh_gmsh(dim=2, shape=dolfinx.cpp.mesh.CellType.triangle, order=1):
    """
    Create a channel of length 2, height one, rotated pi/6 degrees
    around origo, and corresponding facet markers:
    Walls: 1
    Outlet: 2
    Inlet: 3
    """
    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(0.0, 2.0, 0.0, 1.0, 0.0, lcar=0.2)
    geom.rotate(rect, [0, 0, 0], np.pi/6, [0, 0, 1])

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
    facet_data = np.array([mesh.cell_data_dict["gmsh:physical"][k]
                           for k in mesh.cell_data_dict["gmsh:physical"].keys()
                           if k == "line"])[0]

    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": [facet_data]})

    # Write mesh
    meshio.xdmf.write("mesh.xdmf", triangle_mesh)
    meshio.xdmf.write("facet_mesh.xdmf", facet_mesh)


def parallel_read_fix():
    # Fix due to: https://github.com/FEniCS/dolfinx/issues/818
    if dolfinx.MPI.size(dolfinx.MPI.comm_world) == 1:
        with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
            mesh = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none)

        with dolfinx.io.XDMFFile(mesh.mpi_comm(), "facet_mesh.xdmf") as xdmf:
            mvc = xdmf.read_mvc_size_t(mesh, "name_to_read")
        mvc.name = "name_to_read"
        mf = dolfinx.MeshFunction("size_t", mesh, mvc, 0)
        mf.name = "facets"
        with dolfinx.io.XDMFFile(mesh.mpi_comm(), "dolfin_mvc.xdmf") as xdmf:
            xdmf.write(mesh)
            xdmf.write(mf)


# Create cache for numba functions
dolfinx_mpc.utils.cache_numba(matrix=True, vector=True, backsubstitution=True)

# Create mesh
# geom.rotate only in pygmsh master, not stable release yet.
# See: https://github.com/nschloe/pygmsh/commit/d978fa18
# if dolfinx.MPI.size(dolfinx.MPI.comm_world) == 1:
#     create_mesh_gmsh()

# parallel_read_fix()

# Load mesh and corresponding facet markers
with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
    mesh = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none)

with dolfinx.io.XDMFFile(mesh.mpi_comm(), "dolfin_mvc.xdmf") as xdmf:
    read_function = getattr(xdmf, "read_mf_size_t")
    mf = read_function(mesh, "facets")

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
def set_master_slave_slip_relationship(W, V, mf, value):
    """
    Set a slip condition for all dofs in W (where V is the collapsed
    0th subspace of W) that corresponds to the facets in mf marked with value
    """
    x = W.tabulate_dof_coordinates()
    wall_facets = np.where(mf.values == value)[0]

    # Find all dofs of each sub space that are on the facets
    dofx = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(0),
                                                V.sub(0).collapse()),
                                               1, wall_facets)[:, 0]

    dofy = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(1),
                                                V.sub(1).collapse()),
                                               1, wall_facets)[:, 0]

    slaves = []
    masters = []
    global_indices = W.dofmap.index_map.global_indices(False)

    x_g = np.zeros((W.dim(), 3))
    bs = W.dofmap.index_map.block_size
    l_r = W.dofmap.index_map.local_range
    l_s = bs * (l_r[1]-l_r[0])

    # Map dof coordinates from local to global array
    for i in range(l_s):
        x_g[global_indices[i], :] = x[i, :]

    x_g = sum(dolfinx.MPI.comm_world.allgather(x_g))
    dofx_global = [global_indices[d_x] for d_x in dofx]
    dofy_global = [global_indices[d_y] for d_y in dofy]
    dofx_global = list(set(np.hstack(
        dolfinx.MPI.comm_world.allgather(dofx_global))))
    dofy_global = list(set(np.hstack(
        dolfinx.MPI.comm_world.allgather(dofy_global))))

    bc_dofs = []
    for bc in bcs:
        bc_g = [global_indices[bdof] for bdof in bc.dof_indices[:, 0]]
        bc_dofs.append(np.hstack(dolfinx.MPI.comm_world.allgather(bc_g)))
    bc_dofs = np.hstack(bc_dofs)

    # Find index of each pair of x and y components.
    for i, slave_dof in enumerate(dofx_global):
        for j, master_dof in enumerate(dofy_global):
            if np.allclose(x_g[master_dof], x_g[slave_dof]):
                # Check for duplicates with Dirichlet BCS and remove them
                # FIXME: https://gitlab.asimov.cfms.org.uk/wp2/mpc/issues/1
                if master_dof not in bc_dofs:
                    slaves.append(slave_dof)
                    masters.append(master_dof)
    return np.array(masters), np.array(slaves)


start = time.time()
masters, slaves = set_master_slave_slip_relationship(W, V, mf, 1)
end = time.time()
print("Setup master slave relationship: {0:.2e}".format(end-start))

offsets = np.linspace(0, len(slaves), len(slaves)+1, dtype=np.int)
coeffs = np.sqrt(3)*np.ones(len(slaves))

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
u_out.write_checkpoint(u, "u",  0.0)
u_out.close()
p_out = dolfinx.io.XDMFFile(dolfinx.cpp.MPI.comm_world, "p.xdmf")
p_out.write_checkpoint(p, "p",  0.0)
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

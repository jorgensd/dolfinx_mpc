import dolfinx
import dolfinx.geometry as geometry
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from create_and_export_mesh import mesh_2D_rot

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def find_line_function(p0, p1):
    """
    Find line y=ax+b for each of the lines in the mesh
    https://mathworld.wolfram.com/Two-PointForm.html
    """
    return lambda x: np.isclose(x[1],
                                p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0]))


def over_line(p0, p1):
    """
    Check if a point is over or under y=ax+b for each of the lines in the mesh
    https://mathworld.wolfram.com/Two-PointForm.html
    """
    return lambda x: x[1] > p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0])


def demo_stacked_cubes(theta):
    if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
        mesh_2D_rot(theta)

    r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [1, 1, 0], [0, 1, 0]]).T)
    bottom = find_line_function(bottom_points[:, 0], bottom_points[:, 1])
    interface = find_line_function(bottom_points[:, 2], bottom_points[:, 3])
    top_points = np.dot(r_matrix, np.array([[0, 1, 0], [1, 1, 0],
                                            [1, 2, 0], [0, 2, 0]]).T)
    top = find_line_function(top_points[:, 2], top_points[:, 3])
    top_cube = over_line(bottom_points[:, 2], bottom_points[:, 3])

    left_side = find_line_function(top_points[:, 0], top_points[:, 3])

    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                             "meshes/mesh_rot.xdmf") as xdmf:
        mesh = xdmf.read_mesh()

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Move top in y-dir
    V0 = V.sub(0).collapse()
    V1 = V.sub(1).collapse()
    # Traction on top of domain
    fdim = mesh.topology.dim - 1
    markers = {top: 3, interface: 4, bottom: 5, left_side: 6}
    mf = dolfinx.MeshFunction("size_t", mesh, fdim, 0)
    for key in markers.keys():
        mf.mark(key, markers[key])

    # dolfinx.io.VTKFile("mf.pvd").write(mf)
    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = dolfinx.Constant(mesh, g_vec[:2])

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bottom_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                           bottom)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    # def top_v(x):
    #     values = np.empty((2, x.shape[1]))
    #     values[0] = g_vec[0]
    #     values[1] = g_vec[1]
    #     return values
    # u_top = dolfinx.function.Function(V)
    # u_top.interpolate(top_v)
    # top_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
    #                                                     top)
    # top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    # bc_top = fem.DirichletBC(u_top, top_dofs)
    bcs = [bc_bottom]  # , bc_top]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e3
    nu = 0.1
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create MPC
    slaves = []
    masters = []
    coeffs = []
    offsets = [0]

    # Locate dofs on both interfaces
    # FIXME: For some strange reason this doesnt work for the interface
    i_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                      interface)

    # Slicing of list is due to the fact that we only require y-components
    interface_x_dofs = fem.locate_dofs_topological((V.sub(0), V0),
                                                   fdim, i_facets)[:, 0]
    interface_y_dofs = fem.locate_dofs_topological((V.sub(1), V1),
                                                   fdim,
                                                   i_facets)[:, 0]

    # Get some cell info
    global_indices = V.dofmap.index_map.global_indices(False)
    num_cells = mesh.num_entities(mesh.topology.dim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim,
                                                range(num_cells))
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    x_coords = V.tabulate_dof_coordinates()
    tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    mesh.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    mesh.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    cell_to_facet = mesh.topology.connectivity(mesh.topology.dim,
                                               mesh.topology.dim-1)
    nh = dolfinx_mpc.facet_normal_approximation(V, mf, markers[interface])
    n_vec = nh.vector.getArray()
    # dolfinx.io.VTKFile("results/nh.pvd").write(nh)

    # Set normal sliding of a single on the left side of the cube to zero to
    # avoid translational invariant problem
    left_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                         left_side)
    left_dofs_x = fem.locate_dofs_topological(
        (V.sub(0), V0), fdim, left_facets)[:, 0]
    left_dofs_y = fem.locate_dofs_topological(
        (V.sub(1), V1), fdim, left_facets)[:, 0]

    top_cube_left_x = np.flatnonzero(top_cube(x_coords[left_dofs_x].T))
    top_cube_left_y = np.flatnonzero(top_cube(x_coords[left_dofs_y].T))
    slip = False
    n_left = dolfinx_mpc.facet_normal_approximation(V, mf, markers[left_side])
    n_left_vec = n_left.vector.getArray()
    # dolfinx.io.VTKFile("results/nleft.pvd").write(n_left)

    for id_x in top_cube_left_x:
        x_dof = left_dofs_x[id_x]
        corner_1 = np.allclose(x_coords[x_dof], top_points[:, 0])
        corner_2 = np.allclose(x_coords[x_dof], top_points[:, 3])
        if corner_1 or corner_2:
            continue
        for id_y in top_cube_left_y:
            y_dof = left_dofs_y[id_y]
            same_coord = np.allclose(x_coords[x_dof], x_coords[y_dof])
            corner_1 = np.allclose(x_coords[y_dof], top_points[:, 0])
            corner_2 = np.allclose(x_coords[y_dof], top_points[:, 3])
            if not corner_1 and not corner_2 and same_coord:
                slaves.append(global_indices[x_dof])
                masters.append(global_indices[y_dof])
                coeffs.append(-n_left_vec[y_dof] /
                              n_left_vec[x_dof])
                offsets.append(len(masters))
                slip = True
                break
        if slip:
            break
    for cell_index in range(num_cells):
        midpoint = cell_midpoints[cell_index]
        cell_dofs = V.dofmap.cell_dofs(cell_index)

        # Check if any dofs in cell is on interface
        has_dofs_on_interface = np.isin(cell_dofs, interface_y_dofs)
        x_on_interface = np.isin(cell_dofs, interface_x_dofs)
        local_x_indices = np.flatnonzero(x_on_interface)

        if np.any(has_dofs_on_interface):
            local_indices = np.flatnonzero(has_dofs_on_interface)
            for dof_index in local_indices:
                slave_l = cell_dofs[dof_index]
                slave_coords = x_coords[slave_l]
                is_slave_cell = not top_cube(midpoint)
                global_slave = global_indices[slave_l]
                if is_slave_cell and global_slave not in slaves:
                    slaves.append(global_slave)
                    # Find corresponding x-coordinate of this slave
                    master_x = -1
                    for x_index in local_x_indices:
                        if np.allclose(x_coords[cell_dofs[x_index]],
                                       x_coords[slave_l]):
                            master_x = cell_dofs[x_index]
                            break
                    assert master_x != -1
                    global_first_master = global_indices[master_x]
                    global_first_coeff = -\
                        n_vec[master_x]/n_vec[slave_l]
                    masters.append(global_first_master)
                    coeffs.append(global_first_coeff)
                    # Need some parallel comm here
                    # Now find which master cell is close to the slave dof
                    possible_cells = comp_col_pts(tree, slave_coords)
                    for other_index in possible_cells[0]:
                        facets = cell_to_facet.links(other_index)
                        facet_in_cell = np.any(np.isin(facets,
                                                       i_facets))
                        other_dofs = V.dofmap.cell_dofs(other_index)
                        dofs_on_interface = np.isin(other_dofs,
                                                    interface_y_dofs)
                        other_midpoint = cell_midpoints[other_index]
                        is_master_cell = ((other_index != cell_index) and
                                          np.any(dofs_on_interface) and
                                          top_cube(other_midpoint) and
                                          facet_in_cell)

                        if is_master_cell:
                            # Get basis values, and add coeffs of sub space 1
                            # to masters with corresponding coeff
                            basis_values = get_basis(V._cpp_object,
                                                     slave_coords, other_index)
                            flatten_values = np.sum(basis_values, axis=1)
                            for local_idx, dof in enumerate(other_dofs):
                                if not np.isclose(flatten_values[local_idx],
                                                  0):

                                    masters.append(global_indices[dof])
                                    local_coeff = flatten_values[local_idx]
                                    # Check if x coordinate scale by normal y
                                    if dof in interface_x_dofs:
                                        local_coeff *= (n_vec[master_x] /
                                                        n_vec[slave_l])

                                    coeffs.append(local_coeff)
                            offsets.append(len(masters))
                            break

    slaves, masters, coeffs, offsets = (np.array(slaves), np.array(masters),
                                        np.array(coeffs), np.array(offsets))

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc"
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                        "uh_rot.xdmf").write(u_h)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = fem.assemble_vector(lhs)
    fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L_org, bcs)

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
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
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                          uh.owner_range[1]])
    print(uh.norm())


if __name__ == "__main__":
    demo_stacked_cubes(theta=np.pi/7)

import dolfinx.geometry as geometry
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from create_and_export_mesh import mesh_3D_rot

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def find_plane_function(p0, p1, p2):
    """
    Find plane function given three points:
    http://www.nabla.hr/CG-LinesPlanesIn3DA3.htm
    """
    v1 = np.array(p1)-np.array(p0)
    v2 = np.array(p2)-np.array(p0)

    n = np.cross(v1, v2)
    D = -(n[0]*p0[0]+n[1]*p0[1]+n[2]*p0[2])
    return lambda x: np.isclose(0,
                                np.dot(n, x) + D)


def over_plane(p0, p1, p2):
    v1 = np.array(p1)-np.array(p0)
    v2 = np.array(p2)-np.array(p0)

    n = np.cross(v1, v2)
    D = -(n[0]*p0[0]+n[1]*p0[1]+n[2]*p0[2])
    return lambda x: n[0]*x[0] + n[1]*x[1] + D > -n[2]*x[2]


def demo_stacked_cubes(theta):
    if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
        mesh_3D_rot(theta)

    r_matrix = pygmsh.helpers.rotation_matrix(
        [0, 1/np.sqrt(2), 1/np.sqrt(2)], -theta)
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [1, 1, 0], [0, 1, 0]]).T)
    interface_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1],
                                                  [1, 1, 1], [0, 1, 1]]).T)
    top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2],
                                            [1, 1, 2], [0, 1, 2]]).T)
    left_points = np.dot(r_matrix, np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]]).T)

    bottom = find_plane_function(
        bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2])
    interface = find_plane_function(
        interface_points[:, 0], interface_points[:, 1], interface_points[:, 2])
    top = find_plane_function(
        top_points[:, 0], top_points[:, 1], top_points[:, 2])
    left_side = find_plane_function(
        left_points[:, 0], left_points[:, 1], left_points[:, 2])
    top_cube = over_plane(
        interface_points[:, 0], interface_points[:, 1], interface_points[:, 2])

    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                             "meshes/mesh3D_rot.xdmf") as xdmf:
        mesh = xdmf.read_mesh()
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Move top in y-dir
    V0 = V.sub(0).collapse()
    V1 = V.sub(1).collapse()
    V2 = V.sub(2).collapse()

    # Create a cell function with markers = 3 for top cube
    cf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
    num_cells = mesh.num_entities(mesh.topology.dim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim,
                                                range(num_cells))
    top_cube_marker = 3
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            cf.values[cell_index] = top_cube_marker

    # Traction on top of domain
    tdim = mesh.topology.dim

    fdim = tdim - 1
    markers = {top: 3, interface: 4, bottom: 5, left_side: 6}
    mf = dolfinx.MeshFunction("size_t", mesh, fdim, 0)
    for key in markers.keys():
        mf.mark(key, markers[key])

    # dolfinx.io.VTKFile("mf.pvd").write(mf)
    g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
    g = dolfinx.Constant(mesh, g_vec)

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bottom_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                           bottom)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = dolfinx.function.Function(V)
    u_top.interpolate(top_v)
    top_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                        top)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.DirichletBC(u_top, top_dofs)
    bcs = [bc_bottom, bc_top]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e3
    nu = 0
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
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mf,
                     subdomain_id=markers[top])
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create MPC
    slaves = []
    masters = []
    coeffs = []
    offsets = [0]

    # Locate dofs on both interfaces
    i_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                      interface)

    # Slicing of list is due to the fact that we only require y-components
    interface_x_dofs = fem.locate_dofs_topological((V.sub(0), V0),
                                                   fdim, i_facets)[:, 0]
    interface_y_dofs = fem.locate_dofs_topological((V.sub(1), V1),
                                                   fdim,
                                                   i_facets)[:, 0]
    interface_z_dofs = fem.locate_dofs_topological((V.sub(2), V2),
                                                   fdim,
                                                   i_facets)[:, 0]
    # Get some cell info
    global_indices = V.dofmap.index_map.global_indices(False)
    x_coords = V.tabulate_dof_coordinates()

    mesh.create_connectivity(fdim, tdim)
    facet_tree = geometry.BoundingBoxTree(mesh, fdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mf, markers[interface])
    n_vec = nh.vector.getArray()

    # Extract dofmap info to avoid calling non-numpy functions
    dofs = V.dofmap.list.array()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    for i_facet in i_facets:
        cell_indices = facet_to_cell.links(i_facet)
        for cell_index in cell_indices:
            # Local dof position
            cell_dofs = dofs[num_dofs_per_element * cell_index:
                             num_dofs_per_element * cell_index
                             + num_dofs_per_element]
            # Check if any dofs in cell is on interface
            has_dofs_on_interface = np.isin(cell_dofs, interface_y_dofs)
            if np.any(has_dofs_on_interface):
                x_on_interface = np.isin(cell_dofs, interface_x_dofs)
                local_x_indices = np.flatnonzero(x_on_interface)

                z_on_interface = np.isin(cell_dofs, interface_z_dofs)
                local_z_indices = np.flatnonzero(z_on_interface)

                local_indices = np.flatnonzero(has_dofs_on_interface)
                for dof_index in local_indices:
                    slave_l = cell_dofs[dof_index]
                    slave_coords = x_coords[slave_l]
                    is_slave_cell = not (cf.values[cell_index] ==
                                         top_cube_marker)
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
                        # Find corresponding z-coordinate of this slave
                        master_z = -1
                        for z_index in local_z_indices:
                            if np.allclose(x_coords[cell_dofs[z_index]],
                                           x_coords[slave_l]):
                                master_z = cell_dofs[z_index]
                                break
                        assert master_z != -1
                        global_second_master = global_indices[master_z]
                        global_second_coeff = -\
                            n_vec[master_z]/n_vec[slave_l]
                        masters.append(global_second_master)
                        coeffs.append(global_second_coeff)
                        # Need some parallel comm here
                        # Need to add check that point is on a facet in cell.
                        possible_facets = comp_col_pts(
                            facet_tree, slave_coords)[0]
                        for facet in possible_facets:
                            facet_on_interface = facet in i_facets
                            c_cells = facet_to_cell.links(facet)

                            interface_facet = len(c_cells) == 1
                            other_index = c_cells[0]
                            in_top_cube = (cf.values[other_index] ==
                                           top_cube_marker)

                            is_master_cell = (facet_on_interface and
                                              interface_facet and
                                              in_top_cube)

                            if is_master_cell:
                                other_dofs = dofs[num_dofs_per_element
                                                  * other_index:
                                                  num_dofs_per_element
                                                  * other_index
                                                  + num_dofs_per_element]

                                # Get basis values, and add coeffs of sub space
                                # to masters with corresponding coeff
                                basis_values = get_basis(V._cpp_object,
                                                         slave_coords,
                                                         other_index)
                                for local_idx, dof in enumerate(other_dofs):
                                    l_coeff = 0
                                    if dof in interface_x_dofs:
                                        l_coeff = basis_values[local_idx,
                                                               0]
                                        l_coeff *= (n_vec[master_x] /
                                                    n_vec[slave_l])
                                    elif dof in interface_y_dofs:
                                        l_coeff = basis_values[local_idx,
                                                               1]
                                    elif dof in interface_z_dofs:
                                        l_coeff = basis_values[local_idx,
                                                               2]
                                        l_coeff *= (n_vec[master_z] /
                                                    n_vec[slave_l])
                                    if not np.isclose(l_coeff, 0):
                                        masters.append(global_indices[dof])
                                        coeffs.append(l_coeff)
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
                        "results/uh3D_rot.xdmf").write(u_h)
    print("FIN")
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
    demo_stacked_cubes(theta=np.pi/3)

import dolfinx
import dolfinx.geometry as geometry
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from petsc4py import PETSc

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def demo_stacked_cubes(celltype="quad"):
    # with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
    #     mesh = xdmf.read_mesh()

    # Using built in meshes, stacking cubes on top of each other
    if celltype == "quad":
        N = 2
        ct = dolfinx.cpp.mesh.CellType.quadrilateral
        mesh0 = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, N, N, ct)
        mesh1 = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 2*N, 2*N, ct)
        nv = 4
    else:
        N = 2
        ct = dolfinx.cpp.mesh.CellType.triangle
        mesh0 = dolfinx.RectangleMesh(dolfinx.MPI.comm_world,
                                      [np.array([0.0, 0.0, 0]),
                                       np.array([1.0, 1.0, 0])], [N, N],
                                      ct, diagonal="crossed")

        mesh1 = dolfinx.RectangleMesh(dolfinx.MPI.comm_world,
                                      [np.array([0.0, 0.0, 0]),
                                       np.array([1.0, 1.0, 0])], [2*N, 2*N],
                                      ct, diagonal="crossed")
        nv = 3

    x0 = mesh0.geometry.x
    x0 = x0[:, :-1]
    x0[:, 1] += 1
    points = np.vstack([x0, mesh1.geometry.x[:, :-1]])

    # Transform topology info into geometry info
    c2v = mesh0.topology.connectivity(mesh0.topology.dim, 0)
    x_dofmap = mesh0.geometry.dofmap()
    imap = mesh0.topology.index_map(0)
    num_mesh_vertices = imap.size_local + imap.num_ghosts
    vertex_to_node = np.zeros(num_mesh_vertices, dtype=np.int64)
    for c in range(c2v.num_nodes):
        vertices = c2v.links(c)
        x_dofs = x_dofmap.links(c)
        for i in range(vertices.shape[0]):
            vertex_to_node[vertices[i]] = x_dofs[i]
    cells0 = np.zeros((c2v.num_nodes, nv), dtype=np.int64)
    for cell in range(c2v.num_nodes):
        for v in range(nv):
            cells0[cell, v] = vertex_to_node[c2v.links(cell)[v]]
    # Transform topology info into geometry info
    c2v = mesh1.topology.connectivity(mesh1.topology.dim, 0)
    x_dofmap = mesh1.geometry.dofmap()
    imap = mesh1.topology.index_map(0)
    num_mesh_vertices = imap.size_local + imap.num_ghosts
    vertex_to_node = np.zeros(num_mesh_vertices, dtype=np.int64)
    for c in range(c2v.num_nodes):
        vertices = c2v.links(c)
        x_dofs = x_dofmap.links(c)
        for i in range(vertices.shape[0]):
            vertex_to_node[vertices[i]] = x_dofs[i]
    cells1 = np.zeros((c2v.num_nodes, nv), dtype=np.int64)
    for cell in range(c2v.num_nodes):
        for v in range(nv):
            cells1[cell, v] = vertex_to_node[c2v.links(cell)[v]] + x0.shape[0]
    cells = np.vstack([cells0, cells1])
    mesh = dolfinx.Mesh(dolfinx.MPI.comm_world,
                        ct, points, cells, [],
                        dolfinx.cpp.mesh.GhostMode.none)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    if celltype == "unstr":
        with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
            mesh = xdmf.read_mesh()

    mesh.geometry.coord_mapping = cmap
    dolfinx.io.VTKFile("mesh_{0:s}.pvd".format(celltype)).write(mesh)

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Move top in y-dir
    V0 = V.sub(0).collapse()
    V1 = V.sub(1).collapse()

    # Traction on top of domain
    def top(x):
        return np.isclose(x[1], 2)
    fdim = mesh.topology.dim - 1
    mf = dolfinx.MeshFunction("size_t", mesh, fdim, 0)
    mf.mark(top, 3)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
    g = dolfinx.Constant(mesh, (0, -9.81e2))

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    # Fix bottom in all directions
    def bottom(x):
        return np.isclose(x[1], 0)

    bottom_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                           bottom)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    # Fix top corner in x-direction
    zero = dolfinx.Function(V0)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)

    def in_corner(x):
        return np.logical_or(np.isclose(x.T, [0, 2, 0]).all(axis=1),
                             np.isclose(x.T, [1, 2, 0]).all(axis=1))
    dofs = fem.locate_dofs_geometrical((V.sub(0), V0), in_corner)
    bc_corner = dolfinx.DirichletBC(zero, dofs, V.sub(0))
    bcs = [bc_bottom, bc_corner]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
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
    # FIXME: Add facet integrals
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create MPC
    slaves = []
    masters = []
    coeffs = []
    offsets = [0]

    # Locate dofs on both interfaces
    def interface_locater(x):
        return np.isclose(x[1], 1)
    i_facets = dmesh.compute_marked_boundary_entities(mesh, fdim,
                                                      interface_locater)
    # Slicing of list is due to the fact that we only require y-components
    # interface_x_dofs = fem.locate_dofs_topological((V.sub(0), V0),
    #                    fdim, i_facets)[:, 0]
    interface_y_dofs = fem.locate_dofs_topological((V.sub(1), V1),
                                                   fdim,
                                                   i_facets)[:, 0]

    # Get some cell info
    # (range should be reduced with mesh function and markers)
    # cmap = fem.create_coordinate_map(mesh.ufl_domain())
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
    for cell_index in range(num_cells):
        midpoint = cell_midpoints[cell_index]
        cell_dofs = V.dofmap.cell_dofs(cell_index)

        # Fix single central node in x-direction to avoid sliding
        # if cell_midpoints[cell_index][1] < 1:
        #     has_x_dofs_on_interface = np.isin(cell_dofs, interface_x_dofs)
        #     x_dofs_local = np.flatnonzero(has_x_dofs_on_interface)
        #     for dof_index in x_dofs_local:
        #         global_slave = global_indices[cell_dofs[dof_index]]
        #         new_slave = global_slave not in slaves
        #         if np.allclose(x_coords[cell_dofs[dof_index]],
        #                        [0.5, 1, 0]) and new_slave:
        #             slaves.append(global_slave)
        #             # Find master_cell
        #             possible_cells = comp_col_pts(tree,
        #                                           x_coords[cell_dofs[
        #                                               dof_index]])
        #             for other_index in possible_cells[0]:
        #                 facets = cell_to_facet.links(other_index)
        #                 facet_in_cell = np.any(np.isin(facets,
        #                                                i_facets))
        #                 other_dofs = V.dofmap.cell_dofs(other_index)
        #                 dofs_on_interface = np.isin(other_dofs,
        #                                             interface_x_dofs)
        #                 other_midpoint = cell_midpoints[other_index]
        #                 is_master_cell = ((other_index != cell_index) and
        #                                   np.any(dofs_on_interface) and
        #                                   (other_midpoint[1] > 1) and
        #                                   facet_in_cell)
        #                 if is_master_cell:
        #                     x_dofs = V.sub(0).dofmap.cell_dofs(other_index)
        #                     other_local_indices = np.flatnonzero(
        #                         np.isin(other_dofs, x_dofs))
        #                     for local_idx in other_local_indices:
        #                         if np.allclose(x_coords[x_dofs[local_idx]],
        #                                        [0.5, 1, 0]):
        #                             masters.append(
        #                                 global_indices[other_dofs[local_idx]])
        #                             coeffs.append(1)
        #                     offsets.append(len(masters))
        #                     break

        # Check if any dofs in cell is on interface
        has_dofs_on_interface = np.isin(cell_dofs, interface_y_dofs)
        if np.any(has_dofs_on_interface):
            # Check if it is a slave node (x[1]>1)
            local_indices = np.flatnonzero(has_dofs_on_interface)
            for dof_index in local_indices:
                slave_coords = x_coords[cell_dofs[dof_index]]
                # NOTE, we should rather use mesh markers here to determine if
                # the dof is a slave dof, or use something similar to what is
                # used for masters
                is_slave_cell = midpoint[1] < 1
                global_slave = global_indices[cell_dofs[dof_index]]
                if is_slave_cell and global_slave not in slaves:
                    slaves.append(global_slave)
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
                                          (other_midpoint[1] > 1) and
                                          facet_in_cell)

                        if is_master_cell:
                            # Get basis values, and add coeffs of sub space 1
                            # to masters with corresponding coeff
                            basis_values = get_basis(V._cpp_object,
                                                     slave_coords, other_index)
                            # flatten_values = np.sum(basis_values, axis=1)
                            y_values = basis_values[:, 1]
                            y_dofs = V.sub(1).dofmap.cell_dofs(other_index)
                            other_local_indices = np.flatnonzero(
                                np.isin(other_dofs, y_dofs))
                            for local_idx in other_local_indices:
                                if not np.isclose(y_values[local_idx], 0):
                                    masters.append(
                                        global_indices[other_dofs[local_idx]])
                                    coeffs.append(y_values[local_idx])
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
                        "uh_{0:s}.xdmf".format(celltype)).write(u_h)

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
    # solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    # solver.setType(PETSc.KSP.Type.PREONLY)
    # solver.getPC().setType(PETSc.PC.Type.LU)
    # solver.setOperators(A_org)
    # u_ = dolfinx.Function(V)
    # solver.solve(L_org, u_.vector)
    # u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
    #                       mode=PETSc.ScatterMode.FORWARD)
    # u_.name = "u_unperturbed"
    # dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "u_.xdmf").write(u_)

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
    demo_stacked_cubes("quad")
    demo_stacked_cubes("triangle")
    demo_stacked_cubes("unstr")

import dolfinx
import dolfinx.geometry
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from petsc4py import PETSc

comp_col_pts = dolfinx.geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def demo_stacked_cubes():
    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh3D.xdmf") as xdmf:
        mesh = xdmf.read_mesh()
    mesh.create_connectivity_all()
    # To inspect partitioning
    # dolfinx.io.VTKFile("mesh.pvd").write(mesh)

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    # Move top in y-dir
    V0 = V.sub(0).collapse()
    V2 = V.sub(2).collapse()
    uz_disp = dolfinx.function.Function(V2)

    # u_disp = dolfinx.function.Function(V)
    with uz_disp.vector.localForm() as u_local:
        u_local.set(-0.04)

    def top(x):
        return np.isclose(x[2], 2)

    # FIXME Add surface integrals
    # mf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim-1, 0)
    bdofsV2 = dolfinx.fem.locate_dofs_geometrical((V.sub(2), V2), top)
    bc_top = dolfinx.fem.dirichletbc.DirichletBC(uz_disp, bdofsV2, V.sub(2))

    ux_disp = dolfinx.function.Function(V0)
    with ux_disp.vector.localForm() as u_local:
        u_local.set(0)

    def top_corner(x):
        return np.logical_and(np.isclose(x[2], 2), np.isclose(x[0], 0))

    bdofsV1 = dolfinx.fem.locate_dofs_geometrical((V.sub(0), V0), top_corner)
    bc_corner = dolfinx.fem.dirichletbc.DirichletBC(ux_disp, bdofsV1, V.sub(0))

    # bdofs = dolfinx.fem.locate_dofs_geometrical(V, top)
    # bc_top = dolfinx.fem.dirichletbc.DirichletBC(u_disp, bdofs)

    # Fix bottom in all directons
    def boundaries(x):
        return np.isclose(x[2], np.finfo(float).eps)
    facets = dolfinx.mesh.compute_marked_boundary_entities(mesh, 1,
                                                           boundaries)
    topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bc_bottom = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc_bottom, bc_top, bc_corner]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e5
    nu = 0.0
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
    # ds_top = ufl.Measure("ds", domain=mesh,
    #                      subdomain_data=mf, subdomain_id=2)
    # g = dolfinx.Constant(mesh, (0,1e5))
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v) * ufl.dx
    # + ufl.inner(g,v)*ds_top

    # Create MPC

    slaves = []
    masters = []
    coeffs = []
    offsets = [0]

    # Locate dofs on both interfaces
    def boundaries(x):
        return np.isclose(x[2], 1)

    facets = dolfinx.mesh.compute_marked_boundary_entities(mesh,
                                                           mesh.topology.dim-1,
                                                           boundaries)
    # Slicing of list is due to the fact that we only require y-components
    interface_dofs = dolfinx.fem.locate_dofs_topological((V.sub(2), V2),
                                                         mesh.topology.dim-1,
                                                         facets)[:, 0]

    # Get some cell info
    # (range should be reduced with mesh function and markers)
    # cmap = dolfinx.fem.create_coordinate_map(mesh.ufl_domain())
    global_indices = V.dofmap.index_map.global_indices(False)
    num_cells = mesh.num_entities(mesh.topology.dim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim,
                                                range(num_cells))
    x_coords = V.tabulate_dof_coordinates()
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    mesh.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    mesh.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    cell_to_facet = mesh.topology.connectivity(mesh.topology.dim,
                                               mesh.topology.dim-1)
    for cell_index in range(num_cells):
        midpoint = cell_midpoints[cell_index]
        # NOTE, we should rather use mesh markers here to
        # determine if this could be a slave cell
        is_slave_cell = midpoint[2] > 1

        cell_dofs = V.dofmap.cell_dofs(cell_index)

        # Check if any dofs in cell is on interface
        has_dofs_on_interface = np.isin(cell_dofs, interface_dofs)
        if np.any(has_dofs_on_interface) and is_slave_cell:
            # Check if it is a slave node (x[1]>1)
            local_indices = np.flatnonzero(has_dofs_on_interface)
            for dof_index in local_indices:
                slave_coords = x_coords[cell_dofs[dof_index]]
                global_slave = global_indices[cell_dofs[dof_index]]
                if global_slave not in slaves:
                    slaves.append(global_slave)
                    # Need some parallel comm here
                    # Now find which master cell is close to the slave dof
                    possible_cells = comp_col_pts(tree, slave_coords)[0]
                    for other_cell in possible_cells:
                        if other_cell == cell_index:
                            continue
                        if cell_midpoints[other_cell][2] > 1:
                            continue
                        other_facets = cell_to_facet.links(other_cell)
                        has_boundary_facets = np.any(np.isin(other_facets,
                                                             facets))
                        if has_boundary_facets:
                            other_dofs = V.dofmap.cell_dofs(other_cell)
                            # Get basis values, and add coeffs of sub space 2
                            # to masters with corresponding coeff
                            basis_values = get_basis(V._cpp_object,
                                                     slave_coords, other_cell)
                            z_values = basis_values[:, 2]
                            z_dofs = V.sub(2).dofmap.cell_dofs(other_cell)
                            other_local_indices = np.flatnonzero(
                                np.isin(other_dofs, z_dofs))
                            for local_idx in other_local_indices:
                                if not np.isclose(z_values[local_idx], 0):
                                    masters.append(
                                        global_indices[other_dofs[local_idx]])
                                    coeffs.append(z_values[local_idx])

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
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

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
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "uh.xdmf").write(u_h)

    # Transfer data from the MPC problem to numpy arrays for comparison
    # A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    # mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    # A_org = dolfinx.fem.assemble_matrix(a, bcs)

    # A_org.assemble()
    # L_org = dolfinx.fem.assemble_vector(lhs)
    # dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    # L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
    #                   mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(L_org, bcs)
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
    # K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
    #                                                    masters, coeffs,
    #                                                    offsets)
    # Create reduced A
    # A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    # reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    # # Created reduced L
    # vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    # reduced_L = np.dot(K.T, vec)
    # # Solve linear system
    # d = np.linalg.solve(reduced_A, reduced_L)
    # # Back substitution to full solution vector
    # uh_numpy = np.dot(K, d)

    # # Compare LHS, RHS and solution with reference values
    # dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    # dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    # if uh.owner_range[0] < masters[0] and masters[0] < uh.owner_range[1]:
    #     print("MASTER DOF, MPC {0:.4e} Unconstrained {1:.4e}"
    #           .format(uh.array[masters[0]-uh.owner_range[0]],
    #                   u_.vector.array[masters[0]-uh.owner_range[0]]))
    #     print("Slave (given as master*coeff) {0:.4e}".
    #           format(uh.array[masters[0]-uh.owner_range[0]]*coeffs[0]))

    # if uh.owner_range[0] < slaves[0] and slaves[0] < uh.owner_range[1]:
    #     print("SLAVE  DOF, MPC {0:.4e} Unconstrained {1:.4e}"
    #           .format(uh.array[slaves[0]-uh.owner_range[0]],
    #                   u_.vector.array[slaves[0]-uh.owner_range[0]]))
    # assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
    #                                       uh.owner_range[1]])


if __name__ == "__main__":
    demo_stacked_cubes()

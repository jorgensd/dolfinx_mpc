import dolfinx
import dolfinx.geometry
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from petsc4py import PETSc
def demo_stacked_cubes():
    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "mesh.xdmf") as xdmf:
        mesh = xdmf.read_mesh()

    # To inspect partitioning
    # dolfinx.io.VTKFile("mesh.pvd").write(mesh)

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    # Move top in y-dir
    V1 = V.sub(1).collapse()
    # u_disp = dolfinx.function.Function(V1)

    u_disp = dolfinx.function.Function(V)
    with u_disp.vector.localForm() as u_local:
        u_local.set(-0.05)
    def top(x):
        return np.isclose(x[1], 2)
    # FIXME Add surface integrals
    # mf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim-1, 0)
    # mf.mark(top, 2)
    # bdofsV1 = dolfinx.fem.locate_dofs_geometrical((V.sub(1), V1), top)
    bdofs = dolfinx.fem.locate_dofs_geometrical(V, top)
    bc_top = dolfinx.fem.dirichletbc.DirichletBC(u_disp, bdofs)

    # bc_top = dolfinx.fem.dirichletbc.DirichletBC(u_disp, bdofsV1, V.sub(1))

    # Fix bottom in all directons
    def boundaries(x):
        return np.isclose(x[1], np.finfo(float).eps)
    facets = dolfinx.mesh.compute_marked_boundary_entities(mesh, 1,
                                                           boundaries)
    topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bc_bottom = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc_bottom, bc_top]

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
    x = ufl.SpatialCoordinate(mesh)
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    # FIXME: Add facet integrals
    # ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
    # g = dolfinx.Constant(mesh, (0,1e5))
    lhs = ufl.inner(dolfinx.Constant(mesh,(0,0)), v) * ufl.dx# + ufl.inner(g,v)*ds_top

    # Create MPC

    slaves = []
    masters = []
    coeffs = []
    offsets = [0]

    # Locate dofs on both interfaces
    def boundaries(x):
        return np.isclose(x[1], 1)
    facets = dolfinx.mesh.compute_marked_boundary_entities(mesh, 1,
                                                           boundaries)
    interface_dofs = dolfinx.fem.locate_dofs_topological((V.sub(1),V1), 1, facets)[:,0] # Only use dofs in full space

    # Get some cell info
    # (range should be reduced with mesh function and markers)
    cmap = dolfinx.fem.create_coordinate_map(mesh.ufl_domain())
    global_indices = V.dofmap.index_map.global_indices(False)
    num_cells = mesh.num_entities(mesh.topology.dim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, range(num_cells))
    x_coords = V.tabulate_dof_coordinates()
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)

    for cell_index in range(num_cells):
        midpoint = cell_midpoints[cell_index]
        cell_dofs = V.dofmap.cell_dofs(cell_index)

        # Check if any dofs in cell is on interface
        has_dofs_on_interface = np.isin(cell_dofs, interface_dofs)
        if np.any(has_dofs_on_interface):
            # Check if it is a slave node (x[1]>1)
            local_indices = np.flatnonzero(has_dofs_on_interface == True)
            for dof_index in local_indices:
                slave_coords = x_coords[cell_dofs[dof_index]]
                is_slave_dof = midpoint[1] > 1 # This mapping should use mesh markers
                if is_slave_dof and global_indices[cell_dofs[dof_index]] not in slaves:
                    print("Slave cell")
                    slaves.append(global_indices[cell_dofs[dof_index]])
                    # Now project slave to other interface and
                    # get basis function contributions
                    possible_cells = dolfinx.geometry.compute_collisions_point(tree, x_coords[cell_dofs[dof_index]]) # need some parallel comm here
                    for other_index in possible_cells[0]:
                        other_dofs = V.dofmap.cell_dofs(other_index)
                        dofs_on_interface = np.isin(other_dofs, interface_dofs)
                        other_midpoint = cell_midpoints[other_index]
                        is_master_cell = (other_index != cell_index) and np.any(dofs_on_interface) and other_midpoint[1] < 1
                        if is_master_cell:
                            # Check if any dofs in cell is on interface
                            basis_values = dolfinx_mpc.cpp.mpc.get_basis_functions(V._cpp_object, slave_coords, other_index)
                            flatten_values = np.sum(basis_values, axis=1)
                            other_local_indices = np.flatnonzero(dofs_on_interface == True)
                            for local_index in other_local_indices:
                                masters.append(global_indices[other_dofs[local_index]])
                                coeffs.append(flatten_values[local_index])
                            offsets.append(len(masters))
                            break

    print(slaves)
    print(masters)
    print(coeffs)
    print(offsets)

    slaves, masters, coeffs, offsets = np.array(slaves),np.array(masters),np.array(coeffs),np.array(offsets)
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
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)
    solver.solve(L_org, u_.vector)
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    u_.name = "u_unperturbed"
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "u_.xdmf").write(u_)

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
    if uh.owner_range[0] < masters[0] and masters[0] < uh.owner_range[1]:
        print("MASTER DOF, MPC {0:.4e} Unconstrained {1:.4e}"
              .format(uh.array[masters[0]-uh.owner_range[0]],
                      u_.vector.array[masters[0]-uh.owner_range[0]]))
        print("Slave (given as master*coeff) {0:.4e}".
              format(uh.array[masters[0]-uh.owner_range[0]]*coeffs[0]))

    if uh.owner_range[0] < slaves[0] and slaves[0] < uh.owner_range[1]:
        print("SLAVE  DOF, MPC {0:.4e} Unconstrained {1:.4e}"
              .format(uh.array[slaves[0]-uh.owner_range[0]],
                      u_.vector.array[slaves[0]-uh.owner_range[0]]))
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])


if __name__ == "__main__":
    demo_stacked_cubes()

import time

import numpy as np

from petsc4py import PETSc
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


def demo_elasticity(mesh, master_space, slave_space):
    # Create mesh and function space

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    def boundary(x):
        return np.isclose(x.T, [0, 0, 0]).all(axis=1)

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bdofsV = dolfinx.fem.locate_dofs_geometrical(V, boundary)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsV)
    bcs = [bc]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((-5*x[1], x[0]))
    # d = dolfinx.Constant(mesh, 2)
    # f = d*dolfinx.Function(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(f, v)*ufl.dx

    # Generate reference matrices and unconstrained solution
    A1 = dolfinx.fem.assemble_matrix(a, bcs)
    A1.assemble()
    L1 = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L1, [a], [bcs])
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L1, bcs)
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A1)
    u_ = dolfinx.Function(V)
    solver.solve(L1, u_.vector)
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "u_.xdmf").write(u_)

    # Create MPC
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0]): {lambda x: dof_at(x, [1, 1]): 0.1,
                                           lambda x: dof_at(x, [0.5, 1]): 0.3}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c,
                                                           slave_space,
                                                           master_space)

    print(slaves, masters)
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Setup MPC system
    for i in range(2):
        start = time.time()
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        end = time.time()
        print("Runtime: {0:.2e}".format(end-start))
    for i in range(2):
        b = dolfinx_mpc.assemble_vector(lhs, mpc)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = b.copy()
    uh.set(1)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)
    # Create functionspace and function for mpc vector
    modified_dofmap = dolfinx.cpp.fem.DofMap(V.dofmap.dof_layout,
                                             mpc.index_map(),
                                             V.dofmap.dof_array)
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  modified_dofmap)
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "uh.xdmf").write(u_h)

    # Generate global K matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
                                                       masters, coeffs,
                                                       offsets)

    vec = np.zeros(V.dim())
    mpc_vec = np.zeros(V.dim())
    vec[L1.owner_range[0]:L1.owner_range[1]] += L1.array
    vec = sum(dolfinx.MPI.comm_world.allgather(
        np.array(vec, dtype=np.float32)))
    mpc_vec[b.owner_range[0]:b.owner_range[1]] += b.array
    mpc_vec = sum(dolfinx.MPI.comm_world.allgather(
        np.array(mpc_vec, dtype=np.float32)))
    reduced_L = np.dot(K.T, vec)

    count = 0
    for i in range(V.dim()):
        if i in slaves:
            count += 1
            continue
        assert(np.isclose(reduced_L[i-count], mpc_vec[i]))

    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i, col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)

    # Transfer A to numpy matrix for comparison with globally built matrix
    A_mpc_np = np.zeros((V.dim(), V.dim()))
    for i in range(A.getOwnershipRange()[0], A.getOwnershipRange()[1]):
        cols, vals = A.getRow(i)
        # print("Line: ", i, " ", end="")
        for col, val in zip(cols, vals):
            A_mpc_np[i, col] = val

    A_mpc_np = sum(dolfinx.MPI.comm_world.allgather(A_mpc_np))

    # exit(1)
    # Pad globally reduced system with 0 rows and columns for each slave entry
    A_numpy_padded = np.zeros((V.dim(), V.dim()))
    count = 0
    for i in range(V.dim()):
        if i in slaves:
            A_numpy_padded[i, i] = 1
            count += 1
            continue
        m = 0
        for j in range(V.dim()):
            if j in slaves:
                m += 1
                continue
            else:
                A_numpy_padded[i, j] = reduced_A[i-count, j-m]

    # Check that all entities are close
    assert np.allclose(A_mpc_np, A_numpy_padded)

    # Compute globally reduced system and compare norms with A
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)
    print("MPC", uh.array)
    print("NUMPY", uh_numpy[uh.owner_range[0]:uh.owner_range[1]])
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])


for mesh in [dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 4, 4)]:
    for i in [0, 1]:
        for j in [0, 1]:
            demo_elasticity(mesh, i, j)

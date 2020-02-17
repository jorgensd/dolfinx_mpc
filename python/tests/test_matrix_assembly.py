import numpy as np
import pytest

import dolfinx
import dolfinx_mpc
import ufl


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.quadrilateral,
                                      dolfinx.cpp.mesh.CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 5, 3, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    # Create multi point constraint using geometrical mappings
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0]):
             {lambda x: dof_at(x, [0, 1]): 0.43,
              lambda x: dof_at(x, [1, 1]): 0.11},
             lambda x: dof_at(x, [0, 0]):
             {lambda x: dof_at(x, master_point): 0.69}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c)

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # No dirichlet condition as this is just assembly
    bcs = np.array([])

    # Assemble custom MPC assembler
    import time
    for i in range(2):
        start = time.time()
        A2 = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        end = time.time()
        print("Runtime: {0:.2e}".format(end-start))

    # Transfer A2 to numpy matrix for comparison with globally built matrix
    A_mpc_np = np.zeros((V.dim(), V.dim()))
    for i in range(A2.getOwnershipRange()[0], A2.getOwnershipRange()[1]):
        cols, vals = A2.getRow(i)
        for col, val in zip(cols, vals):
            A_mpc_np[i, col] = val
    A_mpc_np = sum(dolfinx.MPI.comm_world.allgather(A_mpc_np))

    # Generate global K matrix
    K = np.zeros((V.dim(), V.dim() - len(slaves)))
    for i in range(K.shape[0]):
        if i in slaves:
            index = np.argwhere(slaves == i)[0, 0]
            masters_index = masters[offsets[index]: offsets[index+1]]
            coeffs_index = coeffs[offsets[index]: offsets[index+1]]
            for master, coeff in zip(masters_index, coeffs_index):
                count = sum(master > np.array(slaves))
                K[i, master - count] = coeff
        else:
            count = sum(i > slaves)
            K[i, i-count] = 1

    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i, col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))
    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)

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


# Check if ordering of connected dofs matter
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.quadrilateral,
                                      dolfinx.cpp.mesh.CellType.triangle])
def test_slave_on_same_cell(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 1, 8, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Build master slave map
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0]):
             {lambda x: dof_at(x, [0, 1]): 0.43,
              lambda x: dof_at(x, [1, 1]): 0.11},
             lambda x: dof_at(x, [0, 0]):
             {lambda x: dof_at(x, master_point): 0.69}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c)

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    # Generating normal matrix for comparison
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # No dirichlet condition as this is just assembly
    bcs = np.array([])

    # Assemble custom MPC assembler
    import time
    for i in range(2):
        start = time.time()
        A2 = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        end = time.time()
        print("Runtime: {0:.2e}".format(end-start))
    # Transfer A2 to numpy matrix for comparison with globally built matrix
    A_mpc_np = np.zeros((V.dim(), V.dim()))

    for i in range(A2.getOwnershipRange()[0], A2.getOwnershipRange()[1]):
        cols, vals = A2.getRow(i)
        for col, val in zip(cols, vals):
            A_mpc_np[i, col] = val
    A_mpc_np = sum(dolfinx.MPI.comm_world.allgather(A_mpc_np))

    # Build global K matrix from slave_master_dict
    K = np.zeros((V.dim(), V.dim()-len(slaves)))
    for i in range(K.shape[0]):
        if i in slaves:
            index = np.argwhere(slaves == i)[0, 0]
            masters_index = masters[offsets[index]: offsets[index+1]]
            coeffs_index = coeffs[offsets[index]: offsets[index+1]]
            for master, coeff in zip(masters_index, coeffs_index):
                count = sum(master > np.array(slaves))
                K[i, master - count] = coeff
        else:
            count = sum(i > slaves)
            K[i, i-count] = 1

    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i, col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))

    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)

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

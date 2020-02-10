import dolfinx
import dolfinx_mpc
import numba
from numba.typed import List
import numpy as np
import ufl
import pytest

def master_dofs(x):
    dofs, vals = [],[]
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.43
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    logical = np.logical_and(np.isclose(x[:,0],1), np.isclose(x[:,1],1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.11
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    return dofs, vals

def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []

def slaves_2(x):
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []

def masters_2(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []

def masters_3(x):
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []

@pytest.mark.parametrize("masters_2", [masters_2, masters_3])
def test_mpc_assembly(masters_2):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 5,3)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    x = V.tabulate_dof_coordinates()
    # Build slaves and masters in parallel with global dof numbering scheme
    masters, coeffs = master_dofs(x)
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(np.array(masters, dtype=np.int32))
    coeffs = dolfinx.MPI.comm_world.allgather(np.array(coeffs, dtype=np.float32))
    masters = np.hstack(masters)
    coeffs = np.hstack(coeffs)
    slaves = slave_dofs(x)
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+V.dofmap.index_map.local_range[0]
    slaves = dolfinx.MPI.comm_world.allgather(np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters,coeffs):
            slave_master_dict[slave][master] = coeff
    # Second constraint
    masters2, coeffs2 = masters_2(x)
    slaves2 = slaves_2(x)
    for i, slave in enumerate(slaves2):
        slaves2[i] = slaves2[i]+V.dofmap.index_map.local_range[0]
    slaves2 = dolfinx.MPI.comm_world.allgather(np.array(slaves2, dtype=np.int32))
    slaves2 = np.hstack(slaves2)
    for i, master in enumerate(masters2):
        masters2[i] = masters2[i]+V.dofmap.index_map.local_range[0]
    masters2 = dolfinx.MPI.comm_world.allgather(np.array(masters2, dtype=np.int32))
    masters2 = np.hstack(masters2)
    coeffs2 = dolfinx.MPI.comm_world.allgather(np.array(coeffs2, dtype=np.float32))
    coeffs2 = np.hstack(coeffs2)

    for slave in slaves2:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters2,coeffs2):
            slave_master_dict[slave][master] = coeff
    print(slave_master_dict)

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    # Generating normal matrix for comparison
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)


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
            A_mpc_np[i,col] = val
    A_mpc_np = sum(dolfinx.MPI.comm_world.allgather(A_mpc_np))

    # Build global K matrix from slave_master_dict
    K = np.zeros((V.dim(), V.dim()-len(slave_master_dict.keys())))
    for i in range(K.shape[0]):
        if i in slave_master_dict.keys():
            for master in slave_master_dict[i].keys():
                l = sum(master>np.array(list(slave_master_dict.keys())))
                K[i, master - l] = slave_master_dict[i][master]
        else:
            l = sum(i>np.array(list(slave_master_dict.keys())))
            K[i, i-l] = 1


    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i,col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))
    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T,A_global),K)
    assert(np.isclose(np.linalg.norm(reduced_A), A2.norm()))

    # Pad globally reduced system with 0 rows and columns for each slave entry
    A_numpy_padded = np.zeros((V.dim(),V.dim()))
    l = 0
    for i in range(V.dim()):
        if i in slave_master_dict.keys():
            l += 1
            continue
        m = 0
        for j in range(V.dim()):
            if j in slave_master_dict.keys():
                m += 1
                continue
            else:
                A_numpy_padded[i, j] = reduced_A[i-l, j-m]

    # Check that all entities are close
    assert np.allclose(A_mpc_np, A_numpy_padded)

#Check if ordering of connected dofs matter
@pytest.mark.parametrize("masters_2", [masters_2, masters_3])
def test_slave_on_same_cell(masters_2):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 1,4)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    x = V.tabulate_dof_coordinates()
    # Build slaves and masters in parallel with global dof numbering scheme
    masters, coeffs = master_dofs(x)
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(np.array(masters, dtype=np.int32))
    coeffs = dolfinx.MPI.comm_world.allgather(np.array(coeffs, dtype=np.float32))
    masters = np.hstack(masters)
    coeffs = np.hstack(coeffs)
    slaves = slave_dofs(x)
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+V.dofmap.index_map.local_range[0]
    slaves = dolfinx.MPI.comm_world.allgather(np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters,coeffs):
            slave_master_dict[slave][master] = coeff

    # Second constraint
    masters2, coeffs2 = masters_2(x)

    slaves2 = slaves_2(x)
    for i, slave in enumerate(slaves2):
        slaves2[i] = slaves2[i]+V.dofmap.index_map.local_range[0]
    slaves2 = dolfinx.MPI.comm_world.allgather(np.array(slaves2, dtype=np.int32))
    slaves2 = np.hstack(slaves2)
    for i, master in enumerate(masters2):
        masters2[i] = masters2[i]+V.dofmap.index_map.local_range[0]
    masters2 = dolfinx.MPI.comm_world.allgather(np.array(masters2, dtype=np.int32))
    masters2 = np.hstack(masters2)
    coeffs2 = dolfinx.MPI.comm_world.allgather(np.array(coeffs2, dtype=np.float32))
    coeffs2 = np.hstack(coeffs2)
    for slave in slaves2:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters2,coeffs2):
            slave_master_dict[slave][master] = coeff

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    # Generating normal matrix for comparison
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)


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
    K = np.zeros((V.dim(), V.dim()-len(slave_master_dict.keys())))
    for i in range(K.shape[0]):
        if i in slave_master_dict.keys():
            for master in slave_master_dict[i].keys():
                l = sum(master>np.array(list(slave_master_dict.keys())))
                K[i, master - l] = slave_master_dict[i][master]
        else:
            l = sum(i>np.array(list(slave_master_dict.keys())))
            K[i, i-l] = 1

    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i,col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))

    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T,A_global),K)
    assert(np.isclose(np.linalg.norm(reduced_A), A2.norm()))

    # Pad globally reduced system with 0 rows and columns for each slave entry
    A_numpy_padded = np.zeros((V.dim(),V.dim()))
    l = 0
    for i in range(V.dim()):
        if i in slave_master_dict.keys():
            l += 1
            continue
        m = 0
        for j in range(V.dim()):
            if j in slave_master_dict.keys():
                m += 1
                continue
            else:
                A_numpy_padded[i, j] = reduced_A[i-l, j-m]

    # Check that all entities are close
    assert np.allclose(A_mpc_np, A_numpy_padded)

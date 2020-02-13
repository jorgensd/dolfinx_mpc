import time

import numpy as np
import pytest

from petsc4py import PETSc

import dolfinx
import dolfinx_mpc
import ufl


def master_dofs(x):
    dofs, vals = [], []
    logical = np.logical_and(np.isclose(x[:, 0], 0), np.isclose(x[:, 1], 1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.43
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    logical = np.logical_and(np.isclose(x[:, 0], 1), np.isclose(x[:, 1], 1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.11
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    return dofs, vals


def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:, 0], 1), np.isclose(x[:, 1], 0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []


def slaves_2(x):
    logical = np.logical_and(np.isclose(x[:, 0], 0), np.isclose(x[:, 1], 0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []


def masters_2(x):
    logical = np.logical_and(np.isclose(x[:, 0], 1), np.isclose(x[:, 1], 1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []


def masters_3(x):
    logical = np.logical_and(np.isclose(x[:, 0], 0), np.isclose(x[:, 1], 1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []


@pytest.mark.parametrize("masters_2", [masters_2, masters_3])
def test_mpc_assembly(masters_2):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 5)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    x = V.tabulate_dof_coordinates()
    masters, coeffs = master_dofs(x)
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(
        np.array(masters, dtype=np.int32))
    coeffs = dolfinx.MPI.comm_world.allgather(
        np.array(coeffs, dtype=np.float32))
    masters = np.hstack(masters)
    coeffs = np.hstack(coeffs)
    slaves = slave_dofs(x)
    local_min = V.dofmap.index_map.local_range[0]
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+local_min
    slaves = dolfinx.MPI.comm_world.allgather(
        np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters, coeffs):
            slave_master_dict[slave][master] = coeff
    # Second constraint
    masters2, coeffs2 = masters_2(x)
    slaves2 = slaves_2(x)
    for i, slave in enumerate(slaves2):
        slaves2[i] = slaves2[i] + local_min
    slaves2 = dolfinx.MPI.comm_world.allgather(
        np.array(slaves2, dtype=np.int32))
    slaves2 = np.hstack(slaves2)
    for i, master in enumerate(masters2):
        masters2[i] = masters2[i] + local_min
    masters2 = dolfinx.MPI.comm_world.allgather(
        np.array(masters2, dtype=np.int32))
    masters2 = np.hstack(masters2)
    coeffs2 = dolfinx.MPI.comm_world.allgather(
        np.array(coeffs2, dtype=np.float32))
    coeffs2 = np.hstack(coeffs2)

    for slave in slaves2:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters2, coeffs2):
            slave_master_dict[slave][master] = coeff

    # Define bcs
    bcs = np.array([])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = dolfinx.Constant(mesh, 1)
    # f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    lhs = ufl.inner(f, v)*ufl.dx
    # Generate reference matrices
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()
    L1 = dolfinx.fem.assemble_vector(lhs)
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)

    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)

    # Setup MPC system
    for i in range(2):
        start = time.time()
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        end = time.time()
        print("Runtime: {0:.2e}".format(end-start))
    mf = dolfinx.MeshFunction("size_t", mesh, 1, 0)

    def boundary(x):
        return np.full(x.shape[1], True)
    mf.mark(boundary, 1)

    for i in range(2):
        b = dolfinx_mpc.assemble_vector(lhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)

    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    if dolfinx.MPI.comm_world.rank == 0:
        print(slave_master_dict)

    mpc.backsubstitution(uh, V.dofmap)
    K = np.zeros((V.dim(), V.dim() - len(slave_master_dict.keys())))
    vec = np.zeros(V.dim())
    mpc_vec = np.zeros(V.dim())
    for i in range(K.shape[0]):
        if i in slave_master_dict.keys():
            for master in slave_master_dict[i].keys():
                count = sum(master > np.array(list(slave_master_dict.keys())))
                K[i, master - count] = slave_master_dict[i][master]
        else:
            count = sum(i > np.array(list(slave_master_dict.keys())))
            K[i, i-count] = 1

    vec[L1.owner_range[0]:L1.owner_range[1]] += L1.array
    vec = sum(dolfinx.MPI.comm_world.allgather(
        np.array(vec, dtype=np.float32)))
    mpc_vec[b.owner_range[0]:b.owner_range[1]] += b.array
    mpc_vec = sum(dolfinx.MPI.comm_world.allgather(
        np.array(mpc_vec, dtype=np.float32)))
    reduced_L = np.dot(K.T, vec)

    count = 0

    for i in range(V.dim()):
        if i in slave_master_dict.keys():
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

    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)

    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

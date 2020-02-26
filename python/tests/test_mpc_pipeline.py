import time

import numpy as np
import pytest

from petsc4py import PETSc
import dolfinx
import dolfinx.io
import dolfinx_mpc
import ufl

from utils import create_transformation_matrix


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
def test_pipeline(master_point):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 5)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Solve Problem without MPC for reference
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx.Constant(mesh, 1)
    # x = ufl.SpatialCoordinate(mesh)
    # f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    lhs = ufl.inner(f, v)*ufl.dx
    # Generate reference matrices
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()
    L1 = dolfinx.fem.assemble_vector(lhs)
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)

    # Create MPC
    dof_at = dolfinx_mpc.dof_close_to
    # s_m_c = {lambda x: dof_at(x, [0, 0]):
    #          {lambda x: dof_at(x, [1, 1]): 0.1}}
    s_m_c = {lambda x: dof_at(x, [1, 0]):
             {lambda x: dof_at(x, [0, 1]): 0.43,
              lambda x: dof_at(x, [1, 1]): 0.11},
             lambda x: dof_at(x, [0, 0]):
             {lambda x: dof_at(x, master_point): 0.69}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c)
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Setup MPC system
    for i in range(2):
        start = time.time()
        A = dolfinx_mpc.assemble_matrix(a, mpc)
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

    # Create global transformation matrix
    K = create_transformation_matrix(V.dim(), slaves, masters, coeffs, offsets)

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

    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)

    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

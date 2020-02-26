import numpy as np
import pytest
from petsc4py import PETSc

import dolfinx
import dolfinx_mpc
import ufl

from utils import create_transformation_matrix


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.quadrilateral,
                                      dolfinx.cpp.mesh.CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 5, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Generate reference vector
    v = ufl.TestFunction(V)
    f = dolfinx.Constant(mesh, 1)
    # x = ufl.SpatialCoordinate(mesh)
    # f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    lhs = ufl.inner(f, v)*ufl.dx

    L1 = dolfinx.fem.assemble_vector(lhs)
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)

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
    for i in range(2):
        b = dolfinx_mpc.assemble_vector(lhs, mpc)

    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)

    # Generate global K matrix
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

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import dolfinx
import dolfinx_mpc
import dolfinx_mpc.utils
import time
import ufl


dolfinx_mpc.utils.cache_numba(matrix=True)


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

    # Assemble custom MPC assembler
    start = time.time()
    A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)
    end = time.time()
    print("Runtime: {0:.2e}".format(end-start))

    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_mpc)

    # Create globally reduced system with numpy
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
                                                       masters, coeffs,
                                                       offsets)
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    A_org_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_org_np), K)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)


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

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Assemble custom MPC assembler
    import time
    start = time.time()
    A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)
    end = time.time()
    print("Runtime: {0:.2e}".format(end-start))

    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_mpc)

    # Create globally reduced system with numpy
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
                                                       masters, coeffs,
                                                       offsets)
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    A_org_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_org_np), K)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)

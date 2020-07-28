# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import time

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import ufl
from mpi4py import MPI
import dolfinx
import dolfinx.log


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.quadrilateral,
                                      dolfinx.cpp.mesh.CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 3, celltype)
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
     coeffs, offsets,
     master_owners) = dolfinx_mpc.slave_master_structure(V, s_m_c)

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets,
                                                   master_owners)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c_new = {l2b([1, 0]): {l2b([0, 1]): 0.43,
                               l2b([1, 1]): 0.11},
                 l2b([0, 0]):
                 {l2b(master_point): 0.69}}
    cc = dolfinx_mpc.create_dictionary_constraint(V, s_m_c_new)

    # Assemble custom MPC assembler
    with dolfinx.common.Timer("~Test: Assemble old"):
        A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)

    with dolfinx.common.Timer("~Test: Assemble new"):
        A_cc = dolfinx_mpc.assemble_matrix_local(a, cc)

    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_mpc)
    A_cc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_cc)
    assert(np.allclose(A_mpc_np, A_cc_np))
    # Create globally reduced system with numpy
    K = dolfinx_mpc.utils.create_transformation_matrix(V, cc)
    A_org = dolfinx.fem.assemble_matrix(a)
    A_org.assemble()
    A_org_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_org_np), K)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_cc_np, cc)


# Check if ordering of connected dofs matter
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.triangle,
                                      dolfinx.cpp.mesh.CellType.quadrilateral])
def test_slave_on_same_cell(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 1, 8, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Build master slave map
    s_m_c = {np.array([1, 0], dtype=np.float64).tobytes():
             {np.array([0, 1], dtype=np.float64).tobytes(): 0.43,
              np.array([1, 1], dtype=np.float64).tobytes(): 0.11},
             np.array([0, 0], dtype=np.float64).tobytes(): {
        np.array(master_point, dtype=np.float64).tobytes(): 0.69}}
    with dolfinx.common.Timer("~TEST: MPC INIT"):
        mpc = dolfinx_mpc.create_dictionary_constraint(V, s_m_c)

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    with dolfinx.common.Timer("~TEST: Assemble matrix"):
        A_mpc = dolfinx_mpc.assemble_matrix_local(a, mpc)
    with dolfinx.common.Timer("~TEST: Compare with numpy"):

        A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_mpc)
        # Create globally reduced system with numpy
        K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)

        A_org = dolfinx.fem.assemble_matrix(a)
        A_org.assemble()
        A_org_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
        reduced_A = np.matmul(np.matmul(K.T, A_org_np), K)

        dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, mpc)

    # dolfinx.common.list_timings(MPI.COMM_WORLD,
    #                             [dolfinx.common.TimingType.wall])

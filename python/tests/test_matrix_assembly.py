# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import ufl
from mpi4py import MPI

import dolfinx
import dolfinx.log

root = 0


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
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]): {l2b([0, 1]): 0.43,
                           l2b([1, 1]): 0.11},
             l2b([0, 0]):
             {l2b(master_point): 0.69}}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()
    with dolfinx.common.Timer("~TEST: Assemble matrix"):
        A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (cached)"):
        A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)

    with dolfinx.common.Timer("~TEST: Assemble matrix C++"):
        Acpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc)

    with dolfinx.common.Timer("~TEST: Compare with numpy"):
        A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(Acpp, root=root)
        A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A_mpc, root=root)
        if MPI.COMM_WORLD.rank == root:
            dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)

        # Create globally reduced system
        A_org = dolfinx.fem.assemble_matrix(a)
        A_org.assemble()
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A_mpc, mpc)


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
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_general_constraint(s_m_c)
        mpc.finalize()

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    with dolfinx.common.Timer("~TEST: Assemble matrix"):
        A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (cached)"):
        A_mpc = dolfinx_mpc.assemble_matrix(a, mpc)

    with dolfinx.common.Timer("~TEST: Assemble matrix C++"):
        Acpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc)

    with dolfinx.common.Timer("~TEST: Compare with numpy"):
        A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(Acpp, root=root)
        A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A_mpc, root=root)
        if MPI.COMM_WORLD.rank == root:
            dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)

        # Create globally reduced system
        A_org = dolfinx.fem.assemble_matrix(a)
        A_org.assemble()
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A_mpc, mpc)

    # dolfinx.common.list_timings(MPI.COMM_WORLD,
    #                             [dolfinx.common.TimingType.wall])

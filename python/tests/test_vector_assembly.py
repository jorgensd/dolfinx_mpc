# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import dolfinx
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import ufl
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.mesh.CellType.quadrilateral,
                                      dolfinx.mesh.CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype, get_assemblers):  # noqa: F811

    _, assemble_vector = get_assemblers

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 5, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Generate reference vector
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    rhs = ufl.inner(f, v) * ufl.dx

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]): {l2b([0, 1]): 0.43,
                           l2b([1, 1]): 0.11},
             l2b([0, 0]): {l2b(master_point): 0.69}}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()
    b = assemble_vector(rhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Reduce system with global matrix K after assembly
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])

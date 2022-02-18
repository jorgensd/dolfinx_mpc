# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as fem
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pytest
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import CellType, create_unit_square
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [CellType.quadrilateral,
                                      CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype, get_assemblers):  # noqa: F811

    _, assemble_vector = get_assemblers

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, celltype)
    V = fem.FunctionSpace(mesh, ("Lagrange", degree))

    # Generate reference vector
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    rhs = ufl.inner(f, v) * ufl.dx
    linear_form = fem.form(rhs)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {l2b([1, 0]): {l2b([0, 1]): 0.43,
                           l2b([1, 1]): 0.11},
             l2b([0, 0]): {l2b(master_point): 0.69}}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()
    b = assemble_vector(linear_form, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Reduce system with global matrix K after assembly
    L_org = fem.petsc.assemble_vector(linear_form)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    root = 0
    comm = mesh.comm
    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)

    list_timings(comm, [TimingType.wall])

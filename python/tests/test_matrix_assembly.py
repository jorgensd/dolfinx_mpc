# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI

import dolfinx.fem as fem
import numpy as np
import pytest
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.mesh import CellType, create_unit_square

import dolfinx_mpc
import dolfinx_mpc.utils
from dolfinx_mpc.utils import get_assemblers  # noqa: F401

root = 0


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [CellType.quadrilateral, CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype, get_assemblers):  # noqa: F811
    assemble_matrix, _ = get_assemblers

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 3, celltype)
    V = fem.functionspace(mesh, ("Lagrange", degree))

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    bilinear_form = fem.form(a)

    def l2b(li):
        return np.array(li, dtype=mesh.geometry.x.dtype).tobytes()

    s_m_c = {
        l2b([1, 0]): {l2b([0, 1]): 0.43, l2b([1, 1]): 0.11},
        l2b([0, 0]): {l2b(master_point): 0.69},
    }
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()
    with Timer("~TEST: Assemble matrix"):
        A_mpc = assemble_matrix(bilinear_form, mpc)

    with Timer("~TEST: Compare with numpy"):
        # Create globally reduced system
        A_org = fem.petsc.assemble_matrix(bilinear_form)
        A_org.assemble()
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A_mpc, mpc)


# Check if ordering of connected dofs matter
@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [CellType.triangle, CellType.quadrilateral])
def test_slave_on_same_cell(master_point, degree, celltype, get_assemblers):  # noqa: F811
    assemble_matrix, _ = get_assemblers

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 1, 8, celltype)
    V = fem.functionspace(mesh, ("Lagrange", degree))

    # Build master slave map
    s_m_c = {
        np.array([1, 0], dtype=mesh.geometry.x.dtype).tobytes(): {
            np.array([0, 1], dtype=mesh.geometry.x.dtype).tobytes(): 0.43,
            np.array([1, 1], dtype=mesh.geometry.x.dtype).tobytes(): 0.11,
        },
        np.array([0, 0], dtype=mesh.geometry.x.dtype).tobytes(): {
            np.array(master_point, dtype=mesh.geometry.x.dtype).tobytes(): 0.69
        },
    }
    with Timer("~TEST: MPC INIT"):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_general_constraint(s_m_c)
        mpc.finalize()

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    bilinear_form = fem.form(a)

    with Timer("~TEST: Assemble matrix"):
        A_mpc = assemble_matrix(bilinear_form, mpc)

    with Timer("~TEST: Compare with numpy"):
        # Create globally reduced system
        A_org = fem.petsc.assemble_matrix(bilinear_form)
        A_org.assemble()
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A_mpc, mpc)

    list_timings(mesh.comm, [TimingType.wall])

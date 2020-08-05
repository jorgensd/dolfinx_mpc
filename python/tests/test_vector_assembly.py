# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


@pytest.mark.parametrize("master_point", [[1, 1], [0, 1]])
@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.quadrilateral,
                                      dolfinx.cpp.mesh.CellType.triangle])
def test_mpc_assembly(master_point, degree, celltype):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 5, celltype)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

    # Generate reference vector
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    lhs = ufl.inner(f, v)*ufl.dx

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
    b = dolfinx_mpc.assemble_vector(lhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c_new = {l2b([1, 0]): {l2b([0, 1]): 0.43,
                               l2b([1, 1]): 0.11},
                 l2b([0, 0]): {l2b(master_point): 0.69}}
    cc = dolfinx_mpc.create_dictionary_constraint(V, s_m_c_new)
    bcc = dolfinx_mpc.assemble_vector_local(lhs, cc)
    bcc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)

    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
    cc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(bcc)

    assert(np.allclose(mpc_vec_np, cc_vec_np))

    # Reduce system with global matrix K after assembly
    L_org = dolfinx.fem.assemble_vector(lhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

    K = dolfinx_mpc.utils.create_transformation_matrix(V, cc)

    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)

    dolfinx_mpc.utils.compare_vectors(reduced_L, cc_vec_np, cc)

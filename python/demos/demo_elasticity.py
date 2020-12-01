# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl


def demo_elasticity():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def boundaries(x):
        return np.isclose(x[0], np.finfo(float).eps)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, 1,
                                                   boundaries)
    topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc]

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v))
                + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    x = ufl.SpatialCoordinate(mesh)
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(ufl.as_vector((0, (x[0] - 0.5) * 10**4 * x[1])), v) * ufl.dx

    # Create MPC
    with dolfinx.common.Timer("~Elasticity: Initialize MPC"):
        def l2b(li):
            return np.array(li, dtype=np.float64).tobytes()
        s_m_c = {l2b([1, 0]): {l2b([1, 1]): 0.9}}
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_general_constraint(s_m_c, 1, 1)
        mpc.finalize()

    # Setup MPC system
    with dolfinx.common.Timer("~Elasticity: Assemble LHS and RHS"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
        dolfinx.fem.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc"
    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                  "results/demo_elasticity.xdmf", "w")
    outfile.write_mesh(mesh)
    outfile.write_function(u_h)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)
    solver.solve(L_org, u_.vector)
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    u_.name = "u_unconstrained"
    outfile.write_function(u_)
    outfile.close()

    # Create reduced LHS and RHS
    K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)

    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    # Solve linear system and backsubstitute
    d = np.linalg.solve(reduced_A, reduced_L)
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
    dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
    assert(np.allclose(
        uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]]))

    # Print out master-slave connectivity for the first slave
    master_owner = None
    master_data = None
    slave_owner = None
    if mpc.num_local_slaves() > 0:
        slave_owner = MPI.COMM_WORLD.rank
        l2g = np.array(mpc.index_map().global_indices())
        bs = mpc.function_space().dofmap.index_map_bs
        slave = mpc.slaves()[0]
        print("Constrained: {0:.5e}\n Unconstrained: {1:.5e}"
              .format(uh.array[slave], u_.vector.array[slave]))
        master_owner = mpc._cpp_object.owners().links(0)[0]
        master = mpc.masters_local().array[0]
        master_data = [l2g[master // bs] * bs + master % bs,
                       mpc.coefficients()[0]]
        # If master not on proc send info to this processor
        if MPI.COMM_WORLD.rank != master_owner:
            MPI.COMM_WORLD.send(master_data, dest=master_owner, tag=1)
        else:
            print("Master*Coeff: {0:.5e}".format(mpc.coefficients()[0] * uh.array[mpc.masters_local().array[0]]))
    # As a processor with a master is not aware that it has a master,
    # Determine this so that it can receive the global dof and coefficient
    master_recv = MPI.COMM_WORLD.allgather(master_owner)
    for master in master_recv:
        if master is not None:
            master_owner = master
            break
    if slave_owner != master_owner and MPI.COMM_WORLD.rank == master_owner:
        bs = mpc.function_space().dofmap.index_map_bs
        in_data = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=1)
        l2g = np.array(mpc.index_map().global_indices())
        l_index = np.flatnonzero(l2g == in_data[0] // bs)[0]
        print("Master*Coeff (on other proc): {0:.5e}"
              .format(uh.array[l_index * bs + in_data[0] % bs] * in_data[1]))


if __name__ == "__main__":
    demo_elasticity()
    # dolfinx.common.list_timings(
    #     MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

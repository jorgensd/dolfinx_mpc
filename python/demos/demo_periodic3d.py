
# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
#
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False


def demo_periodic3D(celltype, out_periodic):
    # Create mesh and finite element
    if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
        # Tet setup
        N = 4
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        V = dolfinx.FunctionSpace(mesh, ("CG", 2))
        M = 2*N
    else:
        # Hex setup
        N = 12
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N,
                                    dolfinx.cpp.mesh.CellType.hexahedron)
        V = dolfinx.FunctionSpace(mesh, ("CG", 1))
        M = N

    # Create Dirichlet boundary condition
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def DirichletBoundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0),
                                           np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0),
                                           np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = dolfinx.fem.locate_dofs_geometrical(
        V, DirichletBoundary)
    bc = dolfinx.fem.DirichletBC(u_bc, geometrical_dofs)
    bcs = [bc]

    # Create MPC (map all left dofs to right
    dof_at = dolfinx_mpc.dof_close_to

    def slave_locater(i, j, N):
        return lambda x: dof_at(x, [1, float(i/N), float(j/N)])

    def master_locater(i, j, N):
        return lambda x: dof_at(x, [0, float(i/N), float(j/N)])

    s_m_c = {}
    for i in range(1, M):
        for j in range(1, M):
            s_m_c[slave_locater(i, j, M)] = {
                master_locater(i, j, M): 1}

    with dolfinx.common.Timer("~Periodic: Old init"):
        (slaves, masters,
         coeffs,
         offsets, owner_ranks) = dolfinx_mpc.slave_master_structure(V,
                                                                    s_m_c)
        mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                       masters, coeffs,
                                                       offsets,
                                                       owner_ranks)

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim-1, PeriodicBoundary)
    mt = dolfinx.MeshTags(mesh, mesh.topology.dim-1,
                          facets, np.full(len(facets), 2, dtype=np.int32))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1-x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x

    with dolfinx.common.Timer("~Periodic: New init"):
        cc = dolfinx_mpc.create_periodic_condition(
            V, mt, 2, periodic_relation, bcs)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    x = ufl.SpatialCoordinate(mesh)
    dx = x[0] - 0.9
    dy = x[1] - 0.5
    dz = x[2] - 0.1
    f = x[0]*ufl.sin(5.0*ufl.pi*x[1]) \
        + 1.0*ufl.exp(-(dx*dx + dy*dy + dz*dz)/0.02)

    lhs = ufl.inner(f, v)*ufl.dx

    # Compute solution
    u = dolfinx.Function(V)
    u.name = "uh"

    # Setup MPC system
    with dolfinx.common.Timer("~Periodic: NUMBA compile old"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(lhs, mpc)

    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    with dolfinx.common.Timer("~Periodic: NUMBA compile new"):
        Acc = dolfinx_mpc.assemble_matrix_local(a, cc, bcs=bcs)
        bcc = dolfinx_mpc.assemble_vector_local(lhs, cc)

    # Apply boundary conditions
    dolfinx.fem.apply_lifting(bcc, [a], [bcs])
    bcc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(bcc, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)

    if complex:
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
    else:
        opts = PETSc.Options()
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-6
        opts["pc_type"] = "hypre"
        opts['pc_hypre_type'] = 'boomeramg'
        opts["pc_hypre_boomeramg_max_iter"] = 1
        opts["pc_hypre_boomeramg_cycle_type"] = "v"
        # opts["pc_hypre_boomeramg_print_statistics"] = 1
        solver.setFromOptions()

    with dolfinx.common.Timer("~PERIODIC: Solve old"):
        solver.setOperators(A)
        uh = b.copy()
        uh.set(0)
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        # solver.view()
        it = solver.getIterationNumber()
        print("Iterations {0:d}".format(it))

    with dolfinx.common.Timer("~PERIODIC: Solve new"):
        solver.setOperators(Acc)
        uhcc = bcc.copy()
        uhcc.set(0)
        solver.solve(bcc, uhcc)
        uhcc.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
        # solver.view()
        it = solver.getIterationNumber()
        print("Iterations {0:d}".format(it))

    # Back substitute to slave dofs
    with dolfinx.common.Timer("~Periodic: Backsubstitute old"):
        dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)
    with dolfinx.common.Timer("~Periodic: Backsubstitute new"):
        dolfinx_mpc.backsubstitution_local(cc, uhcc)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  cc.dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # Write solution to file
    u_hcc = dolfinx.Function(Vmpc)
    u_hcc.vector.setArray(uhcc.array)
    if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
        ext = "tet"
    else:
        ext = "hex"

    mesh.name = "mesh_" + ext
    u_hcc.name = "u_" + ext

    out_periodic.write_mesh(mesh)
    out_periodic.write_function(u_hcc, 0.0,
                                "Xdmf/Domain/"
                                + "Grid[@Name='{0:s}'][1]"
                                .format(mesh.name))

    print("----Verification----")
    # --------------------VERIFICATION-------------------------
    A_org = dolfinx.fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)
    with dolfinx.common.Timer("~Periodic: Original solve"):
        solver.solve(L_org, u_.vector)
        it = solver.getIterationNumber()
    print("Original Iterations {0:d}".format(it))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    u_.name = "u_" + ext + "_unconstrained"
    out_periodic.write_function(u_, 0.0,
                                "Xdmf/Domain/"
                                + "Grid[@Name='{0:s}'][1]"
                                .format(mesh.name))

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    A_cc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(Acc)
    cc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(bcc)

    assert(np.allclose(A_cc_np, A_mpc_np))
    assert(np.allclose(mpc_vec_np, cc_vec_np))
    assert(np.allclose(uhcc.array, uh.array))
    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V, cc)
    # Create reduced A
    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)

    # Created reduced L
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    # Solve linear system
    d = np.linalg.solve(reduced_A, reduced_L)
    # Back substitution to full solution vector
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, cc)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, cc)
    assert np.allclose(
        uhcc.array, uh_numpy[uhcc.owner_range[0]:uhcc.owner_range[1]])


if __name__ == "__main__":
    fname = "results/demo_periodic3d.xdmf"
    out_periodic = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                       fname, "w")
    for celltype in [dolfinx.cpp.mesh.CellType.tetrahedron,
                     dolfinx.cpp.mesh.CellType.hexahedron]:
        demo_periodic3D(celltype, out_periodic)
    out_periodic.close()
    dolfinx.common.list_timings(
        MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

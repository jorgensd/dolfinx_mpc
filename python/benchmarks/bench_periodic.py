
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
import dolfinx.common
import dolfinx_mpc
import dolfinx_mpc.utils
import dolfinx.log
import ufl
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import refine
# from dolfinx_mpc.utils import cache_numba
# cache_numba(True, True, True)

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)


def demo_periodic3D(celltype, out_periodic, r_lvl=0):
    # Create mesh and finite element
    if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
        # Tet setup
        N = 4
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        for i in range(r_lvl):
            mesh = refine(mesh, redistribute=True)
            N *= 2
        V = dolfinx.FunctionSpace(mesh, ("CG", 2))
        M = 2*N
    else:
        # Hex setup
        N = 12
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N,
                                    dolfinx.cpp.mesh.CellType.hexahedron)
        V = dolfinx.FunctionSpace(mesh, ("CG", 1))
        for i in range(r_lvl):
            mesh = refine(mesh, redistribute=True)
            N *= 2
        M = N
    if MPI.COMM_WORLD.rank == 0:
        print("Rank {} R_level {} Num dofs {}".format(MPI.COMM_WORLD.rank,
                                                      r_lvl, V.dim()))
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

    def slaves_locater(x):
        eps = 1e-14
        not_edges_y = np.logical_and(x[1] > eps, x[1] < 1-eps)
        not_edges_z = np.logical_and(x[2] > eps, x[2] < 1-eps)
        not_edges = np.logical_and(not_edges_y, not_edges_z)
        return np.logical_and(np.isclose(x[0], 1), not_edges)

    def master_locater(x):
        eps = 1e-14
        not_edges_y = np.logical_and(x[1] > eps, x[1] < 1-eps)
        not_edges_z = np.logical_and(x[2] > eps, x[2] < 1-eps)
        not_edges = np.logical_and(not_edges_y, not_edges_z)
        return np.logical_and(np.isclose(x[0], 0), not_edges)

    # Find all masters and slaves
    slave_dofs = dolfinx.fem.locate_dofs_geometrical(V, slaves_locater).T[0]
    master_dofs = dolfinx.fem.locate_dofs_geometrical(V, master_locater).T[0]

    def create_master_slave_map(master_dofs, slave_dofs):
        timer = dolfinx.common.Timer("MPC: Create slave-master relationship")
        x = V.tabulate_dof_coordinates()
        local_min = (V.dofmap.index_map.local_range[0]
                     * V.dofmap.index_map.block_size)
        local_max = (V.dofmap.index_map.local_range[1]
                     * V.dofmap.index_map.block_size)
        x_slaves = x[slave_dofs, :]
        x_masters = x[master_dofs, :]
        masters = np.zeros((M-1)*(M-1), dtype=np.int64)
        slaves = np.zeros((M-1)*(M-1), dtype=np.int64)
        for i in range(1, M):
            for j in range(1, M):
                if len(slave_dofs) > 0:
                    slave_coord = [1, i/M, j/M]
                    slave_diff = np.abs(x_slaves-slave_coord).sum(axis=1)
                    if np.isclose(slave_diff[slave_diff.argmin()], 0):
                        # Only add if owned by processor
                        if (slave_dofs[slave_diff.argmin()]
                                < local_max - local_min):
                            slaves[(i-1)*(M-1)+j -
                                   1] = (slave_dofs[slave_diff.argmin()]
                                         + local_min)
                if len(master_dofs) > 0:
                    master_coord = [0, i/M, j/M]
                    master_diff = np.abs(x_masters-master_coord).sum(axis=1)
                    if np.isclose(master_diff[master_diff.argmin()], 0):
                        # Only add if owned by processor
                        if (master_dofs[master_diff.argmin()] <
                                local_max-local_min):
                            masters[(i-1)*(M-1)+j -
                                    1] = (master_dofs[master_diff.argmin()]
                                          + local_min)
        timer.stop()
        return slaves, masters
    snew, mnew = create_master_slave_map(master_dofs, slave_dofs)
    masters = sum(MPI.COMM_WORLD.allgather(mnew))
    slaves = sum(MPI.COMM_WORLD.allgather(snew))
    offsets = np.array(range(len(slaves)+1), dtype=np.int64)
    coeffs = np.ones(len(masters), dtype=np.float64)

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
    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Setup MPC system
    with dolfinx.common.Timer("MPC: Assemble matrix (Total time)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    with dolfinx.common.Timer("MPC: Assemble vector (Total time)"):
        b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # Solve Linear problem
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    # opts["ksp_rtol"] = 1.0e-12
    opts["pc_type"] = "gamg"
    opts["pc_gamg_type"] = "agg"
    opts["pc_gamg_sym_graph"] = True

    # Use Chebyshev smoothing for multigrid
    # opts["mg_levels_ksp_type"] = "richardson"
    # opts["mg_levels_pc_type"] = "sor"

    # Create nullspace
    nullspace = PETSc.NullSpace().create(constant=True)
    PETSc.Mat.setNearNullSpace(A, nullspace)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)

    # pc = solver.getPC()
    def monitor(ksp, its, rnorm, r_lvl=-1):
        print("{}: Iteration: {}, rel. residual: {}".format(r_lvl, its, rnorm))

    def pmonitor(ksp, its, rnorm): return monitor(ksp, its, rnorm, r_lvl=r_lvl)

    solver.setFromOptions()
    solver.setOperators(A)
    solver.setMonitor(pmonitor)
    uh = b.copy()
    uh.set(0)
    with dolfinx.common.Timer("MPC: Solve") as t:
        solver.solve(b, uh)
    # solver.view()

    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    with dolfinx.common.Timer("MPC: Backsubstitute") as t:
        dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)
        t.elapsed()

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    # if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
    #     ext = "tet"
    # else:
    #     ext = "hex"

    # mesh.name = "mesh_" + ext
    # u_h.name = "u_" + ext
    # print(u_h.vector.norm())
    # out_periodic.write_mesh(mesh)
    # out_periodic.write_function(u_h, 0.0,
    #                             "Xdmf/Domain/"
    #                             + "Grid[@Name='{0:s}'][1]"
    #                             .format(mesh.name))

    # print("----Verification----")
    # --------------------VERIFICATION-------------------------
    A_org = dolfinx.fem.assemble_matrix(a, bcs)

    A_org.assemble()
    PETSc.Mat.setNearNullSpace(A_org, nullspace)
    L_org = dolfinx.fem.assemble_vector(lhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)

    def umonitor(ksp, its, rnorm, r_lvl=-1):
        print("UNCONSTRAINED{}: Iteration: {}, rel. residual: {}".format(
            r_lvl, its, rnorm))

    def upmonitor(ksp, its, rnorm):
        return umonitor(ksp, its, rnorm, r_lvl=r_lvl)
    solver.setMonitor(upmonitor)
    solver.solve(L_org, u_.vector)
    # u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
    #                       mode=PETSc.ScatterMode.FORWARD)
    # u_.name = "u_" + ext + "_unconstrained"
    # out_periodic.write_function(u_, 0.0,
    #                             "Xdmf/Domain/"
    #                             + "Grid[@Name='{0:s}'][1]"
    #                             .format(mesh.name))


if __name__ == "__main__":
    fname = "results/demo_periodic3d.xdmf"
    out_periodic = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                       fname, "w")
    for i in range(6):  # range(4, 5):
        fname = "results/demo_periodic3d_{0:d}.xdmf".format(i)
        out_periodic = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                           fname, "w")
        for celltype in [dolfinx.cpp.mesh.CellType.tetrahedron]:
            demo_periodic3D(celltype, out_periodic, r_lvl=i)
        out_periodic.close()
        dolfinx.common.list_timings(
            MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

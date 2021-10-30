# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.

import argparse

import dolfinx.fem as fem
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from create_and_export_mesh import gmsh_3D_stacked, mesh_3D_dolfin

comm = MPI.COMM_WORLD


def demo_stacked_cubes(outfile, theta, gmsh=False,
                       ct=dolfinx.cpp.mesh.CellType.tetrahedron, compare=True, res=0.1, noslip=False):
    celltype = "hexahedron" if ct == dolfinx.cpp.mesh.CellType.hexahedron else "tetrahedron"
    type_ext = "no_slip" if noslip else "slip"
    mesh_ext = "_gmsh_" if gmsh else "_"
    dolfinx_mpc.utils.log_info(f"Run theta:{theta:.2f}, Cell: {celltype}, GMSH {gmsh}, Noslip: {noslip}")

    # Read in mesh
    if gmsh:
        mesh, mt = gmsh_3D_stacked(celltype, theta, res)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
    else:
        mesh_3D_dolfin(theta, ct, celltype, res)
        comm.barrier()
        with dolfinx.io.XDMFFile(comm, f"meshes/mesh_{celltype}_{theta:.2f}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            mt = xdmf.read_meshtags(mesh, "facet_tags")
    mesh.name = f"mesh_{celltype}_{theta:.2f}{type_ext}{mesh_ext}"
    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    g_vec = [0, 0, -4.25e-1]
    if not noslip:
        # Helper for orienting traction
        r_matrix = dolfinx_mpc.utils.rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])

    g = dolfinx.Constant(mesh, PETSc.ScalarType(g_vec))

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = dolfinx.Function(V)
    u_top.interpolate(top_v)

    top_facets = mt.indices[np.flatnonzero(mt.values == 3)]
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.DirichletBC(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e3)
    nu = 0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    # NOTE: Traction deactivated until we have a way of fixing nullspace
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)
    rhs = ufl.inner(dolfinx.Constant(mesh, PETSc.ScalarType((0, 0, 0))), v) * ufl.dx
    + ufl.inner(g, v) * ds

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    if noslip:
        with dolfinx.common.Timer("~~Contact: Create non-elastic constraint"):
            mpc.create_contact_inelastic_condition(mt, 4, 9)
    else:
        with dolfinx.common.Timer("~Contact: Create facet normal approximation"):
            nh = dolfinx_mpc.utils.create_normal_approximation(V, mt.indices[mt.values == 4])
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/nh.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(nh)
        with dolfinx.common.Timer("~Contact: Create contact constraint"):
            mpc.create_contact_slip_condition(mt, 4, 9, nh)
    with dolfinx.common.Timer("~~Contact: Add data and finialize MPC"):
        mpc.finalize()
    null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    with dolfinx.common.Timer(f"~~Contact: Assemble matrix ({num_dofs})"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    with dolfinx.common.Timer(f"~~Contact: Assemble vector ({num_dofs})"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve Linear problem
    opts = PETSc.Options()
    opts["ksp_rtol"] = 1.0e-8
    opts["pc_type"] = "gamg"
    opts["pc_gamg_type"] = "agg"
    opts["pc_gamg_coarse_eq_limit"] = 1000
    opts["pc_gamg_sym_graph"] = True
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["matptap_via"] = "scalable"
    # opts["pc_gamg_square_graph"] = 2
    # opts["pc_gamg_threshold"] = 1e-2
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver
    # Create functionspace and build near nullspace

    A.setNearNullSpace(null_space)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setFromOptions()
    uh = b.copy()
    uh.set(0)
    with dolfinx.common.Timer("~~Contact: Solve"):
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    with dolfinx.common.Timer("~~Contact: Backsubstitution"):
        mpc.backsubstitution(uh)

    it = solver.getIterationNumber()
    unorm = uh.norm()
    num_slaves = MPI.COMM_WORLD.allreduce(mpc.num_local_slaves(), op=MPI.SUM)
    if comm.rank == 0:
        num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        print(f"Number of dofs: {num_dofs}")
        print(f"Number of slaves: {num_slaves}")
        print(f"Number of iterations: {it}")
        print(f"Norm of u {unorm:.5e}")

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = f"u_{celltype}_{theta:.2f}{mesh_ext}{type_ext}".format(celltype, theta, type_ext, mesh_ext)
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")
    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    if not compare:
        return

    dolfinx_mpc.utils.log_info("Solving reference problem with global matrix (using scipy)")
    with dolfinx.common.Timer("~~Contact: Reference problem"):
        A_org = fem.assemble_matrix(a, bcs)
        A_org.assemble()
        L_org = fem.assemble_vector(rhs)
        fem.apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)

    root = 0
    with dolfinx.common.Timer("~~Contact: Compare LHS, RHS and solution"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)

    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res", help="Resolution of Mesh")
    parser.add_argument("--theta", default=np.pi / 3, type=np.float64, dest="theta",
                        help="Rotation angle around axis [1, 1, 0]")
    hex = parser.add_mutually_exclusive_group(required=False)
    hex.add_argument('--hex', dest='hex', action='store_true',
                     help="Use hexahedron mesh", default=False)
    slip = parser.add_mutually_exclusive_group(required=False)
    slip.add_argument('--no-slip', dest='noslip', action='store_true',
                      help="Use no-slip constraint", default=False)
    gmsh = parser.add_mutually_exclusive_group(required=False)
    gmsh.add_argument('--gmsh', dest='gmsh', action='store_true',
                      help="Gmsh mesh instead of built-in grid", default=False)
    comp = parser.add_mutually_exclusive_group(required=False)
    comp.add_argument('--compare', dest='compare', action='store_true',
                      help="Compare with global solution", default=False)
    time = parser.add_mutually_exclusive_group(required=False)
    time.add_argument('--timing', dest='timing', action='store_true',
                      help="List timings", default=False)

    args = parser.parse_args()
    compare = args.compare
    timing = args.timing
    res = args.res
    noslip = args.noslip
    gmsh = args.gmsh
    theta = args.theta
    hex = args.hex

    outfile = dolfinx.io.XDMFFile(comm, "results/demo_contact_3D.xdmf", "w")

    if hex:
        ct = dolfinx.cpp.mesh.CellType.hexahedron
    else:
        ct = dolfinx.cpp.mesh.CellType.tetrahedron
    demo_stacked_cubes(outfile, theta=theta, gmsh=gmsh, ct=ct, compare=compare, res=res, noslip=noslip)

    outfile.close()

    dolfinx_mpc.utils.log_info("Simulation finished")
    if timing:
        dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])

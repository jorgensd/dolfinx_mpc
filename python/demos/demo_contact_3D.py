# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import dolfinx.fem as fem
import numpy as np
import scipy.sparse.linalg
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType
from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)
from dolfinx_mpc.utils import (compare_mpc_lhs, compare_mpc_rhs,
                               create_normal_approximation, gather_PETScMatrix,
                               gather_PETScVector,
                               gather_transformation_matrix, log_info,
                               rigid_motions_nullspace, rotation_matrix)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import Identity, TestFunction, TrialFunction, dx, grad, inner, sym, tr

from create_and_export_mesh import gmsh_3D_stacked, mesh_3D_dolfin


def demo_stacked_cubes(outfile, theta, gmsh: bool = False, ct: CellType = CellType.tetrahedron,
                       compare: bool = True, res: np.float64 = 0.1, noslip: bool = False):
    celltype = "hexahedron" if ct == CellType.hexahedron else "tetrahedron"
    type_ext = "no_slip" if noslip else "slip"
    mesh_ext = "_gmsh_" if gmsh else "_"
    log_info(f"Run theta:{theta:.2f}, Cell: {celltype}, GMSH {gmsh}, Noslip: {noslip}")

    # Read in mesh
    if gmsh:
        mesh, mt = gmsh_3D_stacked(celltype, theta, res)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
    else:
        mesh_3D_dolfin(theta, ct, celltype, res)
        MPI.COMM_WORLD.barrier()
        with XDMFFile(MPI.COMM_WORLD, f"meshes/mesh_{celltype}_{theta:.2f}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            mt = xdmf.read_meshtags(mesh, "facet_tags")
    mesh.name = f"mesh_{celltype}_{theta:.2f}{type_ext}{mesh_ext}"

    # Create functionspaces
    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions

    # Bottom boundary is fixed in all directions
    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    u_bc = np.array((0, ) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_bottom = fem.dirichletbc(u_bc, bottom_dofs, V)

    g_vec = np.array([0, 0, -4.25e-1], dtype=PETSc.ScalarType)
    if not noslip:
        # Helper for orienting traction
        r_matrix = rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, g_vec)

    top_facets = mt.indices[np.flatnonzero(mt.values == 3)]
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.dirichletbc(g_vec, top_dofs, V)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e3)
    nu = 0
    mu = fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v)))

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    # NOTE: Traction deactivated until we have a way of fixing nullspace
    # g = fem.Constant(mesh, PETSc.ScalarType(g_vec))
    # ds = Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)
    rhs = inner(fem.Constant(mesh, PETSc.ScalarType((0, 0, 0))), v) * dx
    # + inner(g, v) * ds
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    mpc = MultiPointConstraint(V)
    if noslip:
        with Timer("~~Contact: Create non-elastic constraint"):
            mpc.create_contact_inelastic_condition(mt, 4, 9)
    else:
        with Timer("~Contact: Create contact constraint"):
            nh = create_normal_approximation(V, mt, 4)
            mpc.create_contact_slip_condition(mt, 4, 9, nh)

    with Timer("~~Contact: Add data and finialize MPC"):
        mpc.finalize()

    # Create null-space
    null_space = rigid_motions_nullspace(mpc.function_space)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    with Timer(f"~~Contact: Assemble matrix ({num_dofs})"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    with Timer(f"~~Contact: Assemble vector ({num_dofs})"):
        b = assemble_vector(linear_form, mpc)

    apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

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
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setFromOptions()

    u_h = fem.Function(mpc.function_space)
    with Timer("~~Contact: Solve"):
        solver.solve(b, u_h.vector)
        u_h.x.scatter_forward()
    with Timer("~~Contact: Backsubstitution"):
        mpc.backsubstitution(u_h.vector)

    it = solver.getIterationNumber()
    unorm = u_h.vector.norm()
    num_slaves = MPI.COMM_WORLD.allreduce(mpc.num_local_slaves, op=MPI.SUM)
    if mesh.comm.rank == 0:
        num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        print(f"Number of dofs: {num_dofs}")
        print(f"Number of slaves: {num_slaves}")
        print(f"Number of iterations: {it}")
        print(f"Norm of u {unorm:.5e}")

    # Write solution to file
    u_h.name = f"u_{celltype}_{theta:.2f}{mesh_ext}{type_ext}".format(celltype, theta, type_ext, mesh_ext)
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")
    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    if not compare:
        return

    log_info("Solving reference problem with global matrix (using scipy)")
    with Timer("~~Contact: Reference problem"):
        A_org = fem.petsc.assemble_matrix(bilinear_form, bcs)
        A_org.assemble()
        L_org = fem.petsc.assemble_vector(linear_form)
        fem.petsc.apply_lifting(L_org, [bilinear_form], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(L_org, bcs)

    root = 0
    with Timer("~~Contact: Compare LHS, RHS and solution"):
        compare_mpc_lhs(A_org, A, mpc, root=root)
        compare_mpc_rhs(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = gather_PETScMatrix(A_org, root=root)
        K = gather_transformation_matrix(mpc, root=root)
        L_np = gather_PETScVector(L_org, root=root)
        u_mpc = gather_PETScVector(u_h.vector, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)

    list_timings(mesh.comm, [TimingType.wall])


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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

    outfile = XDMFFile(MPI.COMM_WORLD, "results/demo_contact_3D.xdmf", "w")

    ct = CellType.hexahedron if args.hex else CellType.tetrahedron

    demo_stacked_cubes(outfile, theta=args.theta, gmsh=args.gmsh, ct=ct,
                       compare=args.compare, res=args.res, noslip=args.noslip)

    outfile.close()

    log_info("Simulation finished")
    if args.timing:
        list_timings(MPI.COMM_WORLD, [TimingType.wall])

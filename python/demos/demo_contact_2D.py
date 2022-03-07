# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# This demo demonstrates how to solve a contact problem between
# two stacked cubes.
# The bottom cube is fixed at the bottom surface
# The top cube has a force applied normal to its to surface.
# A slip condition is implemented at the interface of the cube.
# Additional constraints to avoid tangential movement is
# added to the to left corner of the top cube.


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import scipy.sparse.linalg
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Constant, VectorFunctionSpace, apply_lifting,
                         assemble_matrix, assemble_vector, dirichletbc, form,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import (compare_mpc_lhs, compare_mpc_rhs,
                               create_normal_approximation,
                               facet_normal_approximation, gather_PETScMatrix,
                               gather_PETScVector,
                               gather_transformation_matrix, log_info,
                               rigid_motions_nullspace, rotation_matrix)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, Measure, TestFunction, TrialFunction, dx, grad,
                 inner, sym, tr)

from create_and_export_mesh import gmsh_2D_stacked, mesh_2D_dolfin

set_log_level(LogLevel.ERROR)


def demo_stacked_cubes(outfile, theta, gmsh=True, quad=False, compare=False, res=0.1):
    log_info(f"Run theta:{theta:.2f}, Quad: {quad}, Gmsh {gmsh}, Res {res:.2e}")

    celltype = "quadrilateral" if quad else "triangle"
    if gmsh:
        mesh, mt = gmsh_2D_stacked(celltype, theta)
        mesh.name = f"mesh_{celltype}_{theta:.2f}_gmsh"

    else:
        mesh_name = "mesh"
        filename = f"meshes/mesh_{celltype}_{theta:.2f}.xdmf"

        mesh_2D_dolfin(celltype, theta)
        with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = f"mesh_{celltype}_{theta:.2f}"
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            mt = xdmf.read_meshtags(mesh, name="facet_tags")

    # Helper until meshtags can be read in from xdmf
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))

    r_matrix = rotation_matrix([0, 0, 1], theta)
    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = Constant(mesh, PETSc.ScalarType(g_vec[:2]))

    def bottom_corner(x):
        return np.isclose(x, [[0], [0], [0]]).all(axis=0)

    # Fix bottom corner
    bc_value = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bottom_dofs = locate_dofs_geometrical(V, bottom_corner)
    bc_bottom = dirichletbc(bc_value, bottom_dofs, V)
    bcs = [bc_bottom]

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e3)
    nu = 0
    mu = Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v)))

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    ds = Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)
    rhs = inner(Constant(mesh, PETSc.ScalarType((0, 0))), v) * dx + inner(g, v) * ds

    def left_corner(x):
        return np.isclose(x.T, np.dot(r_matrix, [0, 2, 0])).all(axis=1)

    # Create multi point constraint
    mpc = MultiPointConstraint(V)

    with Timer("~Contact: Create contact constraint"):
        nh = create_normal_approximation(V, mt, 4)
        mpc.create_contact_slip_condition(mt, 4, 9, nh)

    with Timer("~Contact: Add non-slip condition at bottom interface"):
        bottom_normal = facet_normal_approximation(V, mt, 5)
        mpc.create_slip_constraint(V, (mt, 5), bottom_normal, bcs=bcs)

    with Timer("~Contact: Add tangential constraint at one point"):
        vertex = locate_entities_boundary(mesh, 0, left_corner)

        tangent = facet_normal_approximation(V, mt, 3, tangent=True)
        mtv = meshtags(mesh, 0, vertex, np.full(len(vertex), 6, dtype=np.int32))
        mpc.create_slip_constraint(V, (mtv, 6), tangent, bcs=bcs)

    mpc.finalize()
    rtol = 1e-9
    petsc_options = {"ksp_rtol": 1e-9,
                     "pc_type": "gamg", "pc_gamg_type": "agg", "pc_gamg_square_graph": 2,
                     "pc_gamg_threshold": 0.02, "pc_gamg_coarse_eq_limit": 1000, "pc_gamg_sym_graph": True,
                     "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
                     "mg_levels_esteig_ksp_type": "cg"
                     #  , "help": None, "ksp_view": None
                     }

    # Solve Linear problem
    problem = LinearProblem(a, rhs, mpc, bcs=bcs, petsc_options=petsc_options)

    # Build near nullspace
    null_space = rigid_motions_nullspace(mpc.function_space)
    problem.A.setNearNullSpace(null_space)
    u_h = problem.solve()

    it = problem.solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    unorm = u_h.vector.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(f"Norm of u: {unorm}")

    # Write solution to file
    ext = "_gmsh" if gmsh else ""
    u_h.name = "u_mpc_{0:s}_{1:.2f}{2:s}".format(celltype, theta, ext)

    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    if not compare:
        return
    log_info("Solving reference problem with global matrix (using numpy)")
    with Timer("~MPC: Reference problem"):
        # Generate reference matrices and unconstrained solution
        A_org = assemble_matrix(form(a), bcs)
        A_org.assemble()
        L_org = assemble_vector(form(rhs))
        apply_lifting(L_org, [form(a)], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L_org, bcs)

    root = 0
    with Timer("~MPC: Verification"):
        compare_mpc_lhs(A_org, problem.A, mpc, root=root)
        compare_mpc_rhs(L_org, problem.b, mpc, root=root)
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

            assert np.allclose(uh_numpy, u_mpc, rtol=rtol)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res",
                        help="Resolution of Mesh")
    parser.add_argument("--theta", default=np.pi / 3, type=np.float64, dest="theta",
                        help="Rotation angle around axis [1, 1, 0]")
    quad = parser.add_mutually_exclusive_group(required=False)
    quad.add_argument('--quad', dest='quad', action='store_true',
                      help="Use quadrilateral mesh", default=False)
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

    # Create results file
    outfile = XDMFFile(MPI.COMM_WORLD, "results/demo_contact_2D.xdmf", "w")

    # Run demo for input parameters
    demo_stacked_cubes(outfile, theta=args.theta, gmsh=args.gmsh, quad=args.quad,
                       compare=args.compare, res=args.res)

    outfile.close()
    if args.timing:
        list_timings(MPI.COMM_WORLD, [TimingType.wall])

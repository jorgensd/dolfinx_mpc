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
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Constant, DirichletBC, Function, VectorFunctionSpace,
                         locate_dofs_geometrical, set_bc)

from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities_boundary
from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)
from dolfinx_mpc.utils import (create_normal_approximation, log_info,
                               rigid_motions_nullspace, rotation_matrix, facet_normal_approximation,
                               compare_MPC_LHS, compare_MPC_RHS, gather_PETScMatrix, gather_transformation_matrix,
                               gather_PETScVector)
import scipy.sparse.linalg
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, TestFunction, TrialFunction,
                 dx, grad, inner, sym, tr, Measure)
from dolfinx.log import set_log_level, LogLevel
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

    # Helper until MeshTags can be read in from xdmf
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))

    r_matrix = rotation_matrix([0, 0, 1], theta)
    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = Constant(mesh, PETSc.ScalarType(g_vec[:2]))

    def bottom_corner(x):
        return np.isclose(x, [[0], [0], [0]]).all(axis=0)
    # Fix bottom corner
    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bottom_dofs = locate_dofs_geometrical(V, bottom_corner)
    bc_bottom = DirichletBC(u_bc, bottom_dofs)
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

    mpc = MultiPointConstraint(V)

    with Timer("~Contact: Create contact constraint"):
        nh = create_normal_approximation(V, mt, 4)
        mpc.create_contact_slip_condition(mt, 4, 9, nh)

    with Timer("~Contact: Add non-slip condition at bottom interface"):
        bottom_normal = facet_normal_approximation(V, mt, 5)
        with XDMFFile(mesh.comm, "results/nh.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(bottom_normal)
        mpc.create_slip_constraint((mt, 5), bottom_normal, bcs=bcs)

    with Timer("~Contact: Add tangential constraint at one point"):
        vertex = locate_entities_boundary(mesh, 0, left_corner)

        tangent = facet_normal_approximation(V, mt, 3, tangent=True)
        mtv = MeshTags(mesh, 0, vertex, np.full(len(vertex), 6, dtype=np.int32))
        mpc.create_slip_constraint((mtv, 6), tangent, bcs=bcs)

    mpc.finalize()

    with Timer("~Contact: Assemble LHS and RHS"):
        A = assemble_matrix(a, mpc, bcs=bcs)
        b = assemble_vector(rhs, mpc)

    apply_lifting(b, [a], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

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
    opts["pc_gamg_square_graph"] = 2
    opts["pc_gamg_threshold"] = 0.02
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    # Create functionspace and build near nullspace
    null_space = rigid_motions_nullspace(mpc.function_space)
    A.setNearNullSpace(null_space)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    unorm = uh.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(f"Norm of u: {unorm}")

    # Write solution to file
    ext = "gmsh" if gmsh else ""

    u_h = Function(mpc.function_space)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc_{0:s}_{1:.2f}_{2:s}".format(celltype, theta, ext)

    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    if not compare:
        return
    log_info("Solving reference problem with global matrix (using numpy)")
    with Timer("~MPC: Reference problem"):
        # Generate reference matrices and unconstrained solution
        A_org = assemble_matrix(a, bcs)

        A_org.assemble()
        L_org = assemble_vector(rhs)
        apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L_org, bcs)

    root = 0
    with Timer("~MPC: Verification"):
        compare_MPC_LHS(A_org, A, mpc, root=root)
        compare_MPC_RHS(L_org, b, mpc, root=root)
        # Gather LHS, RHS and solution on one process
        A_csr = gather_PETScMatrix(A_org, root=root)
        K = gather_transformation_matrix(mpc, root=root)
        L_np = gather_PETScVector(L_org, root=root)
        u_mpc = gather_PETScVector(uh, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)


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
    compare = args.compare
    timing = args.timing
    res = args.res
    gmsh = args.gmsh
    theta = args.theta
    quad = args.quad

    outfile = XDMFFile(MPI.COMM_WORLD, "results/demo_contact_2D.xdmf", "w")
    demo_stacked_cubes(outfile, theta=theta, gmsh=gmsh,
                       quad=quad, compare=compare, res=res)

    outfile.close()
    if timing:
        list_timings(MPI.COMM_WORLD, [TimingType.wall])

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# This demo demonstrates how to solve a contact problem between
# two stacked cubes.
# The bottom cube is fixed at the bottom surface
# The top cube has a force applied normal to its to surface.
# A slip condition is implemented at the interface of the cube.
# Additional constraints to avoid tangential movement is
# added to the to left corner of the top cube.
from contextlib import ExitStack
import argparse

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as fem
import dolfinx.geometry as geometry
import dolfinx.io
import dolfinx.la
import dolfinx.log
from create_and_export_mesh import mesh_2D_dolfin, mesh_2D_rot
from helpers import find_master_slave_relationship

dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions

comm = MPI.COMM_WORLD


def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [dolfinx.cpp.la.create_vector(
        V.dofmap.index_map) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm())
                     for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(gdim):
            dofs = V.sub(i).dofmap.list
            basis[i][dofs.array()] = 1.0

        # Build rotational null space basis
        if gdim == 2:
            V.sub(0).set_x(basis[2], -1.0, 1)
            V.sub(1).set_x(basis[2], 1.0, 0)
        elif gdim == 3:
            V.sub(0).set_x(basis[3], -1.0, 1)
            V.sub(1).set_x(basis[3], 1.0, 0)

            V.sub(0).set_x(basis[4], 1.0, 2)
            V.sub(2).set_x(basis[4], -1.0, 0)

            V.sub(2).set_x(basis[5], 1.0, 1)
            V.sub(1).set_x(basis[5], -1.0, 2)

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(dim)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


def demo_stacked_cubes(outfile, theta, gmsh=True, triangle=True):
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run theta:{0:.2f}, Triangle: {1:b}, Gmsh {2:b}"
                        .format(theta, triangle, gmsh))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    if gmsh:
        mesh_name = "Grid"
        if MPI.COMM_WORLD.size == 1:
            mesh_2D_rot(theta)
        filename = "meshes/mesh_rot.xdmf"
        facet_file = "meshes/facet_rot.xdmf"
        ext = "gmsh" + "{0:.2f}".format(theta)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_" + ext
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, "Grid")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 facet_file, "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, name="Grid")
    else:
        mesh_name = "mesh"
        if triangle:
            if MPI.COMM_WORLD.size == 1:
                mesh_2D_dolfin("tri", theta)
            filename = "meshes/mesh_tri.xdmf"
            ext = "tri" + "{0:.2f}".format(theta)
        else:
            if MPI.COMM_WORLD.size == 1:
                mesh_2D_dolfin("quad", theta)
            filename = "meshes/mesh_quad.xdmf"
            ext = "quad" + "{0:.2f}".format(theta)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_" + ext
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, name="mesh_tags")
            mt = xdmf.read_meshtags(mesh, name="facet_tags")

    # dolfinx.io.VTKFile("results/mesh.pvd").write(mesh)

    # Helper until MeshTags can be read in from xdmf
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    V0 = V.sub(0).collapse()
    V1 = V.sub(1).collapse()

    r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = dolfinx.Constant(mesh, g_vec[:2])

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    bcs = [bc_bottom]

    # Elasticity parameters
    E = 1.0e3
    nu = 0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt,
                     subdomain_id=3)
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create standard master slave relationsship
    (slaves, masters, coeffs,
     offsets, owner_ranks) = find_master_slave_relationship(
        V, (mt, 4, 9), (ct, 2))

    def left_corner(x):
        return np.isclose(x.T, np.dot(r_matrix, [0, 2, 0])).all(axis=1)

    if np.isclose(theta, 0):
        # Restrict normal sliding by restricting one dof in the top corner
        V0 = V.sub(0).collapse()
        zero = dolfinx.Function(V0)
        with zero.vector.localForm() as zero_local:
            zero_local.set(0.0)
        dofs = fem.locate_dofs_geometrical((V.sub(0), V0), left_corner)
        bc_corner = dolfinx.DirichletBC(zero, dofs, V.sub(0))
        bcs.append(bc_corner)
    else:
        # approximate tangent with normal on left side.
        tangent = dolfinx_mpc.facet_normal_approximation(V, mt, 6)
        t_vec = tangent.vector.getArray()

        dofx = fem.locate_dofs_geometrical((V.sub(0), V0), left_corner)
        dofy = fem.locate_dofs_geometrical((V.sub(1), V1), left_corner)

        indexmap = V.dofmap.index_map
        bs = indexmap.block_size
        global_indices = indexmap.global_indices(False)
        lmin = bs*indexmap.local_range[0]
        lmax = bs*indexmap.local_range[1]
        if (len(dofx) > 0) and (lmin <= global_indices[dofx[0, 0]] < lmax):
            s_side = np.array([global_indices[dofx[0, 0]]], dtype=np.int64)
            m_side = np.array([global_indices[dofy[0, 0]]], dtype=np.int64)
            o_side = np.array([len(masters) + 1], dtype=np.int32)
            c_side = np.array(
                [-t_vec[dofy[0, 0]]/t_vec[dofx[0, 0]]], dtype=np.float64)
            o_r_side = np.array([comm.rank], dtype=np.int32)
        else:
            s_side = np.array([], dtype=np.int64)
            m_side = np.array([], dtype=np.int64)
            o_side = np.array([], dtype=np.int32)
            c_side = np.array([], dtype=np.float64)
            o_r_side = np.array([], dtype=np.int32)

        s_side_g = np.hstack(comm.allgather(s_side))
        m_side_g = np.hstack(comm.allgather(m_side))
        o_side_g = np.hstack(comm.allgather(o_side))
        c_side_g = np.hstack(comm.allgather(c_side))
        o_r_side_g = np.hstack(comm.allgather(o_r_side))
        masters = np.append(masters, m_side_g)
        slaves = np.append(slaves, s_side_g)
        coeffs = np.append(coeffs, c_side_g)
        offsets = np.append(offsets, o_side_g)
        owner_ranks = np.append(owner_ranks, o_r_side_g)
        assert(len(masters) == len(owner_ranks))
        assert(len(slaves) == len(offsets)-1)
        assert(not np.all(np.isin(slaves, masters)))

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets,
                                                   owner_ranks)

    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
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
    opts["pc_gamg_square_graph"] = 2
    opts["pc_gamg_threshold"] = 0.02
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    # Create functionspace and build near nullspace
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
    null_space = build_elastic_nullspace(Vmpc)
    A.setNearNullSpace(null_space)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))
    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)
    unorm = uh.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(unorm)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc_" + ext
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = fem.assemble_matrix(a, bcs)

    A_org.assemble()
    L_org = fem.assemble_vector(lhs)
    fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L_org, bcs)

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
                                                       masters, coeffs,
                                                       offsets)
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
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    print("DIFF", max(abs(uh.array - uh_numpy[uh.owner_range[0]:
                                              uh.owner_range[1]])))
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                          uh.owner_range[1]], atol=1e-7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=1, type=np.int8, dest="case",
                        help="Which testcase:")
    args = parser.parse_args()
    case = args.case
    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                  "results/rotated_cube.xdmf", "w")
    if case == 1:
        demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=True)
    elif case == 2:
        demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=False)
    elif case == 3:
        demo_stacked_cubes(outfile, theta=0, gmsh=True)
    elif case == 4:
        demo_stacked_cubes(outfile, theta=np.pi/7, gmsh=True)
    elif case == 5:
        demo_stacked_cubes(outfile, theta=np.pi/5, gmsh=True)
    elif case == 6:
        demo_stacked_cubes(outfile, theta=np.pi/7, gmsh=False, triangle=False)
    elif case == 7:
        demo_stacked_cubes(outfile, theta=np.pi/5, gmsh=False, triangle=False)
    else:
        raise ValueError("Cases are 1 to 7")
    outfile.close()

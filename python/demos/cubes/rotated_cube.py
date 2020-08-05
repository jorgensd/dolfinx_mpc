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
from create_and_export_mesh import mesh_2D_dolfin, mesh_2D_gmsh

dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions

comm = MPI.COMM_WORLD


def demo_stacked_cubes(outfile, theta, gmsh=True, triangle=True):
    if MPI.COMM_WORLD.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run theta:{0:.2f}, Triangle: {1:b}, Gmsh {2:b}"
                        .format(theta, triangle, gmsh))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    if gmsh:
        if triangle:
            celltype = "triangle"
        else:
            celltype = "quad"
        mesh_name = "Grid"
        if MPI.COMM_WORLD.size == 1:
            mesh_2D_gmsh(theta, celltype)
        filename = "meshes/mesh_{0:s}_{1:.2f}_gmsh.xdmf".format(
            celltype, theta)
        facet_file = "meshes/facet_{0:s}_{1:.2f}_rot.xdmf".format(
            celltype, theta)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_{0:s}_{1:.2f}_gmsh".format(celltype, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, "Grid")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 facet_file, "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, name="Grid")
    else:
        if triangle:
            celltype = "triangle"
        else:
            celltype = "quadrilateral"
        mesh_name = "mesh"
        filename = "meshes/mesh_{0:s}_{1:.2f}.xdmf".format(celltype, theta)
        if triangle:
            if MPI.COMM_WORLD.size == 1:
                mesh_2D_dolfin(celltype, theta)
        else:
            if MPI.COMM_WORLD.size == 1:
                mesh_2D_dolfin(celltype, theta)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_{0:s}_{1:.2f}".format(celltype, theta)
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
    # Find slave master relationship and initialize MPC class
    with dolfinx.common.Timer("~Contact: Create old constraint"):
        (slaves, masters, coeffs, offsets,
         owner_ranks) = dolfinx_mpc.create_collision_constraint(
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
                c_side = np.array([-t_vec[dofy[0, 0]]/t_vec[dofx[0, 0]]],
                                  dtype=PETSc.ScalarType)
                o_r_side = np.array([comm.rank], dtype=np.int32)
            else:
                s_side = np.array([], dtype=np.int64)
                m_side = np.array([], dtype=np.int64)
                o_side = np.array([], dtype=np.int32)
                c_side = np.array([], dtype=PETSc.ScalarType)
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
                                                       masters, coeffs,
                                                       offsets,
                                                       owner_ranks)
    with dolfinx.common.Timer("~Contact: Create new constraint"):
        cc = dolfinx_mpc.create_contact_condition(V, mt, 4, 9)

    # Setup MPC system
    with dolfinx.common.Timer("~Contact: Assemble old"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(lhs, mpc)
    with dolfinx.common.Timer("~Contact: Assemble old comp"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    with dolfinx.common.Timer("~Contact: Assemble new"):
        Acc = dolfinx_mpc.assemble_matrix_local(a, cc, bcs=bcs)
        bcc = dolfinx_mpc.assemble_vector_local(lhs, cc)
    with dolfinx.common.Timer("~Contact: Assemble new comp"):
        Acc = dolfinx_mpc.assemble_matrix_local(a, cc, bcs=bcs)
        bcc = dolfinx_mpc.assemble_vector_local(lhs, cc)
    fem.apply_lifting(bcc, [a], [bcs])
    bcc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(bcc, bcs)

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
    null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vmpc)
    A.setNearNullSpace(null_space)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    # New implementation

    Vcc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                 cc.dofmap())
    Vcc = dolfinx.FunctionSpace(None, V.ufl_element(), Vcc_cpp)
    null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vcc)
    Acc.setNearNullSpace(null_space)
    solver.setOperators(Acc)
    uhcc = bcc.copy()
    uhcc.set(0)
    solver.solve(bcc, uhcc)
    uhcc.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)
    dolfinx_mpc.backsubstitution_local(cc, uhcc)
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    unorm = uhcc.norm()
    uoldnorm = uh.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(unorm, uoldnorm)

    # Write solution to file
    ext = "gmsh" if gmsh else ""
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc_{0:s}_{1:.2f}_{2:s}".format(celltype, theta, ext)

    u_hcc = dolfinx.Function(Vcc)
    u_hcc.vector.setArray(uhcc.array)
    u_hcc.name = "u_cc_{0:s}_{1:.2f}_{2:s}".format(celltype, theta, ext)

    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))
    outfile.write_function(u_hcc, 0.0,
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

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    dolfinx_mpc.utils.log_info(
        "Solving reference problem with global matrix (using numpy)")

    with dolfinx.common.Timer("~MPC: Reference problem"):
        # Generate reference matrices and unconstrained solution
        A_org = fem.assemble_matrix(a, bcs)

        A_org.assemble()
        L_org = fem.assemble_vector(lhs)
        fem.apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)

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

    with dolfinx.common.Timer("~MPC: Compare"):
        # Compare LHS, RHS and solution with reference values
        dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, cc)
        dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, cc)
        print("DIFF", max(abs(uhcc.array - uh_numpy[uhcc.owner_range[0]:
                                                    uhcc.owner_range[1]])))
        assert np.allclose(uhcc.array, uh_numpy[uhcc.owner_range[0]:
                                                uhcc.owner_range[1]],
                           atol=1e-7)


if __name__ == "__main__":
    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                  "results/rotated_cube.xdmf", "w")
    # Built in meshes aligned with coordinate system
    demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=True)
    demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=False)
    # Built in meshes non-aligned
    demo_stacked_cubes(outfile, theta=np.pi/3, gmsh=False, triangle=True)
    demo_stacked_cubes(outfile, theta=np.pi/3, gmsh=False, triangle=False)
    # Gmsh aligned
    demo_stacked_cubes(outfile, theta=0, gmsh=True, triangle=False)
    demo_stacked_cubes(outfile, theta=0, gmsh=True, triangle=True)
    # Gmsh non-aligned
    demo_stacked_cubes(outfile, theta=np.pi/5, gmsh=True, triangle=False)
    demo_stacked_cubes(outfile, theta=np.pi/5, gmsh=True, triangle=True)

    outfile.close()
    dolfinx.common.list_timings(
        MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
import dolfinx.geometry as geometry
import dolfinx.fem as fem
import dolfinx
import dolfinx.io
import dolfinx.log
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from create_and_export_mesh import mesh_2D_rot, mesh_2D_dolfin
from helpers import find_master_slave_relationship

dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def demo_stacked_cubes(outfile, theta, gmsh=True, triangle=True):
    if gmsh:
        mesh_name = "Grid"
        if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
            mesh_2D_rot(theta)
        filename = "meshes/mesh_rot.xdmf"
        facet_file = "meshes/facet_rot.xdmf"
        ext = "gmsh" + "{0:.2f}".format(theta)
        with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(mesh_name)
            mesh.name = "mesh_" + ext
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.create_connectivity(tdim, tdim)
            mesh.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, "Grid")
        with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                                 facet_file, "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, "Grid")
    else:
        mesh_name = "mesh"
        if triangle:
            if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
                mesh_2D_dolfin("tri", theta)
            filename = "meshes/mesh_tri.xdmf"
            ext = "tri" + "{0:.2f}".format(theta)
        else:
            if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
                mesh_2D_dolfin("quad", theta)
            filename = "meshes/mesh_quad.xdmf"
            ext = "quad" + "{0:.2f}".format(theta)
        with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(mesh_name)

            mesh.name = "mesh_" + ext
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.create_connectivity(tdim, tdim)
            mesh.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, "mesh_tags")
            mt = xdmf.read_meshtags(mesh, "facet_tags")
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
    slaves, masters, coeffs, offsets = find_master_slave_relationship(
        V, (mt, 4), (ct, 2))

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

        dofx = fem.locate_dofs_geometrical((V.sub(0), V0), left_corner)[0, 0]
        dofy = fem.locate_dofs_geometrical((V.sub(1), V1), left_corner)[0, 0]
        s_side = np.array([dofx], dtype=np.int64)
        m_side = np.array([dofy], dtype=np.int64)
        o_side = len(masters) + 1
        c_side = np.array([-t_vec[dofy]/t_vec[dofx]])
        masters = np.append(masters, m_side)
        slaves = np.append(slaves, s_side)
        coeffs = np.append(coeffs, c_side)
        offsets = np.append(offsets, o_side)
        assert(len(slaves) == len(offsets)-1)
        assert(not np.all(np.isin(slaves, masters)))

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)

    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

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
    K = dolfinx_mpc.utils.create_transformation_matrix(V.dim(), slaves,
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
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                          uh.owner_range[1]])
    print(uh.norm())


if __name__ == "__main__":
    outfile = dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                                  "results/rotated_cube.xdmf", "w")
    demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=True)
    demo_stacked_cubes(outfile, theta=0, gmsh=True)
    demo_stacked_cubes(outfile, theta=np.pi/7, gmsh=True)
    demo_stacked_cubes(outfile, theta=0, gmsh=False, triangle=False)
    demo_stacked_cubes(outfile, theta=np.pi/7, gmsh=False, triangle=False)

    outfile.close()

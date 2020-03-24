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
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from petsc4py import PETSc
from create_and_export_mesh import mesh_2D_rot, mesh_2D_dolfin
from helpers import find_master_slave_relationship

comp_col_pts = geometry.compute_collisions_point
get_basis = dolfinx_mpc.cpp.mpc.get_basis_functions


def find_line_function(p0, p1):
    """
    Find line y=ax+b for each of the lines in the mesh
    https://mathworld.wolfram.com/Two-PointForm.html
    """
    # Line aligned with y axis
    if np.isclose(p1[0], p0[0]):
        return lambda x: np.isclose(x[0], p0[0])
    return lambda x: np.isclose(x[1],
                                p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0]))


def over_line(p0, p1):
    """
    Check if a point is over or under y=ax+b for each of the lines in the mesh
    https://mathworld.wolfram.com/Two-PointForm.html
    """
    return lambda x: x[1] > p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0])


def demo_stacked_cubes(theta, gmsh=True, triangle=True):
    if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
        if gmsh:
            mesh_2D_rot(theta)
            filename = "meshes/mesh_rot.xdmf"
        else:
            if triangle:
                mesh_2D_dolfin("tri")
                filename = "meshes/mesh_tri.xdmf"
            else:
                mesh_2D_dolfin("quad")
                filename = "meshes/mesh_quad.xdmf"

    r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [1, 1, 0], [0, 1, 0]]).T)
    bottom = find_line_function(bottom_points[:, 0], bottom_points[:, 1])
    interface = find_line_function(bottom_points[:, 2], bottom_points[:, 3])
    top_points = np.dot(r_matrix, np.array([[0, 1, 0], [1, 1, 0],
                                            [1, 2, 0], [0, 2, 0]]).T)
    top = find_line_function(top_points[:, 2], top_points[:, 3])
    top_cube = over_line(bottom_points[:, 2], bottom_points[:, 3])

    left_side = find_line_function(top_points[:, 0], top_points[:, 3])
    right_side = find_line_function(top_points[:, 1], top_points[:, 2])

    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                             filename) as xdmf:
        mesh = xdmf.read_mesh()

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Traction on top of domain

    tdim = mesh.topology.dim
    cf = dolfinx.MeshFunction("size_t", mesh, tdim, 0)
    num_cells = mesh.num_entities(tdim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    top_cube_marker = 3
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            cf.values[cell_index] = top_cube_marker
    dolfinx.io.VTKFile("results/cf.pvd").write(cf)
    fdim = mesh.topology.dim - 1
    markers = {top: 3, interface: 4, bottom: 5, left_side: 6, right_side: 7}
    mf = dolfinx.MeshFunction("size_t", mesh, fdim, 0)
    for key in markers.keys():
        mf.mark(key, markers[key])

    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = dolfinx.Constant(mesh, g_vec[:2])

    # Define boundary conditions (HAS TO BE NON-MASTER NODES)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bottom_facets = np.flatnonzero(mf.values == markers[bottom])
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
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Create standard master slave relationsship
    slaves, masters, coeffs, offsets = find_master_slave_relationship(
        V, (mf, 4), (cf, 3))

    # Set normal sliding of a single on each side of the cube to zero to
    # avoid translational invariant problem
    if np.isclose(theta, 0):
        def in_corner(x):
            return np.logical_or(np.isclose(x.T, [0, 2, 0]).all(axis=1),
                                 np.isclose(x.T, [1, 2, 0]).all(axis=1))
        V0 = V.sub(0).collapse()
        zero = dolfinx.Function(V0)
        with zero.vector.localForm() as zero_local:
            zero_local.set(0.0)
        dofs = fem.locate_dofs_geometrical((V.sub(0), V0), in_corner)
        bc_corner = dolfinx.DirichletBC(zero, dofs, V.sub(0))
        bcs.append(bc_corner)
    else:
        global_indices = V.dofmap.index_map.global_indices(False)
        x_coords = V.tabulate_dof_coordinates()
        V0 = V.sub(0).collapse()
        V1 = V.sub(1).collapse()
        m_side, s_side, c_side, o_side = [], [], [], []
        for key, corners in zip([left_side, right_side], [[0, 3], [1, 2]]):
            side_facets = np.flatnonzero(mf.values == markers[key])

            dofs_x = fem.locate_dofs_topological(
                (V.sub(0), V0), fdim, side_facets)[:, 0]
            dofs_y = fem.locate_dofs_topological(
                (V.sub(1), V1), fdim, side_facets)[:, 0]

            top_cube_side_x = np.flatnonzero(top_cube(x_coords[dofs_x].T))
            top_cube_side_y = np.flatnonzero(top_cube(x_coords[dofs_y].T))
            slip = False
            n_side = dolfinx_mpc.facet_normal_approximation(
                V, mf, markers[key])
            n_side_vec = n_side.vector.getArray()

            for id_x in top_cube_side_x:
                x_dof = dofs_x[id_x]
                corner_1 = np.allclose(
                    x_coords[x_dof], top_points[:, corners[0]])
                corner_2 = np.allclose(
                    x_coords[x_dof], top_points[:, corners[1]])
                if corner_1 or corner_2:
                    continue
                for id_y in top_cube_side_y:
                    y_dof = dofs_y[id_y]
                    same_coord = np.allclose(x_coords[x_dof], x_coords[y_dof])
                    corner_1 = np.allclose(
                        x_coords[y_dof], top_points[:, corners[0]])
                    corner_2 = np.allclose(
                        x_coords[y_dof], top_points[:, corners[1]])
                    coeff = -n_side_vec[y_dof] / n_side_vec[x_dof]
                    not_zero = not np.isclose(coeff, 0)
                    not_corners = not corner_1 and not corner_2
                    if not_corners and same_coord and not_zero:
                        s_side.append(global_indices[x_dof])
                        m_side.append(global_indices[y_dof])
                        c_side.append(coeff)

                        o_side.append(len(masters)+len(m_side))
                        slip = True
                        break
                if slip:
                    break
        m_side = np.array(m_side, dtype=np.int64)
        s_side = np.array(s_side, dtype=np.int64)
        o_side = np.array(o_side, dtype=np.int64)
        masters = np.append(masters, m_side)
        slaves = np.append(slaves, s_side)
        coeffs = np.append(coeffs, c_side)
        offsets = np.append(offsets, o_side)
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
    u_h.name = "u_mpc"
    dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                        "results/uh_rot.xdmf").write(u_h)

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
    demo_stacked_cubes(theta=0, gmsh=False, triangle=True)
    # FIXME: Does not work due to collision detection
    # demo_stacked_cubes(theta=0, gmsh=False, triangle=False)
    demo_stacked_cubes(theta=0, gmsh=True)
    demo_stacked_cubes(theta=np.pi/7, gmsh=True)

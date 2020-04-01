# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.
import dolfinx
import dolfinx.fem as fem
import dolfinx.io

import dolfinx_mpc
import dolfinx_mpc.utils

import numpy as np
import pygmsh
import ufl

from petsc4py import PETSc

from create_and_export_mesh import mesh_3D_rot
from helpers import find_master_slave_relationship


def find_plane_function(p0, p1, p2):
    """
    Find plane function given three points:
    http://www.nabla.hr/CG-LinesPlanesIn3DA3.htm
    """
    v1 = np.array(p1)-np.array(p0)
    v2 = np.array(p2)-np.array(p0)

    n = np.cross(v1, v2)
    D = -(n[0]*p0[0]+n[1]*p0[1]+n[2]*p0[2])
    return lambda x: np.isclose(0,
                                np.dot(n, x) + D)


def over_plane(p0, p1, p2):
    """
    Returns function that checks if a point is over a plane defined
    by the points p0, p1 and p2.
    """
    v1 = np.array(p1)-np.array(p0)
    v2 = np.array(p2)-np.array(p0)

    n = np.cross(v1, v2)
    D = -(n[0]*p0[0]+n[1]*p0[1]+n[2]*p0[2])
    return lambda x: n[0]*x[0] + n[1]*x[1] + D > -n[2]*x[2]


def demo_stacked_cubes(theta):
    # Create rotated mesh
    if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
        mesh_3D_rot(theta)

    # Read in mesh
    with dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                             "meshes/mesh3D_rot.xdmf") as xdmf:
        mesh = xdmf.read_mesh()
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Helper until MeshTags can be read in from xdmf
    r_matrix = pygmsh.helpers.rotation_matrix(
        [0, 1/np.sqrt(2), 1/np.sqrt(2)], -theta)

    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [1, 1, 0], [0, 1, 0]]).T)

    bottom = find_plane_function(
        bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2])
    bottom_facets = dolfinx.mesh.locate_entities_geometrical(
        mesh, fdim, bottom, boundary_only=True)

    interface_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1],
                                                  [1, 1, 1], [0, 1, 1]]).T)
    interface = find_plane_function(
        interface_points[:, 0], interface_points[:, 1], interface_points[:, 2])
    i_facets = dolfinx.mesh.locate_entities_geometrical(
        mesh, fdim, interface, boundary_only=True)

    top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2],
                                            [1, 1, 2], [0, 1, 2]]).T)
    top = find_plane_function(
        top_points[:, 0], top_points[:, 1], top_points[:, 2])
    top_facets = dolfinx.mesh.locate_entities_geometrical(
        mesh, fdim, top, boundary_only=True)

    left_points = np.dot(r_matrix, np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]]).T)
    left_side = find_plane_function(
        left_points[:, 0], left_points[:, 1], left_points[:, 2])
    left_facets = dolfinx.mesh.locate_entities_geometrical(
        mesh, fdim, left_side, boundary_only=True)

    top_cube = over_plane(
        interface_points[:, 0], interface_points[:, 1], interface_points[:, 2])

    top_cube_marker = 3
    indices = []
    values = []
    tdim = mesh.topology.dim
    num_cells = mesh.num_entities(tdim)
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            indices.append(cell_index)
            values.append(top_cube_marker)
    ct = dolfinx.mesh.MeshTags(mesh, fdim, np.array(
        indices, dtype=np.intc), np.array(values, dtype=np.intc))

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Create meshtags for facet data
    markers = {3: top_facets, 4: i_facets,
               5: bottom_facets, 6: left_facets}
    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)

    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key,
                                           dtype=np.intc))
    mt = dolfinx.mesh.MeshTags(mesh, fdim,
                               indices, values)

    # dolfinx.io.VTKFile("mt.pvd").write(mt)
    g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
    g = dolfinx.Constant(mesh, g_vec)

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    # Top boundary has a given deformation normal to the interface
    g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
    g = dolfinx.Constant(mesh, g_vec)

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = dolfinx.function.Function(V)
    u_top.interpolate(top_v)

    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.DirichletBC(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

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
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    # Find slave master relationship and initialize MPC class
    slaves, masters, coeffs, offsets = find_master_slave_relationship(
        V, (mt, 4), (ct, 3))
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
                        "results/uh3D_rot.xdmf").write(u_h)

    # Transfer data from the MPC problem to numpy arrays for comparison
    A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    print("Solving reference problem with global matrix (using numpy)")
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
    demo_stacked_cubes(theta=np.pi/3)

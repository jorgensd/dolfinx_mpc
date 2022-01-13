# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.

import dolfinx.fem as fem
import dolfinx_mpc
import dolfinx_mpc.utils
import gmsh
import numpy as np
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx_mpc.utils import get_assemblers  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc

theta = np.pi / 5


@pytest.fixture
def generate_hex_boxes():
    """
    Generate the stacked boxes [x0,y0,z0]x[y1,y1,z1] and
    [x0,y0,z1] x [x1,y1,z2] with different resolution in each box.
    The markers are is a list of arrays containing markers
    array of markers for [back, bottom, right, left, top, front] per box
    volume_markers a list of marker per volume
    """

    res = 0.2
    x0, y0, z0, x1, y1, z1, z2 = 0, 0, 0, 1, 1, 1, 2
    facet_markers = [[11, 5, 12, 13, 4, 14], [21, 9, 22, 23, 3, 24]]
    volume_markers = [1, 2]
    r_matrix = dolfinx_mpc.utils.rotation_matrix([1, 1, 0], -theta)

    # Check if GMSH is initialized
    gmsh.initialize()
    gmsh.clear()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("General.Terminal", 0)
        bottom = gmsh.model.occ.addRectangle(x0, y0, z0, x1 - x0, y1 - y0)
        top = gmsh.model.occ.addRectangle(x0, y0, z2, x1 - x0, y1 - y0)

        # Set mesh size at point
        gmsh.model.occ.extrude([(2, bottom)], 0, 0, z1 - z0,
                               numElements=[int(1 / (2 * res))], recombine=True)
        gmsh.model.occ.extrude([(2, top)], 0, 0, z1 - z2 - 1e-12,
                               numElements=[int(1 / (2 * res))], recombine=True)
        # Syncronize to be able to fetch entities
        gmsh.model.occ.synchronize()

        # Create entity -> marker map (to be used after rotation)
        volumes = gmsh.model.getEntities(3)
        volume_entities = {"Top": [None, volume_markers[1]],
                           "Bottom": [None, volume_markers[0]]}
        for i, volume in enumerate(volumes):
            com = gmsh.model.occ.getCenterOfMass(volume[0], volume[1])
            if np.isclose(com[2], (z1 - z0) / 2):
                bottom_index = i
                volume_entities["Bottom"][0] = volume
            elif np.isclose(com[2], (z2 - z1) / 2 + z1):
                top_index = i
                volume_entities["Top"][0] = volume
        surfaces = ["Top", "Bottom", "Left", "Right", "Front", "Back"]
        entities = {"Bottom": {key: [[], None] for key in surfaces},
                    "Top": {key: [[], None] for key in surfaces}}
        # Identitfy entities for each surface of top and bottom cube

        # Physical markers for bottom cube
        bottom_surfaces = gmsh.model.getBoundary([volumes[bottom_index]],
                                                 recursive=False, oriented=False)
        for entity in bottom_surfaces:
            com = gmsh.model.occ.getCenterOfMass(entity[0], entity[1])
            if np.allclose(com, [(x1 - x0) / 2, y1, (z1 - z0) / 2]):
                entities["Bottom"]["Back"][0].append(entity[1])
                entities["Bottom"]["Back"][1] = facet_markers[0][0]
            elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z0]):
                entities["Bottom"]["Bottom"][0].append(entity[1])
                entities["Bottom"]["Bottom"][1] = facet_markers[0][1]
            elif np.allclose(com, [x1, (y1 - y0) / 2, (z1 - z0) / 2]):
                entities["Bottom"]["Right"][0].append(entity[1])
                entities["Bottom"]["Right"][1] = facet_markers[0][2]
            elif np.allclose(com, [x0, (y1 - y0) / 2, (z1 - z0) / 2]):
                entities["Bottom"]["Left"][0].append(entity[1])
                entities["Bottom"]["Left"][1] = facet_markers[0][3]
            elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z1]):
                entities["Bottom"]["Top"][0].append(entity[1])
                entities["Bottom"]["Top"][1] = facet_markers[0][4]
            elif np.allclose(com, [(x1 - x0) / 2, y0, (z1 - z0) / 2]):
                entities["Bottom"]["Front"][0].append(entity[1])
                entities["Bottom"]["Front"][1] = facet_markers[0][5]
        # Physical markers for top
        top_surfaces = gmsh.model.getBoundary([volumes[top_index]],
                                              recursive=False, oriented=False)
        for entity in top_surfaces:
            com = gmsh.model.occ.getCenterOfMass(entity[0], entity[1])
            if np.allclose(com, [(x1 - x0) / 2, y1, (z2 - z1) / 2 + z1]):
                entities["Top"]["Back"][0].append(entity[1])
                entities["Top"]["Back"][1] = facet_markers[1][0]
            elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z1]):
                entities["Top"]["Bottom"][0].append(entity[1])
                entities["Top"]["Bottom"][1] = facet_markers[1][1]
            elif np.allclose(com, [x1, (y1 - y0) / 2, (z2 - z1) / 2 + z1]):
                entities["Top"]["Right"][0].append(entity[1])
                entities["Top"]["Right"][1] = facet_markers[1][2]
            elif np.allclose(com, [x0, (y1 - y0) / 2, (z2 - z1) / 2 + z1]):
                entities["Top"]["Left"][0].append(entity[1])
                entities["Top"]["Left"][1] = facet_markers[1][3]
            elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z2]):
                entities["Top"]["Top"][0].append(entity[1])
                entities["Top"]["Top"][1] = facet_markers[1][4]
            elif np.allclose(com, [(x1 - x0) / 2, y0, (z2 - z1) / 2 + z1]):
                entities["Top"]["Front"][0].append(entity[1])
                entities["Top"]["Front"][1] = facet_markers[1][5]

        # gmsh.model.occ.rotate(volumes, 0, 0, 0,
        #                       1 / np.sqrt(2), 1 / np.sqrt(2), 0, theta)
        # Note: Rotation cannot be used on recombined surfaces
        gmsh.model.occ.synchronize()

        for volume in volume_entities.keys():
            gmsh.model.addPhysicalGroup(
                volume_entities[volume][0][0], [volume_entities[volume][0][1]],
                tag=volume_entities[volume][1])
            gmsh.model.setPhysicalName(volume_entities[volume][0][0],
                                       volume_entities[volume][1], volume)

        for box in entities.keys():
            for surface in entities[box].keys():
                gmsh.model.addPhysicalGroup(2, entities[box][surface][0],
                                            tag=entities[box][surface][1])
                gmsh.model.setPhysicalName(2, entities[box][surface][1], box
                                           + ":" + surface)
        # Set mesh sizes on the points from the surface we are extruding
        bottom_nodes = gmsh.model.getBoundary([(2, bottom)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(bottom_nodes, res)
        top_nodes = gmsh.model.getBoundary([(2, top)],
                                           recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(top_nodes, 2 * res)
        # NOTE: Need to synchronize after setting mesh sizes
        gmsh.model.occ.synchronize()
        # Generate mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
    mesh, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, facet_data=True)
    gmsh.clear()
    gmsh.finalize()
    # NOTE: Hex mesh must be rotated after generation due to gmsh API
    mesh.geometry.x[:] = np.dot(r_matrix, mesh.geometry.x.T).T
    return (mesh, ft)


@pytest.mark.parametrize("get_assemblers", ["C++", "numba"], indirect=True)
@pytest.mark.parametrize("nonslip", [True, False])
def test_cube_contact(generate_hex_boxes, nonslip, get_assemblers):  # noqa: F811
    assemble_matrix, assemble_vector = get_assemblers
    comm = MPI.COMM_WORLD
    root = 0
    # Generate mesh
    mesh_data = generate_hex_boxes
    mesh, mt = mesh_data
    fdim = mesh.topology.dim - 1
    # Create functionspaces
    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Helper for orienting traction

    # Bottom boundary is fixed in all directions
    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.dirichletbc(u_bc, bottom_dofs)

    g_vec = [0, 0, -4.25e-1]
    if not nonslip:
        # Helper for orienting traction
        r_matrix = dolfinx_mpc.utils.rotation_matrix(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])

    # Top boundary has a given deformation normal to the interface
    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = fem.Function(V)
    u_top.interpolate(top_v)

    top_facets = mt.indices[np.flatnonzero(mt.values == 3)]
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.dirichletbc(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e3)
    nu = 0
    mu = fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v))
                + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(fem.Constant(mesh, PETSc.ScalarType((0, 0, 0))), v) * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(rhs)

    # Create LU solver
    solver = PETSc.KSP().create(comm)
    solver.setType("preonly")
    solver.setTolerances(rtol=1.0e-14)
    solver.getPC().setType("lu")

    # Create MPC contact condition and assemble matrices
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    if nonslip:
        with Timer("~Contact: Create non-elastic constraint"):
            mpc.create_contact_inelastic_condition(mt, 4, 9)
    else:
        with Timer("~Contact: Create contact constraint"):
            nh = dolfinx_mpc.utils.create_normal_approximation(V, mt, 4)
            mpc.create_contact_slip_condition(mt, 4, 9, nh)

    mpc.finalize()

    with Timer("~TEST: Assemble bilinear form"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    with Timer("~TEST: Assemble vector"):
        b = assemble_vector(linear_form, mpc)

    dolfinx_mpc.apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    with Timer("~MPC: Solve"):
        solver.setOperators(A)
        uh = b.copy()
        uh.set(0)
        solver.solve(b, uh)

    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    # Write solution to file
    # u_h = fem.Function(mpc.function_space)
    # u_h.vector.setArray(uh.array)
    # u_h.x.scatter_forward()
    # u_h.name = "u_{0:.2f}".format(theta)
    # import dolfinx.io as io
    # with io.XDMFFile(comm, "output/rotated_cube3D.xdmf", "w") as outfile:
    #     outfile.write_mesh(mesh)
    #     outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    dolfinx_mpc.utils.log_info("Solving reference problem with global matrix (using numpy)")

    with Timer("~TEST: Assemble bilinear form (unconstrained)"):
        A_org = fem.assemble_matrix(bilinear_form, bcs)
        A_org.assemble()
        L_org = fem.assemble_vector(linear_form)
        fem.apply_lifting(L_org, [bilinear_form], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)

    with Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_mpc_lhs(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_mpc_rhs(L_org, b, mpc, root=root)

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

    list_timings(comm, [TimingType.wall])

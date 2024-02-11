from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from mpi4py import MPI

import dolfinx.common as _common
import dolfinx.cpp as _cpp
import dolfinx.io as _io
import dolfinx.mesh as _mesh
import gmsh
import numpy as np
import ufl
from basix.ufl import element
from dolfinx.io import gmshio

import dolfinx_mpc.utils as _utils


def gmsh_3D_stacked(celltype: str, theta: float, res: float = 0.1,
                    verbose: bool = False) -> Tuple[_mesh.Mesh, _mesh.MeshTags]:
    if celltype == "tetrahedron":
        mesh, ft = generate_tet_boxes(0, 0, 0, 1, 1, 1, 2, res,
                                      facet_markers=[[11, 5, 12, 13, 4, 14],
                                                     [21, 9, 22, 23, 3, 24]],
                                      volume_markers=[1, 2], verbose=verbose)
    else:
        mesh, ft = generate_hex_boxes(0, 0, 0, 1, 1, 1, 2, res,
                                      facet_markers=[[11, 5, 12, 13, 4, 14],
                                                     [21, 9, 22, 23, 3, 24]],
                                      volume_markers=[1, 2], verbose=verbose)

    r_matrix = _utils.rotation_matrix([1, 1, 0], -theta)
    mesh.geometry.x[:] = np.dot(r_matrix, mesh.geometry.x.T).T
    return mesh, ft


def tag_cube_model(model: gmsh.model, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float,
                   z2: float, facet_markers: Sequence[Sequence[int]], volume_markers: Sequence[int]):
    """
    Helper function to tag a cube and its faces
    """
    # Create entity -> marker map (to be used after rotation)
    volumes = model.getEntities(3)
    volume_entities: Dict[str, List[int]] = {"Top": [-1, volume_markers[1]],
                                             "Bottom": [-1, volume_markers[0]]}
    for i, volume in enumerate(volumes):
        com = model.occ.getCenterOfMass(volume[0], np.abs(volume[1]))
        if np.isclose(com[2], (z1 - z0) / 2):
            bottom_index = i
            volume_entities["Bottom"][0] = volume[1]
        elif np.isclose(com[2], (z2 - z1) / 2 + z1):
            top_index = i
            volume_entities["Top"][0] = volume[1]

    surfaces = ["Top", "Bottom", "Left", "Right", "Front", "Back"]
    entities: Dict[str, Dict[str, List[List[int]]]] = {"Bottom": {key: [[], []] for key in surfaces},
                                                       "Top": {key: [[], []] for key in surfaces}}

    # Identitfy entities for each surface of top and bottom cube

    # Physical markers for bottom cube
    bottom_surfaces = model.getBoundary([volumes[bottom_index]],
                                        oriented=False, recursive=False)
    for entity in bottom_surfaces:
        com = model.occ.getCenterOfMass(entity[0], entity[1])
        if np.allclose(com, [(x1 - x0) / 2, y1, (z1 - z0) / 2]):
            entities["Bottom"]["Back"][0].append(entity[1])
            entities["Bottom"]["Back"][1] = [facet_markers[0][0]]
        elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z0]):
            entities["Bottom"]["Bottom"][0].append(entity[1])
            entities["Bottom"]["Bottom"][1] = [facet_markers[0][1]]
        elif np.allclose(com, [x1, (y1 - y0) / 2, (z1 - z0) / 2]):
            entities["Bottom"]["Right"][0].append(entity[1])
            entities["Bottom"]["Right"][1] = [facet_markers[0][2]]
        elif np.allclose(com, [x0, (y1 - y0) / 2, (z1 - z0) / 2]):
            entities["Bottom"]["Left"][0].append(entity[1])
            entities["Bottom"]["Left"][1] = [facet_markers[0][3]]
        elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z1]):
            entities["Bottom"]["Top"][0].append(entity[1])
            entities["Bottom"]["Top"][1] = [facet_markers[0][4]]
        elif np.allclose(com, [(x1 - x0) / 2, y0, (z1 - z0) / 2]):
            entities["Bottom"]["Front"][0].append(entity[1])
            entities["Bottom"]["Front"][1] = [facet_markers[0][5]]
    # Physical markers for top
    top_surfaces = model.getBoundary([volumes[top_index]],
                                     oriented=False, recursive=False)
    for entity in top_surfaces:
        com = model.occ.getCenterOfMass(entity[0], entity[1])
        if np.allclose(com, [(x1 - x0) / 2, y1, (z2 - z1) / 2 + z1]):
            entities["Top"]["Back"][0].append(entity[1])
            entities["Top"]["Back"][1] = [facet_markers[1][0]]
        elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z1]):
            entities["Top"]["Bottom"][0].append(entity[1])
            entities["Top"]["Bottom"][1] = [facet_markers[1][1]]
        elif np.allclose(com, [x1, (y1 - y0) / 2, (z2 - z1) / 2 + z1]):
            entities["Top"]["Right"][0].append(entity[1])
            entities["Top"]["Right"][1] = [facet_markers[1][2]]
        elif np.allclose(com, [x0, (y1 - y0) / 2, (z2 - z1) / 2 + z1]):
            entities["Top"]["Left"][0].append(entity[1])
            entities["Top"]["Left"][1] = [facet_markers[1][3]]
        elif np.allclose(com, [(x1 - x0) / 2, (y1 - y0) / 2, z2]):
            entities["Top"]["Top"][0].append(entity[1])
            entities["Top"]["Top"][1] = [facet_markers[1][4]]
        elif np.allclose(com, [(x1 - x0) / 2, y0, (z2 - z1) / 2 + z1]):
            entities["Top"]["Front"][0].append(entity[1])
            entities["Top"]["Front"][1] = [facet_markers[1][5]]

    # Note: Rotation cannot be used on recombined surfaces
    model.occ.synchronize()

    for volume in volume_entities.keys():
        model.addPhysicalGroup(3, [volume_entities[volume][0]],
                               tag=volume_entities[volume][1])
        model.setPhysicalName(3, volume_entities[volume][0], volume)

    for box in entities.keys():
        for surface in entities[box].keys():
            model.addPhysicalGroup(2, entities[box][surface][0],
                                   tag=entities[box][surface][1][0])
            model.setPhysicalName(2, entities[box][surface][1][0], box
                                  + ":" + surface)


def generate_tet_boxes(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float,
                       z2: float, res: float, facet_markers: Sequence[Sequence[int]],
                       volume_markers: Sequence[int],
                       verbose: bool = False) -> Tuple[_mesh.Mesh, _mesh.MeshTags]:
    """
    Generate the stacked boxes [x0,y0,z0]x[y1,y1,z1] and
    [x0,y0,z1] x [x1,y1,z2] with different resolution in each box.
    The markers are is a list of arrays containing markers
    array of markers for [back, bottom, right, left, top, front] per box
    volume_markers a list of marker per volume
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    if MPI.COMM_WORLD.rank == 0:
        gmsh.clear()
        # NOTE: Have to reset this until:
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1001
        # is in master
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

        # Added tolerance to ensure that gmsh separates boxes
        tol = 1e-12
        gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        gmsh.model.occ.addBox(x0, y0, z1 + tol, x1 - x0, y1 - y0, z2 - z1)

        # Syncronize to be able to fetch entities
        gmsh.model.occ.synchronize()

        tag_cube_model(gmsh.model, x0, y0, z0, x1, y1, z1, z2, facet_markers, volume_markers)

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", res)
        gmsh.model.mesh.field.setNumber(1, "VOut", 2 * res)
        gmsh.model.mesh.field.setNumber(1, "XMin", 0)
        gmsh.model.mesh.field.setNumber(1, "XMax", 1)
        gmsh.model.mesh.field.setNumber(1, "YMin", 0)
        gmsh.model.mesh.field.setNumber(1, "YMax", 1)
        gmsh.model.mesh.field.setNumber(1, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", 1)

        gmsh.model.mesh.field.setAsBackgroundMesh(1)
        # NOTE: Need to synchronize after setting mesh sizes
        gmsh.model.occ.synchronize()
        # Generate mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)
    gmsh.finalize()
    return mesh, ft


def generate_hex_boxes(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, z2: float, res: float,
                       facet_markers: Sequence[Sequence[int]], volume_markers: Sequence[int],
                       verbose: bool = False) -> Tuple[_mesh.Mesh, _mesh.MeshTags]:
    """
    Generate the stacked boxes [x0,y0,z0]x[y1,y1,z1] and
    [x0,y0,z1] x [x1,y1,z2] with different resolution in each box.
    The markers are is a list of arrays containing markers
    array of markers for [back, bottom, right, left, top, front] per box
    volume_markers a list of marker per volume
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))

    if MPI.COMM_WORLD.rank == 0:
        gmsh.clear()
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        bottom = gmsh.model.occ.addRectangle(x0, y0, z0, x1 - x0, y1 - y0)
        top = gmsh.model.occ.addRectangle(x0, y0, z2, x1 - x0, y1 - y0)

        # Set mesh size at point
        gmsh.model.occ.extrude([(2, bottom)], 0, 0, z1 - z0,
                               numElements=[int(1 / (2 * res))], recombine=True)
        gmsh.model.occ.extrude([(2, top)], 0, 0, z1 - z2 - 1e-12,
                               numElements=[int(1 / (2 * res))], recombine=True)
        # Syncronize to be able to fetch entities
        gmsh.model.occ.synchronize()

        # Tag faces and volume
        tag_cube_model(gmsh.model, x0, y0, z0, x1, y1, z1, z2, facet_markers, volume_markers)

        # Set mesh sizes on the points from the surface we are extruding
        bottom_nodes = gmsh.model.getBoundary([(2, bottom)], oriented=False, recursive=True)
        gmsh.model.occ.mesh.setSize(bottom_nodes, res)
        top_nodes = gmsh.model.getBoundary([(2, top)], oriented=False, recursive=True)
        gmsh.model.occ.mesh.setSize(top_nodes, 2 * res)
        # NOTE: Need to synchronize after setting mesh sizes
        gmsh.model.occ.synchronize()
        # Generate mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)
    gmsh.clear()
    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return mesh, ft


def gmsh_2D_stacked(celltype: str, theta: float, verbose: bool = False) -> Tuple[_mesh.Mesh, _mesh.MeshTags]:
    res = 0.1
    x0, y0, z0 = 0, 0, 0
    x1, y1 = 1, 1
    y2 = 2

    # Check if GMSH is initialized
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.clear()

    if MPI.COMM_WORLD.rank == 0:
        if celltype == "quadrilateral":
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            recombine = True
        else:
            recombine = False

        points = [gmsh.model.occ.addPoint(x0, y0, z0), gmsh.model.occ.addPoint(x1, y0, z0),
                  gmsh.model.occ.addPoint(x0, y2, z0), gmsh.model.occ.addPoint(x1, y2, z0)]
        bottom = gmsh.model.occ.addLine(points[0], points[1])
        top = gmsh.model.occ.addLine(points[2], points[3])
        gmsh.model.occ.extrude([(1, bottom)], 0, y1 - y0, 0, numElements=[int(1 / (res))], recombine=recombine)
        gmsh.model.occ.extrude([(1, top)], 0, y1 - y2 - 1e-12, 0, numElements=[int(1 / (2 * res))], recombine=recombine)
        # Syncronize to be able to fetch entities
        gmsh.model.occ.synchronize()

        # Create entity -> marker map (to be used after rotation)
        volumes = gmsh.model.getEntities(2)
        volume_entities = {"Top": [-1, 1], "Bottom": [-1, 2]}
        for i, volume in enumerate(volumes):
            com = gmsh.model.occ.getCenterOfMass(volume[0], volume[1])
            if np.isclose(com[1], (y1 - y0) / 2):
                bottom_index = i
                volume_entities["Bottom"][0] = volume[1]
            elif np.isclose(com[1], (y2 - y1) / 2 + y1):
                top_index = i
                volume_entities["Top"][0] = volume[1]
        surfaces = ["Top", "Bottom", "Left", "Right"]
        entities: Dict[str, Dict[str, List[List[int]]]] = {"Bottom": {key: [[], []] for key in surfaces},
                                                           "Top": {key: [[], []] for key in surfaces}}
        # Identitfy entities for each surface of top and bottom cube
        # Bottom cube: Top, Right, Bottom, Left
        # Top cube : Top, Right, Bottom, Left
        facet_markers = [[4, 7, 5, 6], [3, 12, 9, 13]]
        bottom_surfaces = gmsh.model.getBoundary([volumes[bottom_index]], oriented=False, recursive=False)

        for entity in bottom_surfaces:
            com = gmsh.model.occ.getCenterOfMass(entity[0], abs(entity[1]))
            if np.allclose(com, [(x1 - x0) / 2, y0, z0]):
                entities["Bottom"]["Bottom"][0].append(entity[1])
                entities["Bottom"]["Bottom"][1] = [facet_markers[0][2]]
            elif np.allclose(com, [x1, (y1 - y0) / 2, z0]):
                entities["Bottom"]["Right"][0].append(entity[1])
                entities["Bottom"]["Right"][1] = [facet_markers[0][1]]
            elif np.allclose(com, [x0, (y1 - y0) / 2, z0]):
                entities["Bottom"]["Left"][0].append(entity[1])
                entities["Bottom"]["Left"][1] = [facet_markers[0][3]]
            elif np.allclose(com, [(x1 - x0) / 2, y1, z0]):
                entities["Bottom"]["Top"][0].append(entity[1])
                entities["Bottom"]["Top"][1] = [facet_markers[0][0]]
        # Physical markers for top
        top_surfaces = gmsh.model.getBoundary([volumes[top_index]], oriented=False, recursive=False)
        for entity in top_surfaces:
            com = gmsh.model.occ.getCenterOfMass(entity[0], abs(entity[1]))
            if np.allclose(com, [(x1 - x0) / 2, y1, z0]):
                entities["Top"]["Bottom"][0].append(entity[1])
                entities["Top"]["Bottom"][1] = [facet_markers[1][2]]
            elif np.allclose(com, [x1, y1 + (y2 - y1) / 2, z0]):
                entities["Top"]["Right"][0].append(entity[1])
                entities["Top"]["Right"][1] = [facet_markers[1][1]]
            elif np.allclose(com, [x0, y1 + (y2 - y1) / 2, z0]):
                entities["Top"]["Left"][0].append(entity[1])
                entities["Top"]["Left"][1] = [facet_markers[1][3]]
            elif np.allclose(com, [(x1 - x0) / 2, y2, z0]):
                entities["Top"]["Top"][0].append(entity[1])
                entities["Top"]["Top"][1] = [facet_markers[1][0]]

        # Note: Rotation cannot be used on recombined surfaces
        gmsh.model.occ.synchronize()

        for volume in volume_entities.keys():
            gmsh.model.addPhysicalGroup(2, [volume_entities[volume][0]], tag=volume_entities[volume][1])
            gmsh.model.setPhysicalName(2, volume_entities[volume][1], volume)
        for box in entities.keys():
            for surface in entities[box].keys():
                gmsh.model.addPhysicalGroup(1, entities[box][surface][0], tag=entities[box][surface][1][0])
                gmsh.model.setPhysicalName(1, entities[box][surface][1][0], box + ":" + surface)
        # Set mesh sizes on the points from the surface we are extruding
        bottom_nodes = gmsh.model.getBoundary([volumes[bottom_index]], oriented=False, recursive=True)
        gmsh.model.occ.mesh.setSize(bottom_nodes, res)
        top_nodes = gmsh.model.getBoundary([volumes[top_index]], oriented=False, recursive=True)
        gmsh.model.occ.mesh.setSize(top_nodes, 2 * res)
        # NOTE: Need to synchronize after setting mesh sizes
        gmsh.model.occ.synchronize()
        # Generate mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    r_matrix = _utils.rotation_matrix([0, 0, 1], theta)

    # NOTE: Hex mesh must be rotated after generation due to gmsh API
    mesh.geometry.x[:] = np.dot(r_matrix, mesh.geometry.x.T).T
    gmsh.clear()
    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return mesh, ft


def mesh_2D_dolfin(celltype: str, theta: float = 0, outdir: Union[str, Path] = Path("meshes")):
    """
    Create two 2D cubes stacked on top of each other,
    and the corresponding mesh markers using dolfin built-in meshes
    """
    def find_line_function(p0, p1):
        """
        Find line y=ax+b for each of the lines in the mesh
        https://mathworld.wolfram.com/Two-PointForm.html
        """
        # Line aligned with y axis
        if np.isclose(p1[0], p0[0]):
            return lambda x: np.isclose(x[0], p0[0])
        return lambda x: np.isclose(x[1], p0[1] + (p1[1] - p0[1]) / (p1[0] - p0[0]) * (x[0] - p0[0]))

    def over_line(p0, p1):
        """
        Check if a point is over or under y=ax+b for each of the
        lines in the mesh https://mathworld.wolfram.com/Two-PointForm.html
        """
        return lambda x: x[1] > p0[1] + (p1[1] - p0[1]) / (p1[0] - p0[0]) * (x[0] - p0[0])

    # Using built in meshes, stacking cubes on top of each other
    N = 15
    if celltype == "quadrilateral":
        ct = _mesh.CellType.quadrilateral
    elif celltype == "triangle":
        ct = _mesh.CellType.triangle
    else:
        raise ValueError("celltype has to be tri or quad")
    if MPI.COMM_WORLD.rank == 0:
        mesh0 = _mesh.create_unit_square(MPI.COMM_SELF, N, N, ct)
        mesh1 = _mesh.create_unit_square(MPI.COMM_SELF, 2 * N, 2 * N, ct)
        mesh0.geometry.x[:, 1] += 1

        # Stack the two meshes in one mesh
        r_matrix = _utils.rotation_matrix([0, 0, 1], theta)
        points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
        points = np.dot(r_matrix, points.T)
        points = points[: 2, :].T

        # Transform topology info into geometry info
        tdim0 = mesh0.topology.dim
        num_cells0 = mesh0.topology.index_map(tdim0).size_local
        cells0 = _cpp.mesh.entities_to_geometry(
            mesh0._cpp_object, tdim0, np.arange(num_cells0, dtype=np.int32), False)
        tdim1 = mesh1.topology.dim
        num_cells1 = mesh1.topology.index_map(tdim1).size_local
        cells1 = _cpp.mesh.entities_to_geometry(
            mesh1._cpp_object, tdim1, np.arange(num_cells1, dtype=np.int32), False)
        cells1 += mesh0.geometry.x.shape[0]

        cells = np.vstack([cells0, cells1])
        domain = ufl.Mesh(element("Lagrange", ct.name, 1, shape=(points.shape[1],)))
        mesh = _mesh.create_mesh(MPI.COMM_SELF, cells, points, domain)
        tdim = mesh.topology.dim
        fdim = tdim - 1

        # Find information about facets to be used in meshtags
        bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        bottom = find_line_function(bottom_points[:, 0], bottom_points[:, 1])
        bottom_facets = _mesh.locate_entities_boundary(mesh, fdim, bottom)

        top_points = np.dot(r_matrix, np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]).T)
        top = find_line_function(top_points[:, 2], top_points[:, 3])
        top_facets = _mesh.locate_entities_boundary(mesh, fdim, top)

        left_side = find_line_function(top_points[:, 0], top_points[:, 3])
        left_facets = _mesh.locate_entities_boundary(mesh, fdim, left_side)

        right_side = find_line_function(top_points[:, 1], top_points[:, 2])
        right_facets = _mesh.locate_entities_boundary(mesh, fdim, right_side)

        top_cube = over_line(bottom_points[:, 2], bottom_points[:, 3])

        num_cells = mesh.topology.index_map(tdim).size_local
        cell_midpoints = _mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))
        interface = find_line_function(bottom_points[:, 2], bottom_points[:, 3])
        i_facets = _mesh.locate_entities_boundary(mesh, fdim, interface)
        bottom_interface = []
        top_interface = []
        mesh.topology.create_connectivity(fdim, tdim)
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        for facet in i_facets:
            i_cells = facet_to_cell.links(facet)
            assert len(i_cells == 1)
            i_cell = i_cells[0]
            if top_cube(cell_midpoints[i_cell]):
                top_interface.append(facet)
            else:
                bottom_interface.append(facet)

        top_cube_marker = 2
        cell_indices = []
        cell_values = []
        for cell_index in range(num_cells):
            if top_cube(cell_midpoints[cell_index]):
                cell_indices.append(cell_index)
                cell_values.append(top_cube_marker)
        ct = _mesh.meshtags(mesh, tdim, np.array(cell_indices, dtype=np.intc), np.array(cell_values, dtype=np.intc))

        # Create meshtags for facet data
        markers: Dict[int, np.ndarray] = {3: top_facets, 4: np.hstack(bottom_interface), 9: np.hstack(top_interface),
                                          5: bottom_facets, 6: left_facets, 7: right_facets}
        all_indices = []
        all_values = []

        for key in markers.keys():
            all_indices.append(markers[key])
            all_values.append(np.full(len(markers[key]), key, dtype=np.intc))
        arg_sort = np.argsort(np.hstack(all_indices))
        mt = _mesh.meshtags(mesh, fdim, np.hstack(all_indices)[arg_sort], np.hstack(all_values)[arg_sort])
        mt.name = "facet_tags"
        outpath = Path(outdir)
        outpath.mkdir(exist_ok=True, parents=True)
        with _io.XDMFFile(MPI.COMM_SELF, outpath / f"mesh_{celltype}_{theta:.2f}.xdmf", "w") as o_f:
            o_f.write_mesh(mesh)
            o_f.write_meshtags(ct, x=mesh.geometry)
            o_f.write_meshtags(mt, x=mesh.geometry)
    MPI.COMM_WORLD.barrier()


def mesh_3D_dolfin(theta: float = 0, ct: _mesh.CellType = _mesh.CellType.tetrahedron,
                   ext: str = "tetrahedron", res: float = 0.1, outdir: Union[str, Path] = Path("meshes")):
    timer = _common.Timer("~~Contact: Create mesh")

    def find_plane_function(p0, p1, p2):
        """
        Find plane function given three points:
        http://www.nabla.hr/CG-LinesPlanesIn3DA3.htm
        """
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)

        n = np.cross(v1, v2)
        D = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2])
        return lambda x: np.isclose(0, np.dot(n, x) + D)

    def over_plane(p0, p1, p2):
        """
        Returns function that checks if a point is over a plane defined
        by the points p0, p1 and p2.
        """
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)

        n = np.cross(v1, v2)
        D = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2])
        return lambda x: n[0] * x[0] + n[1] * x[1] + D > -n[2] * x[2]

    N = int(1 / res)
    if MPI.COMM_WORLD.rank == 0:
        mesh0 = _mesh.create_unit_cube(MPI.COMM_SELF, N, N, N, ct)
        mesh1 = _mesh.create_unit_cube(MPI.COMM_SELF, 2 * N, 2 * N, 2 * N, ct)
        mesh0.geometry.x[:, 2] += 1

        # Stack the two meshes in one mesh
        r_matrix = _utils.rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)
        points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
        points = np.dot(r_matrix, points.T).T

        # Transform topology info into geometry info
        tdim0 = mesh0.topology.dim
        num_cells0 = mesh0.topology.index_map(tdim0).size_local
        cells0 = _cpp.mesh.entities_to_geometry(
            mesh0._cpp_object, tdim0, np.arange(num_cells0, dtype=np.int32), False)
        tdim1 = mesh1.topology.dim
        num_cells1 = mesh1.topology.index_map(tdim1).size_local
        cells1 = _cpp.mesh.entities_to_geometry(
            mesh1._cpp_object, tdim1, np.arange(num_cells1, dtype=np.int32), False)
        cells1 += mesh0.geometry.x.shape[0]

        cells = np.vstack([cells0, cells1])
        domain = ufl.Mesh(element("Lagrange", ct.name, 1, shape=(points.shape[1],)))
        mesh = _mesh.create_mesh(MPI.COMM_SELF, cells, points, domain)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        # Find information about facets to be used in meshtags
        bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T)
        bottom = find_plane_function(bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2])
        bottom_facets = _mesh.locate_entities_boundary(mesh, fdim, bottom)

        top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]]).T)
        top = find_plane_function(top_points[:, 0], top_points[:, 1], top_points[:, 2])
        top_facets = _mesh.locate_entities_boundary(mesh, fdim, top)

        # left_side = find_line_function(top_points[:, 0], top_points[:, 3])
        # left_facets = _mesh.locate_entities_boundary(
        #     mesh, fdim, left_side)

        # right_side = find_line_function(top_points[:, 1], top_points[:, 2])
        # right_facets = _mesh.locate_entities_boundary(
        #     mesh, fdim, right_side)
        if_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1],
                                               [0, 1, 1], [1, 1, 1]]).T)

        interface = find_plane_function(if_points[:, 0], if_points[:, 1], if_points[:, 2])
        i_facets = _mesh.locate_entities_boundary(mesh, fdim, interface)
        mesh.topology.create_connectivity(fdim, tdim)
        top_interface = []
        bottom_interface = []
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        num_cells = mesh.topology.index_map(tdim).size_local
        cell_midpoints = _mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))
        top_cube = over_plane(if_points[:, 0], if_points[:, 1], if_points[:, 2])
        for facet in i_facets:
            i_cells = facet_to_cell.links(facet)
            assert len(i_cells) == 1
            i_cell = i_cells[0]
            if top_cube(cell_midpoints[i_cell]):
                top_interface.append(facet)
            else:
                bottom_interface.append(facet)

        num_cells = mesh.topology.index_map(tdim).size_local
        cell_midpoints = _mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))
        top_cube_marker = 2
        indices = []
        values = []
        for cell_index in range(num_cells):
            if top_cube(cell_midpoints[cell_index]):
                indices.append(cell_index)
                values.append(top_cube_marker)
        ct = _mesh.meshtags(mesh, tdim, np.array(indices, dtype=np.int32), np.array(values, dtype=np.int32))

        # Create meshtags for facet data
        markers: Dict[int, np.ndarray] = {3: top_facets, 4: np.hstack(bottom_interface), 9: np.hstack(top_interface),
                                          5: bottom_facets}  # , 6: left_facets, 7: right_facets}
        all_indices = []
        all_values = []

        for key in markers.keys():
            all_indices.append(np.asarray(markers[key], dtype=np.int32))
            all_values.append(np.full(len(markers[key]), key, dtype=np.int32))
        arg_sort = np.argsort(np.hstack(all_indices))
        sorted_vals = np.asarray(np.hstack(all_values)[arg_sort], dtype=np.int32)
        sorted_indices = np.asarray(np.hstack(all_indices)[arg_sort], dtype=np.int32)
        mt = _mesh.meshtags(mesh, fdim, sorted_indices, sorted_vals)
        mt.name = "facet_tags"
        outpath = Path(outdir)
        outpath.mkdir(exist_ok=True, parents=True)
        fname = outpath / f"mesh_{ext}_{theta:.2f}.xdmf"

        with _io.XDMFFile(MPI.COMM_SELF, fname, "w") as o_f:
            o_f.write_mesh(mesh)
            o_f.write_meshtags(ct, mesh.geometry)
            o_f.write_meshtags(mt, mesh.geometry)
    timer.stop()


import numpy as np
import pygmsh

import meshio


def generate_hex_box(x0, y0, z0, x1, y1, z1, theta, res, facet_markers,
                     volume_marker=None):
    """
    Generate the box [x0,y0,z0]x[y0,y1,z0], rotated theta degrees around
    the origin over axis [0,1,1] with resolution res,
    where markers are is an array containing markers
    array of markers for [left, back, top, bottom, front, right]
    and an optional volume_marker.
    """
    geom = pygmsh.built_in.Geometry()

    rect = geom.add_rectangle(x0, x1, y0, y1, 0.0, res)

    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    geom.add_raw_code("Recombine Surface {:};")
    bottom, volume, sides = geom.extrude(rect, translation_axis=[0, 0, 1],
                                         num_layers=int(0.5/res),
                                         recombine=True)

    geom.add_physical(bottom, facet_markers[4])
    if volume_marker is None:
        geom.add_physical(volume, 0)
    else:
        geom.add_physical(volume, volume_marker)
    geom.add_physical(rect.surface, facet_markers[1])
    geom.add_physical(sides[0], facet_markers[3])
    geom.add_physical(sides[1], facet_markers[5])
    geom.add_physical(sides[2], facet_markers[2])
    geom.add_physical(sides[3], facet_markers[0])
    msh = pygmsh.generate_mesh(geom, verbose=False)

    msh.points[:, -1] += z0
    r_matrix = pygmsh.helpers.rotation_matrix(
        [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)
    msh.points[:, :] = np.dot(r_matrix, msh.points.T).T
    return msh


def merge_msh_meshes(msh0, msh1, celltype, facettype):
    """
    Merge two meshio meshes with the same celltype, and create
    the mesh and corresponding facet mesh with domain markers.
    """
    points0 = msh0.points

    cells0 = np.vstack([cells.data for cells in msh0.cells
                        if cells.type == celltype])
    cell_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh0.cell_data_dict["gmsh:physical"].keys()
                            if key == celltype])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    cells1 = np.vstack([cells.data for cells in msh1.cells
                        if cells.type == celltype])
    cell_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh1.cell_data_dict["gmsh:physical"].keys()
                            if key == celltype])
    cells1 += points0.shape[0]
    cells = np.vstack([cells0, cells1])
    cell_data = np.hstack([cell_data0, cell_data1])
    mesh = meshio.Mesh(points=points,
                       cells=[(celltype, cells)],
                       cell_data={"name_to_read": cell_data})

    # Write line mesh
    facet_cells0 = np.vstack([cells.data for cells in msh0.cells
                              if cells.type == facettype])
    facet_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh0.cell_data_dict["gmsh:physical"].keys()
                             if key == facettype])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    facet_cells1 = np.vstack([cells.data for cells in msh1.cells
                              if cells.type == facettype])
    facet_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh1.cell_data_dict["gmsh:physical"].keys()
                             if key == facettype])
    facet_cells1 += points0.shape[0]
    facet_cells = np.vstack([facet_cells0, facet_cells1])

    facet_data = np.hstack([facet_data0, facet_data1])
    facet_mesh = meshio.Mesh(points=points,
                             cells=[(facettype, facet_cells)],
                             cell_data={"name_to_read": facet_data})
    return mesh, facet_mesh


def mesh_3D_rot(theta):
    res = 0.2

    msh0 = generate_hex_box(0, 0, 0, 1, 1, 1, theta, res,
                            [11, 5, 12, 13, 4, 14])
    msh1 = generate_hex_box(0, 0, 1, 1, 1, 2, theta, 2*res,
                            [11, 9, 12, 13, 3, 14], 2)
    mesh, facet_mesh = merge_msh_meshes(msh0, msh1, "hexahedron", "quad")
    meshio.xdmf.write("mesh_hex_cube{0:.2f}.xdmf".format(theta), mesh)
    meshio.xdmf.write("facet_hex_cube{0:.2f}.xdmf".format(theta), facet_mesh)

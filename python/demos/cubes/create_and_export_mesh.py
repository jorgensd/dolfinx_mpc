import pygmsh
import meshio
import numpy as np


def mesh_2D():
    res = 0.1
    geom0 = pygmsh.built_in.Geometry()
    rect = geom0.add_rectangle(0.0, 1, 0.0, 1.0, 0.0, res)
    geom0.add_physical([rect.surface], 7)

    # Top: 1, Bottom 2, Side walls: 3
    # geom.add_physical([rect.line_loop.lines[2]], 1)
    # geom.add_physical([rect.line_loop.lines[0]], 2)
    # geom.add_physical([rect.line_loop.lines[1], rect.line_loop.lines[3]], 3)
    msh0 = pygmsh.generate_mesh(geom0, prune_z_0=True)
    geom1 = pygmsh.built_in.Geometry()
    rect1 = geom1.add_rectangle(0.0, 1, 1.0, 2.0, 0.0, 2*res)
    geom1.add_physical([rect1.surface], 8)
    msh1 = pygmsh.generate_mesh(geom1, prune_z_0=True)

    # Upper box:
    # Top: 4, Bottom 5, Side walls: 6
    # geom.add_physical([rect1.line_aloop.lines[2]], 4)
    # geom.add_physical([rect1.line_loop.lines[0]], 5)
    # geom.add_physical([rect1.line_loop.lines[1],
    # rect1.line_loop.lines[3]], 6)
    # geom.add_physical([rect1.surface], 8)
    points0 = msh0.points
    tri_cells0 = np.vstack([cells.data for cells in msh0.cells
                            if cells.type == "triangle"])
    tri_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh0.cell_data_dict["gmsh:physical"].keys()
                           if key == "triangle"])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    tri_cells1 = np.vstack([cells.data for cells in msh1.cells
                            if cells.type == "triangle"])
    tri_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh1.cell_data_dict["gmsh:physical"].keys()
                           if key == "triangle"])
    tri_cells1 += points0.shape[0]
    tri_cells = np.vstack([tri_cells0, tri_cells1])

    tri_data = np.hstack([tri_data0, tri_data1])
    triangle_mesh = meshio.Mesh(points=points,
                                cells=[("triangle", tri_cells)],
                                cell_data={"name_to_read": tri_data})
    meshio.write("meshes/mesh.xdmf", triangle_mesh)


def mesh_3D():
    geom = pygmsh.built_in.Geometry()
    res = 0.25
    bbox = geom.add_box(0, 1, 0, 1, 0, 1, res)
    tbox = geom.add_box(0, 1, 0, 1, 1, 2, 2*res)
    geom.add_physical([bbox.volume], 1)
    geom.add_physical([tbox.volume], 2)
    msh = pygmsh.generate_mesh(geom)

    tet_cells = np.vstack([cells.data for cells in msh.cells
                           if cells.type == "tetra"])
    tet_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                          for key in msh.cell_data_dict["gmsh:physical"].keys()
                          if key == "tetra"])

    tetra_mesh = meshio.Mesh(points=msh.points,
                             cells=[("tetra", tet_cells)],
                             cell_data={"name_to_read": tet_data})
    meshio.write("meshes/mesh3D.xdmf", tetra_mesh)


def mesh_2D_rot(theta=np.pi/2):
    res = 0.1
    geom0 = pygmsh.built_in.Geometry()
    rect = geom0.add_rectangle(0.0, 1, 0.0, 1.0, 0.0, res)
    geom0.rotate(rect, [0, 0, 0], theta, [0, 0, 1])
    geom0.add_physical([rect.surface], 7)

    # Top: 1, Bottom 2, Side walls: 3
    # geom.add_physical([rect.line_loop.lines[2]], 1)
    # geom.add_physical([rect.line_loop.lines[0]], 2)
    # geom.add_physical([rect.line_loop.lines[1], rect.line_loop.lines[3]], 3)
    msh0 = pygmsh.generate_mesh(geom0, prune_z_0=True)
    geom1 = pygmsh.built_in.Geometry()
    rect1 = geom1.add_rectangle(0.0, 1, 1.0, 2.0, 0.0, 2*res)
    geom1.rotate(rect1, [0, 0, 0], theta, [0, 0, 1])
    geom1.add_physical([rect1.surface], 8)
    msh1 = pygmsh.generate_mesh(geom1, prune_z_0=True)

    # Upper box:
    # Top: 4, Bottom 5, Side walls: 6
    # geom.add_physical([rect1.line_aloop.lines[2]], 4)
    # geom.add_physical([rect1.line_loop.lines[0]], 5)
    # geom.add_physical([rect1.line_loop.lines[1],
    # rect1.line_loop.lines[3]], 6)
    # geom.add_physical([rect1.surface], 8)
    points0 = msh0.points
    tri_cells0 = np.vstack([cells.data for cells in msh0.cells
                            if cells.type == "triangle"])
    tri_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh0.cell_data_dict["gmsh:physical"].keys()
                           if key == "triangle"])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    tri_cells1 = np.vstack([cells.data for cells in msh1.cells
                            if cells.type == "triangle"])
    tri_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh1.cell_data_dict["gmsh:physical"].keys()
                           if key == "triangle"])
    tri_cells1 += points0.shape[0]
    tri_cells = np.vstack([tri_cells0, tri_cells1])

    tri_data = np.hstack([tri_data0, tri_data1])
    triangle_mesh = meshio.Mesh(points=points,
                                cells=[("triangle", tri_cells)],
                                cell_data={"name_to_read": tri_data})
    meshio.write("meshes/mesh_rot.xdmf", triangle_mesh)


def mesh_3D_rot(theta=np.pi/2):
    res = 0.25
    geom0 = pygmsh.built_in.Geometry()
    bbox = geom0.add_box(0, 1, 0, 1, 0, 1, res)
    bbox.dimension = 3
    bbox.id = bbox.volume.id
    geom0.rotate(bbox, [0, 0, 0], -theta, [0, 1, 1])
    geom0.add_physical([bbox.volume], 1)
    msh0 = pygmsh.generate_mesh(geom0)

    geom1 = pygmsh.built_in.Geometry()
    tbox = geom1.add_box(0, 1, 0, 1, 1, 2, 2*res)
    tbox.id = tbox.volume.id
    tbox.dimension = 3
    geom1.rotate(tbox, [0, 0, 0], -theta, [0, 1, 1])
    geom1.add_physical([tbox.volume], 2)
    msh1 = pygmsh.generate_mesh(geom1)

    points0 = msh0.points
    tetra_cells0 = np.vstack([cells.data for cells in msh0.cells
                              if cells.type == "tetra"])
    tetra_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh0.cell_data_dict["gmsh:physical"].keys()
                             if key == "tetra"])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    tetra_cells1 = np.vstack([cells.data for cells in msh1.cells
                              if cells.type == "tetra"])
    tetra_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh1.cell_data_dict["gmsh:physical"].keys()
                             if key == "tetra"])
    tetra_cells1 += points0.shape[0]
    tetra_cells = np.vstack([tetra_cells0, tetra_cells1])

    tetra_data = np.hstack([tetra_data0, tetra_data1])
    tetra_mesh = meshio.Mesh(points=points,
                             cells=[("tetra", tetra_cells)],
                             cell_data={"name_to_read": tetra_data})
    meshio.write("meshes/mesh3D_rot.xdmf", tetra_mesh)


if __name__ == "__main__":
    mesh_2D()
    mesh_3D()
    mesh_2D_rot()
    mesh_3D_rot()

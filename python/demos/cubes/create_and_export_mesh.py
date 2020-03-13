import pygmsh
import meshio
import numpy as np


def mesh_2D():
    geom = pygmsh.built_in.Geometry()
    res = 0.1
    rect = geom.add_rectangle(0.0, 1, 0.0, 1.0, 0.0, res)
    rect2 = geom.add_rectangle(0.0, 1, 1.0, 2.0, 0.0, 2*res)
    # Top: 1, Bottom 2, Side walls: 3
    # geom.add_physical([rect.line_loop.lines[2]], 1)
    # geom.add_physical([rect.line_loop.lines[0]], 2)
    # geom.add_physical([rect.line_loop.lines[1], rect.line_loop.lines[3]], 3)
    geom.add_physical([rect.surface], 7)

    # Upper box:
    # Top: 4, Bottom 5, Side walls: 6
    # geom.add_physical([rect2.line_aloop.lines[2]], 4)
    # geom.add_physical([rect2.line_loop.lines[0]], 5)
    # geom.add_physical([rect2.line_loop.lines[1],
    # rect2.line_loop.lines[3]], 6)
    geom.add_physical([rect2.surface], 8)

    msh = pygmsh.generate_mesh(geom, prune_z_0=True)
    tri_cells = np.vstack([cells.data for cells in msh.cells
                           if cells.type == "triangle"])
    tri_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                          for key in msh.cell_data_dict["gmsh:physical"].keys()
                          if key == "triangle"])

    triangle_mesh = meshio.Mesh(points=msh.points,
                                cells=[("triangle", tri_cells)],
                                cell_data={"name_to_read": tri_data})
    meshio.write("mesh.xdmf", triangle_mesh)

    # line_cells = np.vstack([cells.data for cells in msh.cells
    #                         if cells.type == "line"])
    # line_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
    #                        for key in
    #                        msh.cell_data_dict["gmsh:physical"].keys()
    #                        if key == "line"])

    # line_mesh = meshio.Mesh(points=msh.points,
    #                         cells=[("line", line_cells)],
    #                         cell_data={"name_to_read": line_data})

    # meshio.xdmf.write("mf.xdmf", line_mesh)


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
    meshio.write("mesh3D.xdmf", tetra_mesh)


if __name__ == "__main__":
    mesh_2D()
    mesh_3D()

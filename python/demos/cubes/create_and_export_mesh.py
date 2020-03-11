import pygmsh
import meshio
import numpy as np

geom = pygmsh.built_in.Geometry()
rect = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, 1/17)
rect2 = geom.add_rectangle(0.0, 1.0, 1.0, 2.0, 0.0, 1/29)

# Top: 1, Bottom 2, Side walls: 3
geom.add_physical([rect.line_loop.lines[2]], 1)
geom.add_physical([rect.line_loop.lines[0]], 2)
geom.add_physical([rect.line_loop.lines[1], rect.line_loop.lines[3]], 3)
geom.add_physical([rect.surface], 7)

# Upper box:
# Top: 4, Bottom 5, Side walls: 6
geom.add_physical([rect2.line_loop.lines[2]], 4)
geom.add_physical([rect2.line_loop.lines[0]], 5)
geom.add_physical([rect2.line_loop.lines[1], rect2.line_loop.lines[3]], 6)
geom.add_physical([rect2.surface], 8)

msh = pygmsh.generate_mesh(geom, prune_z_0=True)

line_cells = np.vstack([cells.data for cells in msh.cells
                        if cells.type == "line"])
line_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                       for key in msh.cell_data_dict["gmsh:physical"].keys()
                       if key == "line"])


tri_cells = np.vstack([cells.data for cells in msh.cells
                       if cells.type == "triangle"])
tri_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                      for key in msh.cell_data_dict["gmsh:physical"].keys()
                      if key == "triangle"])


line_mesh = meshio.Mesh(points=msh.points,
                        cells=[("line", line_cells)],
                        cell_data={"name_to_read": line_data})

triangle_mesh = meshio.Mesh(points=msh.points,
                            cells=[("triangle", tri_cells)],
                            cell_data={"name_to_read": tri_data})
meshio.write("mesh.xdmf", triangle_mesh)

meshio.xdmf.write("mf.xdmf", line_mesh)

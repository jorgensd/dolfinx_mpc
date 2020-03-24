import pygmsh
import meshio
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.io


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
    msh0 = pygmsh.generate_mesh(geom0, prune_z_0=True, verbose=False)
    geom1 = pygmsh.built_in.Geometry()
    rect1 = geom1.add_rectangle(0.0, 1, 1.0, 2.0, 0.0, 2*res)
    geom1.rotate(rect1, [0, 0, 0], theta, [0, 0, 1])
    geom1.add_physical([rect1.surface], 8)
    msh1 = pygmsh.generate_mesh(geom1, prune_z_0=True, verbose=False)

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
    res = 0.125
    geom0 = pygmsh.built_in.Geometry()
    bbox = geom0.add_box(0, 1, 0, 1, 0, 1, res)
    bbox.dimension = 3
    bbox.id = bbox.volume.id
    geom0.rotate(bbox, [0, 0, 0], -theta, [0, 1, 1])
    geom0.add_physical([bbox.volume], 1)
    msh0 = pygmsh.generate_mesh(geom0, verbose=False)

    geom1 = pygmsh.built_in.Geometry()
    tbox = geom1.add_box(0, 1, 0, 1, 1, 2, 2*res)
    tbox.id = tbox.volume.id
    tbox.dimension = 3
    geom1.rotate(tbox, [0, 0, 0], -theta, [0, 1, 1])
    geom1.add_physical([tbox.volume], 2)
    msh1 = pygmsh.generate_mesh(geom1, verbose=False)

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


def mesh_2D_dolfin(celltype):
    # Using built in meshes, stacking cubes on top of each other
    if celltype == "quad":
        N = 2
        ct = dolfinx.cpp.mesh.CellType.quadrilateral
        mesh0 = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, N, N, ct)
        mesh1 = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 2*N, 2*N, ct)
        nv = 4
    elif celltype == "tri":
        N = 2
        ct = dolfinx.cpp.mesh.CellType.triangle
        mesh0 = dolfinx.RectangleMesh(dolfinx.MPI.comm_world,
                                      [np.array([0.0, 0.0, 0]),
                                       np.array([1.0, 1.0, 0])], [N, N],
                                      ct, diagonal="crossed")

        mesh1 = dolfinx.RectangleMesh(dolfinx.MPI.comm_world,
                                      [np.array([0.0, 0.0, 0]),
                                       np.array([1.0, 1.0, 0])], [2*N, 2*N],
                                      ct, diagonal="crossed")
        nv = 3
    else:
        raise ValueError("celltype has to be tri or quad")

    # Stack the two meshes in one mesh
    x0 = mesh0.geometry.x
    x0 = x0[:, :-1]
    x0[:, 1] += 1
    points = np.vstack([x0, mesh1.geometry.x[:, :-1]])

    # Transform topology info into geometry info
    c2v = mesh0.topology.connectivity(mesh0.topology.dim, 0)
    x_dofmap = mesh0.geometry.dofmap()
    imap = mesh0.topology.index_map(0)
    num_mesh_vertices = imap.size_local + imap.num_ghosts
    vertex_to_node = np.zeros(num_mesh_vertices, dtype=np.int64)
    for c in range(c2v.num_nodes):
        vertices = c2v.links(c)
        x_dofs = x_dofmap.links(c)
        for i in range(vertices.shape[0]):
            vertex_to_node[vertices[i]] = x_dofs[i]
    cells0 = np.zeros((c2v.num_nodes, nv), dtype=np.int64)
    for cell in range(c2v.num_nodes):
        for v in range(nv):
            cells0[cell, v] = vertex_to_node[c2v.links(cell)[v]]
    # Transform topology info into geometry info
    c2v = mesh1.topology.connectivity(mesh1.topology.dim, 0)
    x_dofmap = mesh1.geometry.dofmap()
    imap = mesh1.topology.index_map(0)
    num_mesh_vertices = imap.size_local + imap.num_ghosts
    vertex_to_node = np.zeros(num_mesh_vertices, dtype=np.int64)
    for c in range(c2v.num_nodes):
        vertices = c2v.links(c)
        x_dofs = x_dofmap.links(c)
        for i in range(vertices.shape[0]):
            vertex_to_node[vertices[i]] = x_dofs[i]
    cells1 = np.zeros((c2v.num_nodes, nv), dtype=np.int64)
    for cell in range(c2v.num_nodes):
        for v in range(nv):
            cells1[cell, v] = vertex_to_node[c2v.links(cell)[v]] + x0.shape[0]
    cells = np.vstack([cells0, cells1])
    mesh = dolfinx.Mesh(dolfinx.MPI.comm_world,
                        ct, points, cells, [],
                        dolfinx.cpp.mesh.GhostMode.none)
    o_f = dolfinx.io.XDMFFile(dolfinx.MPI.comm_world,
                              "meshes/mesh_{0:s}.xdmf".format(celltype))
    o_f.write(mesh)


if __name__ == "__main__":
    mesh_2D()
    mesh_3D()
    mesh_2D_rot()
    mesh_3D_rot()
    mesh_2D_dolfin("tri")
    mesh_2D_dolfin("quad")

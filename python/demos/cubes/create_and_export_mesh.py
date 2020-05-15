import pygmsh
import meshio
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.io
from mpi4py import MPI


def mesh_2D_rot(theta=np.pi/2):
    res = 0.1

    # Lower Cube
    # Top: 4, Bottom 5, Left 8, Right 9
    geom0 = pygmsh.built_in.Geometry()
    rect = geom0.add_rectangle(0.0, 1, 0.0, 1.0, 0.0, res)
    geom0.rotate(rect, [0, 0, 0], theta, [0, 0, 1])
    geom0.add_physical([rect.surface], 0)

    geom0.add_physical([rect.line_loop.lines[2]], 4)
    geom0.add_physical([rect.line_loop.lines[0]], 5)
    geom0.add_physical([rect.line_loop.lines[1]], 8)
    geom0.add_physical([rect.line_loop.lines[3]], 9)
    msh0 = pygmsh.generate_mesh(geom0, prune_z_0=True, verbose=False)

    # Upper box:
    # Top: 3, Bottom 9, Left 6, Right 7
    geom1 = pygmsh.built_in.Geometry()
    rect1 = geom1.add_rectangle(0.0, 1, 1.0, 2.0, 0.0, 2*res)
    geom1.rotate(rect1, [0, 0, 0], theta, [0, 0, 1])

    geom1.add_physical([rect1.line_loop.lines[2]], 3)
    geom1.add_physical([rect1.line_loop.lines[0]], 9)
    geom1.add_physical([rect1.line_loop.lines[1]], 6)
    geom1.add_physical([rect1.line_loop.lines[3]], 7)
    geom1.add_physical([rect1.surface], 2)
    msh1 = pygmsh.generate_mesh(geom1, prune_z_0=True, verbose=False)

    points0 = msh0.points

    # Write triangle mesh
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

    # Write line mesh
    facet_cells0 = np.vstack([cells.data for cells in msh0.cells
                              if cells.type == "line"])
    facet_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh0.cell_data_dict["gmsh:physical"].keys()
                             if key == "line"])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    facet_cells1 = np.vstack([cells.data for cells in msh1.cells
                              if cells.type == "line"])
    facet_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh1.cell_data_dict["gmsh:physical"].keys()
                             if key == "line"])
    facet_cells1 += points0.shape[0]
    facet_cells = np.vstack([facet_cells0, facet_cells1])

    facet_data = np.hstack([facet_data0, facet_data1])
    facet_mesh = meshio.Mesh(points=points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": facet_data})
    meshio.write("meshes/facet_rot.xdmf", facet_mesh)


def mesh_3D_rot(theta=np.pi/2):
    res = 0.125
    geom0 = pygmsh.built_in.Geometry()
    bbox = geom0.add_box(0, 1, 0, 1, 0, 1, res)
    bbox.dimension = 3
    bbox.id = bbox.volume.id
    # for i in range(len(bbox.surface_loop.surfaces)):
    #     geom0.add_physical([bbox.surface_loop.surfaces[i]], i + 3)
    geom0.add_physical([bbox.surface_loop.surfaces[2]], 4)
    geom0.add_physical([bbox.surface_loop.surfaces[3]], 5)

    geom0.rotate(bbox, [0, 0, 0], -theta, [0, 1, 1])
    geom0.add_physical([bbox.volume], 0)
    msh0 = pygmsh.generate_mesh(geom0, verbose=False)

    geom1 = pygmsh.built_in.Geometry()
    tbox = geom1.add_box(0, 1, 0, 1, 1, 2, 2*res)
    tbox.id = tbox.volume.id
    tbox.dimension = 3
    geom1.rotate(tbox, [0, 0, 0], -theta, [0, 1, 1])
    geom1.add_physical([tbox.volume], 2)
    # for i in range(len(tbox.surface_loop.surfaces)):
    #     geom1.add_physical([tbox.surface_loop.surfaces[i]], i + 9)
    # Top 3, Interface 4
    geom1.add_physical(tbox.surface_loop.surfaces[2], 3)
    geom1.add_physical(tbox.surface_loop.surfaces[3], 9)

    msh1 = pygmsh.generate_mesh(geom1, verbose=False)

    points0 = msh0.points

    def merge_and_write_mesh(msh0, msh1, celltype, meshname):
        """
        Merge two meshio meshes of a given celltype and write to file.
        (The meshio mesh can contain multiple cell types).
        """
        cells0 = np.vstack([cells.data for cells in msh0.cells
                            if cells.type == celltype])
        data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh0.cell_data_dict["gmsh:physical"].keys()
                           if key == celltype])
        points1 = msh1.points
        points = np.vstack([points0, points1])
        cells1 = np.vstack([cells.data for cells in msh1.cells
                            if cells.type == celltype])
        data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                           for key in
                           msh1.cell_data_dict["gmsh:physical"].keys()
                           if key == celltype])
        cells1 += points0.shape[0]
        cells = np.vstack([cells0, cells1])

        data = np.hstack([data0, data1])
        mesh = meshio.Mesh(points=points,
                           cells=[(celltype, cells)],
                           cell_data={"name_to_read": data})
        meshio.xdmf.write(meshname, mesh)

    merge_and_write_mesh(msh0, msh1, "tetra", "meshes/mesh3D_rot.xdmf")
    merge_and_write_mesh(msh0, msh1, "triangle", "meshes/facet3D_rot.xdmf")


def mesh_2D_dolfin(celltype, theta=0):
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
        return lambda x: np.isclose(x[1],
                                    p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])
                                    * (x[0]-p0[0]))

    def over_line(p0, p1):
        """
        Check if a point is over or under y=ax+b for each of the
        lines in the mesh https://mathworld.wolfram.com/Two-PointForm.html
        """
        return lambda x: x[1] > p0[1]+(p1[1]-p0[1])/(p1[0]-p0[0])*(x[0]-p0[0])

    # Using built in meshes, stacking cubes on top of each other
    if celltype == "quad":
        N = 2
        ct = dolfinx.cpp.mesh.CellType.quadrilateral
    elif celltype == "tri":
        N = 2
        ct = dolfinx.cpp.mesh.CellType.triangle
    else:
        raise ValueError("celltype has to be tri or quad")

    nv = dolfinx.cpp.mesh.cell_num_vertices(ct)
    mesh0 = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N, ct)
    mesh1 = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2*N, 2*N, ct)
    mesh0.geometry.x[:, 1] += 1

    # Stack the two meshes in one mesh
    r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
    points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
    points = np.dot(r_matrix, points.T)
    points = points[:2, :].T

    # Transform topology info into geometry info
    c2v = mesh0.topology.connectivity(mesh0.topology.dim, 0)
    x_dofmap = mesh0.geometry.dofmap
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
    x_dofmap = mesh1.geometry.dofmap
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
            cells1[cell, v] = vertex_to_node[c2v.links(
                cell)[v]] + mesh0.geometry.x.shape[0]
    cells = np.vstack([cells0, cells1])
    mesh = dolfinx.Mesh(MPI.COMM_WORLD,
                        ct, points, cells, [], degree=1,
                        ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Find information about facets to be used in MeshTags
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [1, 1, 0], [0, 1, 0]]).T)
    bottom = find_line_function(bottom_points[:, 0], bottom_points[:, 1])
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, bottom)

    top_points = np.dot(r_matrix, np.array([[0, 1, 0], [1, 1, 0],
                                            [1, 2, 0], [0, 2, 0]]).T)
    top = find_line_function(top_points[:, 2], top_points[:, 3])
    top_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, top)

    left_side = find_line_function(top_points[:, 0], top_points[:, 3])
    left_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, left_side)

    right_side = find_line_function(top_points[:, 1], top_points[:, 2])
    right_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, right_side)

    top_cube = over_line(bottom_points[:, 2], bottom_points[:, 3])

    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    interface = find_line_function(bottom_points[:, 2], bottom_points[:, 3])
    i_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, interface)
    bottom_interface = []
    top_interface = []
    mesh.topology.create_connectivity(fdim, tdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    for facet in i_facets:
        i_cells = facet_to_cell.links(facet)
        assert(len(i_cells == 1))
        i_cell = i_cells[0]
        if top_cube(cell_midpoints[i_cell]):
            top_interface.append(facet)
        else:
            bottom_interface.append(facet)

    top_cube_marker = 2
    indices = []
    values = []
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            indices.append(cell_index)
            values.append(top_cube_marker)
    ct = dolfinx.mesh.MeshTags(mesh, tdim, np.array(
        indices, dtype=np.intc), np.array(values, dtype=np.intc))

    # Create meshtags for facet data
    markers = {3: top_facets, 4: bottom_interface, 9: top_interface,
               5: bottom_facets, 6: left_facets, 7: right_facets}
    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)

    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key,
                                           dtype=np.intc))
    mt = dolfinx.mesh.MeshTags(mesh, fdim,
                               indices, values)
    mt.name = "facet_tags"
    o_f = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                              "meshes/mesh_{0:s}.xdmf".format(celltype), "w")
    o_f.write_mesh(mesh)
    o_f.write_meshtags(ct)
    o_f.write_meshtags(mt)
    o_f.close()


def mesh_3D_dolfin(theta=0):
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

    N = 2
    ct = dolfinx.cpp.mesh.CellType.tetrahedron
    nv = dolfinx.cpp.mesh.cell_num_vertices(ct)
    mesh0 = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N, ct)
    mesh1 = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2*N, 2*N, 2*N, ct)
    mesh0.geometry.x[:, 2] += 1

    # Stack the two meshes in one mesh
    r_matrix = pygmsh.helpers.rotation_matrix(
        [0, 1/np.sqrt(2), 1/np.sqrt(2)], -theta)
    points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
    points = np.dot(r_matrix, points.T).T

    # Transform topology info into geometry info
    c2v = mesh0.topology.connectivity(mesh0.topology.dim, 0)
    x_dofmap = mesh0.geometry.dofmap
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
    x_dofmap = mesh1.geometry.dofmap
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
            cells1[cell, v] = vertex_to_node[c2v.links(
                cell)[v]] + mesh0.geometry.x.shape[0]
    cells = np.vstack([cells0, cells1])
    mesh = dolfinx.Mesh(MPI.COMM_WORLD,
                        ct, points, cells, [], degree=1,
                        ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Find information about facets to be used in MeshTags
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                               [0, 1, 0], [1, 1, 0]]).T)
    bottom = find_plane_function(bottom_points[:, 0], bottom_points[:, 1],
                                 bottom_points[:, 2])
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, bottom)
    top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2],
                                            [0, 1, 2], [1, 1, 2]]).T)
    top = find_plane_function(top_points[:, 0], top_points[:, 1],
                              top_points[:, 2])
    top_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, top)

    # left_side = find_line_function(top_points[:, 0], top_points[:, 3])
    # left_facets = dolfinx.mesh.locate_entities_boundary(
    #     mesh, fdim, left_side)

    # right_side = find_line_function(top_points[:, 1], top_points[:, 2])
    # right_facets = dolfinx.mesh.locate_entities_boundary(
    #     mesh, fdim, right_side)
    if_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1],
                                           [0, 1, 1], [1, 1, 1]]).T)

    interface = find_plane_function(if_points[:, 0], if_points[:, 1],
                                    if_points[:, 2])
    i_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, interface)
    mesh.topology.create_connectivity(fdim, tdim)
    top_interface = []
    bottom_interface = []
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    top_cube = over_plane(if_points[:, 0], if_points[:, 1],
                          if_points[:, 2])
    for facet in i_facets:
        i_cells = facet_to_cell.links(facet)
        assert(len(i_cells == 1))
        i_cell = i_cells[0]
        if top_cube(cell_midpoints[i_cell]):
            top_interface.append(facet)
        else:
            bottom_interface.append(facet)

    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                range(num_cells))
    top_cube_marker = 2
    indices = []
    values = []
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            indices.append(cell_index)
            values.append(top_cube_marker)
    ct = dolfinx.mesh.MeshTags(mesh, tdim, np.array(
        indices, dtype=np.intc), np.array(values, dtype=np.intc))

    # Create meshtags for facet data
    markers = {3: top_facets, 4: bottom_interface, 9: top_interface,
               5: bottom_facets}  # , 6: left_facets, 7: right_facets}
    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)

    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key,
                                           dtype=np.intc))
    mt = dolfinx.mesh.MeshTags(mesh, fdim,
                               indices, values)
    mt.name = "facet_tags"
    o_f = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                              "meshes/mesh_tetrahedron.xdmf", "w")
    o_f.write_mesh(mesh)
    o_f.write_meshtags(ct)
    o_f.write_meshtags(mt)
    o_f.close()

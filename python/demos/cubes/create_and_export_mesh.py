import meshio
import numpy as np
import pygmsh
import ufl
from mpi4py import MPI

import dolfinx
import dolfinx.fem
import dolfinx.io


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


def generate_rectangle(x0, x1, y0, y1, theta,  res, markers,
                       volume_marker=None, quad=False):
    """
    Generate the rectangle [x0,y0]x[y0,y1], rotated theta degrees around
    the origin with resolution res,
    where markers are is an array of markers for [bottom, left, top, right].
    Optional arguments is a volume marker (default 0), and a flag for
    specifying triangle (default) or quad elements.
    """
    geom = pygmsh.built_in.Geometry()

    rect = geom.add_rectangle(x0, x1, y0, y1, 0.0, res)
    geom.rotate(rect, [0, 0, 0], theta, [0, 0, 1])
    if volume_marker is None:
        geom.add_physical([rect.surface], 0)
    else:
        geom.add_physical([rect.surface], volume_marker)
    for i in range(len(markers)):
        geom.add_physical([rect.line_loop.lines[i]], markers[i])
    if quad:
        geom.add_raw_code("Recombine Surface {:};")
        geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    mesh = pygmsh.generate_mesh(geom, verbose=False, prune_z_0=True)
    return mesh


def generate_hex_box(x0, y0, z0, x1, y1, z1, theta, res, facet_markers,
                     volume_marker=None):
    """
    Generate the box [x0,y0,z0]x[y0,y1,z0], rotated theta degrees around
    the origin over axis [0,1,1] with resolution res,
    where markers are is an array containing markers
    array of markers for [back, bottom, right, left, top, front]
    and an optional volume_marker.
    """
    geom = pygmsh.built_in.Geometry()

    rect = geom.add_rectangle(x0, x1, y0, y1, 0.0, res)

    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    geom.add_raw_code("Recombine Surface {:};")
    bottom, volume, sides = geom.extrude(rect, translation_axis=[0, 0, 1],
                                         num_layers=int(1/res), recombine=True)

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


def generate_box(x0, y0, z0, x1, y1, z1, theta, res, facet_markers,
                 volume_marker=None):
    """
    Generate the box [x0,y0,z0]x[y0,y1,z0], rotated theta degrees around
    the origin over axis [0,1,1] with resolution res,
    where markers are is an array containing markers
    array of markers for [left, back, top, bottom, front, right]
    and an optional volume_marker.
    """
    geom = pygmsh.built_in.Geometry()
    bbox = geom.add_box(x0, x1, y0, y1, z0, z1, res)
    bbox.dimension = 3
    bbox.id = bbox.volume.id
    for i in range(len(bbox.surface_loop.surfaces)):
        geom.add_physical([bbox.surface_loop.surfaces[i]], facet_markers[i])

    geom.rotate(bbox, [0, 0, 0], -theta, [1, 1, 0])
    if volume_marker is None:
        geom.add_physical([bbox.volume], 0)
    else:
        geom.add_physical([bbox.volume], volume_marker)
    msh = pygmsh.generate_mesh(geom, verbose=False)
    return msh


def mesh_2D_gmsh(theta, ct="triangle"):
    """
    Create two stacked cubes rotated by degree theta with triangular
    elements
    """
    if ct == "triangle":
        quad = False
    elif ct == "quad":
        quad = True
    else:
        raise ValueError("Invalid cell type: {0:s}", ct)
    res = 0.1
    msh0 = generate_rectangle(
        0, 1, 0, 1, theta, res=res, markers=[5, 8, 4, 10], quad=quad)
    msh1 = generate_rectangle(
        0, 1, 1, 2, theta, res=2*res, markers=[9, 7, 3, 6], volume_marker=2,
        quad=quad)

    mesh, facet_mesh = merge_msh_meshes(msh0, msh1, ct, "line")
    meshio.write("meshes/mesh_{0:s}_{1:.2f}_gmsh.xdmf"
                 .format(ct, theta), mesh)
    meshio.write("meshes/facet_{0:s}_{1:.2f}_rot.xdmf"
                 .format(ct, theta), facet_mesh)


def mesh_3D_rot(theta=np.pi/2, ct="tetrahedron"):
    res = 0.125
    if ct == "tetrahedron":
        msh0 = generate_box(0, 0, 0, 1, 1, 1, theta, res,
                            [11, 12, 4, 5, 13, 14])
        msh1 = generate_box(0, 0, 1, 1, 1, 2, theta, 2*res,
                            [11, 12, 3, 9, 13, 14], 2)
        mesh, facet_mesh = merge_msh_meshes(msh0, msh1, "tetra", "triangle")
    elif ct == "hexahedron":
        msh0 = generate_hex_box(0, 0, 0, 1, 1, 1, theta, res,
                                [11, 5, 12, 13, 4, 14])
        msh1 = generate_hex_box(0, 0, 1, 1, 1, 2, theta, 2*res,
                                [11, 9, 12, 13, 3, 14], 2)
        mesh, facet_mesh = merge_msh_meshes(msh0, msh1, "hexahedron", "quad")
    else:
        raise ValueError("Celltype has to be tetra or hex.")
    meshio.xdmf.write(
        "meshes/mesh_{0:s}_{1:.2f}_gmsh.xdmf".format(ct, theta), mesh)
    meshio.xdmf.write(
        "meshes/facet_{0:s}_{1:.2f}_gmsh.xdmf".format(ct, theta), facet_mesh)


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
    N = 15
    if celltype == "quadrilateral":
        ct = dolfinx.cpp.mesh.CellType.quadrilateral
    elif celltype == "triangle":
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
    cell = ufl.Cell(celltype, geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)
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
                              "meshes/mesh_{0:s}_{1:.2f}.xdmf"
                              .format(celltype, theta), "w")
    o_f.write_mesh(mesh)
    o_f.write_meshtags(ct)
    o_f.write_meshtags(mt)
    o_f.close()


def mesh_3D_dolfin(theta=0, ct=dolfinx.cpp.mesh.CellType.tetrahedron,
                   ext="tetrahedron"):
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

    N = 3

    nv = dolfinx.cpp.mesh.cell_num_vertices(ct)
    mesh0 = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N, ct)
    mesh1 = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2*N, 2*N, 2*N, ct)
    mesh0.geometry.x[:, 2] += 1

    # Stack the two meshes in one mesh
    r_matrix = pygmsh.helpers.rotation_matrix(
        [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)
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
                              "meshes/mesh_{0:s}_{1:.2f}.xdmf"
                              .format(ext, theta), "w")
    o_f.write_mesh(mesh)
    o_f.write_meshtags(ct)
    o_f.write_meshtags(mt)
    o_f.close()

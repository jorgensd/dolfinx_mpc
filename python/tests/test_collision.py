import dolfinx
import dolfinx.geometry as geometry
import numpy as np
import pytest
from mpi4py import MPI


@pytest.mark.parametrize("coordinate", [[0, 0, 0], [1, 0, 0], [1, 1, 1]])
def test_collision_tetrahedron(coordinate):
    """
    Test alternative collision algorithm which reconstructs the finite element
    from dof coordinates
    """
    mesh = dolfinx.UnitCubeMesh(
        MPI.COMM_WORLD, 1, 1, 1)
    tdim = mesh.topology.dim

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    x_coords = V.tabulate_dof_coordinates()

    # Create cell-to-facet-connectivity
    mesh.topology.create_connectivity_all()
    tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim)
    # Find index of coordinate in dof coordinates
    coordinate_index = -1
    for i, coord in enumerate(x_coords):
        if np.allclose(coord, coordinate):
            coordinate_index = i
            break

    if coordinate_index != -1:
        possible_cells = geometry.compute_collisions_point(
            tree, x_coords[coordinate_index])[0]
        # Compute collisions with boundingboxtree
        # Compute actual collisions with mesh
        actual_collisions = dolfinx.geometry.compute_entity_collisions_mesh(
            tree, mesh, x_coords[coordinate_index].T)

    # Find index of coordinate in geometry
    node_index = -1
    for i, coord in enumerate(mesh.geometry.x):
        if np.allclose(coord, coordinate):
            node_index = i
            break
    if node_index != -1:
        # Test same strategy with mesh coordinates
        possible_cells_mesh = geometry.compute_collisions_point(
            tree, mesh.geometry.x[node_index])[0]
        actual_collisions_mesh = geometry.compute_entity_collisions_mesh(
            tree, mesh, mesh.geometry.x[node_index].T)
    if node_index != -1 and coordinate_index != -1:
        assert(np.allclose(possible_cells_mesh, possible_cells))
        if len(actual_collisions[0]) == len(actual_collisions_mesh[0]):
            print("\n Found points in mesh in standard way for tetrahedron")
            assert(np.allclose(
                actual_collisions[0], actual_collisions_mesh[0]))
    elif node_index == coordinate_index:
        pass
    else:
        print(node_index, coordinate_index)
        raise RuntimeError("Dofs and geometry lives on different processors.")


@pytest.mark.parametrize("celltype", [dolfinx.cpp.mesh.CellType.triangle,
                                      dolfinx.cpp.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("coordinate", [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
def test_collision_2D(celltype, coordinate):
    """
    Test alternative collision algorithm which reconstructs the finite element
    from dof coordinates
    """
    mesh = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, 2, 2, celltype)
    tdim = mesh.topology.dim

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    x_coords = V.tabulate_dof_coordinates()

    # Create cell-to-facet-connectivity
    mesh.topology.create_connectivity_all()
    tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim)
    # Find index of coordinate in dof coordinates
    coordinate_index = -1
    for i, coord in enumerate(x_coords):
        if np.allclose(coord, coordinate):
            coordinate_index = i
            break

    if coordinate_index != -1:
        # Compute collisions with boundingboxtree
        possible_cells = geometry.compute_collisions_point(
            tree, x_coords[coordinate_index])[0]
        # Compute actual collisions with mesh
        actual_collisions = dolfinx.geometry.compute_entity_collisions_mesh(
            tree, mesh, x_coords[coordinate_index].T)

    # Find index of coordinate in geometry
    node_index = -1
    for i, coord in enumerate(mesh.geometry.x):
        if np.allclose(coord, coordinate):
            node_index = i
            break
    if node_index != -1:
        # Test same strategy with mesh coordinates
        possible_cells_mesh = geometry.compute_collisions_point(
            tree, mesh.geometry.x[node_index])[0]
        actual_collisions_mesh = geometry.compute_entity_collisions_mesh(
            tree, mesh, mesh.geometry.x[node_index].T)
    if node_index != -1 and coordinate_index != -1:
        assert(np.allclose(possible_cells_mesh, possible_cells))
        if len(actual_collisions[0]) == len(actual_collisions_mesh[0]):
            print("\n Found points in mesh in standard way for", celltype)
            assert(np.allclose(
                actual_collisions[0], actual_collisions_mesh[0]))
    elif node_index == coordinate_index:
        pass
    else:
        print(node_index, coordinate_index)
        raise RuntimeError("Dofs and geometry lives on different processors.")


@pytest.mark.parametrize("coordinate", [[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]])
def test_collision_1D(coordinate):
    """
    Test alternative collision algorithm which reconstructs the finite element
    from dof coordinates
    """
    mesh = dolfinx.UnitIntervalMesh(
        MPI.COMM_WORLD, 6)
    tdim = mesh.topology.dim

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    x_coords = V.tabulate_dof_coordinates()

    # Create cell-to-facet-connectivity
    mesh.topology.create_connectivity_all()
    tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim)
    # Find index of coordinate in dof coordinates
    coordinate_index = -1
    for i, coord in enumerate(x_coords):
        if np.allclose(coord, coordinate):
            coordinate_index = i
            break

    if coordinate_index != -1:
        # Compute collisions with boundingboxtree
        possible_cells = geometry.compute_collisions_point(
            tree, x_coords[coordinate_index])[0]
        # Compute actual collisions with mesh
        actual_collisions = dolfinx.geometry.compute_entity_collisions_mesh(
            tree, mesh, x_coords[coordinate_index].T)

    # Find index of coordinate in geometry
    node_index = -1
    for i, coord in enumerate(mesh.geometry.x):
        if np.allclose(coord, coordinate):
            node_index = i
            break
    if node_index != -1:
        # Test same strategy with mesh coordinates
        possible_cells_mesh = geometry.compute_collisions_point(
            tree, mesh.geometry.x[node_index])[0]
        actual_collisions_mesh = geometry.compute_entity_collisions_mesh(
            tree, mesh, mesh.geometry.x[node_index].T)
    if node_index != -1 and coordinate_index != -1:
        assert(np.allclose(possible_cells_mesh, possible_cells))
        if len(actual_collisions[0]) == len(actual_collisions_mesh[0]):
            print("\n Found points in mesh in standard way for interval")
            assert(np.allclose(
                actual_collisions[0], actual_collisions_mesh[0]))
    elif node_index == coordinate_index:
        pass
    else:
        print(node_index, coordinate_index)
        raise RuntimeError("Dofs and geometry lives on different processors.")

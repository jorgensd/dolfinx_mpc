import dolfinx_mpc.cpp
import dolfinx
import dolfinx.geometry
import dolfinx.fem
import numpy as np
import pytest


@pytest.mark.parametrize("coordinate", [[0, 0, 0], [1, 0, 0], [1, 1, 1]])
def test_collision_tetrahedron(coordinate):
    """
    Test alternative collision algorithm which reconstructs the finite element
    from dof coordinates
    """
    mesh = dolfinx.UnitCubeMesh(dolfinx.MPI.comm_world, 1, 1, 1)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    x_coords = V.tabulate_dof_coordinates()

    # Find index of coordinate in dof coordinates
    coordinate_index = -1
    for i, coord in enumerate(x_coords):
        if np.allclose(coord, coordinate):
            coordinate_index = i
            break
    assert coordinate_index != -1

    # Find index of coordinate in geometry
    node_index = -1
    for i, coord in enumerate(mesh.geometry.x):
        if np.allclose(coord, coordinate):
            node_index = i
            break
    assert(node_index != -1)

    # Create cell-to-facet-connectivity
    mesh.create_connectivity(tdim, fdim)
    mesh.create_connectivity_all()
    # Compute collisions with boundingboxtree
    tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim)
    possible_cells = dolfinx.geometry.compute_collisions_point(
        tree, x_coords[coordinate_index])[0]

    # Compute actual collisions with mesh
    actual_collisions = dolfinx.geometry.compute_entity_collisions_mesh(
        tree, mesh, x_coords[coordinate_index].T)

    col_2 = dolfinx_mpc.cpp.mpc.check_cell_point_collision(
        possible_cells, V._cpp_object, x_coords[coordinate_index])
    second_cells = possible_cells[col_2 == 1]

    # Test same strategy with mesh coordinates
    possible_cells_mesh = dolfinx.geometry.compute_collisions_point(
        tree, mesh.geometry.x[node_index])[0]
    actual_collisions_mesh = dolfinx.geometry.compute_entity_collisions_mesh(
        tree, mesh, mesh.geometry.x[node_index].T)

    assert(np.allclose(possible_cells_mesh, possible_cells))
    if len(actual_collisions[0]) == len(actual_collisions_mesh[0]):
        print("Found points in mesh in standard way")
        assert(np.allclose(actual_collisions[0], actual_collisions_mesh[0]))
    assert(np.allclose(second_cells, actual_collisions_mesh[0]))

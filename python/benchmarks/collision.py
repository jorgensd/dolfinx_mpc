import dolfinx
import dolfinx.geometry
from mpi4py import MPI
mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
point = [0.2, 0.3, 0.2]
point2 = [0, 0, 0]
first_cell = dolfinx.geometry.compute_colliding_cells(tree, mesh, point)
all_cells = dolfinx.geometry.compute_colliding_cells(tree, mesh, point2, 10)
print(first_cell, all_cells)

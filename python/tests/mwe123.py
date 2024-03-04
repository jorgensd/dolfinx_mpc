from __future__ import annotations

from mpi4py import MPI

import numpy as np
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

import dolfinx_mpc

mesh = create_unit_square(MPI.COMM_WORLD, 1, 1)
V = fem.functionspace(mesh, ("Lagrange", 1))

dolfinx_mpc.cpp.mpc.test_tabulate(V._cpp_object)


# Create multipoint constraint
def periodic_relation(x):
    out_x = np.copy(x)
    out_x[0] = 1 - x[0]
    return out_x


def PeriodicBoundary(x):
    return np.isclose(x[0], 1)


facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
arg_sort = np.argsort(facets)
mt = meshtags(mesh, mesh.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))

mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, mt, 2, periodic_relation, [], default_scalar_type(1.0))

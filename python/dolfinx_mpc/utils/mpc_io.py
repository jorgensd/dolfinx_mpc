# Copyright (C) 2021 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import gmsh as _gmsh
import numpy
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.cpp.io import distribute_entity_data, perm_gmsh
from dolfinx.cpp.mesh import cell_entity_type, to_type
from dolfinx.io import (extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh, create_meshtags
from mpi4py import MPI as _MPI


def read_from_msh(filename: str, cell_data=False, facet_data=False, gdim=None):
    """
    Reads a mesh from a msh-file and returns the dolfin-x mesh.
    Input:
        filename: Name of msh file
        cell_data: Boolean, True of a mesh tag for cell data should be returned
                   (Default: False)
        facet_data: Boolean, True if a mesh tag for facet data should be
                    returned (Default: False)
        gdim: Geometrical dimension of problem (Default: 3)
    """
    if _MPI.COMM_WORLD.rank == 0:
        # Check if gmsh is already initialized
        try:
            current_model = _gmsh.model.getCurrent()
        except ValueError:
            current_model = None
            _gmsh.initialize()

        _gmsh.model.add("Mesh from file")
        _gmsh.merge(filename)
    output = gmsh_model_to_mesh(_gmsh.mode.getCurrent(), cell_data=cell_data, facet_data=facet_data, gdim=gdim)
    if _MPI.COMM_WORLD.rank == 0:
        if current_model is None:
            _gmsh.finalize()
        else:
            _gmsh.model.setCurrent(current_model)
    return output


def gmsh_model_to_mesh(model, cell_data=False, facet_data=False, gdim=None):
    """
    Given a GMSH model, create a DOLFIN-X mesh and meshtags.
        model: The GMSH model
        cell_data: Boolean, True of a mesh tag for cell data should be returned
                   (Default: False)
        facet_data: Boolean, True if a mesh tag for facet data should be
                    returned (Default: False)
        gdim: Geometrical dimension of problem (Default: 3)
    """

    if gdim is None:
        gdim = 3

    if _MPI.COMM_WORLD.rank == 0:
        # Get mesh geometry
        x = extract_gmsh_geometry(model)

        # Get mesh topology for each element
        topologies = extract_gmsh_topology_and_markers(model)

        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
        for i, element in enumerate(topologies.keys()):
            properties = model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = numpy.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = _MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)

        # Check for facet data and broadcast if found
        if facet_data:
            if tdim - 1 in cell_dimensions:
                num_facet_nodes = _MPI.COMM_WORLD.bcast(
                    cell_information[perm_sort[-2]]["num_nodes"], root=0)
                gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
                marked_facets = topologies[gmsh_facet_id]["topology"]
                facet_values = numpy.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=numpy.int32)
            else:
                raise ValueError("No facet data found in file.")

        cells = topologies[cell_id]["topology"]
        cell_values = numpy.asarray(topologies[cell_id]["cell_data"], dtype=numpy.int32)

    else:
        cell_id, num_nodes = _MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = numpy.empty([0, num_nodes]), numpy.empty([0, gdim])
        cell_values = numpy.empty((0,), dtype=numpy.int32)
        if facet_data:
            num_facet_nodes = _MPI.COMM_WORLD.bcast(None, root=0)
            marked_facets = numpy.empty((0, num_facet_nodes), numpy.int32)
            facet_values = numpy.empty((0,), dtype=numpy.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = numpy.asarray(cells[:, gmsh_cell_perm], dtype=numpy.int64)
    mesh = create_mesh(_MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)
    # Create meshtags for cells
    if cell_data:
        local_entities, local_values = distribute_entity_data(mesh, mesh.topology.dim, cells, cell_values)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = AdjacencyList_int32(local_entities)
        ct = create_meshtags(mesh, mesh.topology.dim, adj, numpy.asarray(local_values, dtype=numpy.int32))
        ct.name = "Cell tags"

    # Create meshtags for facets
    if facet_data:
        # Permute facets from MSH to Dolfin-X ordering
        # FIXME: We need to account for multiple facet types in prism meshes, then last argument has to change
        facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())), mesh.topology.dim - 1, 0)
        gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
        marked_facets = numpy.asarray(marked_facets[:, gmsh_facet_perm], dtype=numpy.int32)
        local_entities, local_values = distribute_entity_data(mesh, mesh.topology.dim - 1, marked_facets, facet_values)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        adj = AdjacencyList_int32(local_entities)
        ft = create_meshtags(mesh, mesh.topology.dim - 1, adj, numpy.asarray(local_values, dtype=numpy.int32))
        ft.name = "Facet tags"

    if cell_data and facet_data:
        return mesh, ct, ft
    elif cell_data and not facet_data:
        return mesh, ct
    elif not cell_data and facet_data:
        return mesh, ft
    else:
        return mesh

# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numba
import numpy


@numba.njit(fastmath=True, cache=True)
def extract_slave_cells(cell_offset: "numpy.ndarray[numpy.int32]") -> "numpy.ndarray[numpy.int32]":
    """ 
    From an offset determine which entries are nonzero.
    Parameters
    ----------
    cell_offset
        The offsets of an adjacency list describing how many elements there are per node.

    Example
    -------
    .. code-block:: python

       offsets = [0,0,1,3,4,4,6,6,8]
       extract_slave_cells(offsets) -> [1,2,3,5,7]
    """
    slave_cells = numpy.zeros(len(cell_offset) - 1, dtype=numpy.int32)
    c = 0
    for cell in range(len(cell_offset) - 1):
        num_cells = cell_offset[cell + 1] - cell_offset[cell]
        if num_cells > 0:
            slave_cells[c] = cell
            c += 1
    return slave_cells[:c]


@numba.njit(fastmath=True, cache=True)
def pack_slave_facet_info(facets: 'numpy.ndarray[numpy.int32, int]',
                          cells: 'numpy.ndarray[numpy.int32]') -> 'numpy.ndarray[numpy.int32, int]':
    """
    Given an MPC and a set of facets (cell index, local_facet_index),
    compress the set to those that are related to a certain set of cells.

    Parameters
    ----------
    facets
       Array of tuples `(cell_index, facet_index)` where `cell_index` is local to process, 
       `facet_index` is local to cell.
    cells
       List of cells
    """
    facet_info = numpy.zeros((len(facets), 2), dtype=numpy.int32)
    i = 0
    for facet in facets:
        if sum(cells == facet[0]) > 0:
            facet_info[i, :] = [facet[0], facet[1]]
            i += 1
    return facet_info[:i, :]

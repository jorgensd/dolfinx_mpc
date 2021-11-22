# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

import numba
import numpy


@numba.njit(fastmath=True, cache=True)
def extract_slave_cells(cell_offset: "numpy.ndarray[numpy.int32]") -> "numpy.ndarray[numpy.int32]":
    """ From an offset determine which entries are nonzero"""
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
                          slave_cells: 'numpy.ndarray[numpy.int32]') -> 'numpy.ndarray[numpy.int32, int]':
    """
    Given an MPC and a set of facets (cell index, local_facet_index),
    compress the set to those that only contain slave cells
    """
    facet_info = numpy.zeros((len(facets), 2), dtype=numpy.int32)
    i = 0
    for facet in facets:
        if sum(slave_cells == facet[0]) > 0:
            facet_info[i, :] = [facet[0], facet[1]]
            i += 1
    return facet_info[:i, :]

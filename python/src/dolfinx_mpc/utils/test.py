# Copyright (C) 2021-2022 Jørgen Schartum Dokken and Connor D. Pierce
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

__all__ = [
    "gather_PETScVector",
    "gather_PETScMatrix",
    "compare_mpc_lhs",
    "compare_mpc_rhs",
    "gather_transformation_matrix",
    "compare_CSR",
]

from typing import Any

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.common
import numpy as np
import scipy.sparse

import dolfinx_mpc


def _gather_slaves_global(constraint):
    """
    Given a multi point constraint, return slaves for all processors with global dof numbering
    """
    imap = constraint.function_space.dofmap.index_map
    num_local_slaves = constraint.num_local_slaves
    block_size = constraint.function_space.dofmap.index_map_bs
    _slaves = constraint.slaves
    if num_local_slaves > 0:
        slave_blocks = _slaves[:num_local_slaves] // block_size
        slave_rems = _slaves[:num_local_slaves] % block_size
        glob_slaves = np.asarray(imap.local_to_global(slave_blocks), dtype=np.int64) * block_size + slave_rems
    else:
        glob_slaves = np.array([], dtype=np.int64)

    slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    return slaves


def gather_constants(constraint, root=0):
    """
    Given a multi-point constraint, gather all constants
    """
    imap = constraint.index_map()
    constants = constraint._cpp_object.constants
    l_range = imap.local_range
    ranges = MPI.COMM_WORLD.gather(np.asarray(l_range, dtype=np.int64), root=root)
    g_consts = MPI.COMM_WORLD.gather(constants[: l_range[1] - l_range[0]], root=root)
    if MPI.COMM_WORLD.rank == root:
        block_size = constraint.function_space().dofmap.index_map_bs
        global_consts = np.zeros(imap.size_global * block_size, dtype=constraint.coefficients()[0].dtype)
        for r, vals in zip(ranges, g_consts):
            global_consts[r[0] : r[1]] = vals
        return global_consts
    else:
        return


def gather_transformation_matrix(constraint, root=0):
    """
    Creates the transformation matrix K (dim x dim-len(slaves)) for a given MPC
    and gathers it as a scipy CSR matrix on process 'root'.

    Example:

    For dim=3, where:
      u_1 = alpha u_0 + beta u_2

    Input:
      slaves = [1]
      masters = [0, 2]
      coeffs = [alpha, beta]
      offsets = [0, 1]

    Output:
      K = [[1,0], [alpha beta], [0,1]]
    """
    # Gather slaves from all procs
    V = constraint.V
    imap = constraint.function_space.dofmap.index_map
    block_size = V.dofmap.index_map_bs
    num_local_slaves = constraint.num_local_slaves
    # Gather all global_slaves
    slaves = constraint.slaves[:num_local_slaves]
    if num_local_slaves > 0:
        local_blocks = slaves // block_size
        local_rems = slaves % block_size
        glob_slaves = np.asarray(imap.local_to_global(local_blocks), dtype=np.int64) * block_size + local_rems
    else:
        glob_slaves = np.array([], dtype=np.int64)
    all_slaves = np.hstack(MPI.COMM_WORLD.allgather(glob_slaves))
    masters = constraint.masters.array
    master_blocks = masters // block_size
    master_rems = masters % block_size
    coeffs = constraint.coefficients()[0]
    offsets = constraint.masters.offsets
    # Create sparse K matrix
    K_val, rows, cols = [], [], []

    # Add local contributions to K from local slaves
    for slave, global_slave in zip(slaves, glob_slaves):
        masters_index = (
            np.asarray(
                imap.local_to_global(master_blocks[offsets[slave] : offsets[slave + 1]]),
                dtype=np.int64,
            )
            * block_size
            + master_rems[offsets[slave] : offsets[slave + 1]]
        )
        coeffs_index = coeffs[offsets[slave] : offsets[slave + 1]]
        # If we have a simply equality constraint (dirichletbc)
        if len(masters_index) > 0:
            for master, coeff in zip(masters_index, coeffs_index):
                count = sum(master > all_slaves)
                K_val.append(coeff)
                rows.append(global_slave)
                cols.append(master - count)
        else:
            K_val.append(1)
            count = sum(global_slave > all_slaves)
            rows.append(global_slave)
            cols.append(global_slave - count)

    # Add identity for all dofs on diagonal
    l_range = V.dofmap.index_map.local_range
    global_dofs = np.arange(l_range[0] * block_size, l_range[1] * block_size)
    is_slave = np.isin(global_dofs, glob_slaves)
    for i, dof in enumerate(global_dofs):
        if not is_slave[i]:
            K_val.append(1)
            rows.append(dof)
            cols.append(dof - sum(dof > all_slaves))

    # Gather K to root
    K_vals = MPI.COMM_WORLD.gather(np.asarray(K_val, dtype=coeffs.dtype), root=root)
    rows_g = MPI.COMM_WORLD.gather(np.asarray(rows, dtype=np.int64), root=root)
    cols_g = MPI.COMM_WORLD.gather(np.asarray(cols, dtype=np.int64), root=root)

    if MPI.COMM_WORLD.rank == root:
        K_sparse = scipy.sparse.coo_matrix((np.hstack(K_vals), (np.hstack(rows_g), np.hstack(cols_g)))).tocsr()
        return K_sparse


def petsc_to_local_CSR(A: PETSc.Mat, mpc: dolfinx_mpc.MultiPointConstraint):  # type: ignore
    """
    Convert a PETSc matrix to a local CSR matrix (scipy) including ghost entries
    """
    global_indices = np.asarray(mpc.function_space.dofmap.index_map.global_indices(), dtype=PETSc.IntType)  # type: ignore
    sort_index = np.argsort(global_indices)
    is_A = PETSc.IS().createGeneral(global_indices[sort_index])  # type: ignore
    A_loc = A.createSubMatrices(is_A)[0]
    ai, aj, av = A_loc.getValuesCSR()
    A_csr = scipy.sparse.csr_matrix((av, aj, ai))
    return A_csr[global_indices[:, None], global_indices]


def gather_PETScMatrix(A: PETSc.Mat, root=0) -> scipy.sparse.csr_matrix:  # type: ignore
    """
    Given a distributed PETSc matrix, gather in on process 'root' in
    a scipy CSR matrix
    """
    ai, aj, av = A.getValuesCSR()
    aj_all = MPI.COMM_WORLD.gather(aj, root=root)  # type: ignore
    av_all = MPI.COMM_WORLD.gather(av, root=root)  # type: ignore
    ai_all = MPI.COMM_WORLD.gather(ai, root=root)  # type: ignore
    if MPI.COMM_WORLD.rank == root:
        ai_cum = [0]
        for ai in ai_all:  # type: ignore
            offsets = ai[1:] + ai_cum[-1]
            ai_cum.extend(offsets)
        return scipy.sparse.csr_matrix((np.hstack(av_all), np.hstack(aj_all), ai_cum), shape=A.getSize())  # type: ignore


def gather_PETScVector(vector: PETSc.Vec, root=0) -> np.ndarray:  # type: ignore
    """
    Gather a PETScVector from different processors on
    process 'root' as an numpy array
    """
    if vector.handle == 0:
        raise RuntimeError("Vector has been destroyed prior to this call")
    numpy_vec = np.zeros(vector.size, dtype=vector.array.dtype)
    l_min = vector.owner_range[0]
    l_max = vector.owner_range[1]
    numpy_vec[l_min:l_max] += vector.array
    return np.asarray(sum(MPI.COMM_WORLD.allgather(numpy_vec)))


def compare_CSR(A: scipy.sparse.csr_matrix, B: scipy.sparse.csr_matrix, atol=1e-10):
    """Compare CSR matrices A and B"""
    diff = np.abs(A - B)
    assert diff.max() < atol


def compare_mpc_lhs(
    A_org: PETSc.Mat,  # type: ignore
    A_mpc: PETSc.Mat,  # type: ignore
    mpc: dolfinx_mpc.MultiPointConstraint,
    root: int = 0,
    atol: np.floating[Any] = 5e3 * np.finfo(dolfinx.default_scalar_type).resolution,
):
    """
    Compare an unmodified matrix for the problem with the one assembled with a
    multi point constraint.

    The unmodified matrix is multiplied with K^T A K, where K is the global transformation matrix.
    """
    timer = dolfinx.common.Timer("~MPC: Compare matrices")
    comm = mpc.V.mesh.comm
    V = mpc.V
    assert root < comm.size
    is_complex = np.issubdtype(mpc.coefficients()[0].dtype, np.complexfloating)  # type: ignore
    scipy_dtype = np.complex128 if is_complex else np.float64
    K = gather_transformation_matrix(mpc, root=root)
    A_csr = gather_PETScMatrix(A_org, root=root)
    # Get global slaves
    glob_slaves = _gather_slaves_global(mpc)
    A_mpc_csr = gather_PETScMatrix(A_mpc, root=root)
    if MPI.COMM_WORLD.rank == root:
        K = K.astype(scipy_dtype)
        A_csr = A_csr.astype(scipy_dtype)
        KTAK = np.conj(K.T) * A_csr * K

        # Remove identity rows of MPC matrix
        all_cols = np.arange(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

        cols_except_slaves = np.flatnonzero(np.isin(all_cols, glob_slaves, invert=True).astype(np.int32))
        mpc_without_slaves = A_mpc_csr[cols_except_slaves[:, None], cols_except_slaves]

        # Compute difference
        compare_CSR(KTAK, mpc_without_slaves, atol=atol)

    timer.stop()


def compare_mpc_rhs(
    b_org: PETSc.Vec,  # type: ignore
    b: PETSc.Vec,  # type: ignore
    constraint: dolfinx_mpc.MultiPointConstraint,
    root: int = 0,
):
    """
    Compare an unconstrained RHS with an MPC rhs.
    """
    glob_slaves = _gather_slaves_global(constraint)
    b_org_np = gather_PETScVector(b_org, root=root)
    b_np = gather_PETScVector(b, root=root)
    K = gather_transformation_matrix(constraint, root=root)
    # constants = gather_constants(constraint)
    comm = constraint.V.mesh.comm
    if comm.rank == root:
        reduced_b = np.conj(K.T) @ b_org_np  # - constants for RHS mpc
        all_cols = np.arange(constraint.V.dofmap.index_map.size_global * constraint.V.dofmap.index_map_bs)
        cols_except_slaves = np.flatnonzero(np.isin(all_cols, glob_slaves, invert=True).astype(np.int32))
        assert np.allclose(b_np[glob_slaves], 0)
        assert np.allclose(b_np[cols_except_slaves], reduced_b)

# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Express a scalar integral condition as a multi point constraint."""

from __future__ import annotations

import typing

from mpi4py import MPI

import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import default_scalar_type, la


def create_integral_constraint(
    V: _fem.FunctionSpace,
    weight_form: ufl.Form,
    value: np.floating | np.complexfloating | float | complex,
    bcs: typing.Optional[typing.Sequence[_fem.DirichletBC]] = None,
    rtol: np.floating | float = 1e-14,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    _fem.Function,
]:
    r"""Build the constraint data enforcing a scalar integral condition.

    A linear functional :math:`L` applied to :math:`u_h=\sum_i u_i\phi_i` is a
    single linear equation between all the coefficients,

    .. math::
        L(u_h) = \sum_i w_i u_i = \gamma, \qquad w_i = L(\phi_i),

    where :math:`w` is the assembled vector of ``weight_form``. Solving it for
    one degree of freedom :math:`s` gives the affine relation

    .. math::
        u_s = \sum_{i \neq s} \left(-\frac{w_i}{w_s}\right) u_i
              + \frac{\gamma}{w_s},

    that is a constraint with a single slave, every other degree of freedom in
    the support of the functional as a master, and the inhomogeneity
    :math:`\gamma/w_s`. Typical uses are fixing the mean value of a solution,
    :math:`\int_\Omega u~\mathrm{d}x = \gamma`, which makes a pure Neumann
    problem non-singular, or prescribing a boundary average or a flow rate,
    :math:`\int_\Gamma u\cdot n~\mathrm{d}s = \gamma`, without prescribing the
    profile that carries it.

    Args:
        V: The function space the constraint acts on.
        weight_form: A linear form in ``ufl.TestFunction(V)`` defining the
            functional, for instance ``v * ufl.dx`` or
            ``ufl.dot(v, n) * ds(tag)``.
        value: The prescribed value :math:`\gamma` of the functional.
        bcs: Dirichlet conditions on ``V``. A constrained degree of freedom is
            never chosen as the slave, but may still appear as a master, in
            which case passing the same conditions to
            :class:`MultiPointConstraint` folds its contribution into the
            constraint offset.
        rtol: Discard a master whose coefficient is below this fraction of the
            largest one. The weights of degrees of freedom outside the support
            of the functional are returned by quadrature as roundoff rather than
            as exact zeros, so the threshold is relative rather than a test
            against zero.

    Returns:
        The ``slaves``, ``masters``, ``coeffs``, ``owners`` and ``offsets``
        arrays accepted by :meth:`MultiPointConstraint.add_constraint`, and a
        :class:`dolfinx.fem.Function` holding the inhomogeneity, to be passed to
        :class:`MultiPointConstraint` as ``rhs_coeffs``.

    Note:
        Collective. Every process contributes its owned weights to the slave's
        owner, which is the only one that needs the whole functional; the owner
        then reduces them to the (generally far smaller) list of significant
        masters and forwards that, not the raw weights, to any process ghosting
        the slave. Those are the only processes that declare the constraint.
    """
    bcs = [] if bcs is None else list(bcs)
    comm = V.mesh.comm
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    num_owned = imap.size_local * bs
    dtype = np.dtype(default_scalar_type)
    mpi_scalar = MPI._typedict[dtype.char]

    arguments = ufl.algorithms.extract_arguments(weight_form)
    if len(arguments) != 1 or arguments[0].number() != 0:
        raise ValueError(f"weight_form must be linear in a single test function, got {len(arguments)} argument(s)")
    if arguments[0].ufl_function_space() != V:
        raise ValueError("The test function of weight_form must be in the function space of the constraint")

    # Assemble the functional and accumulate ghost contributions onto the owner
    w = _fem.assemble_vector(_fem.form(weight_form, dtype=dtype))
    w.scatter_reverse(la.InsertMode.add)
    w_owned = w.array[:num_owned]

    # A Dirichlet degree of freedom may be a master, but must not be the slave.
    # Zero those weights in a scratch copy: `set` writes alpha * (value - x0), so
    # alpha=0 masks them whatever the condition prescribes.
    masked = w.array.copy()
    for bc in bcs:
        bc.set(masked, alpha=0.0)

    # The largest weight makes the best slave, since it bounds every coefficient
    # by one in magnitude. That matters, because cond(K^H A K) grows with the
    # square of the coefficient norm and the weights can span many orders of
    # magnitude. Reducing over the masked weights picks it without communicating
    # either the weights or the Dirichlet markers.
    local_best = int(np.argmax(np.abs(masked[:num_owned]))) if num_owned > 0 else 0
    local_magnitude = float(abs(masked[local_best])) if num_owned > 0 else -1.0
    best_magnitude, slave_global = comm.allreduce(
        (local_magnitude, int(imap.local_range[0] * bs) + local_best if num_owned > 0 else -1),
        op=MPI.MAXLOC,
    )
    # `best_magnitude` is |w_s|, never the signed weight, so it is negative only
    # for the sentinel a process with no owned dofs contributes. Comparing it
    # against zero would be too weak: a weight outside the support of the
    # functional comes back from quadrature as roundoff rather than as an exact
    # zero, so a slave whose weight is negligible *relative to the functional*
    # would otherwise be accepted and give coefficients of order 1/rtol.
    scale = comm.allreduce(float(np.abs(w_owned).max(initial=0.0)), op=MPI.MAX)
    if best_magnitude <= rtol * scale:
        raise RuntimeError(
            "No admissible slave: the functional vanishes, to within rtol, on "
            "every degree of freedom that is not constrained by a Dirichlet condition"
        )

    # Only the processes holding the slave declare the constraint, so only they
    # need the weights: gather to one of them and forward to the few others.
    #
    # `global_to_local` gives the slave's local block on every rank that owns
    # or ghosts it; `imap.local_range`/`size_local` then say which of the two
    # for free, no communication, since both are purely local properties of
    # the index map. A rank that only ghosts the slave also already knows who
    # owns it: `owners` is aligned with the ghost list, so the position of the
    # slave's block within it (past `size_local`) is the owning rank directly,
    # again with no communication -- the whole point of ghost ownership info
    # existing on the index map in the first place.
    slave_block = int(imap.global_to_local(np.array([slave_global // bs], dtype=np.int64))[0])
    holds_slave = slave_block != -1
    is_owner = holds_slave and slave_block < imap.size_local
    owner_hint = comm.rank if is_owner else (int(imap.owners[slave_block - imap.size_local]) if holds_slave else -1)
    # Every other process still needs to know root, if only to call the
    # Gatherv below with a value of `root` that agrees with everyone else's;
    # propagating the one fact a holding rank already has to the rest of the
    # communicator is a single scalar reduction (there is exactly one owner,
    # so MAX recovers it) rather than an allgather of one integer per rank.
    root = comm.allreduce(owner_hint, op=MPI.MAX)
    assert root != -1, "No process holds the slave, but one must"

    # The owned degrees of freedom of a process form a contiguous global
    # range, and the processes concatenate in rank order, so a gathered
    # entry's position is already its global index; `counts` is needed for
    # the Gatherv below regardless, to size and place each process's part.
    counts = np.array(comm.allgather(num_owned), dtype=np.int32)
    displ = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    total = int(counts.sum())

    # Root needs to know who else to Send to, i.e. who ghosts the slave. Rather
    # than every process announcing whether it holds the slave (an allgather
    # over the whole communicator), only the ranks that ghost it declare an
    # edge, to the single rank they already know is the owner; root discovers
    # them by building the graph and reading back its incoming edges. Every
    # process must still call this (it is collective), but only a handful ever
    # declare a nonzero degree, so the underlying exchange is between just
    # those ranks and root instead of all of them.
    destinations = [root] if (holds_slave and not is_owner) else []
    graph = comm.Create_dist_graph([comm.rank], [len(destinations)], destinations, reorder=False)
    ghost_ranks, _, _ = graph.Get_dist_neighbors()
    graph.Free()

    all_weights = np.empty(total, dtype=dtype) if is_owner else None
    # mpi4py reads a three-entry tuple as (buffer, counts, datatype), so the
    # datatype has to be spelled out whenever displacements are given
    recvbuf = (all_weights, counts, displ, mpi_scalar) if is_owner else None
    comm.Gatherv(np.ascontiguousarray(w_owned, dtype=dtype), recvbuf, root=root)

    rhs_coeffs = _fem.Function(V, dtype=dtype)
    rhs_coeffs.x.array[:] = 0
    slaves = np.zeros(0, dtype=np.int32)
    masters = np.zeros(0, dtype=np.int64)
    coeffs = np.zeros(0, dtype=dtype)
    owners = np.zeros(0, dtype=np.int32)
    offsets = np.zeros(1, dtype=np.int32)
    if is_owner:
        assert all_weights is not None
        w_slave = all_weights[slave_global]
        # A global master index is its position in `all_weights` (established
        # above), so building an index array and deleting the slave's entry from
        # it -- and separately from the coefficients and owners -- is three
        # total-sized allocate-and-copy passes for something one boolean mask
        # does in one. The slave's own would-be coefficient is exactly ±1 (it
        # has the largest |w| by construction), i.e. the largest a coefficient
        # can ever be, so it must be zeroed before the max below, not just
        # excluded from the final selection, or it would silently set the
        # threshold instead of being subject to it.
        all_coeffs = -all_weights / w_slave
        all_coeffs[slave_global] = 0.0
        # Drop negligible coefficients; they change nothing in the constraint but
        # each costs a ghost, a row of the sparsity pattern and an entry in every
        # element matrix modification. Filtering here, once, rather than after
        # forwarding the raw weights to each ghost, means every ghost gets the
        # already-reduced answer instead of redoing this reduction on its own
        # copy of the (generally much larger) dense weight vector.
        significant = np.abs(all_coeffs) >= rtol * np.abs(all_coeffs).max(initial=0.0)
        significant[slave_global] = False  # never a master of itself, even if threshold == 0
        all_owners_full = np.repeat(np.arange(comm.size, dtype=np.int32), counts)
        masters = np.flatnonzero(significant).astype(np.int64)
        coeffs = np.ascontiguousarray(all_coeffs[significant], dtype=dtype)
        owners = np.ascontiguousarray(all_owners_full[significant], dtype=np.int32)
        # Non-blocking: issuing every send before waiting on any of them means
        # root pays for the slowest destination once, not the sum of all of
        # them. ghost_ranks stays bounded by the slave's local mesh valence
        # (how many subdomains meet at that dof), not by the size of comm, but
        # there is no reason to pay sequential latency for it regardless.
        requests = [comm.isend((masters, coeffs, owners, w_slave), dest=int(other), tag=0) for other in ghost_ranks]
        MPI.Request.waitall(requests)
    elif holds_slave:
        masters, coeffs, owners, w_slave = comm.recv(source=root, tag=0)

    if holds_slave:
        slaves = np.array([slave_block * bs + slave_global % bs], dtype=np.int32)
        offsets = np.array([0, masters.size], dtype=np.int32)
        rhs_coeffs.x.array[slaves[0]] = dtype.type(value) / w_slave
    rhs_coeffs.x.scatter_forward()
    return slaves, masters, coeffs, owners, offsets, rhs_coeffs

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
import matplotlib.transforms as mtransforms
import h5py
import mpi4py
import argparse
import sys
import numpy as np


def visualize_elasticity():
    f = h5py.File('bench_edge_output.hdf5', 'r',
                  driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    iterations = f.get("its")[:]
    dofs = f.get("num_dofs")[:]
    slaves = np.sum(f.get("num_slaves")[:], axis=1)
    solver = f.get("solve_time").attrs["solver"].decode("utf-8")
    ct = f.get("solve_time").attrs["ct"].decode("utf-8")
    degree = f.get("solve_time").attrs["degree"].decode("utf-8")
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(dofs, iterations, "-ro", label="MPC", markersize=12)

    f_ref = h5py.File('elasticity_ref.hdf5', 'r', driver='mpio',
                      comm=mpi4py.MPI.COMM_WORLD)
    iterations_ref = f_ref.get("its")[:]
    dofs_ref = f_ref.get("num_dofs")[:]
    f_ref.close()

    plt.plot(dofs_ref, iterations_ref, "-bs", label="Unconstrained")

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale("log")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("# DOFS", fontsize=20)
    plt.ylabel("# Iterations", fontsize=20)
    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                           x=0.025, y=-0.1, units='inches')
    for i in range(len(iterations)):
        plt.text(dofs[i], iterations[i], slaves[i],
                 transform=trans_offset, fontsize=20)
    plt.title("Linear elasticity (CG{0:s}, {1:s}) with {2:s}".format(
        degree, ct, solver), fontsize=25)
    plt.legend(fontsize=15)
    ax.minorticks_on()
    ax.set_ylim([0, max([max(iterations), max(iterations_ref)]) + 1])
    ax.set_xlim([1e2, 1e8])
    locmax = LogLocator(
        base=10.0,
        numticks=8)
    ax.xaxis.set_major_locator(locmax)
    locmin = LogLocator(
        base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        numticks=9)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.grid(True, which="both", axis="both")
    plt.savefig("elasticity_iterations_CG{0:s}_{1:s}.png".format(
        degree, ct), bbox_inches='tight')


def visualize_periodic():
    f = h5py.File('periodic_output.hdf5', 'r',
                  driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    iterations = f.get("its")[:]
    dofs = f.get("num_dofs")[:]
    slaves = np.sum(f.get("num_slaves")[:], axis=1)

    solver = f.get("solve_time").attrs["solver"].decode("utf-8")
    ct = f.get("solve_time").attrs["ct"].decode("utf-8")
    degree = f.get("solve_time").attrs["degree"].decode("utf-8")
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(dofs, iterations, "-ro", label="MPC", markersize=12)

    f_ref = h5py.File('periodic_ref_output.hdf5', 'r',
                      driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    iterations_ref = f_ref.get("its")[:]
    dofs_ref = f_ref.get("num_dofs")[:]
    f_ref.close()

    plt.plot(dofs_ref, iterations_ref, "-bs", label="Unconstrained")

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale("log")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0, max(iterations) + 1])
    ax.set_xlim([1e2, max(dofs) + 1])
    plt.xlabel("# DOFS", fontsize=20)
    plt.ylabel("# Iterations", fontsize=20)

    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                           x=0.025, y=0.025, units='inches')
    for i in range(len(iterations)):
        plt.text(dofs[i], iterations[i], slaves[i],
                 transform=trans_offset, fontsize=20)
    plt.title("Periodic Poisson (CG {0:s}, {1:s}) with {2:s}".format(
        degree, ct, solver), fontsize=25)
    plt.legend(fontsize=15)
    ax.minorticks_on()
    ax.set_ylim([0, max(iterations) + 1])
    ax.set_xlim([1e2, 1e8])
    locmax = LogLocator(
        base=10.0,
        numticks=8)
    ax.xaxis.set_major_locator(locmax)
    locmin = LogLocator(
        base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        numticks=9)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.grid(True, which="both", axis="both")
    plt.savefig("periodic_iterations_CG{0:s}_{1:s}.png".format(
        degree, ct), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--elasticity", action="store_true",
                        dest="elasticity", default=False,
                        help="Visualize iterations for elasticity")
    parser.add_argument("--periodic", action="store_true",
                        dest="periodic", default=False,
                        help="Visualize iterations for periodic")
    args = parser.parse_args()
    thismodule = sys.modules[__name__]
    periodic = elasticity = None
    for key in vars(args):
        setattr(thismodule, key, getattr(args, key))
    if elasticity:
        visualize_elasticity()

    if periodic:
        visualize_periodic()

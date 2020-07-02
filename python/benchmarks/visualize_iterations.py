import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as mtransforms
import numpy as np
import h5py
import mpi4py
import argparse
import sys


def visualize_elasticity():
    f = h5py.File('bench_edge_output.hdf5', 'r',
                  driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    iterations = f.get("its")[:]
    dofs = f.get("num_dofs")[:]
    slaves = f.get("num_slaves")[:]
    solver = f.get("solve_time").attrs["solver"]
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(dofs, iterations, "-ro", label="MPC", markersize=12)

    f_ref = h5py.File('ref_output.hdf5', 'r', driver='mpio',
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
                                           x=0.025, y=0.025, units='inches')
    for i in range(len(iterations)):
        plt.text(dofs[i], iterations[i], slaves[i],
                 transform=trans_offset, fontsize=20)
    plt.title("Linear elasticity with {0:s}".format(solver), fontsize=25)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("fig.png", bbox_inches='tight')


def visualize_periodic():
    f = h5py.File('periodic_output.hdf5', 'r',
                  driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    iterations = f.get("its")[:]
    dofs = f.get("num_dofs")[:]
    slaves = f.get("num_slaves")[:]
    solver = f.get("solve_time").attrs["solver"]
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
    ax.set_ylim([0, max(iterations)+1])
    ax.set_xlim([1e2, max(dofs)+1])

    plt.xlabel("# DOFS", fontsize=20)
    plt.ylabel("# Iterations", fontsize=20)

    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                           x=0.025, y=0.025, units='inches')
    for i in range(len(iterations)):
        plt.text(dofs[i], iterations[i], slaves[i],
                 transform=trans_offset, fontsize=20)
    plt.title("Periodic Poisson with " + solver, fontsize=25)
    plt.legend(fontsize=15)
    plt.grid(True, which="both")
    plt.savefig("periodic_iterations.png", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elasticity", action="store_true", dest="elasticity", default=False,
                        help="Visualize iterations for elasticity")
    parser.add_argument("--periodic", action="store_true", dest="periodic", default=False,
                        help="Visualize iterations for periodic")
    args = parser.parse_args()
    thismodule = sys.modules[__name__]
    for key in vars(args):
        setattr(thismodule, key, getattr(args, key))
    if elasticity:
        visualize_elasticity()

    if periodic:
        visualize_periodic()

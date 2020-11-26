from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


dofs = [31776, 67281]  # , 234546, 1801086]


def visualize_side_by_side(dofs):
    fig, ax = plt.subplots()
    plt.grid("on", zorder=1)
    procs = []
    first = True
    slaves = []
    for i, dof in enumerate(dofs):
        infile = open("results_bench_{0:d}.txt".format(dof), "r")

        # Read problem info
        procs.append(infile.readline().split(": ")[-1].strip("\n"))
        # Skip num dofs
        infile.readline()
        slaves .append(int(infile.readline().split(": ")[-1].strip("\n")))
        solve_iterations = int(infile.readline().split(": ")[-1].strip("\n"))
        # Skip info line
        infile.readline()
        # Read timings
        operations = infile.readlines()
        colors = ["tab:blue", "tab:brown", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:olive"]
        total_time = 0
        for (j, line) in enumerate(operations):
            data = line.strip("\n").split(" ")
            if data[0] == "Backsubstitution":
                continue
            if first:
                plt.bar(i, float(data[2]), 0.5, bottom=total_time, label=data[0], color=colors[j], zorder=2)
            else:
                plt.bar(i, float(data[2]), 0.5, bottom=total_time, color=colors[j], zorder=2)
            if data[0] == "Solve":
                ax.annotate('{0:d}'.format(solve_iterations),
                            xy=(i, total_time + float(data[2]) / 2),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='bottom')

            total_time += float(data[2])
        first = False

    ax.set_xticks(range(len(dofs)))
    labels = ["#DOFS {0:d}\n#Slaves {2:d}\n#Procs {1:s}".format(
        dof, proc, slave) for dof, proc, slave in zip(dofs, procs, slaves)]
    ax.set_xticklabels(labels)
    # ax.set_yscale("log")
    plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(0.95, 0.5))
    plt.title("Average runtime of operations")
    plt.ylabel("Runtime (s)")

    plt.savefig("test.png")


def visualize_single(dof):
    infile = open("results_bench_{0:d}.txt".format(dof), "r")

    # Read problem info
    procs = infile.readline().split(": ")[-1].strip("\n")
    # Skip num dofs
    infile.readline()
    slaves = int(infile.readline().split(": ")[-1].strip("\n"))
    infile.readline()
    # solve_iterations = int(infile.readline().split(": ")[-1].strip("\n"))
    # Skip info line
    infile.readline()
    # Read timings
    operations = infile.readlines()
    colors = ["tab:blue", "tab:brown", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:olive"]
    fig, ax = plt.subplots()
    plt.grid("on", zorder=1)

    for (j, line) in enumerate(operations):
        data = line.strip("\n").split(" ")
        if data[0] == "Backsubstitution":
            continue
        plt.bar(j, float(data[2]), 0.5, label=data[0], color=colors[j], zorder=2)

    ax.set_xticks([])
    # ax.set_yscale("log")
    plt.legend()
    plt.ylabel("Runtime (s)")
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(bbox_to_anchor=(0.95, 0.5))
    plt.title("Average runtime of operations with {0:d} slave DOFs of {1:s} processors".format(slaves, procs))

    plt.savefig("test_single.png")


visualize_single(dofs[1])
visualize_side_by_side(dofs)

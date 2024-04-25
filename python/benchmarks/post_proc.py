from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Res 0.1 31776, 0.05 234546, 0.025 1801086, 0.02 3488856, 0.0175 5147961,0.015 7960200
dofs = [31776, 234546, 1801086, 3488856, 5147961, 7960200]


def visualize_side_by_side(dofs):
    fig, ax = plt.subplots()
    plt.grid("on", zorder=1)
    procs = []
    first = True
    slaves = []
    totals = np.zeros(len(dofs))
    for i, dof in enumerate(dofs):
        infile = open("results_bench_{0:d}.txt".format(dof), "r")

        # Read problem info
        procs.append(int(infile.readline().split(": ")[-1].strip("\n")))
        # Skip num dofs
        infile.readline()
        slaves.append(int(infile.readline().split(": ")[-1].strip("\n")))
        solve_iterations = int(infile.readline().split(": ")[-1].strip("\n"))
        # Skip info line
        infile.readline()
        # Read timings
        operations = infile.readlines()
        colors = [
            "tab:blue",
            "tab:brown",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:cyan",
            "tab:olive",
        ]
        total_time = 0
        for j, line in enumerate(operations):
            data = line.strip("\n").split(" ")
            if data[0] == "Backsubstitution":
                continue
            if first:
                plt.bar(
                    i,
                    float(data[2]),
                    0.5,
                    bottom=total_time,
                    label=data[0],
                    color=colors[j],
                    zorder=2,
                )
            else:
                plt.bar(i, float(data[2]), 0.5, bottom=total_time, color=colors[j], zorder=2)
            if data[0] == "Solve":
                ax.annotate(
                    "{0:d}".format(solve_iterations),
                    xy=(i, total_time + float(data[2]) / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            total_time += float(data[2])
        first = False
        totals[i] = total_time
    ax.set_xticks(range(len(dofs)))
    labels = dofs
    ax.set_xticklabels(labels)
    assert np.allclose(procs, procs[0])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(0.95, 0.5))
    plt.xlabel("Degrees of freedom")

    plt.title(f"Average runtime of operations with {procs[0]} MPI ranks")
    plt.ylabel("Runtime (s)")
    plt.savefig("comparison_bars.png")
    # Second figure
    plt.figure()
    plt.ylabel("Runtime (s)")
    plt.xlabel("Degrees of freedom")
    power_min = int(np.log10(min(dofs)))
    power_max = int(np.log10(max(dofs)) + 1)
    power_tmin = int(np.log10(min(totals)) - 1)
    power_tmax = int(np.log10(max(totals)) + 1)

    plt.axis((10**power_min, 10**power_max, 10**power_tmin, 10**power_tmax))
    plt.plot(dofs, totals, "-ro", label="Simulations")
    # 7
    plt.plot([3.1 * 10**4, 3.1 * 10**7], [4.3 * 10**-1, 4.3 * 10**2], "--g", label="Order 1")
    # 6 procs
    # plt.plot([3.1 * 10**4, 3.1 * 10**7], [4.8 * 10**-1, 4.8 * 10**2], "--g", label="Order 1")
    # 4 procs
    # plt.plot([3.1 * 10**4, 3.1 * 10**7], [5.7 * 10**-1, 5.7 * 10**2], "--g", label="Order 1")
    plt.title(f"Total runtime of core operations with {procs[0]} MPI ranks")
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("comparison.png")


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
    colors = [
        "tab:blue",
        "tab:brown",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:cyan",
        "tab:olive",
    ]
    fig, ax = plt.subplots()
    plt.grid("on", zorder=1)

    for j, line in enumerate(operations):
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
    plt.title(f"Average runtime of operations with {dof} ({slaves} slaves) on {procs} MPI ranks")

    plt.savefig(f"comparison_{slaves}.png")


visualize_single(max(dofs))
visualize_side_by_side(dofs)

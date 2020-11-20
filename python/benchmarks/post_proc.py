from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


dofs = [31776, 234546, 1801086]
fig, ax = plt.subplots()
procs = []
first = True

for i, dof in enumerate(dofs):
    infile = open("results_bench_{0:d}.txt".format(dof), "r")

    # Read problem info
    procs.append(infile.readline().split(": ")[-1].strip("\n"))
    # Skip num dofs
    infile.readline()
    slaves = int(infile.readline().split(": ")[-1].strip("\n"))
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
            p1 = plt.bar(i, float(data[2]), 0.5, bottom=total_time, label=data[0], color=colors[j])
        else:
            p1 = plt.bar(i, float(data[2]), 0.5, bottom=total_time, color=colors[j])
        if data[0] == "Solve":
            ax.annotate('{0:d}'.format(solve_iterations),
                        xy=(i, total_time + float(data[2]) / 2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom')

        total_time += float(data[2])
    first = False

ax.set_xticks(range(len(dofs)))
labels = ["{0:d}\n ({1:s} procs)".format(dof, proc) for dof, proc in zip(dofs, procs)]
ax.set_xticklabels(labels)
# ax.set_yscale("log")
plt.legend()
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(0.95, 0.5))
plt.title("Histogram of average runtime of operations")

plt.savefig("test.png")

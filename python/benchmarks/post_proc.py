from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


dofs = [31776, 234546]
fig, ax = plt.subplots()

first = True

for i, dof in enumerate(dofs):

    infile = open("results_bench_{0:d}.txt".format(dof), "r")

    # Read problem info
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
        if first:
            p1 = plt.bar(i, float(data[2]), 0.5, bottom=total_time, label=data[0], color=colors[j])
        else:
            p1 = plt.bar(i, float(data[2]), 0.5, bottom=total_time, color=colors[j])
        total_time += float(data[2])
    first = False

ax.set_xticks(range(len(dofs)))
ax.set_xticklabels(dofs)
# ax.set_yscale("log")
plt.legend()
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1., 0.5))
plt.timings("Histogram of average runtime of operations")

plt.savefig("test.png")

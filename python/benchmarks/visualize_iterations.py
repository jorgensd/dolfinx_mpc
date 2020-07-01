import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as mtransforms
import numpy as np
import h5py 
import mpi4py

f = h5py.File('bench_edge_output.hdf5', 'r', driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
iterations = f.get("its")[:]
dofs = f.get("num_dofs")[:]
slaves = f.get("num_slaves")[:]
f.close()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(dofs, iterations, "ro")
ax.set_xscale("log")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("# DOFS", fontsize=25)
plt.ylabel("# Iterations",fontsize=25)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.025, y=0.025, units='inches')
for i in range(len(iterations)):
    plt.text(dofs[i], iterations[i], slaves[i], transform=trans_offset, fontsize=20)
plt.title("MPC Elasticity with BoomerAMG", fontsize=25)
plt.grid()
plt.savefig("fig.png")


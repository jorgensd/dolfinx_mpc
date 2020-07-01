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

plt.plot(dofs, iterations, "-ro", label="MPC", markersize=12)

f_ref = h5py.File('ref_output.hdf5', 'r', driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
iterations_ref = f_ref.get("its")[:]
dofs_ref = f_ref.get("num_dofs")[:]
f_ref.close()

plt.plot(dofs_ref, iterations_ref, "-bs", label="Unconstrained")


ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xscale("log")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("# DOFS", fontsize=20)
plt.ylabel("# Iterations",fontsize=20)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.025, y=0.025, units='inches')
for i in range(len(iterations)):
    plt.text(dofs[i], iterations[i], slaves[i], transform=trans_offset, fontsize=20)
plt.title("Linear elasticity with BoomerAMG", fontsize=25)
plt.legend(fontsize=15)
plt.grid()
plt.savefig("fig.png", bbox_inches='tight')


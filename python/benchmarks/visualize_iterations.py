import matplotlib.pyplot as plt
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
plt.savefig("fig.png")


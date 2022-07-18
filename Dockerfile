FROM dolfinx/dolfinx:v0.4.1 as dolfinx-mpc
WORKDIR /tmp
# Set env variables
ENV HDF5_MPI="ON" \
    CC=mpicc \
    HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/"

# Install dolfinx_mpc
RUN  git clone https://github.com/jorgensd/dolfinx_mpc.git  && \
    cd dolfinx_mpc && \
    git checkout a6fa82ac0160ac1a5d91682c4c05dd63a5489f2c && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build-dir cpp/ && \
    ninja install -j4  -C build-dir && \
    pip3 install python/. --upgrade

# Install h5py
RUN pip3 install --no-binary=h5py h5py meshio

WORKDIR /root

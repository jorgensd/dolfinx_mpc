FROM dolfinx/dolfinx:v0.6.0 as dolfinx-mpc
WORKDIR /tmp
# Set env variables
ENV HDF5_MPI="ON" \
    CC=mpicc \
    HDF5_DIR="/usr/local"

# Install dolfinx_mpc
RUN git clone -b v0.6.0 --single-branch --depth 1 https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build-dir cpp/ && \
    ninja install -j4  -C build-dir && \
    python3 -m pip install python/. --upgrade

# Install h5py https://github.com/h5py/h5py/issues/2222
RUN python3 -m pip install mpi4py cython numpy && \
    python3 -m pip install --no-cache-dir --no-binary=h5py h5py  --no-build-isolation
RUN python3 -m pip install --no-binary=h5py meshio 

WORKDIR /root

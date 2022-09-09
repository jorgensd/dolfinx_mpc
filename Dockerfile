FROM dolfinx/dolfinx:v0.5.1 as dolfinx-mpc
WORKDIR /tmp
# Set env variables
ENV HDF5_MPI="ON" \
    CC=mpicc \
    HDF5_DIR="/usr/local"

# Install dolfinx_mpc
RUN git clone -b v0.5.0.post0 --single-branch --depth 1 https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build-dir cpp/ && \
    ninja install -j4  -C build-dir && \
    pip3 install python/. --upgrade

# Install h5py
RUN pip3 install --no-binary=h5py h5py meshio

WORKDIR /root

FROM dolfinx/dolfinx:v0.6.0-r1 as dolfinx-mpc

WORKDIR /tmp

COPY . dolfinx_mpc

# Install dolfinx_mpc
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir dolfinx_mpc/cpp/ && \
    ninja install -j4  -C build-dir && \
    python3 -m pip install dolfinx_mpc/python/. --upgrade

# Install h5py https://github.com/h5py/h5py/issues/2222
ENV HDF5_MPI="ON" \
    HDF5_DIR="/usr/local"
RUN python3 -m pip install --no-binary=h5py git+https://github.com/h5py/h5py.git@master
RUN python3 -m pip install meshio 

WORKDIR /root

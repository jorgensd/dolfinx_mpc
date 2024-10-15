FROM ghcr.io/fenics/dolfinx/dolfinx:v0.8.0

WORKDIR /tmp

# This argument should be the same as what-ever the python version of the dol
ARG PYTHON_VERSION=3.12

# Set env variables
ENV HDF5_MPI="ON" \
    HDF5_DIR="/usr/local"

RUN python3 -m pip install  -U pip setuptools

# Install h5py https://github.com/h5py/h5py/issues/2222
RUN python3 -m pip install --no-cache-dir --no-binary=h5py git+https://github.com/h5py/h5py.git
RUN python3 -m pip install  meshio 

# Copy DOLFINX_MPC source dir
COPY . dolfinx_mpc
RUN python3 -m pip install -U pip setuptools

# Install real mode
RUN . /usr/local/bin/dolfinx-real-mode && \
    . /usr/local/dolfinx-real/lib/dolfinx/dolfinx.conf && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=Developer -B build-dir-real dolfinx_mpc/cpp/ && \
    ninja install -j4 -C build-dir-real && \
    python3 -m pip install -v --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-real/lib/python${PYTHON_VERSION}/dist-packages --no-dependencies --no-cache-dir ./dolfinx_mpc/python

# Clean repo to remove build dir from pip
RUN rm -rf dolfinx_mpc/python/build 

# Install complex mode
RUN . /usr/local/bin/dolfinx-complex-mode && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=Developer -B build-dir-complex dolfinx_mpc/cpp/ && \
    ninja install -j4 -C build-dir-complex && \
    python3 -m pip install -v --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-complex/lib/python${PYTHON_VERSION}/dist-packages --no-dependencies --no-cache-dir ./dolfinx_mpc/python

WORKDIR /root

FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

SHELL ["/bin/bash", "-c"]
WORKDIR /tmp

# Set env variables
ENV HDF5_MPI="ON" \
    CC=mpicc \
    CXX=mpicxx \
    PIP_NO_BUILD_ISOLATION=0 \
    HDF5_MPI="ON" \
    HDF5_INCLUDEDIR="/usr/include/hdf5/openmpi" \
    H5PY_SETUP_REQUIRES=0

RUN python3 -m pip install  -U pip setuptools

# Install arch dependent h5py
# Calculate the path and save the export command to a dedicated file
RUN ARCH_TRIPLET=$(gcc -print-multiarch) && \
    echo "export HDF5_LIBDIR=/usr/lib/${ARCH_TRIPLET}/hdf5/openmpi" >> /opt/build_env.sh
# Tell Docker to automatically source this file at the start of every subsequent RUN command
ENV BASH_ENV=/opt/build_env.sh
RUN python3 -m pip install --no-cache-dir --no-binary=h5py h5py

# Install meshio
RUN python3 -m pip install  meshio 

# Copy DOLFINX_MPC source dir
COPY . dolfinx_mpc

# Install real mode
RUN . /usr/local/bin/dolfinx-real-mode && \
    . /usr/local/dolfinx-real/lib/dolfinx/dolfinx.conf && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=Developer -B build-dir-real dolfinx_mpc/cpp/ && \
    ninja install -j4 -C build-dir-real && \
    python3 -m pip install -v --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-real/lib/python/dist-packages --no-dependencies --no-cache-dir ./dolfinx_mpc/python

# Clean repo to remove build dir from pip
RUN rm -rf dolfinx_mpc/python/build 

# Install complex mode
RUN . /usr/local/bin/dolfinx-complex-mode && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=Developer -B build-dir-complex dolfinx_mpc/cpp/ && \
    ninja install -j4 -C build-dir-complex && \
    python3 -m pip install -v --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-complex/lib/python/dist-packages --no-dependencies --no-cache-dir ./dolfinx_mpc/python

WORKDIR /root

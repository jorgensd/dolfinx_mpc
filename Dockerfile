FROM dolfinx/dev-env as dolfinx-mpc
WORKDIR /tmp

ARG DOLFINX_VERSION=v0.2.0
# Set env variables
ENV PETSC_ARCH="linux-gnu-real-32"\
    HDF5_MPI="ON" \
    CC=mpicc \
    HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/"

#Install basix, ufl, ffcx
RUN git clone https://github.com/FEniCS/basix.git && \
    cd basix && \
    git checkout ${DOLFINX_VERSION} && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build-dir -S .  && \
    cmake --build build-dir && \    
    cmake --install build-dir && \
    pip3 install -v -e ./python  && \
    pip3 install git+https://github.com/FEniCS/ufl.git@2021.1.0 --no-cache-dir &&\
    pip3 install git+https://github.com/FEniCS/ffcx.git@${DOLFINX_VERSION} --no-cache-dir
# Install dolfinx
RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    git checkout ${DOLFINX_VERSION} && \
    cmake -G Ninja cpp -B build-dir && \
    ninja ${MAKEFLAGS} -j12 install -C build-dir && \
    . /usr/local/lib/dolfinx/dolfinx.conf && \
    pip3 install python/.
# Install dolfinx_mpc
RUN  git clone https://github.com/jorgensd/dolfinx_mpc.git  && \
    cd dolfinx_mpc && \
    git checkout ${DOLFINX_VERSION}} && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build-dir cpp/ && \
    ninja install -j12  -C build-dir && \
    pip3 install python/. --upgrade
# Install h5py
RUN pip3 install --no-binary=h5py h5py

WORKDIR /root

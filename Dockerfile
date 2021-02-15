FROM dolfinx/dev-env as dolfinx-mpc-debug
WORKDIR /tmp
ENV PETSC_ARCH "linux-gnu-complex-32"
RUN pip3 install git+https://github.com/FEniCS/basix.git --no-cache-dir &&\
    pip3 install git+https://github.com/FEniCS/ufl.git --no-cache-dir &&\
    pip3 install git+https://github.com/FEniCS/ffcx.git --no-cache-dir
RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    git checkout dokken/debug_pattern_assemble && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    . /usr/local/lib/dolfinx/dolfinx.conf && \
    pip3 install .
RUN  git clone https://github.com/jorgensd/dolfinx_mpc.git  && \
    cd dolfinx_mpc && \
    git checkout dokken/debug_pattern_assemble && \
    mkdir -p build && \
    cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/ && \
    ninja -j3 install && \
    cd ../python && \
    pip3 install . --upgrade

WORKDIR /root
# FROM dolfinx/dolfinx as dolfinx-mpc-real
# LABEL description="DOLFIN-X in real mode with MPC"
# USER root

# WORKDIR /tmp

# RUN pip3 install --no-cache-dir ipython

# RUN git clone https://github.com/jorgensd/dolfinx_mpc.git && \
#     cd dolfinx_mpc && \
#     mkdir build && \
#     cd build && \
#     cmake -G Ninja  -DCMAKE_BUILD_TYPE=Release ../cpp && \
#     ninja install && \
#     cd ../python && \
#     pip3 install .

# WORKDIR /root

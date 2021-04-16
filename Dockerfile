# FROM dolfinx/dev-env as dolfinx-mpc-debug
# WORKDIR /tmp
# ENV PETSC_ARCH "linux-gnu-complex-32"
# RUN  git clone  git+https://github.com/FEniCS/basix.git && \
#     cd basix && \
#     rm -rf build-dir && \
#     cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build-dir -S .  && \
#     cmake --build build-dir --parallel 3 && \
#     cmake --install build-dir && \
#     pip3 install -v -e ./python  && \
#     pip3 install git+https://github.com/FEniCS/ufl.git --no-cache-dir &&\
#     pip3 install git+https://github.com/FEniCS/ffcx.git --no-cache-dir
# RUN git clone https://github.com/fenics/dolfinx.git && \
#     cd dolfinx && \
#     mkdir build && \
#     cd build && \
#     cmake -G Ninja ../cpp && \
#     ninja ${MAKEFLAGS} install && \
#     cd ../python && \
#     . /usr/local/lib/dolfinx/dolfinx.conf && \
#     pip3 install .
# RUN  git clone https://github.com/jorgensd/dolfinx_mpc.git  && \
#     cd dolfinx_mpc && \
#     mkdir -p build && \
#     cd build && \
#     cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/ && \
#     ninja -j3 install && \
#     cd ../python && \
#     pip3 install . --upgrade

# WORKDIR /root
FROM dolfinx/dolfinx as dolfinx-mpc-real
LABEL description="DOLFIN-X in real mode with MPC"
USER root

WORKDIR /tmp

RUN pip3 install --no-cache-dir ipython

RUN git clone https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
    mkdir build && \
    cd build && \
    cmake -G Ninja  -DCMAKE_BUILD_TYPE=Release ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

WORKDIR /root

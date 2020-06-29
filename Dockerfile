FROM dolfinx/dolfinx as dolfinx-mpc-real
LABEL description="DOLFIN-X in real mode with MPC"
USER root

WORKDIR /tmp

RUN pip3 install --no-cache-dir ipython

RUN git clone https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
	git checkout dokken/gamg-debug && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

WORKDIR /root

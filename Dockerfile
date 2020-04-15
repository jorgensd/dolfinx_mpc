FROM dolfinx/dev-env-real as dolfinx-mpc-real
LABEL description="DOLFIN-X in real mode with MPC"
USER root

WORKDIR /tmp

RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git && \
	git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && \
	git checkout dokken/collision-tol && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

RUN git clone https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
	git checkout dokken/element-collision && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

WORKDIR /root


FROM dolfinx/dev-env-complex as dolfinx-mpc-complex

LABEL description="DOLFIN-X in complex mode with MPC"
USER root

WORKDIR /tmp

RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git && \
    git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && \
	git checkout dokken/collision-tol && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

RUN git clone https://github.com/jorgensd/dolfinx_mpc.git && \
    cd dolfinx_mpc && \
	git checkout dokken/element-collision && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install .

WORKDIR /root

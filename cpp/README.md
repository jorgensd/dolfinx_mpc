# Installation

```
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/
ninja -j3 install

source dolfinx_mpc.conf
cd ../python3
pip3 install .

```
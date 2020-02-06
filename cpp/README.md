# Installation

```
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/
ninja -j3 install

cd ../python3
pip3 install .

```
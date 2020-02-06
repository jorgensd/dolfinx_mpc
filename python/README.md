Shortcut until CI is up and running


mpc_rebuild(){
    BASEDIR=$(pwd)
    cd /home/shared/mpc
    rm -rf build
    mkdir -p build
    cd build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/
    ninja -j3 install
    cd ../python
    pip3 install -e . --upgrade
    cd tests
    python3 -m pytest .
    mpirun -n 3 python3 -m pytest .
    cd $BASEDIR

}

git submodule init
git submodule update
# build in ./build
mkdir -p build
cd build

cmake ..
make

# call in lib folder
mkdir -p ../lib
cp *.so ../lib/
cd ..
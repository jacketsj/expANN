mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=/usr/bin/clang++ ..
make
cd ..

mkdir -p build
cd build
#cmake -D CMAKE_CXX_COMPILER=/usr/bin/clang++ ..
cmake -D CMAKE_CXX_COMPILER=/usr/bin/g++ ..
make
cd ..

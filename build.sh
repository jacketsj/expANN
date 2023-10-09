# Default values for CMake flags
CXX_COMPILER="/usr/bin/g++"
ENABLE_STACK_INFO="OFF"
ENABLE_GCOV="OFF"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clang)
            CXX_COMPILER="/usr/bin/clang++"
            shift
            ;;
        --stack-info)
            ENABLE_STACK_INFO="ON"
            shift
            ;;
        --gcov)
            ENABLE_GCOV="ON"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create build directory
mkdir -p build
cd build

# Run CMake with specified flags
cmake -D CMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -D ENABLE_STACK_INFO="$ENABLE_STACK_INFO" \
      -D ENABLE_GCOV="$ENABLE_GCOV" \
      ..

# Build the project
make

cd ..


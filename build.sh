# Default values for CMake flags
CXX_COMPILER="/usr/bin/g++"
ENABLE_STACK_INFO="OFF"
ENABLE_GCOV="OFF"
ENABLE_ASAN="OFF" # Ensure this is defined
GENERATE_COMPILE_COMMANDS="OFF"
ENABLE_PGO_USE="OFF"
ENABLE_PGO_GENERATE="OFF"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clang)
            echo "Enabling clang"
            CXX_COMPILER="/usr/bin/clang++"
            shift
            ;;
        --stack-info)
            ENABLE_STACK_INFO="ON"
            shift
            ;;
        --asan)
            ENABLE_ASAN="ON"
            shift
            ;;
        --gcov)
            ENABLE_GCOV="ON"
            shift
            ;;
        --compile-commands)
            GENERATE_COMPILE_COMMANDS="ON"
            shift
            ;;
        --pgo-use)
            ENABLE_PGO_USE="ON"
            shift
            ;;
        --pgo-gen)
            ENABLE_PGO_GENERATE="ON"
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
      -D ENABLE_ASAN="$ENABLE_ASAN" \
      -D ENABLE_PGO_USE="$ENABLE_PGO_USE" \
      -D ENABLE_PGO_GENERATE="$ENABLE_PGO_GENERATE" \
      $( [ "$GENERATE_COMPILE_COMMANDS" = "ON" ] && echo "-DCMAKE_EXPORT_COMPILE_COMMANDS=1" ) \
      ..
      #-G Ninja \

# Build the project
cmake --build .
#make

# Optionally, symlink compile_commands.json to the project root
if [ "$GENERATE_COMPILE_COMMANDS" = "ON" ]; then
    ln -sf "$PWD/compile_commands.json" ../compile_commands.json
fi

cd ..

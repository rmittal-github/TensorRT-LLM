#!/bin/bash
# Build script for Categorical Sampling TensorRT Plugin

set -e  # Exit on error

echo "=== Building Categorical Sampling TensorRT Plugin ==="

# Find TensorRT
if [ -z "$TENSORRT_ROOT" ]; then
    echo "TENSORRT_ROOT not set. Trying to find TensorRT..."

    # Common TensorRT locations
    POSSIBLE_PATHS=(
        "/usr/local/TensorRT"
        "/opt/tensorrt"
        "$HOME/TensorRT"
        "$CUDA_HOME/../TensorRT"
    )

    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ]; then
            export TENSORRT_ROOT="$path"
            echo "Found TensorRT at: $TENSORRT_ROOT"
            break
        fi
    done

    if [ -z "$TENSORRT_ROOT" ]; then
        echo "ERROR: Could not find TensorRT. Please set TENSORRT_ROOT environment variable."
        echo "Example: export TENSORRT_ROOT=/path/to/TensorRT"
        exit 1
    fi
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DTENSORRT_ROOT="$TENSORRT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLE=ON

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Summary
echo ""
echo "=== Build Complete ==="
echo "Plugin library: $BUILD_DIR/libcategorical_sampling_plugin.so"
if [ -f "categorical_sampling_example" ]; then
    echo "Example binary: $BUILD_DIR/categorical_sampling_example"
    echo ""
    echo "To run the example:"
    echo "  cd $BUILD_DIR"
    echo "  LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./categorical_sampling_example"
fi

echo ""
echo "To install system-wide (optional):"
echo "  cd $BUILD_DIR"
echo "  sudo make install"

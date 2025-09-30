#!/bin/bash

# Parse command line arguments
DEBUG_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -debug|-d)
            DEBUG_FLAG="--build_type=Debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-debug|-d]"
            exit 1
            ;;
    esac
done

python3 ./scripts/build_wheel.py --cuda_architectures "89-real" --benchmarks --trt_root /usr/local/tensorrt $DEBUG_FLAG

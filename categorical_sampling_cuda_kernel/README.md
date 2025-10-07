# Categorical Sampling CUDA Kernel

A standalone CUDA implementation of categorical sampling for sampling indices from probability distributions. This project is designed to be later integrated as a TensorRT plugin.

## Overview

Categorical sampling is a fundamental operation in language model inference, where we need to sample the next token based on probability distributions output by the model. This implementation uses the inverse transform sampling method on the GPU.

## Features

- **Efficient GPU implementation**: Parallel sampling across batch dimensions
- **Two sampling modes**:
  - Standalone mode: Initializes random states per kernel call
  - Persistent state mode: Maintains random states across multiple calls for better performance
- **Inverse transform sampling**: Simple and efficient algorithm
- **Comprehensive tests**: Statistical validation and performance tests included

## Algorithm

The kernel uses **inverse transform sampling**:
1. Generate a uniform random number `u ~ Uniform(0, 1)`
2. Compute cumulative sum of probabilities
3. Find the first index `i` where `cumsum[i] >= u`

## Project Structure

```
sampling/
├── include/
│   └── categorical_sampling.cuh    # Header file with API
├── src/
│   └── categorical_sampling.cu     # CUDA kernel implementation
├── test/
│   └── test_categorical_sampling.cu # Test suite
├── CMakeLists.txt                  # Build configuration
└── README.md                       # This file
```

## Building

### Prerequisites

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++17 compatible compiler
- NVIDIA GPU with compute capability 7.0 or higher

### Build Instructions

```bash
cd /code/tensorrt_llm/sampling
mkdir build && cd build
cmake ..
make
```

### Specifying CUDA Architecture

To build for specific GPU architectures:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" ..
make
```

Common architectures:
- 70: Tesla V100
- 75: Tesla T4, RTX 2080 Ti
- 80: A100
- 86: RTX 3090, RTX 3080
- 89: RTX 4090

## Running Tests

```bash
cd build
./test_categorical_sampling
```

The test suite includes:
1. **Basic Sampling**: Tests with various probability distributions
2. **Statistical Distribution**: Validates that sampling matches expected probabilities
3. **Persistent States**: Tests the performance-optimized mode with reusable random states

## API Usage

### Simple Standalone Sampling

```cpp
#include "categorical_sampling.cuh"

// Allocate device memory
float* d_probs;  // [batch_size, vocab_size]
int* d_output;   // [batch_size]

cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float));
cudaMalloc(&d_output, batch_size * sizeof(int));

// Copy probabilities to device
cudaMemcpy(d_probs, h_probs, ...);

// Sample
categoricalSampling(d_probs, d_output, batch_size, vocab_size, seed, offset);

// Copy results back
cudaMemcpy(h_output, d_output, ...);
```

### Optimized Sampling with Persistent States

```cpp
// Initialize random states once
curandState* d_states;
cudaMalloc(&d_states, batch_size * sizeof(curandState));
initializeRandomStates(d_states, batch_size, seed);

// Sample multiple times (e.g., in a generation loop)
for (int step = 0; step < num_steps; ++step)
{
    categoricalSamplingWithStates(d_probs, d_output, batch_size, vocab_size, d_states);
    // States are automatically updated for next iteration
}
```

## Performance Considerations

- **Batch size**: Larger batch sizes provide better GPU utilization
- **Persistent states**: Reusing random states avoids initialization overhead
- **Memory layout**: Probabilities should be row-major (contiguous vocab dimension)
- **Normalization**: Input probabilities must be normalized (sum to 1.0)

## Future Work

- [ ] Integration as TensorRT plugin
- [ ] Support for top-k and top-p (nucleus) sampling
- [ ] Temperature scaling
- [ ] Optimized implementations for small vocab sizes using shared memory
- [ ] Support for FP16 probabilities
- [ ] Benchmarking against other implementations

## License

Part of the TensorRT-LLM project.

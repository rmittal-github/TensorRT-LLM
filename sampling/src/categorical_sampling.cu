#include "categorical_sampling.cuh"
#include <stdio.h>

// Kernel to initialize random states
__global__ void initRandomStatesKernel(curandState* states, int batch_size, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Categorical sampling kernel using inverse transform sampling
// Each thread handles one batch element
// Accepts unnormalized probabilities and normalizes them on-the-fly
__global__ void categoricalSamplingKernel(
    float const* probs, int* output, int batch_size, int vocab_size, curandState* rand_states)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size)
    {
        return;
    }

    // Get random state for this thread
    curandState localState = rand_states[batch_idx];

    // Generate uniform random number [0, 1)
    float random_val = curand_uniform(&localState);

    // Compute sum of probabilities for normalization
    float const* prob_row = probs + batch_idx * vocab_size;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i)
    {
        sum += prob_row[i];
    }

    // Perform inverse transform sampling with normalization
    // Find the first index where normalized cumulative sum >= random_val
    float cumsum = 0.0f;
    int sampled_idx = vocab_size - 1; // Default to last index

    for (int i = 0; i < vocab_size; ++i)
    {
        cumsum += prob_row[i] / sum;
        if (random_val <= cumsum)
        {
            sampled_idx = i;
            break;
        }
    }

    output[batch_idx] = sampled_idx;

    // Save updated random state
    rand_states[batch_idx] = localState;
}

// Standalone kernel that initializes its own random states
// Accepts unnormalized probabilities and normalizes them on-the-fly
__global__ void categoricalSamplingStandaloneKernel(
    float const* probs, int* output, int batch_size, int vocab_size, unsigned long long seed, unsigned long long offset)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size)
    {
        return;
    }

    // Initialize random state for this thread
    curandState localState;
    curand_init(seed, batch_idx, offset, &localState);

    // Generate uniform random number [0, 1)
    float random_val = curand_uniform(&localState);

    // Compute sum of probabilities for normalization
    float const* prob_row = probs + batch_idx * vocab_size;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i)
    {
        sum += prob_row[i];
    }

    // Perform inverse transform sampling with normalization
    float cumsum = 0.0f;
    int sampled_idx = vocab_size - 1;

    for (int i = 0; i < vocab_size; ++i)
    {
        cumsum += prob_row[i] / sum;
        if (random_val <= cumsum)
        {
            sampled_idx = i;
            break;
        }
    }

    output[batch_idx] = sampled_idx;
}

// Host function implementations
void initializeRandomStates(curandState* states, int batch_size, unsigned long long seed)
{
    int const threads_per_block = 256;
    int const num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    initRandomStatesKernel<<<num_blocks, threads_per_block>>>(states, batch_size, seed);
}

void categoricalSamplingWithStates(
    float const* probs, int* output, int batch_size, int vocab_size, curandState* rand_states)
{
    int const threads_per_block = 256;
    int const num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    categoricalSamplingKernel<<<num_blocks, threads_per_block>>>(probs, output, batch_size, vocab_size, rand_states);
}

void categoricalSampling(
    float const* probs, int* output, int batch_size, int vocab_size, unsigned long long seed, unsigned long long offset)
{
    int const threads_per_block = 256;
    int const num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    categoricalSamplingStandaloneKernel<<<num_blocks, threads_per_block>>>(
        probs, output, batch_size, vocab_size, seed, offset);
}

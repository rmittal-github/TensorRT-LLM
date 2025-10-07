#include "categorical_sampling.cuh"
#include <stdio.h>

// Categorical sampling kernel that initializes random states internally using clock
// Each thread handles one batch element
// Accepts unnormalized probabilities and normalizes them on-the-fly
// Uses clock() for non-reproducible random seeding
// FP16 version for memory efficiency
__global__ void categoricalSamplingKernel(half const* probs, int* output, int batch_size, int vocab_size)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size)
    {
        return;
    }

    // Initialize random state for this thread using clock for unique seed
    // This provides non-reproducible randomness which is sufficient for sampling
    curandState localState;
    unsigned long long seed = clock64() + batch_idx;
    curand_init(seed, 0, 0, &localState);

    // Generate uniform random number [0, 1)
    float random_val = curand_uniform(&localState);

    // Compute sum of probabilities for normalization (convert to float for accuracy)
    half const* prob_row = probs + batch_idx * vocab_size;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i)
    {
        sum += __half2float(prob_row[i]);
    }

    // Perform inverse transform sampling with normalization
    // Find the first index where normalized cumulative sum >= random_val
    float cumsum = 0.0f;
    int sampled_idx = vocab_size - 1; // Default to last index

    for (int i = 0; i < vocab_size; ++i)
    {
        cumsum += __half2float(prob_row[i]) / sum;
        if (random_val <= cumsum)
        {
            sampled_idx = i;
            break;
        }
    }

    output[batch_idx] = sampled_idx;
}

// Host function implementation
void categoricalSampling(half const* probs, int* output, int batch_size, int vocab_size)
{
    int const threads_per_block = 256;
    int const num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    categoricalSamplingKernel<<<num_blocks, threads_per_block>>>(probs, output, batch_size, vocab_size);
}

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Categorical sampling kernel that samples indices from probability distributions
// Each thread samples one index from its corresponding probability vector

/**
 * @brief Performs categorical sampling on GPU
 *
 * @param probs Input probabilities [batch_size, vocab_size] - can be unnormalized (will be normalized internally)
 * @param output Sampled indices [batch_size]
 * @param batch_size Number of probability distributions to sample from
 * @param vocab_size Size of each probability distribution
 * @param seed Random seed for cuRAND
 * @param offset Offset for cuRAND sequence
 */
void categoricalSampling(float const* probs, int* output, int batch_size, int vocab_size, unsigned long long seed,
    unsigned long long offset);

/**
 * @brief Performs categorical sampling with separate random states
 *
 * @param probs Input probabilities [batch_size, vocab_size] - can be unnormalized (will be normalized internally)
 * @param output Sampled indices [batch_size]
 * @param batch_size Number of probability distributions to sample from
 * @param vocab_size Size of each probability distribution
 * @param rand_states Pre-initialized cuRAND states [batch_size]
 */
void categoricalSamplingWithStates(
    float const* probs, int* output, int batch_size, int vocab_size, curandState* rand_states);

/**
 * @brief Initialize random states for sampling
 *
 * @param states Output random states [batch_size]
 * @param batch_size Number of states to initialize
 * @param seed Random seed
 */
void initializeRandomStates(curandState* states, int batch_size, unsigned long long seed);

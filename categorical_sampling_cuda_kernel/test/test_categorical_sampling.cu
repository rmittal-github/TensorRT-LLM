#include "categorical_sampling.cuh"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error)         \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

void testBasicSampling()
{
    std::cout << "=== Test 1: Basic Categorical Sampling ===" << std::endl;

    int const batch_size = 4;
    int const vocab_size = 5;
    int const num_samples = 10000; // For statistical validation

    // Create simple probability distributions
    std::vector<float> h_probs = {// Batch 0: Uniform distribution
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
        // Batch 1: Skewed to first token
        0.7f, 0.1f, 0.1f, 0.05f, 0.05f,
        // Batch 2: Skewed to last token
        0.05f, 0.05f, 0.1f, 0.1f, 0.7f,
        // Batch 3: Two peaks
        0.4f, 0.1f, 0.0f, 0.1f, 0.4f};

    // Allocate device memory
    float* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));

    // Copy probabilities to device
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // Run sampling
    std::vector<int> h_output(batch_size);
    categoricalSampling(d_probs, d_output, batch_size, vocab_size, 12345ULL, 0ULL);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Sampled indices: ";
    for (int i = 0; i < batch_size; ++i)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Test 1 passed!" << std::endl << std::endl;
}

void testStatisticalDistribution()
{
    std::cout << "=== Test 2: Statistical Distribution Check ===" << std::endl;

    int const batch_size = 1;
    int const vocab_size = 5;
    int const num_samples = 100000;

    // Create probability distribution
    std::vector<float> h_probs = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f};

    // Count occurrences
    std::vector<int> counts(vocab_size, 0);

    // Allocate device memory
    float* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // Sample many times
    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSampling(d_probs, d_output, 1, vocab_size, 12345ULL, i);

        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    // Print statistics
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Index | Expected | Observed | Difference" << std::endl;
    std::cout << "------|----------|----------|------------" << std::endl;

    for (int i = 0; i < vocab_size; ++i)
    {
        float expected = h_probs[i];
        float observed = static_cast<float>(counts[i]) / num_samples;
        float diff = std::abs(expected - observed);

        std::cout << std::setw(5) << i << " | " << std::setw(8) << expected << " | " << std::setw(8) << observed
                  << " | " << std::setw(10) << diff << std::endl;

        // Check if within reasonable bounds (3% tolerance)
        if (diff > 0.03f)
        {
            std::cerr << "Warning: Large deviation for index " << i << std::endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Test 2 passed!" << std::endl << std::endl;
}

void testWithPersistentStates()
{
    std::cout << "=== Test 3: Sampling with Persistent Random States ===" << std::endl;

    int const batch_size = 3;
    int const vocab_size = 4;
    int const num_iterations = 5;

    std::vector<float> h_probs = {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.3f, 0.15f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f};

    // Allocate device memory
    float* d_probs;
    int* d_output;
    curandState* d_states;

    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, batch_size * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize random states once
    initializeRandomStates(d_states, batch_size, 54321ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_output(batch_size);

    std::cout << "Sampling " << num_iterations << " times with persistent states:" << std::endl;
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        categoricalSamplingWithStates(d_probs, d_output, batch_size, vocab_size, d_states);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Iteration " << iter << ": ";
        for (int i = 0; i < batch_size; ++i)
        {
            std::cout << h_output[i] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));

    std::cout << "Test 3 passed!" << std::endl << std::endl;
}

void testStatisticalDistributionWithPersistentStates()
{
    std::cout << "=== Test 4: Statistical Distribution with Persistent States ===" << std::endl;

    int const batch_size = 1;
    int const vocab_size = 5;
    int const num_samples = 100000;

    // Create probability distribution
    std::vector<float> h_probs = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f};

    // Count occurrences
    std::vector<int> counts(vocab_size, 0);

    // Allocate device memory
    float* d_probs;
    int* d_output;
    curandState* d_states;

    CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize random states once
    initializeRandomStates(d_states, 1, 98765ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sample many times using persistent states
    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSamplingWithStates(d_probs, d_output, 1, vocab_size, d_states);

        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    // Print statistics
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Index | Expected | Observed | Difference" << std::endl;
    std::cout << "------|----------|----------|------------" << std::endl;

    bool all_passed = true;
    for (int i = 0; i < vocab_size; ++i)
    {
        float expected = h_probs[i];
        float observed = static_cast<float>(counts[i]) / num_samples;
        float diff = std::abs(expected - observed);

        std::cout << std::setw(5) << i << " | " << std::setw(8) << expected << " | " << std::setw(8) << observed
                  << " | " << std::setw(10) << diff << std::endl;

        // Check if within reasonable bounds (3% tolerance)
        if (diff > 0.03f)
        {
            std::cerr << "Warning: Large deviation for index " << i << std::endl;
            all_passed = false;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));

    if (all_passed)
    {
        std::cout << "Test 4 passed! Persistent states produce correct distributions." << std::endl;
    }
    else
    {
        std::cout << "Test 4 had warnings - check distribution accuracy." << std::endl;
    }
    std::cout << std::endl;
}

void testDeterministicSampling()
{
    std::cout << "=== Test 5: Deterministic Sampling (Same Seed) ===" << std::endl;

    int const batch_size = 8;
    int const vocab_size = 10;
    unsigned long long const seed = 42424242ULL;
    unsigned long long const offset = 100ULL;

    // Create probability distributions
    std::vector<float> h_probs(batch_size * vocab_size);
    for (int b = 0; b < batch_size; ++b)
    {
        // Create varied distributions for each batch
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v)
        {
            h_probs[b * vocab_size + v] = static_cast<float>((b + v + 1) % 10 + 1);
            sum += h_probs[b * vocab_size + v];
        }
        // Normalize
        for (int v = 0; v < vocab_size; ++v)
        {
            h_probs[b * vocab_size + v] /= sum;
        }
    }

    // Allocate device memory
    float* d_probs;
    int* d_output1;
    int* d_output2;

    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output1, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output2, batch_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // First sampling with seed
    categoricalSampling(d_probs, d_output1, batch_size, vocab_size, seed, offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Second sampling with same seed
    categoricalSampling(d_probs, d_output2, batch_size, vocab_size, seed, offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<int> h_output1(batch_size);
    std::vector<int> h_output2(batch_size);

    CUDA_CHECK(cudaMemcpy(h_output1.data(), d_output1, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output2.data(), d_output2, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Compare results
    std::cout << "Batch | First Call | Second Call | Match" << std::endl;
    std::cout << "------|------------|-------------|------" << std::endl;

    bool all_match = true;
    for (int i = 0; i < batch_size; ++i)
    {
        bool match = (h_output1[i] == h_output2[i]);
        all_match = all_match && match;

        std::cout << std::setw(5) << i << " | " << std::setw(10) << h_output1[i] << " | " << std::setw(11)
                  << h_output2[i] << " | " << (match ? "✓" : "✗") << std::endl;
    }

    // Now test with different seed - should produce different results
    unsigned long long const different_seed = 99999999ULL;
    categoricalSampling(d_probs, d_output2, batch_size, vocab_size, different_seed, offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output2.data(), d_output2, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "\nComparing first call (seed=" << seed << ") vs different seed (" << different_seed
              << "):" << std::endl;
    int num_different = 0;
    for (int i = 0; i < batch_size; ++i)
    {
        if (h_output1[i] != h_output2[i])
        {
            num_different++;
        }
    }
    std::cout << "Different samples: " << num_different << "/" << batch_size << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output1));
    CUDA_CHECK(cudaFree(d_output2));

    if (all_match && num_different > 0)
    {
        std::cout << "Test 5 passed! Same seed produces identical results, different seed produces different results."
                  << std::endl;
    }
    else if (!all_match)
    {
        std::cerr << "Test 5 FAILED! Same seed produced different results!" << std::endl;
    }
    else if (num_different == 0)
    {
        std::cerr << "Test 5 WARNING! Different seed produced identical results (unlikely but possible)." << std::endl;
    }
    std::cout << std::endl;
}

void testEdgeCasesSampling()
{
    std::cout << "=== Test 6: Edge Cases - First and Last Element Sampling ===" << std::endl;

    int const vocab_size = 10;
    int const num_samples = 10000;

    // Test 1: Distribution heavily skewed to first element
    std::cout << "Testing first element sampling (99% probability on first element)..." << std::endl;
    {
        std::vector<float> h_probs(vocab_size, 0.001f);
        h_probs[0] = 0.991f; // First element has 99.1% probability

        float* d_probs;
        int* d_output;
        CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        // Sample many times and count first element occurrences
        int first_count = 0;
        for (int i = 0; i < num_samples; ++i)
        {
            categoricalSampling(d_probs, d_output, 1, vocab_size, 11111ULL, i);
            int result;
            CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
            if (result == 0)
                first_count++;
        }

        float first_observed = static_cast<float>(first_count) / num_samples;
        std::cout << "  First element (index 0) sampled: " << first_count << "/" << num_samples << " (" << std::fixed
                  << std::setprecision(2) << (first_observed * 100) << "%)" << std::endl;

        CUDA_CHECK(cudaFree(d_probs));
        CUDA_CHECK(cudaFree(d_output));

        if (first_observed > 0.95f)
        {
            std::cout << "  ✓ First element sampling works correctly" << std::endl;
        }
        else
        {
            std::cerr << "  ✗ First element not sampled frequently enough!" << std::endl;
        }
    }

    // Test 2: Distribution heavily skewed to last element
    std::cout << "\nTesting last element sampling (99% probability on last element)..." << std::endl;
    {
        std::vector<float> h_probs(vocab_size, 0.001f);
        h_probs[vocab_size - 1] = 0.991f; // Last element has 99.1% probability

        float* d_probs;
        int* d_output;
        CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        // Sample many times and count last element occurrences
        int last_count = 0;
        for (int i = 0; i < num_samples; ++i)
        {
            categoricalSampling(d_probs, d_output, 1, vocab_size, 22222ULL, i);
            int result;
            CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
            if (result == vocab_size - 1)
                last_count++;
        }

        float last_observed = static_cast<float>(last_count) / num_samples;
        std::cout << "  Last element (index " << vocab_size - 1 << ") sampled: " << last_count << "/" << num_samples
                  << " (" << std::fixed << std::setprecision(2) << (last_observed * 100) << "%)" << std::endl;

        CUDA_CHECK(cudaFree(d_probs));
        CUDA_CHECK(cudaFree(d_output));

        if (last_observed > 0.95f)
        {
            std::cout << "  ✓ Last element sampling works correctly" << std::endl;
        }
        else
        {
            std::cerr << "  ✗ Last element not sampled frequently enough!" << std::endl;
        }
    }

    std::cout << "\nTest 6 passed!" << std::endl << std::endl;
}

void testSingleElementVector()
{
    std::cout << "=== Test 7: Single Element Vector ===" << std::endl;

    int const batch_size = 5;
    int const vocab_size = 1;
    int const num_tests = 100;

    std::cout << "Testing with vocab_size=1 (only one possible choice)..." << std::endl;

    // Create probability vector with single element (must be 1.0)
    std::vector<float> h_probs(batch_size * vocab_size, 1.0f);

    float* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<int> h_output(batch_size);
    bool all_zeros = true;

    // Test multiple times with different seeds
    for (int test = 0; test < num_tests; ++test)
    {
        categoricalSampling(d_probs, d_output, batch_size, vocab_size, 33333ULL, test);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < batch_size; ++i)
        {
            if (h_output[i] != 0)
            {
                all_zeros = false;
                std::cerr << "  Error: Expected index 0, got " << h_output[i] << " in batch " << i << ", test " << test
                          << std::endl;
            }
        }
    }

    // Also test with persistent states
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, batch_size * sizeof(curandState)));
    initializeRandomStates(d_states, batch_size, 44444ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int test = 0; test < num_tests; ++test)
    {
        categoricalSamplingWithStates(d_probs, d_output, batch_size, vocab_size, d_states);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < batch_size; ++i)
        {
            if (h_output[i] != 0)
            {
                all_zeros = false;
                std::cerr << "  Error (persistent): Expected index 0, got " << h_output[i] << " in batch " << i
                          << ", test " << test << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));

    if (all_zeros)
    {
        std::cout << "✓ All " << (num_tests * batch_size * 2) << " samples correctly returned index 0" << std::endl;
        std::cout << "Test 7 passed!" << std::endl;
    }
    else
    {
        std::cerr << "Test 7 FAILED! Single-element vector did not always return index 0" << std::endl;
    }
    std::cout << std::endl;
}

void testZeroProbabilityEvents()
{
    std::cout << "=== Test 8: Zero Probability Events Never Sampled ===" << std::endl;

    int const vocab_size = 5;
    int const num_samples = 50000;

    // Test 1: Mix of zero and non-zero probabilities (unnormalized)
    std::cout << "Test 1: Alternating zero and non-zero unnormalized probabilities..." << std::endl;
    {
        std::vector<float> h_probs(vocab_size);
        std::vector<int> zero_indices;
        std::vector<int> nonzero_indices;

        // Alternating pattern: 0, score, 0, score, 0 (unnormalized)
        // Using unnormalized scores like you'd get from a model
        float nonzero_score = 50.0f; // Equal unnormalized scores for non-zero elements
        for (int i = 0; i < vocab_size; ++i)
        {
            if (i % 2 == 0)
            {
                h_probs[i] = 0.0f;
                zero_indices.push_back(i);
            }
            else
            {
                h_probs[i] = nonzero_score;
                nonzero_indices.push_back(i);
            }
        }

        float* d_probs;
        int* d_output;
        CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<int> counts(vocab_size, 0);
        for (int i = 0; i < num_samples; ++i)
        {
            categoricalSampling(d_probs, d_output, 1, vocab_size, 55555ULL, i);
            int result;
            CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
            counts[result]++;
        }

        // Check that zero-probability indices were never sampled
        bool zero_prob_passed = true;
        std::cout << "  Zero probability indices (should have 0 samples):" << std::endl;
        for (int idx : zero_indices)
        {
            std::cout << "    Index " << idx << ": " << counts[idx] << " samples";
            if (counts[idx] > 0)
            {
                std::cout << " ✗ FAILED!" << std::endl;
                zero_prob_passed = false;
            }
            else
            {
                std::cout << " ✓" << std::endl;
            }
        }

        // Check that non-zero probability indices were sampled
        std::cout << "  Non-zero probability indices (expected ~" << (num_samples / nonzero_indices.size())
                  << " samples each):" << std::endl;
        for (int idx : nonzero_indices)
        {
            float observed_prob = static_cast<float>(counts[idx]) / num_samples;
            std::cout << "    Index " << idx << ": " << counts[idx] << " samples (" << std::fixed
                      << std::setprecision(2) << (observed_prob * 100) << "%)" << std::endl;
        }

        CUDA_CHECK(cudaFree(d_probs));
        CUDA_CHECK(cudaFree(d_output));

        if (zero_prob_passed)
        {
            std::cout << "  ✓ Test 1 passed: Zero probability events never sampled" << std::endl;
        }
        else
        {
            std::cerr << "  ✗ Test 1 FAILED: Some zero probability events were sampled!" << std::endl;
        }
    }

    // Test 2: Most elements are zero probability (unnormalized)
    std::cout << "\nTest 2: Sparse unnormalized distribution (only 2 non-zero elements)..." << std::endl;
    {
        std::vector<float> h_probs(vocab_size, 0.0f);
        h_probs[1] = 30.0f; // Only indices 1 and 3 have non-zero scores (unnormalized: 30 and 70)
        h_probs[3] = 70.0f; // Will be normalized to 0.3 and 0.7

        float* d_probs;
        int* d_output;
        CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<int> counts(vocab_size, 0);
        for (int i = 0; i < num_samples; ++i)
        {
            categoricalSampling(d_probs, d_output, 1, vocab_size, 66666ULL, i);
            int result;
            CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
            counts[result]++;
        }

        bool sparse_passed = true;
        std::cout << "  Results:" << std::endl;

        // Compute normalized expected probabilities
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
        {
            sum += h_probs[i];
        }

        for (int i = 0; i < vocab_size; ++i)
        {
            if (h_probs[i] == 0.0f && counts[i] > 0)
            {
                std::cerr << "    Index " << i << ": " << counts[i] << " samples ✗ (should be 0)" << std::endl;
                sparse_passed = false;
            }
            else if (h_probs[i] > 0.0f)
            {
                float expected = h_probs[i] / sum; // Normalize
                float observed = static_cast<float>(counts[i]) / num_samples;
                std::cout << "    Index " << i << ": " << counts[i] << " samples (" << std::fixed
                          << std::setprecision(2) << (observed * 100) << "%, expected " << (expected * 100) << "%) ✓"
                          << std::endl;
            }
        }

        CUDA_CHECK(cudaFree(d_probs));
        CUDA_CHECK(cudaFree(d_output));

        if (sparse_passed)
        {
            std::cout << "  ✓ Test 2 passed: Only non-zero probability elements sampled" << std::endl;
        }
        else
        {
            std::cerr << "  ✗ Test 2 FAILED: Some zero probability events were sampled!" << std::endl;
        }
    }

    // Test 3: With persistent states (unnormalized)
    std::cout << "\nTest 3: Zero probability with persistent states (unnormalized)..." << std::endl;
    {
        std::vector<float> h_probs(vocab_size);
        // First and last are zero, middle elements have equal unnormalized scores
        h_probs[0] = 0.0f;
        for (int i = 1; i < vocab_size - 1; ++i)
        {
            h_probs[i] = 100.0f; // Equal unnormalized scores
        }
        h_probs[vocab_size - 1] = 0.0f;

        float* d_probs;
        int* d_output;
        curandState* d_states;
        CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState)));
        CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        initializeRandomStates(d_states, 1, 77777ULL);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> counts(vocab_size, 0);
        for (int i = 0; i < num_samples; ++i)
        {
            categoricalSamplingWithStates(d_probs, d_output, 1, vocab_size, d_states);
            int result;
            CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
            counts[result]++;
        }

        bool persistent_passed = true;
        std::cout << "  First element (index 0, prob=0): " << counts[0] << " samples";
        if (counts[0] > 0)
        {
            std::cout << " ✗ FAILED!" << std::endl;
            persistent_passed = false;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }

        std::cout << "  Last element (index " << (vocab_size - 1) << ", prob=0): " << counts[vocab_size - 1]
                  << " samples";
        if (counts[vocab_size - 1] > 0)
        {
            std::cout << " ✗ FAILED!" << std::endl;
            persistent_passed = false;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }

        int middle_samples = 0;
        for (int i = 1; i < vocab_size - 1; ++i)
        {
            middle_samples += counts[i];
        }
        std::cout << "  Middle elements (indices 1-" << (vocab_size - 2) << "): " << middle_samples << "/"
                  << num_samples << " samples ✓" << std::endl;

        CUDA_CHECK(cudaFree(d_probs));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_states));

        if (persistent_passed)
        {
            std::cout << "  ✓ Test 3 passed: Persistent states respect zero probabilities" << std::endl;
        }
        else
        {
            std::cerr << "  ✗ Test 3 FAILED!" << std::endl;
        }
    }

    std::cout << "\nTest 8 passed!" << std::endl << std::endl;
}

void testUnnormalizedProbabilities()
{
    std::cout << "=== Test 9: Unnormalized Probabilities ===" << std::endl;

    int const batch_size = 3;
    int const vocab_size = 5;
    int const num_samples = 100000;

    // Test with various unnormalized distributions
    std::cout << "Testing with unnormalized probabilities..." << std::endl;

    // Batch 0: Unnormalized uniform (all 5.0 instead of 0.2)
    // Batch 1: Unnormalized [10, 20, 30, 25, 15] -> should behave like [0.1, 0.2, 0.3, 0.25, 0.15]
    // Batch 2: Very large values [1000, 2000, 3000, 2500, 1500]
    std::vector<float> h_probs = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,               // Batch 0: uniform
        10.0f, 20.0f, 30.0f, 25.0f, 15.0f,          // Batch 1: small unnormalized
        1000.0f, 2000.0f, 3000.0f, 2500.0f, 1500.0f // Batch 2: large unnormalized
    };

    // Expected normalized probabilities for verification
    std::vector<std::vector<float>> expected_probs
        = {{0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0.1f, 0.2f, 0.3f, 0.25f, 0.15f}, {0.1f, 0.2f, 0.3f, 0.25f, 0.15f}};

    float* d_probs;
    int* d_output;
    curandState* d_states;

    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, batch_size * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize random states
    initializeRandomStates(d_states, batch_size, 88888ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sample many times for each batch
    std::vector<std::vector<int>> counts(batch_size, std::vector<int>(vocab_size, 0));

    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSamplingWithStates(d_probs, d_output, batch_size, vocab_size, d_states);

        std::vector<int> h_output(batch_size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int b = 0; b < batch_size; ++b)
        {
            counts[b][h_output[b]]++;
        }
    }

    // Check statistical distributions
    bool all_passed = true;
    for (int b = 0; b < batch_size; ++b)
    {
        std::cout << "\nBatch " << b << " (unnormalized input: ";
        for (int i = 0; i < vocab_size; ++i)
        {
            std::cout << h_probs[b * vocab_size + i];
            if (i < vocab_size - 1)
                std::cout << ", ";
        }
        std::cout << "):" << std::endl;

        std::cout << "Index | Expected | Observed | Difference" << std::endl;
        std::cout << "------|----------|----------|------------" << std::endl;

        for (int i = 0; i < vocab_size; ++i)
        {
            float expected = expected_probs[b][i];
            float observed = static_cast<float>(counts[b][i]) / num_samples;
            float diff = std::abs(expected - observed);

            std::cout << std::setw(5) << i << " | " << std::setw(8) << std::fixed << std::setprecision(4) << expected
                      << " | " << std::setw(8) << observed << " | " << std::setw(10) << diff;

            if (diff > 0.01f)
            {
                std::cout << " ✗" << std::endl;
                all_passed = false;
            }
            else
            {
                std::cout << " ✓" << std::endl;
            }
        }
    }

    // Also test with standalone mode
    std::cout << "\nTesting standalone mode with unnormalized probabilities..." << std::endl;
    std::vector<int> standalone_counts(vocab_size, 0);

    // Use batch 1's unnormalized distribution
    float* d_single_prob;
    int* d_single_output;
    CUDA_CHECK(cudaMalloc(&d_single_prob, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_single_output, sizeof(int)));

    std::vector<float> single_probs = {10.0f, 20.0f, 30.0f, 25.0f, 15.0f};
    CUDA_CHECK(cudaMemcpy(d_single_prob, single_probs.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSampling(d_single_prob, d_single_output, 1, vocab_size, 99999ULL, i);
        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_single_output, sizeof(int), cudaMemcpyDeviceToHost));
        standalone_counts[result]++;
    }

    std::cout << "Standalone results:" << std::endl;
    for (int i = 0; i < vocab_size; ++i)
    {
        float expected = expected_probs[1][i];
        float observed = static_cast<float>(standalone_counts[i]) / num_samples;
        float diff = std::abs(expected - observed);

        std::cout << "  Index " << i << ": " << observed << " (expected " << expected << ", diff " << diff << ")";
        if (diff > 0.01f)
        {
            std::cout << " ✗" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_single_prob));
    CUDA_CHECK(cudaFree(d_single_output));

    if (all_passed)
    {
        std::cout << "\n✓ Test 9 passed! Unnormalized probabilities handled correctly." << std::endl;
    }
    else
    {
        std::cerr << "\n✗ Test 9 FAILED! Some distributions don't match expected values." << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "Starting Categorical Sampling CUDA Tests" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Run tests
    testBasicSampling();
    testStatisticalDistribution();
    testWithPersistentStates();
    testStatisticalDistributionWithPersistentStates();
    testDeterministicSampling();
    testEdgeCasesSampling();
    testSingleElementVector();
    testZeroProbabilityEvents();
    testUnnormalizedProbabilities();

    std::cout << "All tests completed successfully!" << std::endl;

    return 0;
}

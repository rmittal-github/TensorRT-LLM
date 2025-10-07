/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CategoricalSamplingPlugin.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// Simple logger for TensorRT
class Logger : public ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        // Only print errors and warnings
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// Helper to check CUDA errors
#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status)        \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

int main(int argc, char** argv)
{
    std::cout << "=== Categorical Sampling Plugin Example ===" << std::endl;

    // Configuration
    int const batchSize = 4;
    int const vocabSize = 10;

    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Vocab size: " << vocabSize << std::endl;

    // Initialize TensorRT plugin registry
    initLibNvInferPlugins(&gLogger, "");
    std::cout << "Initialized TensorRT plugin registry" << std::endl;

    // Register our plugin
    auto registry = getPluginRegistry();
    auto creator = new CategoricalSamplingPluginCreator();
    registry->registerCreator(*creator, "magpie.nvidia.com");
    std::cout << "Registered CategoricalSampling plugin creator" << std::endl;

    // Create plugin (no parameters needed - uses clock-based seeding)
    PluginFieldCollection fc;
    fc.nbFields = 0;
    fc.fields = nullptr;

    IPluginV3* plugin = creator->createPlugin("categorical_sampling", &fc, TensorRTPhase::kBUILD);

    if (!plugin)
    {
        std::cerr << "Failed to create plugin instance!" << std::endl;
        return 1;
    }
    std::cout << "Created plugin instance" << std::endl;

    // Create builder and network
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    if (!builder)
    {
        std::cerr << "Failed to create builder!" << std::endl;
        return 1;
    }

    auto const explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    if (!network)
    {
        std::cerr << "Failed to create network!" << std::endl;
        return 1;
    }
    std::cout << "Created TensorRT network" << std::endl;

    // Add input tensor (probabilities) - FP16
    auto inputDims = Dims2{batchSize, vocabSize};
    auto* inputTensor = network->addInput("probs", DataType::kHALF, inputDims);
    if (!inputTensor)
    {
        std::cerr << "Failed to add input tensor!" << std::endl;
        return 1;
    }

    // Add plugin layer
    ITensor* inputs[] = {inputTensor};
    ITensor* shapeInputs[] = {};
    auto* pluginLayer = network->addPluginV3(inputs, 1, shapeInputs, 0, *plugin);
    if (!pluginLayer)
    {
        std::cerr << "Failed to add plugin layer!" << std::endl;
        return 1;
    }
    pluginLayer->setName("categorical_sampling");
    std::cout << "Added plugin layer to network" << std::endl;

    // Mark output
    auto* outputTensor = pluginLayer->getOutput(0);
    outputTensor->setName("sampled_indices");
    network->markOutput(*outputTensor);

    // Build engine
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20); // 1 MB

    // Enable FP16 mode since we're using FP16 inputs
    config->setFlag(BuilderFlag::kFP16);

    std::cout << "Building TensorRT engine (FP16 mode)..." << std::endl;
    auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    if (!serializedEngine)
    {
        std::cerr << "Failed to build engine!" << std::endl;
        return 1;
    }
    std::cout << "Successfully built engine" << std::endl;

    // Deserialize engine
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));

    if (!engine)
    {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        return 1;
    }

    // Create execution context
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "Failed to create execution context!" << std::endl;
        return 1;
    }
    std::cout << "Created execution context" << std::endl;

    // Prepare input data (simple probability distributions)
    std::vector<float> hostProbsFloat(batchSize * vocabSize);

    // Create different probability distributions for each batch
    for (int b = 0; b < batchSize; ++b)
    {
        float sum = 0.0f;
        for (int v = 0; v < vocabSize; ++v)
        {
            // Create different patterns for each batch
            if (b == 0)
            {
                // Uniform distribution
                hostProbsFloat[b * vocabSize + v] = 1.0f;
            }
            else if (b == 1)
            {
                // Peaked at beginning
                hostProbsFloat[b * vocabSize + v] = vocabSize - v;
            }
            else if (b == 2)
            {
                // Peaked at end
                hostProbsFloat[b * vocabSize + v] = v + 1;
            }
            else
            {
                // Peaked in middle
                hostProbsFloat[b * vocabSize + v] = (v < vocabSize / 2) ? v + 1 : vocabSize - v;
            }
            sum += hostProbsFloat[b * vocabSize + v];
        }
        // Normalize
        for (int v = 0; v < vocabSize; ++v)
        {
            hostProbsFloat[b * vocabSize + v] /= sum;
        }
    }

    // Convert to FP16
    std::vector<half> hostProbs(batchSize * vocabSize);
    for (int i = 0; i < batchSize * vocabSize; ++i)
    {
        hostProbs[i] = __float2half(hostProbsFloat[i]);
    }

    std::cout << "\nInput probability distributions (FP16):" << std::endl;
    for (int b = 0; b < batchSize; ++b)
    {
        std::cout << "Batch " << b << ": [";
        for (int v = 0; v < vocabSize; ++v)
        {
            printf("%.3f", hostProbsFloat[b * vocabSize + v]);
            if (v < vocabSize - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Allocate device memory (FP16 for input)
    void* deviceProbs;
    void* deviceOutput;
    CHECK_CUDA(cudaMalloc(&deviceProbs, batchSize * vocabSize * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * sizeof(int32_t)));

    // Copy input to device (FP16)
    CHECK_CUDA(cudaMemcpy(deviceProbs, hostProbs.data(), batchSize * vocabSize * sizeof(half), cudaMemcpyHostToDevice));

    // Set input/output bindings
    void* bindings[] = {deviceProbs, deviceOutput};

    // Execute inference
    std::cout << "\nExecuting inference..." << std::endl;
    bool status = context->executeV2(bindings);
    if (!status)
    {
        std::cerr << "Inference execution failed!" << std::endl;
        return 1;
    }

    // Copy output back to host
    std::vector<int32_t> hostOutput(batchSize);
    CHECK_CUDA(cudaMemcpy(hostOutput.data(), deviceOutput, batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "\nSampled indices:" << std::endl;
    for (int b = 0; b < batchSize; ++b)
    {
        std::cout << "Batch " << b << ": " << hostOutput[b] << std::endl;
    }

    // Run multiple times to show randomness
    std::cout << "\nRunning 5 more times to demonstrate randomness:" << std::endl;
    for (int run = 0; run < 5; ++run)
    {
        context->executeV2(bindings);
        CHECK_CUDA(cudaMemcpy(hostOutput.data(), deviceOutput, batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::cout << "Run " << (run + 2) << ": [";
        for (int b = 0; b < batchSize; ++b)
        {
            std::cout << hostOutput[b];
            if (b < batchSize - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(deviceProbs));
    CHECK_CUDA(cudaFree(deviceOutput));

    std::cout << "\n=== Example completed successfully ===" << std::endl;
    std::cout << "\nThe FP16 plugin:" << std::endl;
    std::cout << "- Accepts FP16 (half precision) probability inputs" << std::endl;
    std::cout << "- Converts to FP32 internally for accurate computation" << std::endl;
    std::cout << "- Uses clock-based seeding for non-reproducible randomness" << std::endl;
    std::cout << "- Initializes cuRAND states internally within the kernel" << std::endl;
    std::cout << "- Produces different random samples on each inference run" << std::endl;
    std::cout << "- Handles unnormalized probability distributions automatically" << std::endl;
    std::cout << "- No configuration needed - works out of the box!" << std::endl;

    return 0;
}

/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

 #include "tensorrt_llm/kernels/cfgKernels.h"

 namespace tensorrt_llm::kernels
 {
 
 namespace
 {
 
 // The __global__ kernel should be defined in the .cu file and is best placed
 // in an anonymous namespace to limit its visibility to this translation unit.
 template <typename T>
 __global__ void applyCfgKernel(
     T* logits, int const numRequests, int const vocabSize, float const cfgScale)
 {
     // Each block processes one or more requests using a grid-stride loop.
     // This makes the kernel robust to any number of requests.
     for (int reqIdx = blockIdx.x; reqIdx < numRequests; reqIdx += gridDim.x)
     {
         // Each thread in the block processes one or more vocab entries using a block-stride loop.
         // This ensures all vocab entries are processed regardless of vocabSize.
         for (int vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
         {
             // The input tensor is conceptually [numRequests, 2, vocabSize] but laid out as
             // a contiguous [numRequests * 2, vocabSize] tensor.
             // We access the conditional logits at [reqIdx * 2 * vocabSize + vocabIdx] and
             // unconditional logits at [reqIdx * 2 * vocabSize + vocabSize + vocabIdx].
             T* condLogitPtr = logits + reqIdx * 2 * vocabSize + vocabIdx;
             T const* uncondLogitPtr = logits + reqIdx * 2 * vocabSize + vocabSize + vocabIdx;
 
             // Perform calculations in float for precision.
             float condLogitFloat = static_cast<float>(*condLogitPtr);
             float uncondLogitFloat = static_cast<float>(*uncondLogitPtr);
 
             // Apply the CFG formula: guidance * cond + (1 - guidance) * uncond
             float result = cfgScale * condLogitFloat + (1.0f - cfgScale) * uncondLogitFloat;
 
             // Store the result back in place of the conditional logit.
             *condLogitPtr = static_cast<T>(result);
         }
     }
 }
 
 } // anonymous namespace
 
 
 // Definition of the function that launches the kernel.
 // This function must be defined in the .cu file because it uses <<<...>>> syntax.
 template <typename T>
 void invokeCfgKernel(T* logits, int const numRequests, int const vocabSize, float const cfgScale, cudaStream_t stream)
 {
     // A block size of 512 is a good general-purpose choice for memory-bound kernels.
     dim3 block(512);
     // Use a grid-stride loop, so we don't need a grid size equal to numRequests.
     // Capping the grid size can improve efficiency by avoiding launching an excessive
     // number of small blocks. 256 is a safe heuristic.
     dim3 grid(std::min(numRequests, 256));
 
     // Launch the kernel.
     applyCfgKernel<T><<<grid, block, 0, stream>>>(logits, numRequests, vocabSize, cfgScale);
 
     // It is critical to check for errors after launching a kernel.
     TLLM_CUDA_CHECK(cudaGetLastError());
 }
 
 // Explicitly instantiate the templates for the supported data types.
 // This is required because the definition is in a .cu file and would not
 // otherwise be visible to other parts of the program that include the header.
 template void invokeCfgKernel<float>(float* logits, int const numRequests, int const vocabSize, float const cfgScale, cudaStream_t stream);
 template void invokeCfgKernel<half>(half* logits, int const numRequests, int const vocabSize, float const cfgScale, cudaStream_t stream);
 
 } // namespace tensorrt_llm::kernels
 
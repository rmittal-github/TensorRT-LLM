#pragma once

#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels
{

/**
 * @brief Forward declaration for the templated kernel launcher.
 *
 * The full definition of this function resides in the .cu file and is compiled by NVCC.
 * This declaration makes it visible to the inline `invokeCfg` function below.
 */
template <typename T>
void invokeCfgKernel(T* logits, int const numRequests, int const vocabSize, float const cfgScale, cudaStream_t stream);

/**
 * @brief Applies classifier-free guidance (CFG) to the logits tensor in-place on the GPU.
 *
 * This function is the main entry point for the CFG operation. It is a type-dispatcher
 * that calls the appropriate templated kernel launcher based on the data type of the logits tensor.
 *
 * @details The formula applied is:
 * `logits_cond = cfgScale * logits_cond + (1 - cfgScale) * logits_uncond`
 *
 * The input logits tensor is expected to have a shape that can be interpreted as
 * [numRequests, 2, vocabSize], where logits[i, 0, :] are the conditional logits and
 * logits[i, 1, :] are the unconditional logits. The result is written back into the
 * conditional logits' location. For efficiency, the implementation treats the tensor
 * as having the shape [numRequests * 2, vocabSize].
 *
 * @param stream The CUDA stream to execute the kernel on.
 * @param logitsView A shared pointer to the tensor containing both conditional and
 * unconditional logits. The tensor is modified in-place.
 * @param numRequests The number of requests in the batch.
 * @param vocabSize The size of the vocabulary.
 * @param cfgScale The guidance scale factor. A value of 1.0 effectively disables CFG.
 */
inline void invokeCfg(tensorrt_llm::runtime::CudaStream const& stream,
    runtime::ITensor::SharedPtr logitsView, int numRequests, int vocabSize, float cfgScale)
{
    auto const& logitsDataType = logitsView->getDataType();

    if (logitsDataType == nvinfer1::DataType::kFLOAT)
    {
        invokeCfgKernel(runtime::bufferCast<float>(*logitsView),
            numRequests, vocabSize, cfgScale, stream.get());
    }
    else if (logitsDataType == nvinfer1::DataType::kHALF)
    {
        invokeCfgKernel(runtime::bufferCast<half>(*logitsView),
            numRequests, vocabSize, cfgScale, stream.get());
    }
    else
    {
        TLLM_THROW("Unsupported data type for CFG. Only float and half are supported.");
    }
}

} // namespace tensorrt_llm::kernels
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"

#include <NvInferRuntime.h>
#include <memory>

namespace tensorrt_llm::runtime
{
class GptDecoderBatched;
}

namespace tensorrt_llm::batch_manager
{
class DecoderInputBuffers;
class DecoderBuffers;
class MakeDecodingBatchInputOutput;

class TrtLocalTransformer
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorMap = runtime::ITensor::TensorMap;

    static constexpr auto kInHiddenStatesTensorName = "hidden_states";
    static constexpr auto kInTokensTensorName = "tokens";
    static constexpr auto kOutLogitsTensorName = "logits";

    TrtLocalTransformer(
        runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig,
        runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger);
    
    ~TrtLocalTransformer();

    /// \brief Run the local transformer using hiddenStates and current request sets.
    void run(TensorPtr const& hiddenStates,
        RequestVector const& contextRequests,
        std::vector<SizeType32> const& numContextFramesVec,
        RequestVector const& generationRequests,
        // TODO: create those inside local transformer
        std::shared_ptr<runtime::GptDecoderBatched> &decoder,
        std::unique_ptr<runtime::decoder_batch::Input> &decodingInput,
        std::unique_ptr<runtime::decoder_batch::Output> &decodingOutput,
        DecoderInputBuffers &decoderInputBuffers,
        std::shared_ptr<DecoderBuffers> &decoderBuffers,
        SizeType32 maxNumSequences
    );

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const;

private:
    runtime::ModelConfig mModelConfig;
    runtime::WorldConfig mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<runtime::TllmRuntime> mRuntime;
    int hiddenSize;
    int numTokens;
    int vocabSize;

    TensorPtr inHiddenStates;  // [batch x dim]
    TensorPtr inTokens;  // [8 x batch']
    TensorPtr outLogits;  // [batch' x VocabSize]
    TensorPtr outLogitsHost;  // [batch' x VocabSize]
    TensorMap inputMap;
    TensorMap outputMap;

    std::unique_ptr<tensorrt_llm::batch_manager::MakeDecodingBatchInputOutput const> mMakeDecodingBatchInputOutput;

    void HandleLogits(
        RequestVector const& contextRequests,
        RequestVector const& generationRequests,
        std::shared_ptr<DecoderBuffers> &decoderBuffers
    );
};

} // namespace tensorrt_llm::batch_manager



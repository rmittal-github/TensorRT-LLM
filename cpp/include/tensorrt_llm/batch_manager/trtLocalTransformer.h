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

#include <NvInferRuntime.h>
#include <memory>

namespace tensorrt_llm::batch_manager
{

class TrtLocalTransformer
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;

    static constexpr auto kInHiddenStatesTensorName = "hidden_states";

    TrtLocalTransformer(runtime::WorldConfig const& worldConfig,
        runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger);

    /// \brief Run the local transformer using logits and current request sets.
    void run(TensorPtr const& logits,
        RequestVector const& contextRequests,
        RequestVector const& generationRequests);

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const;

private:
    runtime::WorldConfig mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<runtime::TllmRuntime> mRuntime;

    TensorPtr inHiddenStates;  // [batch x dim]
};

} // namespace tensorrt_llm::batch_manager



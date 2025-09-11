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

#include "tensorrt_llm/batch_manager/trtLocalTransformer.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

namespace tensorrt_llm::batch_manager
{

using namespace tensorrt_llm::runtime;

TrtLocalTransformer::TrtLocalTransformer(
    runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine,
    std::shared_ptr<nvinfer1::ILogger> logger)
    , mWorldConfig(worldConfig)
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mRuntime{std::make_shared<TllmRuntime>(rawEngine, logger.get(), /*gpuWeightsPercent*/ std::nullopt)}
{
    auto const statesType = rawEngine.getTensorDataType(batch_manager::TrtLocalTransformer::kInHiddenStatesTensorName);
    inHiddenStates = mRuntime->getBufferManager().emptyTensor(MemoryType::kGPU, statesType);
}

void TrtLocalTransformer::run(TensorPtr const& logits, RequestVector const& contextRequests,
    RequestVector const& generationRequests)
{
    // reshape input hidden states based on the inputs and fill it in
}

runtime::BufferManager const& TrtLocalTransformer::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

runtime::BufferManager::CudaStreamPtr TrtLocalTransformer::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

} // namespace tensorrt_llm::batch_manager



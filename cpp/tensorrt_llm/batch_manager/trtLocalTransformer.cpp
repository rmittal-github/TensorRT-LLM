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

#include <algorithm>

#include "tensorrt_llm/batch_manager/trtLocalTransformer.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"

namespace tensorrt_llm::batch_manager
{

using namespace tensorrt_llm::runtime;

TrtLocalTransformer::TrtLocalTransformer(
    runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine,
    std::shared_ptr<nvinfer1::ILogger> logger,
    SizeType32 maxNumSequences,
    SizeType32 maxSequenceLen,
    SizeType32 numMicroBatches,
    SizeType32 maxBatchSize)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mRuntime{std::make_shared<TllmRuntime>(rawEngine, logger.get(), 1.0f)}
    , hiddenSize{768}  // TODO: change to 768 once switch to use hidden state instead of logits from model
    , numTokens{8}
    , vocabSize{2024}
    , mMaxNumSequences{maxNumSequences}
{
    // create a context for the local transformer engine
    mRuntime->clearContexts();
    mRuntime->addContext(0);

    auto& manager = getBufferManager();
    auto const statesType = mRuntime->getEngine().getTensorDataType(kInHiddenStatesTensorName);
    inHiddenStates = manager.emptyTensor(MemoryType::kGPU, statesType);
    inTokens = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    inTokensSliceHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    auto const logitsType = mRuntime->getEngine().getTensorDataType(kOutLogitsTensorName);
    outLogits = manager.emptyTensor(MemoryType::kGPU, logitsType);
    outLogitsHost = manager.emptyTensor(MemoryType::kCPU, logitsType);

    // create decoder and buffers
    mDecoder = std::make_shared<runtime::GptDecoderBatched>(
        getRuntimeStreamPtr(),
        mModelConfig.getSpeculativeDecodingMode(),
        logitsType
    );
    auto decodingMode = executor::DecodingMode::TopKTopP();
    mDecoder->setup(decodingMode, mMaxNumSequences, 1 /*beam width*/, 0 /*attn window*/,
        0 /*sink token len*/, maxSequenceLen, mModelConfig.getMaxDecodingTokens(),
        logitsType, mModelConfig, mWorldConfig
    );
    for (SizeType32 i = 0; i < numMicroBatches; ++i)
    {
        mDecoderInputBuffers.emplace_back(
            maxBatchSize, mModelConfig.getMaxDecodingTokens(), getBufferManager());
    }
    for (SizeType32 i = 0; i < numTokens; i++) {
        // independent decoder buffer for each token
        mDecoderBuffers.push_back(std::make_shared<DecoderBuffers>(
            mMaxNumSequences, 1 /*beam width*/,
            0 /*attn window*/, maxSequenceLen,
            mModelConfig.getMaxDecodingTokens(), getBufferManager(),
            mModelConfig, mWorldConfig
        ));
    }
    mSlotDecoderBuffers.clear();
    for (SizeType32 i = 0; i < mMaxNumSequences; ++i)
    {
        mSlotDecoderBuffers.emplace_back(std::make_shared<SlotDecoderBuffers>(
            1 /*beam width*/, maxSequenceLen, getBufferManager()));
    }
    mDecodingInputs.resize(numMicroBatches);
}

TrtLocalTransformer::~TrtLocalTransformer()
{
}

void TrtLocalTransformer::HandleLogits(
    RequestVector const& contextRequests,
    RequestVector const& generationRequests,
    std::shared_ptr<DecoderBuffers> &decoderBuffers
) {
    // forward the logits to the decoder buffers
    SizeType32 batchIndex{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const seqSlot = llmReq->mSeqSlots.at(0);
            auto& decoderLogits = decoderBuffers->logits.at(seqSlot);
            TensorPtr logitsView = ITensor::slice(outLogits, batchIndex, 1);
            decoderLogits = ITensor::view(logitsView, ITensor::makeShape({1, 1, vocabSize}));
            batchIndex++;
        }
    }
}


void TrtLocalTransformer::run(TensorPtr const& hiddenStates,
    RequestVector const& contextRequests,
    std::vector<SizeType32> const& numContextFramesVec,
    RequestVector const& generationRequests,
    SizeType32 microBatchId,
    SizeType32 fusedBufferId
) {
    // reshape input hidden states based on the inputs and fill it in using requests
    // check that all requests are actually cfg
    bool allCfg = true;
    bool hasCfg = false;
    for (auto const& req : contextRequests) {
        allCfg &= req->isCfg();
        hasCfg |= req->isCfg();
    }
    for (auto const& req : generationRequests) {
        allCfg &= req->isCfg();
        hasCfg |= req->isCfg();
    }
    if (!allCfg && hasCfg) {
        TLLM_LOG_ERROR("Either all requests should be CFG or none");
    }
    // that specifies whether each request corresponds to dim * 2 or just dim of hidden states
    int cfgMult = allCfg ? 2 : 1;

    // overall batch size
    auto const batchSize = (int)(contextRequests.size() + generationRequests.size()) * cfgMult;

    auto& manager = getBufferManager();
    auto const statesType = mRuntime->getEngine().getTensorDataType(kInHiddenStatesTensorName);
    inHiddenStates = manager.gpu(ITensor::makeShape({batchSize, hiddenSize}), statesType);

    // for context requests, copy the hidden states into the input buffer
    SizeType32 batchIndex{0};
    SizeType32 frameIndex{0};
    for (auto const& llmReq : contextRequests) {

        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "Beam width must be 1 for local transformer");
        auto const contextFrames = numContextFramesVec.at(batchIndex);
        TLLM_CHECK_WITH_INFO(!llmReq->isLastContextChunk() || llmReq->getNumDraftTokens() == 0,
            "Draft tokens are not supported for local transformer");

        // copy hidden states for conditional (and optionally unconditional) generation
        for (SizeType32 i = 0; i < cfgMult; i++) {
            frameIndex += contextFrames;
            auto const numFrames = 1;
            TensorPtr statesView = ITensor::slice(hiddenStates, frameIndex - numFrames, numFrames);
            TensorPtr outStatesView = ITensor::slice(inHiddenStates, batchIndex, numFrames);
            manager.copy(*statesView, *outStatesView);
            batchIndex += numFrames;
        }
    }

    // for generation requests, copy the rest of the runtime buffer to the local transformer buffer
    if (generationRequests.size() > 0) {
        TensorPtr genStatesView = ITensor::slice(hiddenStates, frameIndex, generationRequests.size() * cfgMult);
        TensorPtr outGenStatesView = ITensor::slice(inHiddenStates, batchIndex, generationRequests.size() * cfgMult);
        manager.copy(*genStatesView, *outGenStatesView);
    }

    // create a buffer for the tokens, 0th token reserved for hidden states
    inTokens = manager.gpu(ITensor::makeShape({numTokens + 1, batchSize / cfgMult}), nvinfer1::DataType::kINT32);
    inTokensSliceHost = manager.cpu(ITensor::makeShape({batchSize / cfgMult}), nvinfer1::DataType::kINT32);
    // TODO: set intokens[0] to special token which is expanded to 0 with emb of local transformer
    // for CFG, model folds the tensor in two
    auto const logitsType = mRuntime->getEngine().getTensorDataType(kOutLogitsTensorName);
    outLogits = manager.gpu(ITensor::makeShape({batchSize / cfgMult, vocabSize}), logitsType);
    outLogitsHost = manager.cpu(ITensor::makeShape({batchSize / cfgMult, vocabSize}), logitsType);

    inputMap.clear();
    outputMap.clear();
    // always needs the hidden states as input
    inputMap.insert_or_assign(kInHiddenStatesTensorName, inHiddenStates);
    // puts logits into the same buffer
    outputMap.insert_or_assign(kOutLogitsTensorName, outLogits);
    for (int i = 0; i < numTokens; i++) {

        // provide additional input - previously generated tokens
        TensorPtr prevTokensView = ITensor::slice(inTokens, 0, i + 1);  // i x batch
        inputMap.insert_or_assign(kInTokensTensorName, prevTokensView);

        auto const contextId = 0;
        mRuntime->setInputTensors(contextId, inputMap);
        mRuntime->setOutputTensors(contextId, outputMap);
        auto enqueueSuccessful = mRuntime->executeContext(contextId);
        if (!enqueueSuccessful)
        {
            throw std::runtime_error("Executing local transformer engine failed!");
        }
        sync_check_cuda_error(mRuntime->getStream().get());

        // handle logits, but forwarding them to requests
        HandleLogits(contextRequests, generationRequests, mDecoderBuffers.at(i));
        // prepare inputs for decoder
        auto& decodingInput = mDecodingInputs.at(microBatchId);
        std::tie(decodingInput, mDecodingOutput)
            = (*mMakeDecodingBatchInputOutput)(contextRequests, generationRequests,
                *mDecoderBuffers.at(i), mDecoderInputBuffers.at(fusedBufferId), mDecoder->getDecoderState(),
                mModelConfig, mMaxNumSequences, 1, getBufferManager(),
                mRuntime->getStream(), std::nullopt);
        // actually execute the decoder
        runtime::CudaEvent finishedEvent = mDecoder->forwardAsync(*mDecodingOutput, *decodingInput);
        // update decoder buffers
        manager.getStream().wait(finishedEvent);
        manager.copy(*mDecoder->getDecoderState().getAllNewTokens(), *mDecoderBuffers.at(i)->newOutputTokensHost);
        manager.copy(*mDecoder->getDecoderState().getJointDecodingOutput().lengths, *mDecoderBuffers.at(i)->sequenceLengthsHost);
        auto const finishedSumDevice = mDecoder->getDecoderState().getFinishedSum();
        manager.copy(*finishedSumDevice, *mDecoderBuffers.at(i)->finishedSumHost);
        auto const finishReasonsDevice = mDecoder->getDecoderState().getFinishReasons();
        manager.copy(*finishReasonsDevice, *mDecoderBuffers.at(i)->finishReasonsHost);
        sync_check_cuda_error(mRuntime->getStream().get());

        // should copy newly generated tokens to inTokens.
        // the problem is that `getAllNewTokens` and `newOutputTokensHost` are ordered using seqSlots.
        // need to collect the tokens in order of the batch and copy them to inTokens
        // TODO: is it possible to do it in a more optimized way??
        // we have a buffer of size (1 x batchsize) on cpu, we collect to it the tokens,
        // then copy it to inTokens
        batchIndex = 0;
        auto const hostNewOutputTokensShape = mDecoderBuffers.at(i)->newOutputTokensHost->getShape();
        auto const* const hostNewOutputTokensData
            = bufferCast<TokenIdType const>(*mDecoderBuffers.at(i)->newOutputTokensHost);
        auto* const inTokensSliceHostData = bufferCast<TokenIdType>(*inTokensSliceHost);
        for (auto const& requests : {contextRequests, generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const seqSlot = llmReq->mSeqSlots.at(0);
                auto const newTokenIdx = tensorrt_llm::common::flat_index(hostNewOutputTokensShape.d, 0 /*step*/, seqSlot, 0 /*beam*/);
                auto const newToken = hostNewOutputTokensData[newTokenIdx];
                inTokensSliceHostData[batchIndex] = newToken;
                TLLM_LOG_WARNING(">>>>request ID %ld vocab %d, newToken %d", llmReq->mRequestId, i, newToken);
                batchIndex++;
            }
        }
        manager.copy(*inTokensSliceHost, *ITensor::slice(inTokens, i + 1, 1));
        sync_check_cuda_error(mRuntime->getStream().get());
    }
}

std::shared_ptr<runtime::GptDecoderBatched>& TrtLocalTransformer::getDecoder()
{
    return mDecoder;
}

DecoderInputBuffers& TrtLocalTransformer::getDecoderInputBuffers(SizeType32 microBatchId)
{
    return mDecoderInputBuffers.at(microBatchId);
}

std::shared_ptr<DecoderBuffers>& TrtLocalTransformer::getDecoderBuffers(SizeType32 vocabId)
{
    return mDecoderBuffers.at(vocabId);
}

std::shared_ptr<SlotDecoderBuffers>& TrtLocalTransformer::getSlotDecoderBuffers(SizeType32 seqSlot)
{
    return mSlotDecoderBuffers.at(seqSlot);
}

runtime::BufferManager const& TrtLocalTransformer::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

runtime::BufferManager::CudaStreamPtr TrtLocalTransformer::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

runtime::CudaStream const& TrtLocalTransformer::getRuntimeStream() const
{
    return mRuntime->getStream();
}

} // namespace tensorrt_llm::batch_manager



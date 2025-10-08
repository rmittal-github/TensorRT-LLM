#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0 /usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec --fp16     --onnx=/code/tensorrt_llm/local_transformer.onnx     --saveEngine=/code/tensorrt_llm/local_transformer.trt     --minShapes=hidden_states:2x768      --optShapes=hidden_states:8x768     --maxShapes=hidden_states:32x768      --plugins=/code/tensorrt_llm/build/libcategorical_sampling_plugin.so

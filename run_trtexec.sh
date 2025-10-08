#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0 /usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec --fp16 \
    --onnx=/code/tensorrt_llm/local_transformer.onnx \
    --saveEngine=/code/tensorrt_llm/local_transformer.trt \
   --maxShapes=hidden_states:32x16192,tokens:8x16 \
     --plugins=/code/tensorrt_llm/build/libcategorical_sampling_plugin.so

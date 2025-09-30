#!/bin/bash

docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864                  --gpus=all                 --volume $HOME/trtllm/TensorRT-LLM:/code/tensorrt_llm                 --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache"                 --env "CCACHE_BASEDIR=/code/tensorrt_llm"                 --workdir /code/tensorrt_llm                 --hostname aiapps-021325-devel                 --name tensorrt_llm-devel-rfejgin                 --tmpfs /tmp:exec                 tensorrt_llm/devel:latest-rfejgin

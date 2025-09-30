#!/bin/bash
set -x
python examples/models/contrib/t5tts/convert_checkpoint.py \
    --model_path "checkpoints/model_weights.ckpt" \
    --output_dir checkpoints/magpie_convert --dtype float16

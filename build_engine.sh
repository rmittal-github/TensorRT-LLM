#!/bin/bash
set -x
export PATH=/home/rfejgin/.local/bin/:$PATH
trtllm-build --checkpoint_dir checkpoints/magpie_convert/decoder \
	--output_dir checkpoints/magpie_engine/decoder \
	--moe_plugin disable \
	--max_beam_width 1 \
	--max_batch_size 128 \
	--max_input_len 2048 \
	--max_seq_len 8192 \
	--max_encoder_input_len 256 \
	--gemm_plugin float16 \
	--bert_attention_plugin float16 \
	--gpt_attention_plugin float16 \
	--remove_input_padding enable

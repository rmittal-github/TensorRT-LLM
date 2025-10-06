import json
import os
import time

import click
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import requests
import tensorrt as trt
import torch
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from omegaconf.omegaconf import OmegaConf, open_dict

def update_config(model_cfg, codecmodel_path, legacy_codebooks=False):
    ''' helper function to rename older yamls from t5 to magpie '''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if hasattr(model_cfg, 'use_local_transformer') and model_cfg.use_local_transformer:
        # For older checkpoints trained with a different parameter name
        model_cfg.local_transformer_type = "autoregressive"
        del model_cfg.use_local_transformer

    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print("WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints.")
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2
    if hasattr(model_cfg, 'sample_rate'):
        # This was removed from the config and is now in the model class
        sample_rate = model_cfg.sample_rate
        del model_cfg.sample_rate
    else:
        sample_rate = None
    return model_cfg, sample_rate


def update_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
            new_state_dict[new_key] = state_dict[key]
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def load_model(checkpoint_file, hparams_file, audio_codec, engine_dir, legacy_codebooks=False):
    if hparams_file is not None and checkpoint_file is not None:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(model_cfg, audio_codec, legacy_codebooks)

        model = MagpieTTSModel(cfg=model_cfg)
        model.use_kv_cache_for_inference = True

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(checkpoint_file, weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        state_dict = update_ckpt(ckpt)
        model.load_state_dict(state_dict)
        checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]

    if cfg_sample_rate is not None and cfg_sample_rate != model.sample_rate:
        raise ValueError("Sample rate in config and model do not match")

    print("Loaded weights.")
    model.cuda()
    model.eval() 
    model = model.half()
    return model, model_cfg


class IntLT(torch.nn.Module):
    def __init__(self, model:MagpieTTSModel, dtype, d_model=768, cfg_scale=2.5):
        super().__init__()
        self.local_transformer_in_projections=model.local_transformer_in_projection
        self.local_transformer=model.local_transformer
        self.local_transformer_out_projections=model.local_transformer_out_projections
        
        with torch.no_grad():
            initial_embeddings = torch.nn.Embedding(model.audio_embeddings[0].weight.shape[0], model.audio_embeddings[0].weight.shape[1], _freeze=False).cuda().eval()
            initial_embeddings.weight.data.zero_()
            initial_embeddings = initial_embeddings.half()
            initial_embeddings.weight.requires_grad = False
            self.audio_embeddings = torch.nn.ModuleList([initial_embeddings])
            
            for audio_embedding in model.audio_embeddings:
                audio_embedding.requires_grad = False
                audio_embedding = audio_embedding.half()
                self.audio_embeddings.append(audio_embedding)
            
            self.audio_embeddings = self.audio_embeddings.cuda().half()
        
        self.dtype = dtype
        self.d_model = d_model
        self.cfg_scale = cfg_scale
    
    def forward(self, dec_output, tokens):
        logits= []
        for layer_idx,out_projection_layer in enumerate(self.local_transformer_out_projections):
            audio_embedding = self.audio_embeddings[layer_idx](tokens.transpose(0, 1))
            audio_embedding = audio_embedding.repeat_interleave(2, dim=0)  # 2b x N x dimj
            audio_embedding[:,0,:] += dec_output
            
            local_transformer_input = self.local_transformer_in_projections(audio_embedding)
            
            mask = torch.ones(audio_embedding.shape[0], audio_embedding.shape[1], device=audio_embedding.device, requires_grad=False, dtype=torch.bool)
            local_transformer_output = self.local_transformer(local_transformer_input, mask)["output"]

            projection_layer_out = out_projection_layer(local_transformer_output[:, -1, :])

            cond_logits = projection_layer_out[::2, :]  # select even indices (0,2,4,...)
            uncond_logits = projection_layer_out[1::2, :]  # select odd indices (1,3,5,...)
            final_logits = cond_logits * self.cfg_scale + uncond_logits * (1 - self.cfg_scale)
            logits.append(final_logits.unsqueeze(1))

        return torch.cat(logits, dim=1)

    def export_to_onnx(self, onnx_file, opset_version):
        cfg_bs = 4
        out_bs = int(cfg_bs / 2)
        dtype = torch.float16 if self.dtype == "float16" else torch.float32
        dec_output = torch.rand(cfg_bs, self.d_model).to("cuda").to(dtype=dtype)
        tokens = torch.ones(3, out_bs, dtype=torch.int).to("cuda")

        with torch.no_grad():
            input_names = ["hidden_states", "tokens"]
            output_names = ["logits"]
            dynamic_axes = {
                "hidden_states": {
                    0: "cfg_batch_size",
                },
                "tokens": {
                    0: "num_tokens",
                    1: "batch_size"
                },
                "logits": {
                    0: "batch_size",
                }
            }
            inputs_args = {
                'hidden_states': dec_output,
                'tokens': tokens,
            }
            torch.onnx.export(self,
                              tuple(inputs_args.values()),
                              onnx_file,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes,
                              opset_version=opset_version)

class MagpieLocalTransformerExportTRT:

    def __init__(self,
                 checkpoint_dir,
                 engine_dir,
                 model_cfg,
                 minBS=1,
                 optBS=None,
                 maxBS=2,
                 dtype="float16",
                 opset_version=17):
        self.checkpoint_dir = checkpoint_dir
        self.engine_dir = engine_dir
        self.opset_version = opset_version
        self.lt_config = {}

        self.dtype = dtype

        if optBS is None:
            optBS = minBS + int((maxBS - minBS) / 2)

        if optBS > maxBS or optBS < minBS:
            raise Exception(f"Invalid optBS should be minBS < optBS < maxBS")

        self.minBS = minBS
        self.optBS = optBS
        self.maxBS = maxBS
        if optBS is None:
            self.optBS = minBS + int((maxBS - minBS) / 2)

        print(model_cfg.keys())
        self.n_codebooks = model_cfg.get("num_codebooks", 8)
    
        self.d_model = model_cfg.decoder.d_model

        self.lt_config['min_batch_size'] = minBS
        self.lt_config['opt_batch_size'] = optBS
        self.lt_config['max_batch_size'] = maxBS
        self.lt_config['d_model'] = self.d_model
        self.lt_config['dtype'] = dtype

    def export_lt_to_onnx(self, model):
        int_lt = IntLT(model, dtype=self.dtype)

        onnx_file = os.path.join(self.checkpoint_dir, 'local_transformer/local_transformer.onnx')
        int_lt.export_to_onnx(onnx_file, opset_version=self.opset_version)

    def generate_trt_engine(self):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.dtype == "bfloat16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Using FP32")
        
        print(f"{config.flags=}")

        #config.flags = config.flags
        parser = trt.OnnxParser(network, logger)
        onnx_file = os.path.join(self.checkpoint_dir, 'local_transformer/local_transformer.onnx')

        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        nBS = -1
        nTokens = -1
        nMinBS = self.minBS
        nMaxBS = self.maxBS
        nOptBS = self.optBS
        
        input_feat = network.get_input(0)
        input_tokens = network.get_input(1)
        input_feat.shape = [nBS, self.d_model]
        input_tokens.shape = [nTokens, nBS]
        profile.set_shape(
            input_feat.name,
            [nMinBS*2, self.d_model],
            [nOptBS*2, self.d_model],
            [nMaxBS*2, self.d_model],
        )
        profile.set_shape(
            input_tokens.name,
            [1, nMinBS],
            [4, nOptBS],
            [8, nMaxBS],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        plan_path = os.path.join(self.engine_dir, "local_transformer")
        os.makedirs(plan_path, exist_ok=True)

        plan_file = os.path.join(plan_path, 'local_transformer.plan')
        config_file = os.path.join(plan_path, 'config.json')

        if engineString == None:
            print("Failed building %s" % plan_file)
        else:
            print("Succeeded building %s in %d s" % (plan_file, t1 - t0))
            with open(plan_file, "wb") as f:
                f.write(engineString)
            with open(config_file, 'w') as jf:
                json.dump(self.lt_config, jf)


@click.command()
@click.option("--dtype", type=str, default="float16", help="dataype of model")
@click.option("--model_ckpt", type=str, help="Path to model checkpoint")
@click.option("--audio_codec", type=str, help="Output Path to audio codec")
@click.option("--hparams_file", type=str, help="Path to hparams file")
@click.option("--max_bs", type=int, default=32, help="maximum batch size")
@click.option("--min_bs", type=int, default=1, help="minimum batch size")
@click.option("--opt_bs", type=int, default=None, help="optimal batch size")
@click.option("--opset_version", type=int, default=17, help="onnx opset version")
@click.option("--tllm_checkpoint_dir", default="tllm_checkpoint", type=str)
@click.option("--engine_dir", default="engines", type=str)
def convert_lt_to_trt(model_ckpt, audio_codec, hparams_file,
                           tllm_checkpoint_dir, engine_dir, dtype,
                           max_bs, min_bs, opt_bs, opset_version):
    model, model_cfg = load_model(model_ckpt, hparams_file, audio_codec, engine_dir, legacy_codebooks=False)
    with torch.no_grad():
        lt = MagpieLocalTransformerExportTRT(tllm_checkpoint_dir, engine_dir, model_cfg, dtype=dtype,
                                        maxBS=max_bs, minBS=min_bs, optBS=opt_bs, opset_version=opset_version)
        lt.export_lt_to_onnx(model)
        lt.generate_trt_engine()


if __name__ == "__main__":
    convert_lt_to_trt()
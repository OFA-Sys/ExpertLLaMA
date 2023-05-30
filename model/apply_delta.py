"""
Apply the delta weights on top of a base model.

code adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py

Usage:
python3 apply_delta.py --base-model-path {your_base_model_path} --target-model-path {your_target_model_path} --delta-path {downloaded_delta_weights}
"""
import argparse
import gc
import glob
import json
import os
import shutil
import tempfile

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the delta weights from {delta_path}")
    delta_tokenizer = LlamaTokenizer.from_pretrained(delta_path, use_fast=False)
    delta = LlamaForCausalLM.from_pretrained(
        delta_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    )

    print(f"Loading the base model from {base_model_path}")
    base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)
    base = LlamaForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True
    )

    # following alpaca training recipe, we have added new initialized tokens
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    special_tokens_dict = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
    num_new_tokens = base_tokenizer.add_special_tokens(special_tokens_dict)
    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data

    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Applying the delta")
    target_weights = {}
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]
        target_weights[name] = param.data

    print(f"Saving the target model to {target_model_path}")
    base.load_state_dict(target_weights)
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
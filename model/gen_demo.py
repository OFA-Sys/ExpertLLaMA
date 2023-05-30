"""
Usage:
python3 gen_demo.py --expertllama_path {your_target_model_path}
"""

import transformers
import os
import json
import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM

def gen(expertllama_path):

    tokenizer = transformers.LlamaTokenizer.from_pretrained(expertllama_path)
    model = transformers.LlamaForCausalLM.from_pretrained(expertllama_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.cuda()

    temp = 0
    max_seq_len = 1024

    while True:
        prompt = input("User >>> ").strip().replace("\\n", "\n")
        if prompt == "exit":
            break

        # prompt wrapper, only single-turn is allowed for now
        prompt = f"### Human:\n{prompt}\n\n### Assistant:\n"

        batch = tokenizer(
            prompt,
            return_tensors="pt", 
            add_special_tokens=False
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        generated = model.generate(batch["input_ids"], max_length=max_seq_len, temperature=temp)
        response = tokenizer.decode(generated[0][:-1]).split("### Assistant:\n", 1)[1]
        print(f"ExpertLLaMA >>>", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertllama_path", type=str, required=True)
    args = parser.parse_args()

    gen(args.expertllama_path)
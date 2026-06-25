import os
import pdb
import torch
import argparse
from LLMPruner.evaluator.ppl import PPLMetric
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):

    print(f"[INFO] Loading model from {args.model_path}...")
    if args.model_path.endswith('.bin'):
        sd = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model = sd['model'].to(torch.bfloat16).to(args.device)
        tokenizer = sd['tokenizer']
    elif os.path.isdir(args.model_path):
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    model.eval()
    # max_seq_len = min(8192, model.config.max_position_embeddings)
    max_seq_len = 128

    print("[INFO] Evaluating PPL...")
    ppl_metric = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], max_seq_len, batch_size=1, device=args.device)
    print("[INFO] PPL after pruning: {}".format(ppl_metric))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='llama-2-7b-hf')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
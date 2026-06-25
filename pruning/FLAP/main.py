import os 
import pdb
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.hf_llama.modeling_llama import LlamaForCausalLM
from importlib.metadata import version
from lib.prune import prune_wanda_sp, prune_flap, prune_magnitude_sp, check_sparsity
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    
    for i in range(model.config.num_hidden_layers):
        # Initialize o_proj bias
        if model.model.layers[i].self_attn.o_proj.bias is None:
            out_features = model.model.layers[i].self_attn.o_proj.out_features
            model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        
        # Initialize down_proj bias
        if model.model.layers[i].mlp.down_proj.bias is None:
            out_features = model.model.layers[i].mlp.down_proj.out_features
            model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias) 
        
    # model.seqlen = min(model.config.max_position_embeddings, 8192)
    model.seqlen = 128
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=2048, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    
    # pdb.set_trace()
    
    if args.eval:
        ppl = eval_ppl(model, tokenizer, device)    
        print(f"ppl on wikitext {ppl}")

    # Prune the model
    print("pruning starts")
    if args.prune_method == "flap":
        if args.metrics == 'N/A':
            raise ValueError("For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice.")  
        if args.structure == 'N/A':
            raise ValueError("For FLAP pruning, the compressed model structure parameter must be chosen from ['UL-UM', 'UL-MM', 'AL-MM', 'AL-AM']. 'N/A' is not a valid choice.")  
        prune_flap(args, model, tokenizer, device)
    elif args.prune_method == "wanda_sp":
        prune_wanda_sp(args, model, tokenizer, device)
    elif args.prune_method == "mag_sp":
        prune_magnitude_sp(args, model, tokenizer, device)

    # Check the sparsity of the model
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*"*30)
    
    # Evaluate the model
    if args.eval:
        ppl = eval_ppl(model, tokenizer, device)    
        print(f"ppl on wikitext {ppl}")
        
    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        
        # Mark that model has been modified with bias for proper loading
        model.config.attn_bias = True
        model.config.mlp_bias = True
        
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        
        # Save a note about the modifications
        with open(os.path.join(args.save_model, 'README.md'), 'w') as f:
            f.write("# Pruned Model with Bias Compensation\n\n")
            f.write("This model has been pruned using FLAP with bias compensation.\n")
            f.write("The o_proj and down_proj layers have bias parameters added.\n")
            f.write(f"Sparsity ratio: {sparsity_ratio:.4f}\n")
            f.write(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B\n")
    

if __name__ == '__main__':
    main()
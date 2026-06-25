import pdb
import glob
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/scratch/yx3038/pruning/llm-pruning-collection-fork-flap/pruning/FLAP/llm_weights/flap_p0.5_WIFV_AL-AM_llama3.1_8b"

sd = {}
for fpath in glob.glob(f"{model_path}/*.safetensors"):
    print(f"loading {fpath}")
    sd.update(load_file(fpath))

tot_params = 0
for key, module in sd.items():
    print(f"key {key} module {module.shape}")
    tot_params += module.numel()
    
print(f"total params: {tot_params:.2e}")


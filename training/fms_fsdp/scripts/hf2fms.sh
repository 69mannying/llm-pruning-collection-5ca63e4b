#!/bin/bash

set -euo pipefail

PROJ_DIR=$(pwd)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms_fsdp

ckpt_dir=$PROJ_DIR/../../checkpoints
model_variant='llama3_8b'
# hf_path="meta-llama/Llama-3.1-8B"
hf_path="/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B"
save_dir="${ckpt_dir}/fms"
mkdir -p ${save_dir}

python training/hf2fms.py \
    --model_variant ${model_variant} \
    --hf_path ${hf_path} \
    --save_path ${save_dir}/${model_variant}_fms.pth
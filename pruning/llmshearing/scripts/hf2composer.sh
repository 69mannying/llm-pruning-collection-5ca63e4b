#!/bin/bash

PROJ_DIR=$(pwd)
# HF_MODEL_NAME=meta-llama/Llama-2-7b-hf
# HF_MODEL_NAME=/n/fs/vision-mix/yx1168/model_ckpts/Llama-2-7b-hf
# HF_MODEL_NAME=/scratch/yx3038/cache/Llama-2-7b-hf
# OUTPUT_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-2-7b-composer/state_dict.pt
# HF_MODEL_NAME=/scratch/yx3038/cache/Llama-3.1-8B
HF_MODEL_NAME=~/yufeng/model_ckpts/Llama-3.1-8B
OUTPUT_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-3.1-8B-composer/state_dict.pt

export PYTHONPATH=$(pwd):$PYTHONPATH

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH
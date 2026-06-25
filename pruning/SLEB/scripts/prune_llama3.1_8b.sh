#!/bin/bash
#SBATCH --job-name=prune_llama3.1_8b
#SBATCH --output=logs/prune_llama3.1_8b_%j.out
#SBATCH --error=logs/prune_llama3.1_8b_%j.err
#SBATCH --partition=sfscai
#SBATCH --nodes=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sleb

# Set common variables
model_path="meta-llama/Llama-3.1-8B"
# model_path="/scratch/yx3038/cache/Llama-3.1-8B"

# Define function to run python command
run_python_command () {
    num_remove_blocks=${1:-16}
    python -m sleb \
        --model_name $model_path \
        --num_blocks 32 \
        --num_remove_blocks $num_remove_blocks \
        --eval_ppl True \
        --eval_zeroshot True
}

run_python_command 2
run_python_command 4
run_python_command 8
run_python_command 16
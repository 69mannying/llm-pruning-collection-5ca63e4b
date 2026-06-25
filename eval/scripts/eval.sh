#!/bin/bash

#SBATCH --job-name=eval_%j
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=sfscai
#SBATCH --nodes=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h20:1
#SBATCH --time=12:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue

source ~/.bashrc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate minitron

# python -c "from datasets import load_dataset; load_dataset('allenai/c4', 'en')"

# hf_model_path=meta-llama/Llama-3.1-8B
hf_model_path=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing-2/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_4_mlp_10112_sl8192_bs1_3200ba/hf-pruned
tasks=(
    # c4
    # wikitext
    # wikitext2
    # cnn_dailymail
    winogrande
    arc_easy
    arc_challenge
    hellaswag
    # truthfulqa_mc1
    # truthfulqa_mc2
    piqa
    # sciq
    boolq
    # anli_r1
    # anli_r2
    # anli_r3
    # openbookqa
    # rte
    # mmlu
    # record
)

PROJ_DIR=$(pwd)
export PYTHONPATH=$PROJ_DIR/lm-evaluation-harness:$PYTHONPATH

tasks_str=$(IFS=, ; echo "${tasks[*]}")
# echo "Evaluating tasks: $tasks_str"

python eval.py \
    --hf_path $hf_model_path \
    --tokenizer_path $hf_model_path \
    --tasks $tasks_str

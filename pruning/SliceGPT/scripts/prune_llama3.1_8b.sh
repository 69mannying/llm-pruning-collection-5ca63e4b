#!/bin/bash

# model_path=meta-llama/Llama-3.1-8B
model_path=/scratch/yx3038/cache/Llama-3.1-8B

run_experiment() {
    sparsity=${1:-0.25}
    model_name=$(basename $model_path)
    exp_name=$(echo ${model_name}-sp-${sparsity} | tr '[A-Z]' '[a-z]')
    save_dir=../../checkpoints/slicegpt/${exp_name}
    mkdir -p $save_dir
    python experiments/run_slicegpt.py \
        --model ${model_path} \
        --save-dir ${exp_name} \
        --sparsity ${sparsity} \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb
}

run_experiment 0.1
run_experiment 0.2
run_experiment 0.25
run_experiment 0.3
run_experiment 0.5
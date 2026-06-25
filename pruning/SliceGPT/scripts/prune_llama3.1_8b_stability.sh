#!/bin/bash

set -euo pipefail

# Override with env var if needed.
# Example: MODEL_PATH=meta-llama/Llama-3.1-8B bash scripts/prune_llama3.1_8b_stability.sh
model_path="${MODEL_PATH:-/scratch/yx3038/cache/Llama-3.1-8B}"

# Default sparsity sweep. You can override by passing values as args.
# Example: bash scripts/prune_llama3.1_8b_stability.sh 0.1 0.2 0.25
if [ "$#" -gt 0 ]; then
    sparsities=("$@")
else
    sparsities=(0.1 0.2 0.25 0.3 0.5)
fi

run_experiment() {
    sparsity="$1"
    model_name="$(basename "$model_path")"
    exp_name="$(echo "${model_name}-stable-sp-${sparsity}" | tr '[A-Z]' '[a-z]')"
    save_dir="../../checkpoints/slicegpt/${exp_name}"
    mkdir -p "$save_dir"

    python experiments/run_slicegpt.py \
        --model "${model_path}" \
        --save-dir "${save_dir}" \
        --sparsity "${sparsity}" \
        --device cuda:0 \
        --eval-baseline \
        --final-orientation pca \
        --dtype fp32 \
        --cal-dataset c4 \
        --cal-nsamples 512 \
        --cal-batch-size 8 \
        --no-wandb
}

for sp in "${sparsities[@]}"; do
    run_experiment "$sp"
done

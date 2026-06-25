#!/bin/bash
#
# Smoke test: Llama3.1 8B -> 3B GQA pruning (10 batches only)
# Run from: pruning/llmshearing/
#   bash scripts/test_prune_llama3.sh
#
# Adjust MODEL_PATH and DATA_DIR before running.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llmshearing

PROJ_DIR=$(pwd)
DATA_DIR=${PROJ_DIR}/llmshearing/data/redpajama/for_prune_llama3
OUTPUT_DIR=${PROJ_DIR}/../../checkpoints/llmshearing
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-3.1-8B-composer

from_model=8b
to_model=3b
config_file=${PROJ_DIR}/llmshearing/configs/llama3/${from_model}.yaml
path=${MODEL_PATH}/state_dict.pt

num_gpus=${SLURM_GPUS_ON_NODE:-1}
max_seq_len=8192
device_train_microbatch_size=1
global_train_batch_size=2   # small for smoke test
device_eval_batch_size=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lr=1e-4
max_duration=10ba           # only 10 batches — change for real runs
save_interval=10ba
t_warmup=2ba

dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp]
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15]
update_type=doremi
target_loss=[1.84,0.67,2.00,1.50,1.60,1.33,2.00]
eval_split_name=eval_merge
eval_target_model=false
eval_interval=10ba

lag_lr=1.0
lagr_warmup=4ba

# target: 3B GQA (n_heads=20, n_kv_heads=5)
target_d_model=2560
target_n_heads=20
target_n_kv_heads=5
target_n_layers=32
target_intermediate_size=8960

run_name=llama3_${from_model}_SMOKETEST_to${to_model}_${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir}

SCRIPT_ARGS=(
    "$TRAIN_SCRIPT"
    "$config_file"
    run_name=${run_name}
    data_local=${DATA_DIR}
    eval_loader.dataset.split=${eval_split_name}
    global_train_batch_size=${global_train_batch_size}
    device_train_microbatch_size=${device_train_microbatch_size}
    device_eval_batch_size=${device_eval_batch_size}
    max_seq_len=${max_seq_len}
    max_duration=${max_duration}
    eval_first=false
    scheduler.t_warmup=${t_warmup}
    save_folder=${save_dir}
    loggers.wandb.init_kwargs.dir=${wandb_dir}
    eval_interval=${eval_interval}
    save_interval=${save_interval}
    optimizer.lr=${lr}
    optimizer.lag_lr=${lag_lr}
    model.path=${path}
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup}
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]'
    model.l0_module.eval_target_model=${eval_target_model}
    model.l0_module.target_model.d_model=${target_d_model}
    model.l0_module.target_model.n_heads=${target_n_heads}
    model.l0_module.target_model.n_kv_heads=${target_n_kv_heads}
    model.l0_module.target_model.n_layers=${target_n_layers}
    model.l0_module.target_model.intermediate_size=${target_intermediate_size}
    callbacks.data_loading.dynamic=${dynamic}
    callbacks.data_loading.set_names=${set_names}
    callbacks.data_loading.proportion=${proportion}
    callbacks.data_loading.update_type=${update_type}
    callbacks.data_loading.target_loss=${target_loss}
    train_loader.num_workers=0
    train_loader.prefetch_factor=null
    train_loader.persistent_workers=false
    autoresume=true
)

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
export MASTER_PORT

echo "=== Smoke test: ${max_duration}, $(nvidia-smi -L | wc -l) GPU(s) ==="
torchrun --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} "${SCRIPT_ARGS[@]}"
echo "=== Smoke test PASSED ==="

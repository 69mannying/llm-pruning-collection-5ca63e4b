#!/bin/bash

#SBATCH --job-name=prune_llama3_3b_%j
#SBATCH --output=logs/prune_llama3_3b_%j.out
#SBATCH --error=logs/prune_llama3_3b_%j.err
#SBATCH --partition=sfscai
#SBATCH --nodes=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h20:2
#SBATCH --time=24:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue


# pruning llama3.1 8b -> target architecture

set -euo pipefail
# set +x

# source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llmshearing

# Please specify the working folder
PROJ_DIR=$(pwd)
DATA_DIR=${PROJ_DIR}/llmshearing/data/redpajama/for_prune_llama3
OUTPUT_DIR=${PROJ_DIR}/../../checkpoints/llmshearing
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-3.1-8B-composer

# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

test=False

from_model=8b # source model size
to_model=3b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/llama3/${from_model}.yaml
path=$MODEL_PATH/state_dict.pt

# data setup
data_local=${DATA_DIR}

# basic setup
num_gpus=${SLURM_GPUS_ON_NODE:-1}
max_seq_len=8192
device_train_microbatch_size=1
global_train_batch_size=16
device_eval_batch_size=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=3200ba # 0.42B tokens
save_interval=3200ba # save in the end
t_warmup=320ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $to_model == 3b ]]; then
    target_loss=[1.84,0.67,2.00,1.50,1.60,1.33,2.00] # 3b predicted loss from scaling law (placeholder)
fi
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=800ba # eval every 50 batches and update the loading proportion


# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=640ba # 20% sparsity warmup
if [[ $to_model == 3b ]]; then
    target_d_model=2560; target_n_heads=20; target_n_kv_heads=5; target_n_layers=32; target_intermediate_size=8960
fi

# save directroy
TIME=$(date +%Y%m%d_%H%M%S)
run_name=llama3_${from_model}_pruning_scaling_${update_type}_to${to_model}_sl${max_seq_len}_bs${device_train_microbatch_size}_${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

num_nodes=${SLURM_JOB_NUM_NODES}
echo "SLURM_JOB_NUM_NODES: $num_nodes"
if [[ $num_nodes -gt 1 ]]; then
    node_rank=${SLURM_NODEID}     
    num_gpus=$(nvidia-smi -L | wc -l)
    master_addr=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address | awk '{print $1}')
    export MASTER_ADDR=${head_node_ip}

    echo "SLURM_NODEID (node rank): $node_rank"
    echo "GPUs on this node: $num_gpus"
    echo "Master address: $master_addr"
    echo "Head node ip: $head_node_ip"
fi

# export WANDB_MODE=offline

SCRIPT_ARGS=(
    "$TRAIN_SCRIPT"
    "$config_file"
    run_name=${run_name} 
    data_local=${data_local} 
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

get_random_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"
}

run_experiment() {
    export MASTER_PORT=$(get_random_port)

    echo $num_nodes
    if [[ $num_nodes -gt 1 ]]; then
        srun torchrun \
            --nnodes=${num_nodes} \
            --nproc_per_node=${num_gpus} \
            --rdzv_id=${RANDOM} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
            "${SCRIPT_ARGS[@]}"
    else
        echo "GPUs on this node: $num_gpus"
        export NODE_RANK=0
        torchrun --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} "${SCRIPT_ARGS[@]}"
    fi

}

# run experiments with retries
for i in {1..3}; do
    echo "Attempt $i to run the experiment..."
    if run_experiment; then
        echo "Experiment completed successfully!"
        break
    else
        echo "Experiment failed on attempt $i. Retrying..."
    fi
done
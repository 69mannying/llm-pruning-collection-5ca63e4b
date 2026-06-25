#!/bin/bash

#SBATCH --job-name=prune_llama3_d_3072_kv_6_mlp_9984_%j
#SBATCH --output=logs/prune_llama3_d_3072_kv_6_mlp_9984/%j.out
#SBATCH --error=logs/prune_llama3_d_3072_kv_6_mlp_9984/%j.err
#SBATCH --partition=sfscai
#SBATCH --nodes=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --gres=gpu:h20:4
#SBATCH --time=48:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue


set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llmshearing

PROJ_DIR=$(pwd)
DATA_DIR=${PROJ_DIR}/llmshearing/data/redpajama/for_prune_llama3
OUTPUT_DIR=${PROJ_DIR}/../../checkpoints/llmshearing
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-3.1-8B-composer

test=False

from_model=8b
config_file=${PROJ_DIR}/llmshearing/configs/llama3/${from_model}.yaml
path=$MODEL_PATH/state_dict.pt

data_local=${DATA_DIR}

# num_gpus=${SLURM_GPUS_ON_NODE:-1}

num_gpus=${NUM_GPUS:-4}
source scripts/utils.sh
export CUDA_VISIBLE_DEVICES=$(get_free_gpus $num_gpus)
echo "Using gpus $CUDA_VISIBLE_DEVICES"

max_seq_len=4096
device_train_microbatch_size=1
global_train_batch_size=32
device_eval_batch_size=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL stability flags to avoid intermittent connection errors
# export NCCL_SOCKET_TIMEOUT=600000
# export NCCL_SOCKET_NRETRY=10
# export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=^lo,docker0
# export NCCL_DEBUG=INFO                                                                                                                                                       
# export NCCL_DEBUG_SUBSYS=ALL        
export NCCL_IB_DISABLE=1      

lr=1e-4
max_duration=3200ba
save_interval=400ba
t_warmup=320ba

dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp]
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15]
update_type=doremi
target_loss=[2.2417,0.8230,2.2413,1.8674,1.6774,1.2733,2.3723]
eval_split_name=eval_merge
eval_target_model=false
eval_interval=800ba

lag_lr=1.0
lagr_warmup=640ba

target_d_model=3072; target_n_heads=24; target_n_kv_heads=6; target_n_layers=32; target_intermediate_size=9984

TIME=$(date +%Y%m%d_%H%M%S)
run_name=llama3_8b_pruning_scaling_doremi_h_${target_d_model}_kv_${target_n_kv_heads}_mlp_${target_intermediate_size}_sl4096_bs1_3200ba
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir}

# num_nodes=${SLURM_JOB_NUM_NODES}
num_nodes=1
echo "SLURM_JOB_NUM_NODES: $num_nodes"
if [[ $num_nodes -gt 1 ]]; then
    node_rank=${SLURM_NODEID}
    num_gpus=$(nvidia-smi -L | wc -l)
    master_addr=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address | awk '{print $1}')
    export MASTER_ADDR=${head_node_ip}
fi

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
    model.l0_module.target_model.d_model=3072
    model.l0_module.target_model.n_heads=24
    model.l0_module.target_model.n_kv_heads=6
    model.l0_module.target_model.n_layers=32
    model.l0_module.target_model.intermediate_size=9984
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
    if [[ $num_nodes -gt 1 ]]; then
        srun torchrun \
            --nnodes=${num_nodes} \
            --nproc_per_node=${num_gpus} \
            --rdzv_id=${RANDOM} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
            "${SCRIPT_ARGS[@]}"
    else
        export NODE_RANK=0
        torchrun --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} "${SCRIPT_ARGS[@]}"
    fi
}

for i in {1..1}; do
    echo "Attempt $i to run the experiment..."
    if run_experiment; then
        echo "Experiment completed successfully!"
        break
    else
        echo "Experiment failed on attempt $i. Retrying..."
    fi
done

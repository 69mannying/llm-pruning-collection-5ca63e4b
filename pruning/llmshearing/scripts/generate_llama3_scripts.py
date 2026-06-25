from __future__ import annotations

import os
import re
import pdb
import subprocess
from pathlib import Path
from typing import Sequence, Union


def make_llama3_prune_slurm_script(
    *,
    d_model: int,
    n_kv_heads: int,
    mlp_size: int,
    # paths to your helper scripts
    count_params_py: Union[str, Path],
    scaling_law_py: Union[str, Path],
    # fixed knobs (edit defaults if you want)
    n_layers: int = 32,
    from_model: str = "8b",
    max_seq_len: int = 4096,
    device_train_microbatch_size: int = 1,
    global_train_batch_size: int = 32,
    device_eval_batch_size: int = 1,
    lr: str = "1e-4",
    max_duration: str = "3200ba",
    save_interval: str = "400ba",
    t_warmup: str = "320ba",
    eval_interval: str = "800ba",
    update_type: str = "doremi",
    lag_lr: str = "1.0",
    lagr_warmup: str = "640ba",
    set_names: Sequence[str] = ("cc", "github", "book", "stackexchange", "wiki", "arxiv", "c4-rp"),
    proportion: Sequence[float] = (0.67, 0.045, 0.045, 0.02, 0.045, 0.025, 0.15),
) -> Path:
    """
    Generate ONE slurm bash script (in your exact format) for a target architecture.

    Assumptions:
      - n_heads = 4 * n_kv_heads
      - count_params.py prints a single number (params or B params) to stdout given dims
      - scaling_law.py prints a list of losses (JSON list or python list) in the same order as set_names

    You may need to tweak the CLI flags in the two subprocess calls to match your helper scripts.
    """

    def run(cmd: Sequence[str]) -> str:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        return p.stdout.strip()

    def parse_number(s: str) -> float:
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if not m:
            raise ValueError(f"Cannot parse number from:\n{s}")
        return float(m.group(0))

    def parse_list(s: str) -> list[float]:
        # try JSON / python literal, otherwise regex floats
        try:
            import json
            obj = json.loads(s)
            if isinstance(obj, list):
                return [float(x) for x in obj]
        except Exception:
            pass
        try:
            obj = eval(s, {"__builtins__": {}}, {})
            if isinstance(obj, list):
                return [float(x) for x in obj]
        except Exception:
            pass
        toks = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if not toks:
            raise ValueError(f"Cannot parse float list from:\n{s}")
        return [float(x) for x in toks]

    n_heads = 4 * n_kv_heads
    # if d_model % n_heads != 0:
    #     raise ValueError(f"Invalid: d_model={d_model} must be divisible by n_heads={n_heads} (4*n_kv_heads).")

    # ---- call count_params.py ----
    # Adjust flags here if your script uses different argument names.
    cmd = [
        "python", str(count_params_py),
        "--d_model", str(d_model),
        "--num_heads", str(n_heads),
        "--num_kv_heads", str(n_kv_heads),
        "--num_layers", str(n_layers),
        "--intermediate_size", str(mlp_size), 
        "--vocab_size", str(128256),  # Llama3 vocab size
        "--head_dim", str(128),  # ensure head_dim is consistent with d_model and n_heads
    ]
    model_size_out = run(cmd)
    
    target_model_size = int(parse_number(model_size_out))

    # ---- call scaling_law.py ----
    # Adjust flags here if your script uses different argument names.
    cmd = [
        "python", "-W", "ignore",str(scaling_law_py),
        "--target_model_size", str(target_model_size),
    ]
    # pdb.set_trace()
    losses_out = run(cmd)
    # target_loss = parse_list(losses_out)
    # if len(target_loss) != len(set_names):
    #     raise ValueError(f"scaling_law.py returned {len(target_loss)} losses, expected {len(set_names)}.")

    # bash literals
    set_names_bash = "[" + ",".join(set_names) + "]"
    proportion_bash = "[" + ",".join(f"{p:.6g}" for p in proportion) + "]"
    # target_loss_bash = "[" + ",".join(f"{x:.6g}" for x in target_loss) + "]"
    target_loss_bash = losses_out

    out_path = f"scripts/prune_llama3_d_{d_model}_kv_{n_kv_heads}_mlp_{mlp_size}.sh"
    
    os.makedirs(f"logs/prune_llama3_d_{d_model}_kv_{n_kv_heads}_mlp_{mlp_size}", exist_ok=True)

    script = f"""#!/bin/bash

#SBATCH --job-name=prune_llama3_d_{d_model}_kv_{n_kv_heads}_mlp_{mlp_size}_%j
#SBATCH --output=logs/prune_llama3_d_{d_model}_kv_{n_kv_heads}_mlp_{mlp_size}/%j.out
#SBATCH --error=logs/prune_llama3_d_{d_model}_kv_{n_kv_heads}_mlp_{mlp_size}/%j.err
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
DATA_DIR=${{PROJ_DIR}}/llmshearing/data/redpajama/for_prune_llama3
OUTPUT_DIR=${{PROJ_DIR}}/../../checkpoints/llmshearing
TRAIN_SCRIPT=${{PROJ_DIR}}/llmshearing/train.py
MODEL_PATH=${{PROJ_DIR}}/../../checkpoints/llmshearing/Llama-3.1-8B-composer

test=False

from_model={from_model}
config_file=${{PROJ_DIR}}/llmshearing/configs/llama3/${{from_model}}.yaml
path=$MODEL_PATH/state_dict.pt

data_local=${{DATA_DIR}}

# num_gpus=${{SLURM_GPUS_ON_NODE:-1}}

num_gpus=${{NUM_GPUS:-4}}
source scripts/utils.sh
export CUDA_VISIBLE_DEVICES=$(get_free_gpus $num_gpus)
echo "Using gpus $CUDA_VISIBLE_DEVICES"

max_seq_len={max_seq_len}
device_train_microbatch_size={device_train_microbatch_size}
global_train_batch_size={global_train_batch_size}
device_eval_batch_size={device_eval_batch_size}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL stability flags to avoid intermittent connection errors
# export NCCL_SOCKET_TIMEOUT=600000
# export NCCL_SOCKET_NRETRY=10
# export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=^lo,docker0
# export NCCL_DEBUG=INFO                                                                                                                                                       
# export NCCL_DEBUG_SUBSYS=ALL        
export NCCL_IB_DISABLE=1      
# export NCCL_SOCKET_IFNAME=lo  

lr={lr}
max_duration={max_duration}
save_interval={save_interval}
t_warmup={t_warmup}

dynamic=True
set_names={set_names_bash}
proportion={proportion_bash}
update_type={update_type}
target_loss={target_loss_bash}
eval_split_name=eval_merge
eval_target_model=false
eval_interval={eval_interval}

lag_lr={lag_lr}
lagr_warmup={lagr_warmup}

target_d_model={d_model}; target_n_heads={n_heads}; target_n_kv_heads={n_kv_heads}; target_n_layers={n_layers}; target_intermediate_size={mlp_size}

TIME=$(date +%Y%m%d_%H%M%S)
run_name=llama3_{from_model}_pruning_scaling_{update_type}_h_${{target_d_model}}_kv_${{target_n_kv_heads}}_mlp_${{target_intermediate_size}}_sl{max_seq_len}_bs{device_train_microbatch_size}_{max_duration}
save_dir=${{OUTPUT_DIR}}/${{run_name}}
wandb_dir=${{save_dir}}

# num_nodes=${{SLURM_JOB_NUM_NODES}}
num_nodes=1
echo "SLURM_JOB_NUM_NODES: $num_nodes"
if [[ $num_nodes -gt 1 ]]; then
    node_rank=${{SLURM_NODEID}}
    num_gpus=$(nvidia-smi -L | wc -l)
    master_addr=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address | awk '{{print $1}}')
    export MASTER_ADDR=${{head_node_ip}}
fi

SCRIPT_ARGS=(
    "$TRAIN_SCRIPT"
    "$config_file"
    run_name=${{run_name}}
    data_local=${{data_local}}
    eval_loader.dataset.split=${{eval_split_name}}
    global_train_batch_size=${{global_train_batch_size}}
    device_train_microbatch_size=${{device_train_microbatch_size}}
    device_eval_batch_size=${{device_eval_batch_size}}
    max_seq_len=${{max_seq_len}}
    max_duration=${{max_duration}}
    eval_first=false
    scheduler.t_warmup=${{t_warmup}}
    save_folder=${{save_dir}}
    loggers.wandb.init_kwargs.dir=${{wandb_dir}}
    eval_interval=${{eval_interval}}
    save_interval=${{save_interval}}
    optimizer.lr=${{lr}}
    optimizer.lag_lr=${{lag_lr}}
    model.path=${{path}}
    model.l0_module.lagrangian_warmup_steps=${{lagr_warmup}}
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]'
    model.l0_module.eval_target_model=${{eval_target_model}}
    model.l0_module.target_model.d_model={d_model}
    model.l0_module.target_model.n_heads={n_heads}
    model.l0_module.target_model.n_kv_heads={n_kv_heads}
    model.l0_module.target_model.n_layers={n_layers}
    model.l0_module.target_model.intermediate_size={mlp_size}
    callbacks.data_loading.dynamic=${{dynamic}}
    callbacks.data_loading.set_names=${{set_names}}
    callbacks.data_loading.proportion=${{proportion}}
    callbacks.data_loading.update_type=${{update_type}}
    callbacks.data_loading.target_loss=${{target_loss}}
    train_loader.num_workers=0
    train_loader.prefetch_factor=null
    train_loader.persistent_workers=false
    autoresume=true
)

get_random_port() {{
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"
}}

run_experiment() {{
    export MASTER_PORT=$(get_random_port)
    if [[ $num_nodes -gt 1 ]]; then
        srun torchrun \\
            --nnodes=${{num_nodes}} \\
            --nproc_per_node=${{num_gpus}} \\
            --rdzv_id=${{RANDOM}} \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint=${{head_node_ip}}:${{MASTER_PORT}} \\
            "${{SCRIPT_ARGS[@]}}"
    else
        export NODE_RANK=0
        torchrun --nproc_per_node=${{num_gpus}} --master_port=${{MASTER_PORT}} "${{SCRIPT_ARGS[@]}}"
    fi
}}

for i in {{1..3}}; do
    echo "Attempt $i to run the experiment..."
    if run_experiment; then
        echo "Experiment completed successfully!"
        break
    else
        echo "Experiment failed on attempt $i. Retrying..."
        sleep 10
    fi
done
"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    out_path.chmod(0o755)
    return out_path



if __name__ == "__main__":

    for (d_model, n_kv_heads, mlp_size) in [
        (3072, 4, 10112),
        (3072, 5, 9984),
        (3072, 6, 9984),
        (3072, 7, 9856),
        (3072, 8, 9728),
        
        (3456, 4, 8192),
	    (3456, 5, 8064),
	    (3456, 6, 8064),
	    (3456, 7, 7936),
	    (3456, 8, 7808)
    ]:
        # print(count_params(
        #     d_model=d_model,
        #     num_heads=4*n_kv_heads,
        #     num_layers=32,
        #     vocab_size=128256,
        #     intermediate_size=mlp_size,
        #     num_kv_heads=n_kv_heads,
        #     head_dim=128,
        # ))
        script_path = make_llama3_prune_slurm_script(
            d_model=d_model,
            n_kv_heads=n_kv_heads,
            mlp_size=mlp_size,
            count_params_py="count_params.py",
            scaling_law_py="scaling_law.py",
        )
        print(f"Generated slurm script at: {script_path}")
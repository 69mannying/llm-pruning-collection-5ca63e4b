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
conda activate flap

# Set common variables
model="meta-llama/Llama-3.1-8B"
# model="/scratch/yx3038/cache/Llama-3.1-8B"

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --pruning_ratio $2 \
    --remove_heads $3 \
    --metrics $4 \
    --structure $5 \
    --nsamples 128 \
    --save_model "llm_weights/${1}_p${2}_${4}_${5}_llama3.1_8b/" \
    --eval 
}
# calib samples should be 1024

# llama3.1-8b with flap pruning method (p=0.2/0.3/0.5, adaptive in all layers)
echo "Running with flap pruning method"
run_python_command "flap" 0.2 -1 "WIFV" "AL-AM" 
run_python_command "flap" 0.3 -1 "WIFV" "AL-AM" 
run_python_command "flap" 0.5 -1 "WIFV" "AL-AM" 

# llama3.1-8b with flap pruning method (p=0.2/0.3/0.5, uniform in all layers)
echo "Running with flap pruning method"
# run_python_command "flap" 0.2 -1 "WIFV" "UL-UM"
# run_python_command "flap" 0.3 -1 "WIFV" "UL-UM"
# run_python_command "flap" 0.5 -1 "WIFV" "UL-UM"

# # llama-7b with wanda-sp pruning method (p=0.2/0.3/0.5, uniform in all layers)
# echo "Running with wanda-sp pruning method"
run_python_command "wanda_sp" 0.2 -1 N/A N/A
run_python_command "wanda_sp" 0.3 -1 N/A N/A 
run_python_command "wanda_sp" 0.5 -1 N/A N/A 

# # llama-7b with mag-sp pruning method (p=0.2/0.3/0.5, uniform in all layers)
# echo "Running with magnitude pruning method"
# run_python_command "mag_sp" 0.2 -1 N/A N/A
# run_python_command "mag_sp" 0.3 -1 N/A N/A 
# run_python_command "mag_sp" 0.5 -1 N/A N/A

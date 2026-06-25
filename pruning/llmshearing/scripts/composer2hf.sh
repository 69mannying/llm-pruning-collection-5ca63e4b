#!/bin/bash
set -euo pipefail

# MODEL_DIR=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_5_mlp_9984_sl8192_bs1_3200ba
MODEL_DIR=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing-2/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_4_mlp_10112_sl8192_bs1_3200ba

convert_hf() {
    local MODEL_DIR="$1"
    local MODEL_PATH="$MODEL_DIR/latest-rank0.pt"
    local OUTPUT_PATH="$MODEL_DIR/hf-pruned"
    local MODEL_CLASS="LlamaForCausalLM"
    local MODEL_NAME="Sheared-Llama3"
    local TOKENIZER_NAME="/scratch/yx3038/cache/Llama-3.1-8B"

    local DIR_BASENAME
    DIR_BASENAME="$(basename "$MODEL_DIR")"

    local HIDDEN_SIZE=""
    local NUM_KEY_VALUE_HEADS=""
    local INTERMEDIATE_SIZE=""

    if [[ "$DIR_BASENAME" =~ (^|_)h_([0-9]+)($|_) ]]; then
        HIDDEN_SIZE="${BASH_REMATCH[2]}"
    fi
    if [[ "$DIR_BASENAME" =~ (^|_)kv_([0-9]+)($|_) ]]; then
        NUM_KEY_VALUE_HEADS="${BASH_REMATCH[2]}"
    fi
    if [[ "$DIR_BASENAME" =~ (^|_)mlp_([0-9]+)($|_) ]]; then
        INTERMEDIATE_SIZE="${BASH_REMATCH[2]}"
    fi

    if [[ -z "$HIDDEN_SIZE" || -z "$NUM_KEY_VALUE_HEADS" || -z "$INTERMEDIATE_SIZE" ]]; then
        echo "Failed to parse one or more of hidden_size / kv_heads / mlp_size from dir name: $DIR_BASENAME" >&2
        exit 1
    fi

    local NUM_ATTENTION_HEADS
    NUM_ATTENTION_HEADS=$((NUM_KEY_VALUE_HEADS * 4))

    read NUM_HIDDEN_LAYERS VOCAB_SIZE <<< "$(python3 - <<PY
import torch

w = torch.load("$MODEL_PATH", map_location="cpu", mmap=True)
if "state" in w:
    w = w["state"]["model"]

    python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
            model_class=${MODEL_CLASS} \
            hidden_size=${HIDDEN_SIZE} \
            num_attention_heads=${NUM_ATTENTION_HEADS} \
            num_hidden_layers=${NUM_HIDDEN_LAYERS} \
            intermediate_size=${INTERMEDIATE_SIZE} \
            num_key_value_heads=${NUM_KEY_VALUE_HEADS} \
            vocab_size=${VOCAB_SIZE} \
            tokenizer_name=${TOKENIZER_NAME} \
            rope_theta=500000 \
            _name_or_path=${MODEL_NAME}
}

<<<<<<< HEAD
echo "Converting model in $MODEL_DIR"
convert_hf "$MODEL_DIR"
=======
# for model_dir in /scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b*; do
#     [ -d "$model_dir" ] || continue
#     echo "Converting model in $model_dir"
#     convert_hf "$model_dir"
# done

convert_hf $MODEL_DIR
>>>>>>> 9daf96e2972771f23a46e5703d8f4b574b4975e3

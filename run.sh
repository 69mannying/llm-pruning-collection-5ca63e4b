#!/bin/bash
# Minimal end-to-end reproduction of the granularity-ordering mechanism from
# arXiv 2606.14150 ("Small LLMs: Pruning vs Training from Scratch").
#
# One-shot prune a pretrained LLM at a fixed ratio with a chosen method/granularity,
# measure wikitext-2 perplexity before vs after pruning, optionally save the pruned
# HF checkpoint. The paper's mechanistic claim: at a fixed ratio, finer / data-aware
# pruning preserves more of the parent model's quality (lower post-prune perplexity).
#
# Experiments vary ONLY prune_config.env; this script + the run command stay fixed.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"
mkdir -p "${REPO_ROOT}/.openresearch/artifacts"

# shellcheck disable=SC1091
set -a            # export every variable defined while sourcing the config
source ./prune_config.env
set +a
TF_VER="${TRANSFORMERS_VERSION:-4.55.0}"

echo "[install] python deps (transformers==${TF_VER})"
pip install --quiet --upgrade pip
pip install --quiet \
    "torch" \
    "transformers==${TF_VER}" \
    "datasets" "accelerate" \
    "sentencepiece" "protobuf" "scikit-learn" "tqdm" "huggingface_hub"

# HF auth for gated models (token provided via env).
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[auth] logging in to Hugging Face with HF_TOKEN"
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || \
        python3 -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

echo "[run] MODEL=${MODEL} METHOD=${PRUNE_METHOD} TYPE=${SPARSITY_TYPE} RATIO=${SPARSITY_RATIO} SEQLEN=${SEQLEN}"
python3 "${REPO_ROOT}/run_prune.py"

echo "[done] EVAL.md:"
cat "${REPO_ROOT}/EVAL.md" || true

#!/usr/bin/env python3
"""
Minimal end-to-end reproduction of the granularity-ordering mechanism from
arXiv 2606.14150 ("Small LLMs: Pruning vs Training from Scratch").

One-shot prune a pretrained LLM at a fixed ratio with a chosen method/granularity,
measure wikitext-2 perplexity before vs after pruning, optionally save the pruned
HF checkpoint, and write an EVAL.md scorecard + artifacts.

This driver is deliberately version- and architecture-robust so it works on both
the original Llama/Qwen2 stack and modern transformers (>=5.x) models whose decoder
layers rely on `position_embeddings` (RoPE computed at the model level) and which may
wrap the language model inside a multimodal `*ForConditionalGeneration` container
(e.g. Gemma-4). It reuses the repo's Wanda (WrappedGPT) and SparseGPT scoring code.

Config comes from environment variables (set by prune_config.env via run.sh):
  MODEL, PRUNE_METHOD, SPARSITY_TYPE, SPARSITY_RATIO, NSAMPLES, SEQLEN, SAVE_MODEL
"""
import os
import sys
import json
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

# Reuse the repo's scoring implementations (Wanda metric + SparseGPT/OBS).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "pruning", "wanda", "src", "lib"))
from layerwrapper import WrappedGPT          # noqa: E402
from sparsegpt import SparseGPT               # noqa: E402

ART_DIR = os.path.join(REPO_ROOT, ".openresearch", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def cfg(name, default):
    return os.environ.get(name, default)


MODEL = cfg("MODEL", "Qwen/Qwen2.5-1.5B")
PRUNE_METHOD = cfg("PRUNE_METHOD", "wanda")           # magnitude | wanda | sparsegpt
SPARSITY_TYPE = cfg("SPARSITY_TYPE", "unstructured")  # unstructured | 2:4 | 4:8
SPARSITY_RATIO = float(cfg("SPARSITY_RATIO", "0.5"))
NSAMPLES = int(cfg("NSAMPLES", "128"))
SEQLEN = int(cfg("SEQLEN", "2048"))
SAVE_MODEL = cfg("SAVE_MODEL", "0") == "1"
SEED = int(cfg("SEED", "0"))

prune_n, prune_m = 0, 0
if SPARSITY_TYPE != "unstructured":
    assert SPARSITY_RATIO == 0.5, "N:M sparsity requires ratio 0.5"
    prune_n, prune_m = map(int, SPARSITY_TYPE.split(":"))


# --------------------------------------------------------------------------- #
# Model / layer discovery (architecture-robust)
# --------------------------------------------------------------------------- #
def find_linear_layers(module, prefix=""):
    """Recursively collect nn.Linear submodules -> {name: module}."""
    res = {}
    for n, child in module.named_children():
        full = f"{prefix}.{n}" if prefix else n
        if isinstance(child, nn.Linear):
            res[full] = child
        else:
            res.update(find_linear_layers(child, full))
    return res


def get_decoder_layers(model):
    """
    Locate the transformer decoder layers (a ModuleList) for any architecture
    by searching named_modules for the longest ModuleList of repeated decoder
    blocks. Works for plain causal LMs (model.model.layers) and multimodal
    wrappers (model.language_model.layers, model.model.language_model.layers, ...).
    Returns (layers_module_list, parent_module).
    """
    best = None  # (num_layers, name, module, parent)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            # heuristic: a decoder stack is the longest ModuleList whose entries
            # contain nn.Linear submodules (i.e. real transformer blocks).
            if any(isinstance(s, nn.Linear) for s in mod[0].modules()):
                if best is None or len(mod) > best[0]:
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    parent = model.get_submodule(parent_name) if parent_name else model
                    best = (len(mod), name, mod, parent)
    if best is None:
        raise RuntimeError(
            "Could not locate decoder layers. Model structure:\n" + str(type(model)))
    print(f"[discover] decoder layers via '{best[1]}' ({best[0]} layers)")
    return best[2], best[3]


# --------------------------------------------------------------------------- #
# Data (wikitext2 ppl test set + c4 calibration), self-contained
# --------------------------------------------------------------------------- #
WIKITEXT_REPO = "Salesforce/wikitext"  # canonical mirror; bare "wikitext" fails on newer datasets


def get_wikitext_test(tokenizer):
    from datasets import load_dataset
    testdata = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")


def get_calibration(tokenizer, nsamples, seqlen, seed):
    from datasets import load_dataset
    try:
        data = load_dataset(
            "allenai/c4", "en",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train", verification_mode="no_checks")
        texts = data
        use_c4 = True
    except Exception as e:
        print(f"[calib] c4 unavailable ({e}); falling back to wikitext2 train")
        data = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split="train")
        enc = tokenizer(" ".join(data["text"]), return_tensors="pt")
        use_c4 = False

    random.seed(seed)
    samples = []
    if use_c4:
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(texts) - 1)
                enc = tokenizer(texts[i]["text"], return_tensors="pt")
                if enc.input_ids.shape[1] > seqlen:
                    break
            j = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            samples.append(enc.input_ids[:, j:j + seqlen])
    else:
        for _ in range(nsamples):
            j = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            samples.append(enc.input_ids[:, j:j + seqlen])
    return samples


# --------------------------------------------------------------------------- #
# Perplexity (wikitext2), self-contained
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_ppl(model, tokenizer, seqlen, device):
    testenc = get_wikitext_test(tokenizer).input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1))
        nlls.append(loss * seqlen)
        if i % 20 == 0:
            print(f"[ppl] sample {i}/{nsamples}")
    return torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()


# --------------------------------------------------------------------------- #
# Calibration capture (version-robust): grab the input + ALL kwargs the first
# decoder layer receives, so we can replay every layer in isolation exactly.
# --------------------------------------------------------------------------- #
def set_use_cache(model, value):
    """Set use_cache on whichever config actually carries it (top-level or
    text_config for multimodal wrappers like Gemma-4). Returns the previous
    value, or None if the attribute isn't present anywhere."""
    cfg = model.config
    target = cfg
    if not hasattr(cfg, "use_cache") and hasattr(cfg, "text_config"):
        target = cfg.text_config
    prev = getattr(target, "use_cache", None)
    try:
        target.use_cache = value
    except Exception:
        pass
    return prev


# --------------------------------------------------------------------------- #
# Pruning methods
# --------------------------------------------------------------------------- #
def prune_magnitude(layers, ratio, prune_n, prune_m):
    for i, layer in enumerate(layers):
        for name, lin in find_linear_layers(layer).items():
            W = lin.weight.data
            metric = torch.abs(W)
            if prune_n != 0:
                mask = torch.zeros_like(W, dtype=torch.bool)
                for c in range(0, metric.shape[1], prune_m):
                    tmp = metric[:, c:c + prune_m].float()
                    idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                    mask.scatter_(1, c + idx, True)
            else:
                thresh = torch.sort(metric.flatten())[0][int(W.numel() * ratio)]
                mask = metric <= thresh
            W[mask] = 0
        print(f"[magnitude] layer {i} done")


@torch.no_grad()
def collect_stats(model, layers, samples, device, make_collector):
    """
    Run full-model forwards over the calibration samples with a forward-hook on
    every nn.Linear inside the decoder layers, accumulating per-layer statistics
    via `make_collector(linear_module) -> obj` whose `.add_batch(inp, out)` is
    called with that layer's real inputs/outputs.

    Running the real forward (rather than replaying layers in isolation) makes
    this correct for heterogeneous architectures — each layer naturally receives
    its own attention type, head_dim, position_embeddings, KV-sharing, etc.
    Returns {layer_idx: {linear_name: collector_obj}}.
    """
    collectors = {}
    handles = []
    for i, layer in enumerate(layers):
        collectors[i] = {}
        for name, lin in find_linear_layers(layer).items():
            obj = make_collector(lin)
            collectors[i][name] = obj

            def hook(o):
                def tmp(_, inp, out):
                    o.add_batch(inp[0].data, out.data)
                return tmp
            handles.append(lin.register_forward_hook(hook(obj)))

    for s in samples:
        model(s.to(device))

    for h in handles:
        h.remove()
    return collectors


@torch.no_grad()
def prune_wanda(model, layers, samples, device, ratio, prune_n, prune_m):
    collectors = collect_stats(model, layers, samples, device,
                               lambda lin: WrappedGPT(lin))
    for i, layer in enumerate(layers):
        subset = find_linear_layers(layer)
        for name, lin in subset.items():
            W = lin.weight.data
            metric = torch.abs(W) * torch.sqrt(
                collectors[i][name].scaler_row.reshape((1, -1)).to(W.device))
            mask = torch.zeros_like(metric, dtype=torch.bool)
            if prune_n != 0:
                for c in range(0, metric.shape[1], prune_m):
                    tmp = metric[:, c:c + prune_m].float()
                    idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                    mask.scatter_(1, c + idx, True)
            else:
                sort_idx = torch.sort(metric, dim=-1, stable=True)[1]
                idx = sort_idx[:, :int(metric.shape[1] * ratio)]
                mask.scatter_(1, idx, True)
            W[mask] = 0
        print(f"[wanda] layer {i} done")


@torch.no_grad()
def prune_sparsegpt(model, layers, samples, device, ratio, prune_n, prune_m):
    collectors = collect_stats(model, layers, samples, device,
                               lambda lin: SparseGPT(lin))
    for i, layer in enumerate(layers):
        subset = find_linear_layers(layer)
        for name in subset:
            collectors[i][name].fasterprune(
                ratio, prune_n=prune_n, prune_m=prune_m,
                percdamp=0.01, blocksize=128)
            collectors[i][name].free()
        print(f"[sparsegpt] layer {i} done")


# --------------------------------------------------------------------------- #
# Sparsity check
# --------------------------------------------------------------------------- #
def check_sparsity(layers):
    zeros, total = 0, 0
    for layer in layers:
        for _, lin in find_linear_layers(layer).items():
            W = lin.weight.data
            zeros += (W == 0).sum().item()
            total += W.numel()
    return zeros / total if total else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    print("torch", version("torch"), "| transformers", version("transformers"))
    print(f"# gpus: {torch.cuda.device_count()}")
    print(f"[config] MODEL={MODEL} METHOD={PRUNE_METHOD} TYPE={SPARSITY_TYPE} "
          f"RATIO={SPARSITY_RATIO} NSAMPLES={NSAMPLES} SEQLEN={SEQLEN}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    max_ctx = getattr(model.config, "max_position_embeddings", SEQLEN)
    seqlen = min(SEQLEN, max_ctx if max_ctx else SEQLEN)
    print(f"[model] loaded. effective seqlen={seqlen} (cap {max_ctx})")

    layers, _ = get_decoder_layers(model)

    # --- dense baseline perplexity ---
    ppl_dense = eval_ppl(model, tokenizer, seqlen, device)
    print(f"[result] dense wikitext2 ppl = {ppl_dense:.4f}")

    # --- prune ---
    if SPARSITY_RATIO > 0:
        if PRUNE_METHOD == "magnitude":
            prune_magnitude(layers, SPARSITY_RATIO, prune_n, prune_m)
        else:
            print("[calib] collecting calibration statistics via full-model forwards")
            samples = get_calibration(tokenizer, NSAMPLES, seqlen, SEED)
            prev_uc = set_use_cache(model, False)
            if PRUNE_METHOD == "wanda":
                prune_wanda(model, layers, samples, device,
                            SPARSITY_RATIO, prune_n, prune_m)
            elif PRUNE_METHOD == "sparsegpt":
                prune_sparsegpt(model, layers, samples, device,
                                SPARSITY_RATIO, prune_n, prune_m)
            else:
                raise ValueError(f"unknown method {PRUNE_METHOD}")
            if prev_uc is not None:
                set_use_cache(model, prev_uc)

    actual_sparsity = check_sparsity(layers)
    print(f"[check] actual sparsity = {actual_sparsity:.4f}")

    # --- pruned perplexity ---
    ppl_pruned = eval_ppl(model, tokenizer, seqlen, device)
    print(f"[result] pruned wikitext2 ppl = {ppl_pruned:.4f}")

    elapsed = time.time() - t0
    result = {
        "model": MODEL, "method": PRUNE_METHOD, "sparsity_type": SPARSITY_TYPE,
        "sparsity_ratio": SPARSITY_RATIO, "actual_sparsity": round(actual_sparsity, 4),
        "seqlen": seqlen, "nsamples": NSAMPLES,
        "ppl_dense": round(ppl_dense, 4), "ppl_pruned": round(ppl_pruned, 4),
        "ppl_delta": round(ppl_pruned - ppl_dense, 4),
        "elapsed_sec": round(elapsed, 1),
        "transformers": version("transformers"),
    }
    with open(os.path.join(ART_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[artifact] result.json:", json.dumps(result))

    # --- optional checkpoint product ---
    if SAVE_MODEL:
        out_dir = os.path.join(REPO_ROOT, "pruned_checkpoint")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(ART_DIR, "checkpoint_info.txt"), "w") as f:
            f.write(f"Pruned checkpoint saved to {out_dir}\n")
            f.write(json.dumps(result, indent=2))
        print(f"[product] pruned checkpoint -> {out_dir}")

    # --- EVAL.md ---
    write_eval_md(result)


def write_eval_md(r):
    label = f"{r['method']}-{r['sparsity_type']}"
    retained = 100.0 * r["ppl_dense"] / r["ppl_pruned"] if r["ppl_pruned"] else 0
    md = f"""# Pruning result — {r['model']}

Reproduction of the granularity-ordering mechanism from arXiv 2606.14150
(*Small LLMs: Pruning vs Training from Scratch*). One-shot pruning as an
initialization-quality probe: lower post-prune perplexity ⇒ a stronger
initialization, which the paper argues finer/data-aware granularities preserve.

## Configuration
| field | value |
|---|---|
| model | `{r['model']}` |
| method / granularity | **{label}** |
| sparsity ratio (target / actual) | {r['sparsity_ratio']} / {r['actual_sparsity']} |
| calibration | {r['nsamples']} samples @ seqlen {r['seqlen']} |
| transformers | {r['transformers']} |
| wall time | {r['elapsed_sec']}s |

## Result (wikitext-2 perplexity)
| | perplexity |
|---|---|
| dense (parent) | **{r['ppl_dense']}** |
| pruned ({label}) | **{r['ppl_pruned']}** |
| Δ (pruned − dense) | {r['ppl_delta']} |

Quality retained ≈ {retained:.1f}% (dense/pruned ppl ratio).

Lower `ppl_pruned` / smaller `ppl_delta` ⇒ finer-grained, higher-quality
initialization — the paper's central mechanism.
"""
    with open(os.path.join(REPO_ROOT, "EVAL.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ART_DIR, "EVAL.md"), "w") as f:
        f.write(md)
    print("[artifact] EVAL.md written")


if __name__ == "__main__":
    main()

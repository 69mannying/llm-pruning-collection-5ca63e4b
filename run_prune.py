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
# Optional: push the pruned checkpoint to the HF Hub (requires SAVE_MODEL=1).
HF_UPLOAD_REPO = cfg("HF_UPLOAD_REPO", "")            # e.g. reneeice/gemma-4-31B-sparsegpt-unstructured-0.5
USE_WANDB = cfg("USE_WANDB", "1") == "1"

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
def capture_per_layer_io(model, layers, samples, device):
    """
    One full-model forward per sample, hooking every decoder layer's *forward_pre*
    to record (input_hidden_states, kwargs) for THAT layer specifically. This
    captures each layer's own attention type / head_dim / position_embeddings /
    KV-sharing — correct for heterogeneous architectures — while storing only the
    cheap inputs (not Hessians). Returns per_layer[i] = list of (inp, kwargs).
    """
    store = {i: [] for i in range(len(layers))}
    handles = []
    for i, layer in enumerate(layers):
        def pre_hook(idx):
            def tmp(_, args, kwargs):
                hs = args[0] if args else kwargs.get("hidden_states")
                # keep inputs on CPU to fit all layers; kwargs (masks/pos-emb) stay
                store[idx].append((hs.detach().cpu(),
                                   {k: v for k, v in kwargs.items()}))
            return tmp
        handles.append(layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
    for s in samples:
        model(s.to(device))
    for h in handles:
        h.remove()
    return store


@torch.no_grad()
def prune_sparsegpt(model, layers, samples, device, ratio, prune_n, prune_m):
    # Capture each layer's real inputs+kwargs in a single forward pass (cheap),
    # then prune layer-by-layer: rebuild one layer's Hessians by replaying just
    # that layer over its cached inputs. Memory-safe (one layer's Hessians live
    # at a time) AND fast (one full forward total, not one per layer).
    per_layer = capture_per_layer_io(model, layers, samples, device)
    for i, layer in enumerate(layers):
        subset = find_linear_layers(layer)
        gpts = {name: SparseGPT(lin) for name, lin in subset.items()}

        handles = []
        for name, lin in subset.items():
            def hook(o):
                def tmp(_, inp, out):
                    o.add_batch(inp[0].data, out.data)
                return tmp
            handles.append(lin.register_forward_hook(hook(gpts[name])))

        ldev = next(layer.parameters()).device
        for inp, kwargs in per_layer[i]:
            layer(inp.to(ldev), **kwargs)
        for h in handles:
            h.remove()

        for name in subset:
            gpts[name].fasterprune(ratio, prune_n=prune_n, prune_m=prune_m,
                                   percdamp=0.01, blocksize=128)
            gpts[name].free()
        del gpts, per_layer[i]
        torch.cuda.empty_cache()
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

    wb = None
    if USE_WANDB and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wb = wandb.init(
                project="llm-pruning-granularity-2606.14150",
                name=f"{MODEL.split('/')[-1]}-{PRUNE_METHOD}-{SPARSITY_TYPE}",
                config={"model": MODEL, "method": PRUNE_METHOD,
                        "sparsity_type": SPARSITY_TYPE, "ratio": SPARSITY_RATIO,
                        "nsamples": NSAMPLES, "seqlen": SEQLEN})
            print("[wandb] initialized")
        except Exception as e:
            print(f"[wandb] disabled ({e})")

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

    if wb is not None:
        try:
            wb.log({"ppl_dense": result["ppl_dense"],
                    "ppl_pruned": result["ppl_pruned"],
                    "ppl_delta": result["ppl_delta"],
                    "actual_sparsity": result["actual_sparsity"],
                    "quality_retained_pct":
                        100.0 * result["ppl_dense"] / result["ppl_pruned"]})
            wb.summary.update(result)
        except Exception as e:
            print(f"[wandb] log failed ({e})")

    # --- optional checkpoint product (+ HF Hub upload) ---
    if SAVE_MODEL:
        out_dir = os.path.join(REPO_ROOT, "pruned_checkpoint")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(ART_DIR, "checkpoint_info.txt"), "w") as f:
            f.write(f"Pruned checkpoint saved to {out_dir}\n")
            f.write(json.dumps(result, indent=2))
        print(f"[product] pruned checkpoint -> {out_dir}")

        if HF_UPLOAD_REPO:
            try:
                from huggingface_hub import HfApi
                token = os.environ.get("HF_TOKEN")
                api = HfApi(token=token)
                api.create_repo(HF_UPLOAD_REPO, exist_ok=True, repo_type="model")
                card = build_model_card(result, HF_UPLOAD_REPO)
                with open(os.path.join(out_dir, "README.md"), "w") as f:
                    f.write(card)
                api.upload_folder(folder_path=out_dir, repo_id=HF_UPLOAD_REPO,
                                  repo_type="model")
                print(f"[product] uploaded to https://huggingface.co/{HF_UPLOAD_REPO}")
                with open(os.path.join(ART_DIR, "hf_upload.txt"), "w") as f:
                    f.write(f"https://huggingface.co/{HF_UPLOAD_REPO}\n")
            except Exception as e:
                print(f"[hf-upload] FAILED ({e})")

    # --- EVAL.md ---
    write_eval_md(result)
    if wb is not None:
        wb.finish()


def build_model_card(r, repo):
    label = f"{r['method']}-{r['sparsity_type']}"
    return f"""---
license: gemma
base_model: {r['model']}
tags:
- pruning
- wanda
- sparsegpt
- arxiv-2606.14150
---

# {repo.split('/')[-1]}

One-shot **{label}** pruned (ratio {r['sparsity_ratio']}, actual {r['actual_sparsity']})
version of `{r['model']}`, produced as part of a minimal reproduction of the
granularity-ordering mechanism in arXiv 2606.14150
(*Small LLMs: Pruning vs Training from Scratch*).

| metric | value |
|---|---|
| dense wikitext-2 ppl | {r['ppl_dense']} |
| pruned wikitext-2 ppl | {r['ppl_pruned']} |
| Δ ppl | {r['ppl_delta']} |
| calibration | {r['nsamples']} samples @ seqlen {r['seqlen']} |

Note: unstructured / N:M sparsity zeroes weights **in place** — the parameter
count and file size are unchanged; this is an initialization-quality probe, not
a size-reduction. See the paper for the granularity/hardware trade-off.
"""


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

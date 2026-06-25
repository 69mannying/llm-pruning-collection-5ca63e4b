#!/usr/bin/env python3
"""
Benchmark A — CPU dense-vs-sparse GEMM on the REAL pruned Gemma-4 weights.

DeepSparse (Neural Magic) is deprecated and cannot load a 2026 Gemma-4
architecture, so instead of a dead engine we measure the underlying mechanism it
relied on directly: does skipping the zeros of a 50%-unstructured-sparse weight
matrix actually beat a dense matmul on CPU?

We pull real Linear weights from our uploaded SparseGPT-unstructured checkpoint
(reneeice/gemma-4-31B-sparsegpt-unstructured-0.5), and for each compare:
  - dense  : torch float32 matmul  (W @ x)
  - sparse : scipy CSR sparse matmul on the same W with zeros dropped

Reports per-matrix and aggregate latency + speedup, and the measured nnz.
Honest result either way — this tells you whether unstructured 50% sparsity is
CPU-exploitable at these shapes at all.
"""
import os
import json
import time

import numpy as np
import torch
from scipy import sparse as sp
from safetensors import safe_open
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(REPO_ROOT, ".openresearch", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

SRC_REPO = os.environ.get("BENCH_SRC_REPO",
                          "reneeice/gemma-4-31B-sparsegpt-unstructured-0.5")
NUM_MATRICES = int(os.environ.get("BENCH_NUM_MATRICES", "6"))
BATCH_TOKENS = int(os.environ.get("BENCH_BATCH_TOKENS", "256"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))


def pick_weights(repo, k):
    """Download the index, pick k representative 2D Linear weights, load them."""
    token = os.environ.get("HF_TOKEN")
    idx_path = hf_hub_download(repo, "model.safetensors.index.json", token=token)
    with open(idx_path) as f:
        index = json.load(f)["weight_map"]
    # representative projections from the first decoder layer(s)
    wanted_suffixes = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.o_proj.weight", "mlp.gate_proj.weight",
                       "mlp.up_proj.weight", "mlp.down_proj.weight"]
    chosen = []
    for name in index:
        if any(name.endswith(s) for s in wanted_suffixes) and "layers.0." in name:
            chosen.append(name)
    chosen = sorted(set(chosen))[:k]
    if not chosen:  # fallback: any 2D weights
        chosen = [n for n in index if n.endswith(".weight")][:k]

    shards = sorted(set(index[n] for n in chosen))
    files = {s: hf_hub_download(repo, s, token=token) for s in shards}
    weights = {}
    for s, path in files.items():
        with safe_open(path, framework="pt") as f:
            for n in chosen:
                if index[n] == s:
                    weights[n] = f.get_tensor(n).float()
    return weights


def bench_matrix(W, x, repeats):
    # dense matmul: (out, in) @ (in, tokens) -> (out, tokens)
    for _ in range(2):  # warmup
        _ = W @ x
    t = time.perf_counter()
    for _ in range(repeats):
        dense_out = W @ x
    dense_t = (time.perf_counter() - t) / repeats

    Wnp = W.numpy()
    xnp = x.numpy()
    W_csr = sp.csr_matrix(Wnp)
    nnz_frac = W_csr.nnz / Wnp.size
    for _ in range(2):  # warmup
        _ = W_csr @ xnp
    t = time.perf_counter()
    for _ in range(repeats):
        sparse_out = W_csr @ xnp
    sparse_t = (time.perf_counter() - t) / repeats

    err = float(np.abs(dense_out.numpy() - sparse_out).max())
    return dense_t, sparse_t, nnz_frac, err


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"[bench-cpu] threads={torch.get_num_threads()} "
          f"src={SRC_REPO} tokens={BATCH_TOKENS} repeats={REPEATS}")
    weights = pick_weights(SRC_REPO, NUM_MATRICES)
    print(f"[bench-cpu] loaded {len(weights)} weight matrices")

    rows = []
    tot_dense = tot_sparse = 0.0
    for name, W in weights.items():
        din = W.shape[1]
        x = torch.randn(din, BATCH_TOKENS)
        dense_t, sparse_t, nnz, err = bench_matrix(W, x, REPEATS)
        speedup = dense_t / sparse_t if sparse_t else 0.0
        tot_dense += dense_t
        tot_sparse += sparse_t
        rows.append({"name": name, "shape": list(W.shape),
                     "nnz_frac": round(nnz, 4),
                     "dense_ms": round(dense_t * 1e3, 3),
                     "sparse_ms": round(sparse_t * 1e3, 3),
                     "speedup": round(speedup, 3),
                     "max_abs_err": round(err, 6)})
        print(f"[bench-cpu] {name}: shape={list(W.shape)} nnz={nnz:.3f} "
              f"dense={dense_t*1e3:.2f}ms sparse={sparse_t*1e3:.2f}ms "
              f"speedup={speedup:.2f}x")

    agg_speedup = tot_dense / tot_sparse if tot_sparse else 0.0
    result = {
        "benchmark": "cpu_dense_vs_csr_sparse",
        "source_model": SRC_REPO,
        "threads": torch.get_num_threads(),
        "batch_tokens": BATCH_TOKENS, "repeats": REPEATS,
        "aggregate_dense_ms": round(tot_dense * 1e3, 3),
        "aggregate_sparse_ms": round(tot_sparse * 1e3, 3),
        "aggregate_speedup": round(agg_speedup, 3),
        "per_matrix": rows,
    }
    with open(os.path.join(ART_DIR, "bench_cpu.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[bench-cpu] aggregate speedup (dense/sparse) =", round(agg_speedup, 3))

    write_eval(result)


def write_eval(r):
    lines = [f"| `{m['name'].split('.')[-2]}` | {m['shape']} | {m['nnz_frac']} | "
             f"{m['dense_ms']} | {m['sparse_ms']} | {m['speedup']}× |"
             for m in r["per_matrix"]]
    verdict = ("CSR sparse matmul **beats** dense" if r["aggregate_speedup"] > 1
               else "CSR sparse matmul is **slower than** dense")
    md = f"""# Benchmark A — CPU dense vs unstructured-sparse GEMM

Real Linear weights from `{r['source_model']}` (our SparseGPT-unstructured
Gemma-4-31B), 50% unstructured sparsity. dense (torch fp32) vs CSR
sparse (scipy), {r['threads']} CPU threads, {r['batch_tokens']} tokens,
{r['repeats']} repeats.

| layer | shape | nnz frac | dense (ms) | sparse (ms) | speedup |
|---|---|---|---|---|---|
{chr(10).join(lines)}

**Aggregate:** dense {r['aggregate_dense_ms']} ms vs sparse
{r['aggregate_sparse_ms']} ms → **{r['aggregate_speedup']}× speedup**.

Verdict: at 50% unstructured sparsity and these shapes, {verdict} on CPU.
This is exactly the regime DeepSparse targeted (a specialized engine does far
better than a generic CSR kernel), but it shows the underlying mechanism with a
runnable, supported toolchain rather than the deprecated DeepSparse runtime.
"""
    with open(os.path.join(REPO_ROOT, "EVAL.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ART_DIR, "EVAL.md"), "w") as f:
        f.write(md)
    print("[bench-cpu] EVAL.md written")


if __name__ == "__main__":
    main()

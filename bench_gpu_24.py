#!/usr/bin/env python3
"""
Benchmark B — 2:4 semi-structured GPU speedup on the B200.

Our Wanda-2:4 checkpoint has the one sparsity pattern current NVIDIA GPUs
accelerate in hardware (Sparse Tensor Cores), but it was saved as a dense tensor
that merely contains a 2:4 zero pattern — a vanilla matmul shows no speedup. To
get the acceleration you must repack into the compressed 2:4 layout
(`torch.sparse.to_sparse_semi_structured`, backed by cuSPARSELt) and run the
sparse matmul path.

This measures exactly that: at Gemma-4-31B Linear shapes, build a true 2:4 weight,
compare dense `linear` vs the semi-structured-sparse `linear` on the GPU, in
bf16/fp16, and report latency + speedup. This is the live, supported path (unlike
DeepSparse), so the number is real and reproducible on Ampere/Hopper/Blackwell.
"""
import os
import json
import time

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(REPO_ROOT, ".openresearch", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

BATCH_TOKENS = int(os.environ.get("BENCH_BATCH_TOKENS", "4096"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "50"))
DTYPE = torch.float16  # cuSPARSELt 2:4 path is well supported in fp16

# Representative Gemma-4-31B Linear shapes: (out_features, in_features)
SHAPES = {
    "q_proj":   (5376, 5376),
    "o_proj":   (5376, 5376),
    "gate_proj": (21504, 5376),
    "up_proj":  (21504, 5376),
    "down_proj": (5376, 21504),
}


def make_24_weight(out_f, in_f, dtype, device):
    """Build a weight whose every contiguous group of 4 along in_f has exactly
    2 zeros (true 2:4 pattern), keeping the 2 largest-magnitude of each 4."""
    W = torch.randn(out_f, in_f, dtype=dtype, device=device)
    g = W.view(out_f, in_f // 4, 4)
    # zero the 2 smallest-|.| of each group of 4
    idx = g.abs().argsort(dim=-1)[..., :2]
    mask = torch.ones_like(g, dtype=torch.bool)
    mask.scatter_(-1, idx, False)
    g = g * mask
    return g.view(out_f, in_f).contiguous()


@torch.no_grad()
def bench(out_f, in_f, device):
    x = torch.randn(BATCH_TOKENS, in_f, dtype=DTYPE, device=device)
    W = make_24_weight(out_f, in_f, DTYPE, device)
    actual_sparsity = (W == 0).float().mean().item()

    # dense
    for _ in range(10):
        _ = torch.nn.functional.linear(x, W)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(REPEATS):
        dense_out = torch.nn.functional.linear(x, W)
    torch.cuda.synchronize()
    dense_t = (time.perf_counter() - t) / REPEATS

    # 2:4 semi-structured (compressed) — cuSPARSELt path
    W_sp = to_sparse_semi_structured(W)
    for _ in range(10):
        _ = torch.nn.functional.linear(x, W_sp)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(REPEATS):
        sparse_out = torch.nn.functional.linear(x, W_sp)
    torch.cuda.synchronize()
    sparse_t = (time.perf_counter() - t) / REPEATS

    err = (dense_out.float() - sparse_out.float()).abs().max().item()
    return dense_t, sparse_t, actual_sparsity, err


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    SparseSemiStructuredTensor._FORCE_CUTLASS = False  # prefer cuSPARSELt
    device = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"[bench-gpu] device={name} dtype={DTYPE} tokens={BATCH_TOKENS} "
          f"repeats={REPEATS}")

    rows = []
    tot_dense = tot_sparse = 0.0
    for label, (out_f, in_f) in SHAPES.items():
        dense_t, sparse_t, sparsity, err = bench(out_f, in_f, device)
        speedup = dense_t / sparse_t if sparse_t else 0.0
        tot_dense += dense_t
        tot_sparse += sparse_t
        rows.append({"layer": label, "shape": [out_f, in_f],
                     "sparsity": round(sparsity, 4),
                     "dense_ms": round(dense_t * 1e3, 4),
                     "sparse_ms": round(sparse_t * 1e3, 4),
                     "speedup": round(speedup, 3),
                     "max_abs_err": round(err, 4)})
        print(f"[bench-gpu] {label}: shape=({out_f},{in_f}) "
              f"dense={dense_t*1e3:.3f}ms 2:4={sparse_t*1e3:.3f}ms "
              f"speedup={speedup:.2f}x")

    agg = tot_dense / tot_sparse if tot_sparse else 0.0
    result = {
        "benchmark": "gpu_2to4_semi_structured",
        "device": name, "dtype": str(DTYPE),
        "batch_tokens": BATCH_TOKENS, "repeats": REPEATS,
        "aggregate_dense_ms": round(tot_dense * 1e3, 4),
        "aggregate_sparse_ms": round(tot_sparse * 1e3, 4),
        "aggregate_speedup": round(agg, 3),
        "per_layer": rows,
    }
    with open(os.path.join(ART_DIR, "bench_gpu.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[bench-gpu] aggregate 2:4 speedup =", round(agg, 3))
    write_eval(result)


def write_eval(r):
    lines = [f"| `{m['layer']}` | {m['shape']} | {m['dense_ms']} | "
             f"{m['sparse_ms']} | {m['speedup']}× |" for m in r["per_layer"]]
    md = f"""# Benchmark B — 2:4 semi-structured GPU speedup

Gemma-4-31B Linear shapes, true 2:4 weights, **{r['device']}**, {r['dtype']},
{r['batch_tokens']} tokens, {r['repeats']} repeats. Dense `F.linear` vs the
compressed 2:4 path (`torch.sparse.to_sparse_semi_structured`, cuSPARSELt).

| layer | shape | dense (ms) | 2:4 (ms) | speedup |
|---|---|---|---|---|
{chr(10).join(lines)}

**Aggregate:** dense {r['aggregate_dense_ms']} ms vs 2:4
{r['aggregate_sparse_ms']} ms → **{r['aggregate_speedup']}× speedup** on
{r['device']}'s Sparse Tensor Cores.

This is the live acceleration path our Wanda-2:4 checkpoint can take once repacked
into the compressed 2:4 layout — the concrete payoff of the coarser (2:4) point on
the granularity axis.
"""
    with open(os.path.join(REPO_ROOT, "EVAL.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ART_DIR, "EVAL.md"), "w") as f:
        f.write(md)
    print("[bench-gpu] EVAL.md written")


if __name__ == "__main__":
    main()

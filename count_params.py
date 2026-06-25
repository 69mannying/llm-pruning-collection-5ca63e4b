#!/usr/bin/env python3

def count_params(
    n_layers,
    n_heads,
    n_kv_heads,
    head_dim,
    hidden_size,
    intermediate_size,
    vocab_size,
):
    """Return total params for a decoder-only Transformer (untied embedding/lm_head)."""
    q_hidden_size = n_heads * head_dim
    kv_hidden_size = n_kv_heads * head_dim

    # per-layer params: attention + MLP + 2 RMSNorm weights
    attn = (
        hidden_size * q_hidden_size
        + hidden_size * kv_hidden_size
        + hidden_size * kv_hidden_size
        + q_hidden_size * hidden_size
    )
    mlp = 3 * hidden_size * intermediate_size
    norms = 2 * hidden_size
    layers_total = n_layers * (attn + mlp + norms)

    token_embedding = vocab_size * hidden_size
    lm_head = hidden_size * vocab_size  # not tied with token_embedding
    final_norm = hidden_size

    return token_embedding + layers_total + final_norm + lm_head


if __name__ == "__main__":
    # Example
    total = count_params(
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        intermediate_size=14336,
        vocab_size=128256,
    )
    print(f"Total params: {total:,} ({total / 1e9:.3f}B)")
    
    total = count_params(
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        head_dim=128,
        hidden_size=4096,
        intermediate_size=11008,
        vocab_size=32000,
    )
    print(f"Total params: {total:,} ({total / 1e9:.3f}B)")
    
    total = count_params(
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=2048,
        intermediate_size=14336,
        vocab_size=128256,
    )
    print(f"Total params: {total:,} ({total / 1e9:.3f}B)")

    

#!/usr/bin/env python3
"""
Simple function to count transformer LLM parameters.
Supports MHA and GQA (Grouped Query Attention).
"""
import argparse

def count_params(
    d_model: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int,
    num_kv_heads: int = None,
    head_dim: int = None,
) -> int:
    """
    Count parameters in a transformer LLM.
    
    Args:
        d_model: Hidden dimension
        num_heads: Number of query heads
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size
        intermediate_size: FFN intermediate dimension
        num_kv_heads: Number of KV heads (defaults to num_heads for MHA)
    
    Returns:
        Total parameter count
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
        
    head_dim = head_dim or (d_model // num_heads)
    
    # Per layer
    attn_params = (
        d_model * d_model +                          # Q proj
        d_model * (num_kv_heads * head_dim) +        # K proj
        d_model * (num_kv_heads * head_dim) +        # V proj
        d_model * d_model                            # O proj
    )
    ffn_params = 3 * d_model * intermediate_size    # gate + up + down
    ln_params = 2 * d_model                         # 2 layer norms
    
    per_layer = attn_params + ffn_params + ln_params
    
    # Total
    return (
        2 * vocab_size * d_model +           # embeddings (tied with output)
        per_layer * num_layers +         # transformer blocks
        d_model                          # final layer norm
    )


# Quick test
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count parameters in a transformer LLM.")
    parser.add_argument("--d_model", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of query heads")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--intermediate_size", type=int, required=True, help="FFN intermediate dimension")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of KV heads (defaults to num_heads for MHA)")
    parser.add_argument("--head_dim", type=int, default=None, help="Head dimension (defaults to d_model/num_heads)")
    args = parser.parse_args()
    
    total_params = count_params(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        intermediate_size=args.intermediate_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    print(total_params)

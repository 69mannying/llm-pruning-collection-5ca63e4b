# SliceGPT

This is the official implementation of SliceGPT by Ashkboos et al. For the original README, please see [here](ORIGINAL_README.md).

## Installation
```bash
bash scripts/install.sh
```

## Pruning
```bash
bash scripts/prune_llama2_7b.sh
bash scripts/prune_llama3.1_8b.sh
```

## Results
Notably, the sequence length of eval data significantly affects the perplexity. Generally, if the sequence length is identical with the model's `max_position_embeddings` (4096 for Llama-2-7B, 8192 for Llama-3.1-8B), the perplexity will be much lower. However, in this section we follow the original setups in the paper and set sequence length to 2048.

Notably, the results we obtained on Llama-2-7B are slightly worse than the original paper, yet we have made no adjustments to the codebase at all.

Base Model=Llama2-7B
| Method | Ratio | WikiText2(Tested) | WikiText2(Paper) |
|:--:|:--:|:--:|:--:|
| Dense | 0.00 | 5.47 | 5.47 | 
| SliceGPT | 0.10 | 5.96 | 5.89 |
| SliceGPT | 0.20 | 6.86 | 6.64 |
| SliceGPT | 0.25 | 7.56 | 7.24 |
| SliceGPT | 0.30 | 8.63 | 8.12 |
| SliceGPT | 0.50 | 21.1 | - |

Base Model=Llama3.1-8B
| Method | Ratio | WikiText2(Tested) | WikiText2(Paper) |
|:--:|:--:|:--:|:--:|
| Dense | 0.00 | 6.24 | - |
| SliceGPT | 0.10 | 8.04 | - |
| SliceGPT | 0.20 | 11.31 | - |
| SliceGPT | 0.25 | 13.79 | - |
| SliceGPT | 0.30 | 19.42 | - |
| SliceGPT | 0.50 | 47.86 | - |


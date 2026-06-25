# SLEB

This directory contains the implementation of SLEB by Song et al. For the original README, please see [here](ORIGINAL_README.md).


## Installation
```bash
bash scripts/install.sh
```

## Pruning
```bash
bash scripts/prune_llama2_7b.sh
bash scripts/prune_llama2_13b.sh
bash scripts/prune_llama3.1_8b.sh
```

## Results
Base model=Llama-2-7b-hf
| Method | Layers Removed | WikiText2(Tested) | WikiText2(Paper) | C4(Tested) | C4(Paper) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Dense | 0 |  | - | | - |
| SLEB | 2 | 6.07 | - | 8.18 | - |
| SLEB | 4(~10%) | 6.95 | - | 9.34 | 9.34 |
| SLEB | 7(~20%) | 9.14 | - | 12.32 | 12.32 |
| SLEB | 8 | 10.39 | - | 13.74 | - |
| SLEB | 16 | 106.20 | - | 85.96 | - |

Base model=Llama-2-13b-hf
| Method | Layers Removed | WikiText2(Tested) | WikiText2(Paper) | C4(Tested) | C4(Paper) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Dense | 0 |  |  | | - |
| SLEB | 4(10%) | 5.64 | - | 7.79 | 7.80 |
| SLEB | 8(20%) | 6.75 | - | 9.35 | 9.42 |

<!-- 
Base model=Llama-2-70b-hf
| Method | Layers Removed | WikiText2(Tested) | WikiText2(Paper) | C4(Tested) | C4(Paper) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Dense | 0 |  |  | | - |
| SLEB | 8(10%) |  | - |  | - |
| SLEB | 16(20%) |  | - | | - |
 -->

Base model=Llama-3.1-8B
| Method | Layers Removed | WikiText2(Tested) | WikiText2(Paper) | C4(Tested) | C4(Paper) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Dense | 0 |  | - | | - |
| SLEB | 2 | 7.35 | - | 11.37 | - |
| SLEB | 4 | 8.93 | - | 13.61 | - |
| SLEB | 8 | 16.61 | - | 20.52 | - |
| SLEB | 16 | 159.39 | - | 136.34 | - |

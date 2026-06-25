#!/bin/bash

conda create -n flap python=3.9 -y 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flap

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
pip install transformers==4.43.3 datasets==2.16.0 wandb sentencepiece accelerate==0.31.0

pip install sacrebleu sqlitedict scikit-learn omegaconf pycountry evaluate peft==0.11
#!/bin/bash

conda create -n sleb python=3.10 -y 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sleb

pip install -r requirements.txt
cd lm-evaluation-harness
pip install -e .

mkdir -p sleb_results
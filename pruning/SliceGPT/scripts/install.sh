#!/bin/bash

conda create -n slicegpt python=3.10 -y 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slicegpt

pip install -e .[experiment]
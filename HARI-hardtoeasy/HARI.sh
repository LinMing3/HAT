#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-qwen-deepseed

echo "Starting to run all files..."
accelerate launch HARI.py > HARI.log 2>&1

echo "All HARI.py files have finished."

#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-qwen-deepseed

echo "Starting to run all files..."
# accelerate launch HARI.py > HARI.log 2>&
CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --config_file /home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/GRPO/TRAIN/jr_config.yaml GRPO.py > GRPO.log 2>&1

echo "All GRPO.py files have finished."

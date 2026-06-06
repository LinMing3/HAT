#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-qwen-deepseed

echo "Starting to run all files..."
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file /home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/HARI_random/jr_config.yaml rdsample.py > HAT-2.log 2>&1

echo "All HARI.py files have finished."

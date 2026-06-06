#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-qwen-deepseed

echo "Starting to run all files..."
# accelerate launch VLC.py > VLC.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file /home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/VL-Cogito/jr_config.yaml VLC.py > VLC-medium-hard.log 2>&1

echo "All VLC.py files have finished."
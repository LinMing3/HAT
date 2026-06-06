#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-qwen-deepseed


echo "Starting to run all files..."
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file /home/yangjiacheng/data/jiarui/HARI-sampling/ac_config.yaml HARI.py > HAT2.log 2>&1

echo "All HARI.py files have finished."

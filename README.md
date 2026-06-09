# HAT: High-resolution Adaptive Training for Large Multimodal Models

> **Paper:** HAT: High-resolution Adaptive Training for Large Multimodal Models Inspired by Human Cognition

## Overview

HAT is an adaptive post-training framework that enables efficient learning for Large Multimodal Models (LMMs) under high-resolution settings. Inspired by human cognitive processes (the "easy-to-hard" effect), HAT implements **vision-centric curriculum learning** and introduces **R-GRPO** (Rethinking-based Group Relative Policy Optimization) to address the sparse reward problem in reinforcement fine-tuning.

### Key Contributions

- **Vision-centric Complexity Assessment**: Measures inherent visual complexity using GLCM-entropy, median frequency (MDF), and pixel count — no external model needed.
- **Adaptive Sampling**: Dynamically selects training samples based on visual complexity and the model's real-time performance.
- **R-GRPO**: A reinforcement fine-tuning strategy that allows the model to rethink incorrect responses, providing denser reward signals (`r ∈ {0, 0.5, 1}`).

## Repository Structure

```
HAT/
├── curves/                      # Training curve data and visualization scripts
│   ├── draw.py                  # Plot training curves
│   ├── load.py                  # Load training logs
│   └── *.json                   # Training curve data (GRPO, HAT, random, hard-to-easy)
│
├── difficulty/                  # Complexity assessment module
│   ├── dataset_difficulty_process.py   # Compute complexity scores for dataset
│   ├── diff.py                  # Difficulty analysis utilities
│   ├── count.py                 # Sample count statistics
│   └── difficulty_hist/         # Complexity distribution histograms
│
├── difficulty_export/           # Sample visualization by difficulty level
│   ├── easy/                    # Easy samples (low visual complexity)
│   ├── medium/                  # Medium samples
│   └── hard/                    # Hard samples (high visual complexity)
│
├── preview_imgs/                # Preview images with difficulty scores
│
├── prompt/                      # Training scripts and RL prompts
│   ├── HARI.py                  # Main HAT training script
│   ├── trl_main/                # Modified TRL framework for distributed training
│   └── HARI/                    # Training checkpoints and logs
│
├── Qwen/                        # Qwen2.5-VL-7B evaluation scripts
│   ├── qwen.py                  # Main evaluation script
│   ├── accmme.py                # MME-RealWorld accuracy computation
│   ├── acctree.py               # TreeBench accuracy computation
│   ├── hr4k.py                  # HR-Bench-4K evaluation
│   ├── hr8k.py                  # HR-Bench-8K evaluation
│   └── vstar.py                 # V*Bench evaluation
│
├── visual/                      # Visualization and result analysis
│   ├── HARI.py                  # Visualization script for HAT outputs
│   ├── dataset_image_process.py # Image preprocessing utilities
│   ├── export_correct_samples.py# Export correctly answered samples
│   └── result/                  # Raw inference outputs (JSON)
│
├── VL-Cogito/                   # VL-COGITO baseline experiments
│   ├── VLC.py                   # VL-COGITO evaluation script
│   ├── VLC.sh                   # Training shell script
│   └── EVAL/                    # Evaluation results
│
├── TreeBench.py                 # TreeBench evaluation entry point
├── ret.py                       # Result aggregation utilities
└── rew.py                       # Reward function implementation
```

## Method

### Complexity Assessment

Each image is scored along three dimensions:

| Metric | Description |
|--------|-------------|
| GLCM-Entropy | Texture complexity via gray-level co-occurrence matrix |
| Median Frequency (MDF) | Frequency-domain complexity |
| Pixel Count | Scale of the image |

The overall complexity score is the mean of the three normalized metrics:

$$c_i = \text{mean}(g_i, m_i, n_i)$$

### Adaptive Sampling

Sampling weight for each sample at training step $t$:

$$\omega_i = \exp\left(-\frac{(c_i - \frac{t}{T})^2}{2\sigma^2}\right)$$

Adjusted by real-time model performance:

$$\tilde{\omega}_i = \omega_i + \lambda|\Delta a|, \quad \Delta a = a_t - a_{t'}$$

### R-GRPO

Rewards are defined as:
- `r = 1`: correct on the first attempt
- `r = 0.5`: correct after rethinking
- `r = 0`: incorrect after rethinking

## Main Results

### MME-RealWorld-Lite (In-distribution)

| Method | Overall | RS | MO | AD |
|--------|---------|----|----|-----|
| Qwen2.5-VL-7B (base) | 42.3 | 32.7 | 27.3 | 30.0 |
| GRPO | 53.1 | 58.0 | 45.1 | 57.3 |
| **HAT-7B (Ours)** | **62.9** | **62.0** | **53.9** | **54.9** |

### Other Benchmarks

| Benchmark | Qwen2.5-VL-7B | HAT-7B |
|-----------|---------------|--------|
| V\*Bench | 78.5 | **86.4** |
| HR-Bench-4K | 70.1 | **72.4** |
| HR-Bench-8K | 61.0 | **72.5** |
| MMStar | 59.3 | **63.0** |

HAT with easy-to-hard sampling reaches **58% accuracy in only 2,500 steps**, while vanilla GRPO requires **9,000 steps** to reach the same level.

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-7B |
| Standard Deviation σ | 0.15 |
| Scaling Factor λ | 0.1 |
| Learning Rate | 1e-5 |
| Epochs | 3 |
| Completion Max Length | 512 |
| Prompt Max Length | 4096 |
| Number of Generations | 4 |
| Data Type | bf16 |

## Benchmarks

- **MME-RealWorld / MME-RealWorld-Lite**: 24,000+ samples, avg. resolution 2076×1434
- **TreeBench**: 405 samples, avg. resolution 2152×1615
- **V\*Bench**: 191 samples, avg. resolution 2246×1583
- **HR-Bench-4K / 8K**: 200 samples each, up to 5727×4430
- **MMStar**: 1,500 samples, avg. resolution 511×391

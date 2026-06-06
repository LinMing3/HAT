"""Simple histogram script for dataset difficulty column."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import datasets

DATASET_DIR = Path("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset")
BIN_COUNTS = [100, 50, 20,30,40,60,70,80,90,110]
OUTPUT_DIR = Path("difficulty_hist")


def summarize(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
	edges = np.linspace(0.0, 1.0, bins + 1)
	counts, _ = np.histogram(values, bins=edges)
	print(f"\nBin counts for k={bins}:")
	for left, right, count in zip(edges[:-1], edges[1:], counts):
		print(f"[{left:.3f}, {right:.3f}): {count}")
	return counts, edges


def plot(values: np.ndarray, edges: np.ndarray, bins: int, out_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(9, 5))
	ax.hist(values, bins=edges, color="#4C72B0", edgecolor="black")
	ax.set_title(f"Difficulty Distribution (k={bins})")
	ax.set_xlabel("difficulty")
	ax.set_ylabel("count")
	ax.set_xlim(0.0, 1.0)
	ax.grid(axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)
 

dataset_dir = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"
dataset = datasets.load_from_disk(dataset_dir)
print(dataset[0])
difficulties = np.array(dataset['difficulty'])
print(f"Loaded {difficulties.size} samples from {DATASET_DIR}.")
print(f"Difficulty stats -> min: {difficulties.min():.4f}, max: {difficulties.max():.4f}, "
	    f"mean: {difficulties.mean():.4f}, std: {difficulties.std():.4f}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for k in BIN_COUNTS:
	_, edges = summarize(difficulties, k)
	out = OUTPUT_DIR / f"difficulty_hist_k{k}.png"
	plot(difficulties, edges, k, out)
	print(f"Saved histogram to {out}")



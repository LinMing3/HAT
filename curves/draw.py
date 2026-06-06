"""Plot accuracy curves for GRPO, HARI, and HTE runs."""

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


CURVE_DIR = Path("/home/yangjiacheng/data/jiarui/curves")


def load_curve(path: Path) -> Tuple[List[int], List[float]]:
	with path.open(encoding="utf-8") as f:
		entries = json.load(f)
	steps = [item["step"] for item in entries]
	accs = [item["acc"] for item in entries]
	return steps, accs


def main() -> None:
	curves = {
		"GRPO": CURVE_DIR / "GRPO-Vstar.json",
		"HARI": CURVE_DIR / "HARI-Vstar.json",
		"HTE": CURVE_DIR / "hardtoeasy-Vstar.json",
		"Random": CURVE_DIR / "random-Vstar.json",
	}

	plt.figure(figsize=(9, 5.5))
	for label, path in curves.items():
		steps, accs = load_curve(path)
		plt.plot(steps, accs, marker="o", linewidth=2, label=label)

	plt.xlabel("Step")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Step")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()

	output_path = CURVE_DIR / "curves-Vstar.png"
	plt.savefig(output_path, dpi=300)
	plt.close()
	print(f"Saved plot to {output_path}")


if __name__ == "__main__":
	main()

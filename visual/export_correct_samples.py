import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


DEFAULT_LOG_PATH = (
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/"
    "HARI-halfretry-more/EVAL-all-mme/logs/16_checkpoint-19000_20251222_211250_run1.log"
)
DEFAULT_DATASET_PATH = \
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/test/MME-RealWorld-Lite.json"
DEFAULT_OUTPUT_PATH = \
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/test/hari-test.json"


INDEX_LINE = re.compile(r"^(?P<idx>\d+)\s*-\s*(?P<total>\d+)")


def parse_correct_indices(log_path: Path) -> List[int]:
    """Return the dataset indices that were marked correct in the log."""

    correct_indices: List[int] = []
    expect_prediction = False
    current_index = None

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            match = INDEX_LINE.match(line)
            if match:
                current_index = int(match.group("idx"))
                expect_prediction = True
                continue

            if expect_prediction:
                parts = line.split()
                if len(parts) >= 2 and current_index is not None:
                    pred, gt = parts[0], parts[1]
                    if pred == gt:
                        correct_indices.append(current_index)
                expect_prediction = False

    return correct_indices


def load_dataset(dataset_path: Path) -> List[dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Dataset JSON must be a list or contain a 'data' list.")

    if not isinstance(data, list):
        raise ValueError("Unsupported JSON root. Expected a list of samples.")

    return data


def select_samples(dataset: List[dict], indices: Iterable[int]) -> List[dict]:
    """Collect dataset entries at the provided indices, preserving order."""

    selected = []
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        if 0 <= idx < len(dataset):
            selected.append(dataset[idx])
            seen.add(idx)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract correctly answered samples based on a run log."
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        type=Path,
        help="Path to run1 log that contains prediction outputs.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        type=Path,
        help="Path to the source MME JSON dataset.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help="Where to write the filtered hari-test JSON.",
    )

    args = parser.parse_args()

    correct_indices = parse_correct_indices(args.log)
    print(f"Found {len(correct_indices)} correct samples in log {args.log}.")

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples from {args.dataset}.")

    selected_samples = select_samples(dataset, correct_indices)
    print(f"Selecting {len(selected_samples)} samples for export.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=4)

    print(f"Saved filtered samples to {args.output}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import torch
from PIL import Image
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

DEFAULT_DATASET_ROOT = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TRIALS = 4
BATCH_SIZE = 8  # 根据显存调整
MAX_NEW_TOKENS = 8  # 足够生成 \boxed{A}
TEMPERATURE = None  # 不采样，贪心
TOP_P = 1.0
THRESH_ACC = 0.5  # 丢弃 >50% 的样本
OUT_DATASET = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/VL-Cogito/dataset_filtered_qwen25vl7b_50acc"
LOG_JSONL = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/VL-Cogito/qwen25vl7b_eval.jsonl"  # 如不需日志设为 None

TMP = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/VL-Cogito/tmp_hf"
os.makedirs(TMP, exist_ok=True)
os.environ.setdefault("TMPDIR", TMP)
os.environ.setdefault("HF_DATASETS_CACHE", TMP)
os.environ.setdefault("HF_HOME", TMP)

BOXED_RE = re.compile(r"\\boxed\{([A-E])\}", re.IGNORECASE)
LETTER_RE = re.compile(r"\b([A-E])\b")
SYSTEM_PROMPT = (
    "You are a helpful assistant. Read the image and the question with options. "
    "Answer with ONLY one capital letter from [A, B, C, D, E] enclosed in \\boxed{}, e.g., \\boxed{A}. "
    "Do NOT output any other text."
)

def build_prompt(example: Dict[str, Any]) -> str:
    question = example.get("Text", "")
    choices = example.get("Answer choices", [])
    choice_text = "\n".join(choices)
    return (
        f"{question}\n"
        f"{choice_text}\n"
        "Choose the correct option. Answer with ONLY one boxed letter, e.g., \\boxed{A}."
    )

def extract_letter(text: str) -> str:
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).upper()
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    for ch in text:
        if ch.isalpha():
            return ch.upper()
    return text.strip()[:1].upper() if text.strip() else ""

def resolve_image(example: Dict[str, Any]) -> Image.Image:
    img = example.get("Image") or example.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str) and os.path.exists(img):
        return Image.open(img).convert("RGB")
    img_path = example.get("ImagePath")
    if img_path and os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    raise FileNotFoundError(f"Image not found for Question_id={example.get('Question_id')}")

def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",  # 关键：decoder-only 左补齐
    )
    if processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
    ).eval()
    return processor, model

def run_batch(proc, model, examples: List[Dict[str, Any]], trial_idx: int):
    images = [resolve_image(ex) for ex in examples]
    prompts = [build_prompt(ex) for ex in examples]

    messages_list = [
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prm},
            ]},
        ]
        for img, prm in zip(images, prompts)
    ]
    chats = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    inputs = proc(text=chats, images=images, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 贪心，避免长推理
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id,
        )
    gen_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
    ]
    resps = proc.batch_decode(gen_ids_trimmed, skip_special_tokens=True)

    records = []
    for ex, resp in zip(examples, resps):
        pred_letter = extract_letter(resp)
        gt_letter = str(ex.get("Ground truth", "")).strip().upper()[:1]
        records.append({
            "Question_id": ex.get("Question_id"),
            "trial": trial_idx,
            "prediction_raw": resp,
            "pred_letter": pred_letter,
            "ground_truth": gt_letter,
            "correct": pred_letter == gt_letter,
        })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT),
        help="Path to load_from_disk dataset",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root
    print(f"Loading dataset from: {dataset_root}")

    proc, model = load_model()
    ds = load_from_disk(dataset_root)

    def count_samples(d):
        if isinstance(d, DatasetDict):
            return sum(len(s) for s in d.values())
        return len(d)

    total_samples = count_samples(ds)
    total_steps = total_samples * TRIALS
    start_time = time.time()
    processed_steps = 0

    if LOG_JSONL is not None:
        log_path = Path(LOG_JSONL)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        log_f = log_path.open("w")
    else:
        log_f = None

    counts = defaultdict(int)
    corrects = defaultdict(int)

    def split_iter(d):
        if isinstance(d, DatasetDict):
            for sp, sp_ds in d.items():
                yield sp, sp_ds
        else:
            yield None, d

    for split, split_ds in split_iter(ds):
        n = len(split_ds)
        for t in range(TRIALS):
            for start in range(0, n, BATCH_SIZE):
                batch = split_ds[start:start + BATCH_SIZE]
                examples = [{k: batch[k][i] for k in batch} for i in range(len(batch["Question_id"]))]
                recs = run_batch(proc, model, examples, t)
                for rec in recs:
                    if log_f:
                        if split is not None:
                            rec["split"] = split
                        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    qid = rec["Question_id"]
                    counts[qid] += 1
                    corrects[qid] += int(rec["correct"])
                processed_steps += len(examples)
                if processed_steps % 200 == 0 or processed_steps == total_steps:
                    elapsed = time.time() - start_time
                    sps = processed_steps / max(elapsed, 1e-9)
                    eta = (total_steps - processed_steps) / max(sps, 1e-9)
                    print(f"Progress {processed_steps}/{total_steps} "
                          f"({processed_steps/total_steps:.1%}), "
                          f"elapsed {elapsed/60:.1f} min, "
                          f"ETA {eta/60:.1f} min, "
                          f"throughput {sps:.2f} step/s")
            print(f"Split={split or 'all'}, trial={t}, processed {n} samples")

    if log_f:
        log_f.close()

    drop_ids = set()
    for qid, n in counts.items():
        acc = corrects[qid] / n
        if n >= TRIALS and acc > THRESH_ACC:
            drop_ids.add(qid)
    print(f"Will drop {len(drop_ids)} samples with >{THRESH_ACC*100:.0f}% acc and >= {TRIALS} trials")

    def keep_fn(ex):
        return ex.get("Question_id") not in drop_ids

    if isinstance(ds, DatasetDict):
        filtered = DatasetDict({
            split: split_ds.filter(keep_fn, num_proc=8, desc=f"Filtering {split}")
            for split, split_ds in ds.items()
        })
    else:
        filtered = ds.filter(keep_fn, num_proc=8, desc="Filtering")

    if os.path.exists(OUT_DATASET):
        import shutil
        shutil.rmtree(OUT_DATASET)
    filtered.save_to_disk(OUT_DATASET)
    total_time = time.time() - start_time
    print(f"Saved filtered dataset to {OUT_DATASET}")
    print(f"Evaluation+filter done in {total_time/60:.2f} minutes")
    print(filtered)

if __name__ == "__main__":
    main()

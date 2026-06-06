#!/usr/bin/env python
import os
import json
import shutil
import numpy as np
from datasets import load_from_disk

DATASET_DIR = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"
SRC_PREFIX = "/home/dangyunkai/yunkai/VLM/VIG-Group"
REAL_PREFIX = "/data1/yunkai/VIG_Group"
OUT_DIR = "./difficulty_export"  # 根目录，内部会生成 easy/ hard 子目录
LOW1,LOW2,HIGH = 0.35,0.45,0.65           # easy: < LOW; hard: > HIGH
TYPE_EXCLUDE = "Diagram and Table"

def fix_path(fake_path: str) -> str:
    if fake_path.startswith(SRC_PREFIX):
        return REAL_PREFIX + fake_path[len(SRC_PREFIX):]
    return fake_path

def main():
    ds = load_from_disk(DATASET_DIR)
    required_cols = ["ImagePath", "difficulty", "Text", "Answer choices", "Subtask"]
    for c in required_cols:
        if c not in ds.column_names:
            raise KeyError(f"缺少列: {c}")

    diff = np.asarray(ds["difficulty"], dtype=float)
    types = np.asarray(ds["Subtask"])
    mask_valid = types != TYPE_EXCLUDE

    easy_mask = (diff < LOW2) & mask_valid & (diff > LOW1)
    
    hard_mask = (diff > HIGH) & mask_valid
    rng = np.random.default_rng(42)
    easy_idx = rng.choice(np.nonzero(easy_mask)[0], size=100, replace=False)
    # easy_idx = np.nonzero(easy_mask)[0]
    hard_idx = np.nonzero(hard_mask)[0]

    print(f"easy (difficulty >= {LOW1} and < {LOW2}): {len(easy_idx)}")
    print(f"hard (difficulty > {HIGH}): {len(hard_idx)}")

    # 取列表字段
    image_paths_all = list(ds["ImagePath"])
    texts_all = list(ds["Text"])
    answers_all = list(ds["Answer choices"])

    def export(split_name, indices):
        if indices.size == 0:
            return 0
        subdir = os.path.join(OUT_DIR, split_name)
        os.makedirs(subdir, exist_ok=True)
        manifest = []
        diffs_sel = diff[indices]
        for i, idx in enumerate(indices, start=1):
            img = image_paths_all[idx]
            text = texts_all[idx]
            ans = answers_all[idx]
            dval = diffs_sel[i - 1]

            real_img = fix_path(img)
            if not os.path.isfile(real_img):
                print(f"[WARN] 找不到文件: {real_img}")
                continue

            fname = f"{i:05d}_" + os.path.basename(real_img)
            dst_img = os.path.join(subdir, fname)
            shutil.copy2(real_img, dst_img)

            manifest.append({
                "dst_image": dst_img,
                "difficulty": float(dval),
                "text": text,
                "answer_choices": ans,
                "src_image": real_img,
                "subtask": types[idx],
            })

        meta_path = os.path.join(subdir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {split_name}: 导出 {len(manifest)} 个样本 -> {subdir}")
        return len(manifest)

    export("easy", easy_idx)
    export("hard", hard_idx)

if __name__ == "__main__":
    main()

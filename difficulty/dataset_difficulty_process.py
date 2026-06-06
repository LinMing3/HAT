import os, numpy as np
from datasets import load_from_disk, Image
from PIL import Image as PILImage
from pathlib import Path
from jiarui.difficulty.GLCM import glcm_entropy  # 你已有的实现
from jiarui.difficulty.MDF2 import image_mdf  # 你已有的实现


os.environ["TMPDIR"] = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/temp"  # 确保存在且可写
DATA_DIR = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiacheng/251116-DynamicResolution/resolution_model/dataset/MME-train"
OUT_DIR = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"

ds = load_from_disk(DATA_DIR)
# ds = ds.select(range(min(100, len(ds))))

print("dataset loaded:", ds)

# 若图像列是路径，先 cast；假设列名为 "Image"
if not isinstance(ds.features["Image"], Image):
    ds = ds.cast_column("Image", Image(decode=True))

def add_raw_metrics(example):
    try:
        img = example["Image"]
        arr = np.array(img.convert("L"))
        example["mdf_raw"] = image_mdf(arr)
        example["glcm_raw"] = float(glcm_entropy(np.array(img.convert("RGB")), levels=32, reduction="mean"))
        example["pixel_raw"] = float(img.width * img.height)
    except Exception as e:
        example["mdf_raw"] = 0.0
        example["glcm_raw"] = 0.0
        example["pixel_raw"] = 0.0
        print(f"Error processing image: {e}")
    return example

ds = ds.map(add_raw_metrics, num_proc=32)

# 全局 min/max
m_min, m_max = float(np.min(ds["mdf_raw"])), float(np.max(ds["mdf_raw"]))
e_min, e_max = float(np.min(ds["glcm_raw"])), float(np.max(ds["glcm_raw"]))
p_min, p_max = float(np.min(ds["pixel_raw"])), float(np.max(ds["pixel_raw"]))


def add_difficulty(example):
    m_norm = (example["mdf_raw"] - m_min) / (m_max - m_min + 1e-9)
    e_norm = (example["glcm_raw"] - e_min) / (e_max - e_min + 1e-9)
    p_norm = (example["pixel_raw"] - p_min) / (p_max - p_min + 1e-9)
    d = (m_norm + e_norm + p_norm) / 3.0 
    example["difficulty"] = float(d)
    return example

ds = ds.map(add_difficulty, num_proc=32)

# 分桶（K=5）
qs = np.quantile(ds["difficulty"], np.linspace(0.2, 0.8, 4))
def add_bucket(example):
    d = example["difficulty"]
    bucket = int(np.searchsorted(qs, d, side="right"))
    example["bucket"] = bucket  # 0..9
    return example

ds = ds.map(add_bucket, num_proc=32)

# 保存
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ds.save_to_disk(OUT_DIR)
print("done, saved to", OUT_DIR)

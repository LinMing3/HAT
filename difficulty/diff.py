#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from jiarui.difficulty.MDF2 import image_mdf
from jiarui.difficulty.GLCM import glcm_entropy


def _read_image(path):
    """读取图片为 numpy 数组（优先 imageio，失败用 PIL）。"""
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception:
        from PIL import Image
        return np.array(Image.open(path))


def area_norm(h, w, ref_area=7680 * 5046):
    """尺寸归一化，越大越接近 1."""
    return float(min(1.0, (h * w) / float(ref_area))) if h > 0 and w > 0 else 0.0


def compute_metrics(path, levels=32):
    """单张图片的 MDF、GLCM 熵、尺寸因子。"""
    img = _read_image(path)
    H, W = int(img.shape[0]), int(img.shape[1])

    mdf = float(image_mdf(img, fs_row=float(H), fs_col=float(W)))
    glcm_h = float(glcm_entropy(img, levels=levels, reduction="mean"))
    pix = area_norm(H, W)
    return {"path": path, "mdf": mdf, "glcm": glcm_h, "pix": pix}


def collect_files(root):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = [p for pat in exts for p in glob.glob(os.path.join(root, "**", pat), recursive=True)]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="批量计算 MDF/GLCM 难度并分桶")
    parser.add_argument("root", help="图片根目录")
    parser.add_argument("--levels", type=int, default=32, help="GLCM 灰度级数")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--buckets", type=int, default=10, help="桶数量 K")
    parser.add_argument("--output", default="difficulty_stats.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    files = collect_files(args.root)
    if not files:
        print(f"目录无图片: {args.root}")
        return

    print(f"共 {len(files)} 张图，levels={args.levels}，workers={args.workers}")

    # 第 1 遍：并行计算指标
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(compute_metrics, f, args.levels): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[错误] {f}: {e}")

    if not rows:
        print("全部失败，退出")
        return

    # 统计 min/max
    m_vals = np.array([r["mdf"] for r in rows], dtype=np.float64)
    e_vals = np.array([r["glcm"] for r in rows], dtype=np.float64)
    m_min, m_max = float(m_vals.min()), float(m_vals.max())
    e_min, e_max = float(e_vals.min()), float(e_vals.max())

    def norm(x, lo, hi):
        if hi - lo <= 1e-12:
            return 0.0
        return float((x - lo) / (hi - lo))

    # 归一化与难度
    for r in rows:
        m_n = norm(r["mdf"], m_min, m_max)
        e_n = norm(r["glcm"], e_min, e_max)
        r["mdf_norm"] = m_n
        r["glcm_norm"] = e_n
        r["d"] = 0.5 * m_n + 0.5 * e_n

    # 分桶
    d_vals = np.array([r["d"] for r in rows], dtype=np.float64)
    qs = np.quantile(d_vals, np.linspace(0, 1, args.buckets + 1))
    for r in rows:
        # 桶编号 1..K
        k = int(np.searchsorted(qs, r["d"], side="right"))
        r["bucket"] = min(max(1, k), args.buckets)

    # 写 CSV
    fieldnames = ["path", "mdf", "glcm", "mdf_norm", "glcm_norm", "pix", "d", "bucket"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成！输出: {args.output}")
    print(f"MDF min/max: {m_min:.6f}/{m_max:.6f}, GLCM min/max: {e_min:.6f}/{e_max:.6f}")
    print(f"桶分位点: {qs}")


if __name__ == "__main__":
    main()

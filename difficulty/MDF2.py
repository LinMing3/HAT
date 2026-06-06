import os, sys, glob
import numpy as np

def image_mdf(image: np.ndarray, fs_row: float = 1.0, fs_col: float = 1.0) -> float:
    X = np.asarray(image, dtype=np.float64)
    if X.ndim == 3: X = X.mean(axis=2)
    if X.ndim != 2: raise ValueError("image must be 2D or 3D")
    H, W = X.shape

    # 第1轮：对每列求 MDF
    X = X - X.mean(axis=0, keepdims=True)
    Spec = np.fft.rfft(X, axis=0)
    P = (Spec.real**2 + Spec.imag**2)
    f_r = np.fft.rfftfreq(H, d=1.0 / fs_row)
    df_r = (f_r[1] - f_r[0]) if len(f_r) > 1 else float(fs_row)
    bins = P * df_r
    tot = bins.sum(axis=0)
    half = 0.5 * tot
    csum = np.cumsum(bins, axis=0)
    idx = np.argmax(csum >= half[np.newaxis, :], axis=0)

    prev = np.zeros(W); mask = idx > 0; cols = np.arange(W)
    prev[mask] = csum[idx[mask]-1, cols[mask]]
    cur = bins[idx, cols]
    frac = np.zeros(W); np.divide(half - prev, cur, out=frac, where=(cur > 0))
    fprev = np.full(W, f_r[0]); fprev[mask] = f_r[idx[mask]-1]

    mdf_cols = np.zeros(W); valid = tot > 0
    mdf_cols[valid] = fprev[valid] + frac[valid] * df_r

    # 第2轮：对列向量再求 MDF
    m = mdf_cols - mdf_cols.mean()
    Spec2 = np.fft.rfft(m)
    P2 = (Spec2.real**2 + Spec2.imag**2)
    f_c = np.fft.rfftfreq(W, d=1.0 / fs_col)
    df_c = (f_c[1] - f_c[0]) if len(f_c) > 1 else float(fs_col)

    bins2 = P2 * df_c
    tot2 = bins2.sum()
    if not np.isfinite(tot2) or tot2 <= 0: return 0.0
    csum2 = np.cumsum(bins2); half2 = 0.5 * tot2
    k = int(np.searchsorted(csum2, half2, side="left"))
    prev2 = csum2[k-1] if k > 0 else 0.0
    frac2 = (half2 - prev2) / bins2[k] if bins2[k] > 0 else 0.0
    fprev2 = f_c[k-1] if k > 0 else f_c[0]
    return float(fprev2 + frac2 * df_c)

def _read_image(path: str):
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception:
        from PIL import Image
        return np.array(Image.open(path))

if __name__ == "__main__":
    # root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    # exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    # files = sorted(p for pat in exts for p in glob.glob(os.path.join(root, pat)))
    # if not files:
        # print(f"目录无图片: {root}"); sys.exit(0)
    for f in ['/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/preview_imgs/sample_5_d0.478_b4.png']:
        try:
            img = _read_image(f)
            val = image_mdf(img, fs_row=1.0, fs_col=1.0)
            print(f"{os.path.basename(f)}\tMDF={val:.6f}")
        except Exception as e:
            print(f"{os.path.basename(f)}\t错误: {e}")
    for f in ['preview_imgs/sample_2_d0.115_b0.png']:
        try:
            img = _read_image(f)
            val = image_mdf(img, fs_row=1.0, fs_col=1.0)
            print(f"{os.path.basename(f)}\tMDF={val:.6f}")
        except Exception as e:
            print(f"{os.path.basename(f)}\t错误: {e}")
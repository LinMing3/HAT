import numpy as np

def compute_glcm(image: np.ndarray,
                 levels: int = 32,
                 offsets: tuple = ((0, 1), (-1, 1), (-1, 0), (-1, -1)),
                 symmetric: bool = True,
                 normalized: bool = True) -> np.ndarray:
    """
    计算图像的灰度共生矩阵（GLCM）
    参数:
      - image: HxW 或 HxWxC 图像（RGB 会转灰度）
      - levels: 量化的灰度级数 n
      - offsets: 偏移(dy, dx)元组列表，例如(0,1),( -1,1 ),(-1,0),(-1,-1)
      - symmetric: 若为 True，则计数后与其转置相加，得到对称 GLCM
      - normalized: 若为 True，对每个偏移的矩阵做概率归一化（总和=1）
    返回:
      - glcm: 形状 (levels, levels, K) 的矩阵，K 为偏移数量
    说明:
      - 先将图像转为灰度并量化到 [0, levels-1] 的整数，再按偏移统计 (i,j) 共现次数
    """
    x = np.asarray(image)
    # 转灰度
    if x.ndim == 3:
        if x.shape[-1] == 4:
            x = x[..., :3]
        x = x.astype(np.float32)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        x = x.astype(np.float32)

    # 归一与量化到 levels 个灰度
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = np.zeros_like(x, dtype=np.float32)
    q = np.floor(x_norm * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    H, W = q.shape
    K = len(offsets)
    glcm = np.zeros((levels, levels, K), dtype=np.float64)

    for k, (dy, dx) in enumerate(offsets):
        # 计算有效配对区域索引
        if dy >= 0:
            y0 = slice(0, H - dy)
            y1 = slice(dy, H)
        else:
            y0 = slice(-dy, H)
            y1 = slice(0, H + dy)
        if dx >= 0:
            x0 = slice(0, W - dx)
            x1 = slice(dx, W)
        else:
            x0 = slice(-dx, W)
            x1 = slice(0, W + dx)

        a = q[y0, x0].ravel()
        b = q[y1, x1].ravel()

        # 使用一维直方图累加到二维矩阵
        idx = a * levels + b
        counts = np.bincount(idx, minlength=levels * levels)
        M = counts.reshape(levels, levels)

        if symmetric:
            M = M + M.T

        if normalized:
            s = M.sum()
            if s > 0:
                M = M / s

        glcm[:, :, k] = M

    return glcm

def glcm_entropy(image: np.ndarray,
                 levels: int = 32,
                 offsets: tuple = ((0, 1), (-1, 1), (-1, 0), (-1, -1)),
                 symmetric: bool = True,
                 normalized: bool = True,
                 reduction: str = "mean",
                 eps: float = 1e-12):
    """
    计算 GLCM 熵 H_g = -∑_i ∑_j p(i,j) log p(i,j)
    参数:
      - image: 输入图像
      - levels, offsets, symmetric, normalized: 与 compute_glcm 同
      - reduction: 'mean' 对各偏移的熵取平均并返回标量；
                   'sum' 对各偏移求和并返回标量；
                   'none' 返回每个偏移的熵数组，长度为偏移数
      - eps: 避免 log(0) 的微小常数
    返回:
      - 标量（mean/sum）或长度为K的一维数组（none）
    """
    # 计算每个偏移的 GLCM（最后一维为偏移数 K）
    glcm = compute_glcm(image, levels=levels, offsets=offsets,
                        symmetric=symmetric, normalized=normalized)

    P = glcm.astype(np.float64)
    # 若未归一化，则对每个偏移的矩阵单独归一化
    if not normalized:
        s = P.sum(axis=(0, 1), keepdims=True)
        P = np.where(s > 0, P / s, 0.0)

    # 对每个偏移计算熵：-∑ p log p
    H = -np.sum(P * np.log(P + eps), axis=(0, 1))  # 形状: (K,)

    if reduction == "none":
        return H
    if reduction == "sum":
        return float(np.sum(H))
    return float(np.mean(H))


def _read_image(path: str):
    # 简单读图（imageio 优先，失败则用 PIL）
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception:
        from PIL import Image
        import numpy as np
        return np.array(Image.open(path))

if __name__ == "__main__":
    # import os, sys, glob

    # # 目标目录：命令行传入，否则为当前工作目录
    # root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    # # 收集常见图片扩展名
    # exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    # files = []
    # for pat in exts:
    #     files.extend(glob.glob(os.path.join(root, pat)))
    # files = sorted(files)

    # if not files:
    #     print(f"目录无图片: {root}")
    #     sys.exit(0)

    for f in ['/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/preview_imgs/sample_5_d0.478_b4.png']:
        try:
            img = _read_image(f)
            H = glcm_entropy(img, levels=32, offsets=((0,1),(-1,1),(-1,0),(-1,-1)),
                             symmetric=True, normalized=True, reduction="mean")
            print(f"GLCM_Entropy={H:.6f}")
        except Exception as e:
            print(f"{os.path.basename(f)}\t错误: {e}")

    for f in ['/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/preview_imgs/sample_10_d0.507_b4.png']:
        try:
            img = _read_image(f)
            H = glcm_entropy(img, levels=32, offsets=((0,1),(-1,1),(-1,0),(-1,-1)),
                             symmetric=True, normalized=True, reduction="mean")
            print(f"GLCM_Entropy={H:.6f}")
        except Exception as e:
            print(f"{os.path.basename(f)}\t错误: {e}")
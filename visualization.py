#!/usr/bin/env python3
"""
view_nii_all_slices.py

一次性可视化 NIfTI (.nii/.nii.gz) 文件的所有轴向切片（Axial）为一个 mosaic 图像，
并将结果保存为 PNG。也会同时弹出一个 matplotlib 窗口显示结果。

用法（直接运行即可）：
    python view_nii_all_slices.py

可选参数（命令行）：
    --max-cols  最大列数（默认 16）
    --downsample  下采样因子，用于缩小每个切片（整数，默认 1，不缩放）
    --no-save    不保存 PNG（只显示）
    --cmap       颜色映射（默认 gray）

注意：若切片数量非常大，生成的大图会消耗大量内存/显存，可通过 --downsample 或减小 --max-cols 解决。
"""
import os
import math
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ========== 这里替换为你的文件路径（已使用你提供的路径） ==========
NII_PATH = "/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz"
# ======================================================================

def normalize_image(img, pmin=1, pmax=99):
    """对整个 volume 使用全局百分位归一化，返回 [0,1] 浮点数组。"""
    flat = img.ravel()
    vmin = np.percentile(flat, pmin)
    vmax = np.percentile(flat, pmax)
    if vmax <= vmin:
        # 防止除零
        vmax = vmin + 1e-6
    norm = (img - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    return norm

def make_mosaic(slices, cols):
    """把一组 2D 切片拼成 mosaic（numpy array）。
       slices: list 或 array of 2D arrays (H, W)
       cols: 列数
       返回 (H*rows, W*cols) 的大图
    """
    n = len(slices)
    rows = math.ceil(n / cols)
    h, w = slices[0].shape
    mosaic = np.zeros((rows * h, cols * w), dtype=slices[0].dtype)
    for i, sl in enumerate(slices):
        r = i // cols
        c = i % cols
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = sl
    return mosaic, rows

def main():
    parser = argparse.ArgumentParser(description="一次性可视化 NIfTI 的所有轴向切片为 mosaic")
    parser.add_argument("--nii", type=str, default=NII_PATH, help="nii.gz 文件路径")
    parser.add_argument("--max-cols", type=int, default=16, help="mosaic 最大列数（默认16）")
    parser.add_argument("--downsample", type=int, default=1, help="每个切片下采样因子（整数，默认1）")
    parser.add_argument("--no-save", action="store_true", help="只显示，不保存 PNG")
    parser.add_argument("--cmap", type=str, default="gray", help="matplotlib colormap，默认 gray")
    args = parser.parse_args()

    nii_path = args.nii
    if not os.path.exists(nii_path):
        print(f"[错误] 找不到文件: {nii_path}")
        return

    print(f"Loading: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    if data.ndim == 4:
        print("[Info] 4D 数据，使用第一个时间点（index 0）。")
        data = data[..., 0]
    data = np.asarray(data)
    nx, ny, nz = data.shape
    print(f"Volume shape: (x={nx}, y={ny}, z={nz})  -> axis: axial dim = {nz} slices")

    # 取轴向切片（z 方向），可以对每张切片做旋转以便更直观显示
    slices = []
    # 先做全局归一化，保证灰度一致
    data_norm = normalize_image(data, pmin=1, pmax=99)

    ds = max(1, int(args.downsample))
    for z in range(nz):
        sl = data_norm[:, :, z]
        # 选用 rot90 使显示方向更符合常见习惯（可根据需要更改或删掉）
        sl = np.rot90(sl)
        if ds > 1:
            # 下采样（最近邻，简单快速）
            sl = sl[::ds, ::ds]
        slices.append(sl)

    n_slices = len(slices)
    # 自动选列数：优先使用 max-cols，按接近正方形布局选择实际列数
    ideal_cols = int(math.ceil(math.sqrt(n_slices)))
    cols = min(args.max_cols, max(1, ideal_cols))
    # 如果 slices 比 cols 少，调整
    cols = min(cols, n_slices)
    mosaic, rows = make_mosaic(slices, cols)
    print(f"Creating mosaic: slices={n_slices}, cols={cols}, rows={rows}, mosaic shape={mosaic.shape}")

    # 绘图显示并保存
    fig_w = cols * 1.5
    fig_h = rows * 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.imshow(mosaic, cmap=args.cmap, interpolation='nearest')
    ax.axis('off')
    ax.set_title(os.path.basename(nii_path) + f"  slices={n_slices}  cols={cols} rows={rows}")

    plt.tight_layout()
    out_png = os.path.splitext(nii_path)[0] + "_all_slices.png"
    if args.no_save:
        print("[Info] --no-save 指定：不保存 PNG，仅显示。")
    else:
        # 保存为 PNG，dpi 取 100，可根据需要调整
        print(f"Saving mosaic to: {out_png}")
        fig.savefig(out_png, bbox_inches='tight', dpi=100)
        print("Saved.")

    print("Showing figure window (关闭窗口以结束脚本)。")
    plt.show()


if __name__ == "__main__":
    main()

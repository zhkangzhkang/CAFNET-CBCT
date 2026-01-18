import os
import glob
import nibabel as nib
import numpy as np

# 修改为你的标签路径
label_dir = r"D:/BaiduNetdiskDownload/3D_CBCT/Train-Labeled/labels"

nii_files = glob.glob(os.path.join(label_dir, "*.nii*"))
print(f"找到 {len(nii_files)} 个标签文件")

all_classes = set()

for fp in nii_files:
    arr = nib.load(fp).get_fdata()
    u = np.unique(arr.astype(np.int32))
    print(f"{os.path.basename(fp)}: {u}")
    all_classes.update(u.tolist())

print("\n数据集中所有类别值：", sorted(all_classes))
print("最大类别ID:", max(all_classes))
print("建议 --num_classes =", max(all_classes) + 1, "(包含背景0)")

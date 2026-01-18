import nibabel as nib
import numpy as np

label_path = "G:/githubSource/3DCBCT/MICCAI2024/test_label/STS2024_Test_Labeled_0001_Mask.nii.gz"

lbl_nii = nib.load(label_path)
lbl = lbl_nii.get_fdata()

print("shape:", lbl.shape)
print("dtype:", lbl.dtype)
print("min:", lbl.min())
print("max:", lbl.max())
print("unique values:", np.unique(lbl))


import nibabel as nib
import numpy as np
import os

# 原始 mask 路径（你现在用的那个）
in_path = r"G:/githubSource/3DCBCT/MICCAI2024/test_label/STS2024_Test_Labeled_0001_Mask.nii.gz"

# 输出一个专门给 Slicer 用的新 mask
out_path = r"G:/githubSource/3DCBCT/MICCAI2024/test_label_fixed/STS2024_Test_Labeled_0001_Mask_slicer_uint8.nii.gz"

nii = nib.load(in_path)
data = nii.get_fdata()  # float64, 带 0/11/12/...48

print("原始 dtype:", data.dtype)
print("原始 unique:", np.unique(data))

# 1）先四舍五入到最近的整数（防止浮点误差，比如 10.999999）
data_rounded = np.rint(data)

# 2）转为 uint8（0~255，足够容纳你的 0~48）
data_uint8 = data_rounded.astype(np.uint8)

print("修正后 dtype:", data_uint8.dtype)
print("修正后 unique:", np.unique(data_uint8))

# 3）保持 affine 和 header（除了数据类型之外）
new_nii = nib.Nifti1Image(data_uint8, nii.affine, nii.header)
# 建议顺手把 header 里的数据类型也改掉（可选，很多情况下不改也没事）
new_nii.set_data_dtype(np.uint8)

nib.save(new_nii, out_path)
print("✅ 已保存为:", out_path)

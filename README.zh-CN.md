# CAFNet-CBCT

<p align="center">
  <b>Language:</b>
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a>
</p>

CAFNet-CBCT 是一个面向 3D CBCT 牙齿逐牙实例分割的半监督学习项目。当前主线方法已经从普通 Mean Teacher 扩展为：

1. **Anatomy-Aware CAF-Net Backbone**：使用 DSCA 编码模块和 SFG3D 层级解码结构，同时建模局部牙齿边界、跨切片连续性和牙弓级上下文。
2. **Reliability-Calibrated Complementary Dual-Teacher Pseudo-Labeling**：使用两个互补 EMA teacher 生成伪标签，并用置信度和 teacher disagreement 共同筛选可靠区域。
3. **Instance-Boundary-Aware Consistency**：在可靠区域内约束 student 预测与融合伪标签的 soft boundary 一致，重点减少相邻牙粘连、根尖缺失和 merge/split errors。

当前推荐主入口：

```bash
python train_cafnet_cbct.py
```

推荐单病例推理和可视化入口：

```bash
python infer_cafnet_case.py
```

所有推荐脚本均已改成英文文件名，不再使用中文、括号或 `+`。

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 项目结构](#2-项目结构)
- [3. 数据集组织方式](#3-数据集组织方式)
- [4. 环境安装](#4-环境安装)
- [5. 训练完整方法](#5-训练完整方法)
- [6. 日志怎么看](#6-日志怎么看)
- [7. 输出文件说明](#7-输出文件说明)
- [8. 验证、测试和推理](#8-验证测试和推理)
- [9. 可视化和论文出图](#9-可视化和论文出图)
- [10. 关键参数解释](#10-关键参数解释)
- [11. 推荐消融实验](#11-推荐消融实验)
- [12. 常见问题](#12-常见问题)
- [13. 文件命名和旧脚本说明](#13-文件命名和旧脚本说明)

## 1. 快速开始

如果数据已经放在 `./data`，GPU 环境也已经配置好，可以直接运行完整方法：

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/cafnet_full" \
  --epochs 110 \
  --val_every 10 \
  --num_classes 49 \
  --roi 160 160 160 \
  --batch_l 1 \
  --batch_u 1 \
  --consist_w 4.0 \
  --boundary_w 0.5 \
  --consist_ramp 60 \
  --tau_conf 0.6 \
  --tau_disagree 0.05 \
  --ema_global 0.995 \
  --ema_detail 0.99 \
  --workers 4 \
  --amp
```

Windows PowerShell 可以使用单行命令，或把 Bash 的 `\` 换成 PowerShell 的反引号：

```powershell
python train_cafnet_cbct.py `
  --root ".\data" `
  --outdir "runs\cafnet_full" `
  --epochs 110 `
  --val_every 10 `
  --roi 160 160 160 `
  --batch_l 1 `
  --batch_u 1 `
  --workers 4 `
  --amp
```

训练结束后，主要结果会在：

```text
runs/cafnet_full/
  best_student.pt
  best_teacher_global.pt
  best_teacher_detail.pt
  predTs/
    *_pred.nii.gz
```

## 2. 项目结构

```text
CAFNet-CBCT/
  data/
    Train-Labeled/
      Images/ or images/
      Masks/  or labels/
    Train-Unlabeled/
    dental_CBCT_test_set/
      images/
      labels/
    Validation-Public/

  networks/
    simam_hier_unet.py          # CAFNet3D backbone: DSCA + SFG3D hierarchical decoder
    cafnet_ssl.py               # dual teacher, reliable mask, KL consistency, boundary loss
    __init__.py

  train_cafnet_cbct.py          # 推荐主训练入口：训练 + 验证 + 批量测试推理
  infer_cafnet_case.py          # 单病例 checkpoint 推理和 GT-vs-pred 可视化
  viz_cbct_compare.py           # 高分辨率论文对比图
  visualization.py              # NIfTI 切片 mosaic 可视化
  check_labels.py               # 标签类别检查
  mask.py                       # mask 辅助处理脚本

  train1023.py                  # 旧版 Mean Teacher/UNet 训练脚本
  train_unet_val5_debug.py      # 旧版调试脚本，不是当前主线
  train_unet_simam.py           # 旧版 SimAM 消融脚本
  train_unet_simam_eca_hierarchical.py
  training_strategy_and_methods.docx
```

当前主线只需要关注：

```text
train_cafnet_cbct.py
infer_cafnet_case.py
networks/simam_hier_unet.py
networks/cafnet_ssl.py
README.md
```

## 3. 数据集组织方式

训练脚本默认读取一个 `root` 目录。`root` 可以是 `./data`，也可以是服务器上的数据路径，例如 `/root/autodl-tmp/3D_CBCT`。

推荐结构：

```text
data/
  Train-Labeled/
    Images/
      STS24_Train_Labeled_0001.nii.gz
      STS24_Train_Labeled_0002.nii.gz
      ...
    Masks/
      STS24_Train_Labeled_0001_Mask.nii.gz
      STS24_Train_Labeled_0002_Mask.nii.gz
      ...

  Train-Unlabeled/
    STS24_Train_Unlabeled_0001.nii.gz
    STS24_Train_Unlabeled_0002.nii.gz
    ...

  dental_CBCT_test_set/
    images/
      STS2024_Test_Labeled_0001.nii.gz
      ...
    labels/
      STS2024_Test_Labeled_0001_Mask.nii.gz
      ...

  Validation-Public/
    STS24_Validation_0001.nii.gz
    ...
```

兼容写法：

```text
Train-Labeled/Images + Train-Labeled/Masks
Train-Labeled/images + Train-Labeled/labels
```

标签约定：

```text
0      background
1-48   tooth-wise anatomical instance labels
```

因此默认 `--num_classes 49`。

## 4. 环境安装

推荐环境：

```text
Python 3.10 或 3.11
CUDA GPU
PyTorch
MONAI
nibabel
matplotlib
scikit-image
```

创建 conda 环境：

```bash
conda create -n cafnet-cbct python=3.10 -y
conda activate cafnet-cbct
```

安装 PyTorch。建议到 PyTorch 官方安装页选择与你机器匹配的 CUDA 版本：

```text
https://pytorch.org/get-started/locally/
```

常见 pip 示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果官方页面推荐的是 CUDA 12.6、CUDA 12.4 或 CPU 版本，请优先使用官方页面生成的命令。

安装项目依赖：

```bash
pip install monai nibabel numpy scipy scikit-image matplotlib tqdm
```

检查 GPU 和 Python 包：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import monai, nibabel; print(monai.__version__); print('nibabel ok')"
```

检查项目代码是否能被 Python 解析：

```bash
python -m py_compile networks/simam_hier_unet.py networks/cafnet_ssl.py train_cafnet_cbct.py infer_cafnet_case.py
```

如果 `torch.cuda.is_available()` 输出 `False`，说明当前 PyTorch 没有正确识别 CUDA。3D CBCT 训练强烈建议使用 GPU。

## 5. 训练完整方法

### 5.1 本地数据路径

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/cafnet_full" \
  --epochs 110 \
  --val_every 10 \
  --num_classes 49 \
  --roi 160 160 160 \
  --batch_l 1 \
  --batch_u 1 \
  --workers 4 \
  --amp
```

### 5.2 AutoDL 或 Linux 服务器路径

```bash
python train_cafnet_cbct.py \
  --root "/root/autodl-tmp/3D_CBCT" \
  --outdir "/root/autodl-tmp/CBCT/runs_cafnet_full" \
  --epochs 110 \
  --val_every 10 \
  --num_classes 49 \
  --roi 160 160 160 \
  --batch_l 1 \
  --batch_u 1 \
  --workers 4 \
  --amp
```

### 5.3 显存不够时的轻量训练命令

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/cafnet_full_roi128" \
  --epochs 110 \
  --val_every 10 \
  --roi 128 128 128 \
  --batch_l 1 \
  --batch_u 1 \
  --sw_batch 1 \
  --val_sw_batch 1 \
  --workers 0 \
  --amp
```

### 5.4 快速 smoke test

如果只是想确认数据读取和模型前向能跑，可以先跑 1 个 epoch：

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/smoke_test" \
  --epochs 1 \
  --val_every 1 \
  --roi 128 128 128 \
  --batch_l 1 \
  --batch_u 1 \
  --workers 0
```

注意：这不是正式训练，只用于检查环境、路径、数据尺寸和代码逻辑。

## 6. 日志怎么看

训练过程中会周期性打印类似日志：

```text
[epoch 005] iter 10 | sup 1.2345 | con 0.0321 | bd 0.0188 | mask 0.126 | conf 0.742 | js 0.0142 | w 0.820/0.103 | 3.42s/it
```

字段解释：

| 字段 | 含义 | 观察建议 |
|---|---|---|
| `sup` | labeled 数据上的 DiceCE 监督损失 | 应整体下降 |
| `con` | 可靠区域上的 KL consistency | 太大可能说明伪标签和 student 差距大 |
| `bd` | instance-boundary consistency | 关注牙齿边界一致性 |
| `mask` | reliable region mask 占比 | 长期为 0 说明筛选太严格 |
| `conf` | 融合 teacher 伪标签平均置信度 | 越高通常越稳定 |
| `js` | Teacher-A 和 Teacher-B 的平均 JS disagreement | 越高说明两个 teacher 分歧越大 |
| `w` | 当前 consistency/boundary ramp-up 权重 | 前期小，随后逐渐增大 |

经验判断：

- `mask` 长期接近 `0.000`：降低 `--tau_conf` 或提高 `--tau_disagree`。
- `js` 很高且预测不稳定：降低 `--tau_disagree`，让可靠区域更严格。
- `bd` 很大且可视化边界粗糙：可以适当提高 `--boundary_w`。
- 初期 loss 波动大：增大 `--consist_ramp` 或降低 `--consist_w`。

## 7. 输出文件说明

训练输出目录示例：

```text
runs/cafnet_full/
  best_student.pt
  best_teacher_global.pt
  best_teacher_detail.pt
  predTs/
    STS2024_Test_Labeled_0001_pred.nii.gz
    ...
```

文件含义：

| 文件 | 含义 |
|---|---|
| `best_student.pt` | 验证集 Dice 最优时的 student 模型 |
| `best_teacher_global.pt` | 同一最优 epoch 的 Teacher-A/global-context EMA 模型 |
| `best_teacher_detail.pt` | 同一最优 epoch 的 Teacher-B/local-detail EMA 模型 |
| `predTs/` | 训练结束后自动生成的测试集预测 NIfTI |

训练结束后的批量测试推理优先加载：

```text
best_teacher_global.pt
best_teacher_detail.pt
```

如果这两个 teacher checkpoint 不存在，但存在 `best_student.pt`，脚本会退回用 `best_student.pt` 初始化两个 teacher 做推理。

## 8. 验证、测试和推理

### 8.1 训练过程中的验证

如果存在：

```text
dental_CBCT_test_set/images/
dental_CBCT_test_set/labels/
```

训练脚本会每隔 `--val_every` 个 epoch 自动验证一次，并打印：

```text
val Dice
val HD95
```

### 8.2 训练结束后的批量测试推理

`train_cafnet_cbct.py` 训练结束后会自动对 `dental_CBCT_test_set` 进行滑窗推理，预测结果保存到：

```text
<outdir>/predTs/
```

当前批量推理使用双 teacher ensemble：

```text
0.5 * softmax(Teacher-A(global-context view))
+ 0.5 * softmax(Teacher-B(local-detail view))
```

这与训练阶段的 complementary dual-teacher pseudo-labeling 保持一致。

### 8.3 只重新生成 predTs

如果已经训练完成，只想重新用已有 checkpoint 生成 `predTs/`，可以使用 `--epochs 0`：

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/cafnet_full" \
  --epochs 0 \
  --roi 160 160 160 \
  --sw_batch 1 \
  --workers 0 \
  --amp
```

注意：当前脚本即使 `--epochs 0`，仍会做一次数据和模型 sanity check，因此仍需要 `Train-Labeled/` 和 `Train-Unlabeled/` 存在。

### 8.4 单病例 checkpoint 推理和可视化

使用 `best_student.pt` 对单病例推理并生成 GT-vs-pred 图：

```bash
python infer_cafnet_case.py \
  --ckpt "runs/cafnet_full/best_student.pt" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz" \
  --num_classes 49 \
  --roi 160 160 160 \
  --out "case0001_compare.png" \
  --dpi 600 \
  --amp
```

如果已经有 `predTs` 里的预测文件，可以直接可视化：

```bash
python infer_cafnet_case.py \
  --pred "runs/cafnet_full/predTs/STS2024_Test_Labeled_0001_pred.nii.gz" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz" \
  --out "case0001_predTs_compare.png" \
  --dpi 600
```

## 9. 可视化和论文出图

### 9.1 高 DPI GT-vs-pred 对比图

```bash
python viz_cbct_compare.py \
  --ckpt "runs/cafnet_full/best_student.pt" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0050_Mask.nii.gz" \
  --num_classes 49 \
  --roi 160 160 160 \
  --out "viz_compare50.png" \
  --dpi 600 \
  --amp
```

### 9.2 只显示指定牙齿 ID

```bash
python viz_cbct_compare.py \
  --pred "runs/cafnet_full/predTs/STS2024_Test_Labeled_0050_pred.nii.gz" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0050_Mask.nii.gz" \
  --classes 16 23 37 \
  --class_names Tooth16 Tooth23 Tooth37 \
  --out "viz_selected_teeth.png" \
  --dpi 600
```

### 9.3 NIfTI 切片 mosaic

如果只想快速查看一个 NIfTI 体数据，可以使用：

```bash
python visualization.py --help
```

然后按脚本帮助传入图像或 mask 文件。

## 10. 关键参数解释

### 10.1 数据和训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--root` | `/root/autodl-tmp/3D_CBCT` | 数据集根目录 |
| `--epochs` | `110` | 训练 epoch 数 |
| `--val_every` | `10` | 每隔多少 epoch 验证一次 |
| `--num_classes` | `49` | 类别数，背景 0 + 48 个牙齿 ID |
| `--roi` | `160 160 160` | 随机 3D crop 大小 |
| `--batch_l` | `1` | labeled batch size |
| `--batch_u` | `1` | unlabeled batch size |
| `--workers` | `4` | DataLoader worker 数 |
| `--amp` | 关闭 | 开启 CUDA mixed precision |

### 10.2 半监督损失参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--consist_w` | `4.0` | KL consistency 最大权重 |
| `--boundary_w` | `0.5` | boundary consistency 最大权重 |
| `--consist_ramp` | `60` | 无监督损失 ramp-up epoch |
| `--tau_conf` | `0.6` | reliable mask 的置信度阈值 |
| `--tau_disagree` | `0.05` | Teacher-A/B JS disagreement 阈值 |
| `--temperature` | `0.7` | KL consistency 温度系数 |
| `--boundary_focus` | `2.0` | 对伪标签边界区域的额外加权 |

### 10.3 双 teacher view 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--ema_global` | `0.995` | Teacher-A EMA 更新率 |
| `--ema_detail` | `0.99` | Teacher-B EMA 更新率 |
| `--global_kernel` | `5` | Teacher-A 平滑视图 kernel |
| `--global_mix` | `0.35` | Teacher-A global-context view 混合比例 |
| `--detail_kernel` | `3` | Teacher-B local-detail view kernel |
| `--detail_gain` | `0.5` | Teacher-B unsharp/detail 增强强度 |
| `--student_noise_std` | `0.03` | Student strong view 高斯噪声 |
| `--student_gamma` | `0.7 1.3` | Student strong view gamma 范围 |

## 11. 消融实验

为了验证模型有效性，需要完成以下实验：

| 实验 | 目的 |
|---|---|
| Baseline UNet | 证明基础网络性能 |
| CAF-Net backbone only | 验证 DSCA + SFG3D 主干贡献 |
| CAF-Net + single Mean Teacher | 对比普通半监督框架 |
| CAF-Net + dual teacher | 验证互补 teacher 的价值 |
| CAF-Net + dual teacher + reliability mask | 验证置信度和 disagreement 校准 |
| Full method | 验证 boundary consistency 的最终增益 |

## 12. 常见问题

### 12.1 CUDA out of memory

优先尝试：

```bash
--roi 128 128 128 --batch_l 1 --batch_u 1 --sw_batch 1 --val_sw_batch 1 --amp
```

如果仍然 OOM：

```bash
--workers 0
```

显存足够时再逐步把 `--roi` 调回 `160 160 160`。

### 12.2 reliable mask 一直为 0

日志中如果 `mask 0.000` 持续很多 iteration，说明伪标签筛选过严：

```bash
--tau_conf 0.5 --tau_disagree 0.08
```

### 12.3 伪标签噪声较多

如果可视化看到明显错误伪标签传播，可以更严格：

```bash
--tau_conf 0.7 --tau_disagree 0.03
```

### 12.4 边界粘连仍然明显

可以适当增加 boundary consistency：

```bash
--boundary_w 1.0
```

如果训练不稳定，配合增大 ramp-up：

```bash
--consist_ramp 80
```

### 12.5 验证很慢

减少验证频率：

```bash
--val_every 20
```

降低滑窗 batch：

```bash
--val_sw_batch 1
```

### 12.6 Windows 和 Linux 命令换行

Bash/Linux 使用：

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --amp
```

PowerShell 使用：

```powershell
python train_cafnet_cbct.py `
  --root ".\data" `
  --amp
```

也可以直接写成一行：

```powershell
python train_cafnet_cbct.py --root ".\data" --outdir "runs\cafnet_full" --epochs 110 --amp
```

## 13. 文件命名和旧脚本说明

当前推荐文件命名已经整理为英文：

| 当前文件 | 用途 |
|---|---|
| `train_cafnet_cbct.py` | 当前主训练脚本 |
| `infer_cafnet_case.py` | 单病例推理和可视化 |
| `train_unet_val5_debug.py` | 旧调试训练脚本 |
| `train_unet_simam.py` | 旧 SimAM 消融脚本 |
| `train_unet_simam_eca_hierarchical.py` | 旧 SimAM + ECA + hierarchical 脚本 |
| `training_strategy_and_methods.docx` | 方法设计文档 |

旧脚本不会被当前主流程自动调用。正式训练和论文主结果建议使用：

```bash
python train_cafnet_cbct.py
```

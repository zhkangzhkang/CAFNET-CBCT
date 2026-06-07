# CAFNet-CBCT

[English](README.md) | [Chinese](README.zh-CN.md)

CAFNet-CBCT is a semi-supervised 3D CBCT tooth-wise anatomical instance segmentation project. The current main pipeline implements an anatomy-aware complementary-teacher framework for sparse-label CBCT tooth instance segmentation.

## Highlights

1. **Anatomy-Aware CAF-Net Backbone**  
   CAF-Net uses DSCA encoder blocks and an SFG3D hierarchical decoder to capture local tooth-boundary cues, cross-slice structural continuity, and dental-arch-level context.

2. **Reliability-Calibrated Complementary Dual-Teacher Pseudo-Labeling**  
   Two EMA teachers provide complementary pseudo-label predictions. Teacher-A focuses on global-context views, while Teacher-B focuses on local-detail views. Their fused confidence and inter-teacher disagreement are used to select reliable pseudo-label regions.

3. **Instance-Boundary-Aware Consistency**  
   A reliability-weighted soft-boundary consistency objective encourages the student prediction to preserve instance boundaries, reducing adjacent-tooth adhesion and merge/split errors.

Recommended training entry:

```bash
python train_cafnet_cbct.py
```

Recommended single-case inference and visualization entry:

```bash
python infer_cafnet_case.py
```

All recommended script names are English-only and avoid Chinese characters, parentheses, and shell-sensitive symbols such as `+`.

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Repository Structure](#2-repository-structure)
- [3. Dataset Layout](#3-dataset-layout)
- [4. Environment Setup](#4-environment-setup)
- [5. Training](#5-training)
- [6. Training Logs](#6-training-logs)
- [7. Output Files](#7-output-files)
- [8. Validation, Testing, and Inference](#8-validation-testing-and-inference)
- [9. Visualization and Paper Figures](#9-visualization-and-paper-figures)
- [10. Key Arguments](#10-key-arguments)
- [11. Recommended Ablation Study](#11-recommended-ablation-study)
- [12. Troubleshooting](#12-troubleshooting)
- [13. File Naming and Legacy Scripts](#13-file-naming-and-legacy-scripts)
- [14. Bilingual README on GitHub](#14-bilingual-readme-on-github)

## 1. Quick Start

If the dataset has already been placed under `./data` and the GPU environment is ready, run the full method with:

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

For Windows PowerShell, use backticks for multi-line commands:

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

Main outputs will be saved to:

```text
runs/cafnet_full/
  best_student.pt
  best_teacher_global.pt
  best_teacher_detail.pt
  predTs/
    *_pred.nii.gz
```

## 2. Repository Structure

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
    simam_hier_unet.py              # CAFNet3D backbone: DSCA + SFG3D decoder
    cafnet_ssl.py                   # dual teacher, reliable mask, KL loss, boundary loss
    __init__.py

  train_cafnet_cbct.py              # main training, validation, and batch test inference
  infer_cafnet_case.py              # single-case inference and GT-vs-pred visualization
  viz_cbct_compare.py               # high-resolution qualitative figures
  visualization.py                  # NIfTI slice mosaic visualization
  check_labels.py                   # label class inspection
  mask.py                           # mask utility script

  train1023.py                      # legacy Mean Teacher / UNet script
  train_unet_val5_debug.py          # legacy debug script
  train_unet_simam.py               # legacy SimAM ablation script
  train_unet_simam_eca_hierarchical.py
  training_strategy_and_methods.docx
```

For the current main method, focus on:

```text
train_cafnet_cbct.py
infer_cafnet_case.py
networks/simam_hier_unet.py
networks/cafnet_ssl.py
README.md
README.zh-CN.md
```

## 3. Dataset Layout

The training script reads a dataset root specified by `--root`. This root can be `./data` locally or a server path such as `/root/autodl-tmp/3D_CBCT`.

Recommended layout:

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

Supported labeled-data folder names:

```text
Train-Labeled/Images + Train-Labeled/Masks
Train-Labeled/images + Train-Labeled/labels
```

Label convention:

```text
0      background
1-48   tooth-wise anatomical instance labels
```

Therefore, the default class number is:

```bash
--num_classes 49
```

## 4. Environment Setup

Recommended software:

```text
Python 3.10 or 3.11
CUDA-capable GPU
PyTorch
MONAI
nibabel
matplotlib
scikit-image
```

Create a conda environment:

```bash
conda create -n cafnet-cbct python=3.10 -y
conda activate cafnet-cbct
```

Install PyTorch. Choose the command that matches your CUDA driver from the official PyTorch installation page:

```text
https://pytorch.org/get-started/locally/
```

Common CUDA 12.1 pip example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If the official selector recommends CUDA 12.6, CUDA 12.4, or CPU wheels for your machine, use the selector-generated command instead.

Install the remaining dependencies:

```bash
pip install monai nibabel numpy scipy scikit-image matplotlib tqdm
```

Check CUDA and Python packages:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import monai, nibabel; print(monai.__version__); print('nibabel ok')"
```

Check that the project scripts are syntactically valid:

```bash
python -m py_compile networks/simam_hier_unet.py networks/cafnet_ssl.py train_cafnet_cbct.py infer_cafnet_case.py
```

If `torch.cuda.is_available()` prints `False`, the current PyTorch installation cannot use CUDA. CPU execution is possible for simple checks, but full 3D CBCT training should be performed on a GPU.

## 5. Training

### 5.1 Local Dataset

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

### 5.2 AutoDL or Linux Server

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

### 5.3 Low-Memory Training

If GPU memory is limited, start with:

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

### 5.4 Smoke Test

To verify data loading, preprocessing, and model forward pass, run a 1-epoch smoke test:

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

This is only for debugging. It is not a meaningful training run.

## 6. Training Logs

A typical training log line looks like:

```text
[epoch 005] iter 10 | sup 1.2345 | con 0.0321 | bd 0.0188 | mask 0.126 | conf 0.742 | js 0.0142 | w 0.820/0.103 | 3.42s/it
```

| Field | Meaning | How to interpret it |
|---|---|---|
| `sup` | Supervised DiceCE loss on labeled data | Should generally decrease |
| `con` | KL consistency on reliable pseudo-label regions | Very large values may indicate student-teacher mismatch |
| `bd` | Instance-boundary consistency | Reflects boundary-level agreement |
| `mask` | Ratio of reliable pseudo-label voxels | Long-term zero means filtering is too strict |
| `conf` | Mean fused teacher confidence | Higher usually indicates more stable pseudo-labels |
| `js` | Mean JS disagreement between Teacher-A and Teacher-B | Higher values indicate stronger teacher disagreement |
| `w` | Current ramp-up weights for consistency and boundary terms | Starts small and increases during training |

Useful rules:

- If `mask` stays close to `0.000`, lower `--tau_conf` or increase `--tau_disagree`.
- If `js` is high and predictions are unstable, decrease `--tau_disagree`.
- If boundaries remain rough, increase `--boundary_w`.
- If early training is unstable, increase `--consist_ramp` or reduce `--consist_w`.

## 7. Output Files

Example output directory:

```text
runs/cafnet_full/
  best_student.pt
  best_teacher_global.pt
  best_teacher_detail.pt
  predTs/
    STS2024_Test_Labeled_0001_pred.nii.gz
    ...
```

| File | Description |
|---|---|
| `best_student.pt` | Student model from the epoch with the best validation Dice |
| `best_teacher_global.pt` | Teacher-A/global-context EMA model from the same best epoch |
| `best_teacher_detail.pt` | Teacher-B/local-detail EMA model from the same best epoch |
| `predTs/` | NIfTI predictions automatically generated after training |

The final batch inference stage first tries to load:

```text
best_teacher_global.pt
best_teacher_detail.pt
```

If these teacher checkpoints are unavailable but `best_student.pt` exists, the script falls back by copying the best student weights into both teachers for inference.

## 8. Validation, Testing, and Inference

### 8.1 Validation During Training

If both folders exist:

```text
dental_CBCT_test_set/images/
dental_CBCT_test_set/labels/
```

the training script validates every `--val_every` epochs and reports:

```text
val Dice
val HD95
```

### 8.2 Batch Test Inference After Training

After training, `train_cafnet_cbct.py` automatically runs sliding-window inference on `dental_CBCT_test_set` and saves predictions to:

```text
<outdir>/predTs/
```

The current batch inference uses a dual-teacher ensemble:

```text
0.5 * softmax(Teacher-A(global-context view))
+ 0.5 * softmax(Teacher-B(local-detail view))
```

This matches the complementary dual-teacher design used during training.

### 8.3 Regenerate `predTs` Only

If training has already finished and you only want to regenerate test predictions from existing checkpoints, run:

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

Note: with the current script, `--epochs 0` still performs a data/model sanity check before inference. Therefore, `Train-Labeled/` and `Train-Unlabeled/` should still be available.

### 8.4 Single-Case Inference and Visualization

Run inference with `best_student.pt` and generate a GT-vs-pred comparison figure:

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

If a prediction NIfTI already exists in `predTs/`, visualize it directly:

```bash
python infer_cafnet_case.py \
  --pred "runs/cafnet_full/predTs/STS2024_Test_Labeled_0001_pred.nii.gz" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz" \
  --out "case0001_predTs_compare.png" \
  --dpi 600
```

## 9. Visualization and Paper Figures

### 9.1 High-DPI GT-vs-Pred Figure

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

### 9.2 Visualize Selected Tooth IDs

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

### 9.3 NIfTI Slice Mosaic

For quick NIfTI volume inspection:

```bash
python visualization.py --help
```

Then pass the image or mask path according to the script help message.

## 10. Key Arguments

### 10.1 Data and Training

| Argument | Default | Description |
|---|---:|---|
| `--root` | `/root/autodl-tmp/3D_CBCT` | Dataset root |
| `--epochs` | `110` | Number of training epochs |
| `--val_every` | `10` | Validation interval |
| `--num_classes` | `49` | Background plus 48 tooth IDs |
| `--roi` | `160 160 160` | 3D random crop size |
| `--batch_l` | `1` | Labeled batch size |
| `--batch_u` | `1` | Unlabeled batch size |
| `--workers` | `4` | DataLoader workers |
| `--amp` | disabled | Enable CUDA mixed precision |

### 10.2 Semi-Supervised Losses

| Argument | Default | Description |
|---|---:|---|
| `--consist_w` | `4.0` | Maximum KL consistency weight |
| `--boundary_w` | `0.5` | Maximum boundary consistency weight |
| `--consist_ramp` | `60` | Ramp-up epochs for unsupervised losses |
| `--tau_conf` | `0.6` | Confidence threshold for reliable regions |
| `--tau_disagree` | `0.05` | JS-disagreement threshold between two teachers |
| `--temperature` | `0.7` | KL consistency temperature |
| `--boundary_focus` | `2.0` | Extra weighting on teacher boundary regions |

### 10.3 Dual-Teacher Views

| Argument | Default | Description |
|---|---:|---|
| `--ema_global` | `0.995` | EMA rate for Teacher-A |
| `--ema_detail` | `0.99` | EMA rate for Teacher-B |
| `--global_kernel` | `5` | Smoothing kernel for Teacher-A view |
| `--global_mix` | `0.35` | Mixing ratio for global-context view |
| `--detail_kernel` | `3` | Kernel for Teacher-B detail view |
| `--detail_gain` | `0.5` | Unsharp/detail enhancement strength |
| `--student_noise_std` | `0.03` | Gaussian noise for student strong view |
| `--student_gamma` | `0.7 1.3` | Gamma range for student strong view |

## 11. Recommended Ablation Study

For a paper-ready experimental section, consider the following variants:

| Variant | Purpose |
|---|---|
| Baseline UNet | Establish the baseline segmentation performance |
| CAF-Net backbone only | Evaluate DSCA + SFG3D backbone contribution |
| CAF-Net + single Mean Teacher | Compare against standard semi-supervised learning |
| CAF-Net + dual teacher | Evaluate complementary teacher predictions |
| CAF-Net + dual teacher + reliability mask | Evaluate confidence and disagreement calibration |
| Full method | Evaluate boundary consistency on top of all components |


## 12. Troubleshooting

### 12.1 CUDA Out of Memory

Start with:

```bash
--roi 128 128 128 --batch_l 1 --batch_u 1 --sw_batch 1 --val_sw_batch 1 --amp
```

If OOM persists:

```bash
--workers 0
```

When memory allows, increase `--roi` back to `160 160 160`.

### 12.2 Reliable Mask Is Always Zero

If the log shows `mask 0.000` for many iterations, pseudo-label filtering is too strict:

```bash
--tau_conf 0.5 --tau_disagree 0.08
```

### 12.3 Pseudo-Label Noise Is Too Strong

Use stricter filtering:

```bash
--tau_conf 0.7 --tau_disagree 0.03
```

### 12.4 Adjacent-Tooth Adhesion Remains Obvious

Increase boundary consistency:

```bash
--boundary_w 1.0
```

If training becomes unstable, use a longer ramp-up:

```bash
--consist_ramp 80
```

### 12.5 Validation Is Slow

Validate less frequently:

```bash
--val_every 20
```

Reduce sliding-window batch size:

```bash
--val_sw_batch 1
```

### 12.6 Windows and Linux Line Continuation

Bash/Linux:

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --amp
```

PowerShell:

```powershell
python train_cafnet_cbct.py `
  --root ".\data" `
  --amp
```

One-line PowerShell command:

```powershell
python train_cafnet_cbct.py --root ".\data" --outdir "runs\cafnet_full" --epochs 110 --amp
```

## 13. File Naming and Legacy Scripts

The recommended filenames have been standardized:

| File | Purpose |
|---|---|
| `train_cafnet_cbct.py` | Current main training script |
| `infer_cafnet_case.py` | Single-case inference and visualization |
| `train_unet_val5_debug.py` | Legacy debug training script |
| `train_unet_simam.py` | Legacy SimAM ablation script |
| `train_unet_simam_eca_hierarchical.py` | Legacy SimAM + ECA + hierarchical script |
| `training_strategy_and_methods.docx` | Method design notes |

Legacy scripts are not called by the current main pipeline. For official training and manuscript results, use:

```bash
python train_cafnet_cbct.py
```



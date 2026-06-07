# CAFNet-CBCT

CAFNet-CBCT is a semi-supervised 3D CBCT tooth-wise anatomical instance segmentation project. The current main pipeline implements:

- Anatomy-aware CAF-Net backbone: DSCA encoder blocks and SFG3D hierarchical decoder.
- Reliability-calibrated complementary dual-teacher pseudo-labeling.
- Instance-boundary-aware consistency for reducing adjacent-tooth adhesion and merge/split errors.

The recommended entry point is:

```bash
python train_cafnet_cbct.py
```

All recommended script names are now English-only and avoid parentheses or shell-sensitive symbols.

## 1. Project Structure

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
    Validation-Public/              # optional unlabeled public validation images
  networks/
    simam_hier_unet.py              # CAFNet3D backbone
    cafnet_ssl.py                   # dual-teacher fusion, reliable mask, boundary loss
  train_cafnet_cbct.py              # main training + validation + batch test inference
  infer_cafnet_case.py              # single-case checkpoint inference and visualization
  viz_cbct_compare.py               # high-DPI GT-vs-pred comparison figure
  visualization.py                  # NIfTI slice mosaic visualization
```

## 2. Data Layout

Put the dataset under `data/` or another root directory with the same structure:

```text
data/
  Train-Labeled/
    Images/
      STS24_Train_Labeled_0001.nii.gz
      ...
    Masks/
      STS24_Train_Labeled_0001_Mask.nii.gz
      ...
  Train-Unlabeled/
    STS24_Train_Unlabeled_0001.nii.gz
    ...
  dental_CBCT_test_set/
    images/
      STS2024_Test_Labeled_0001.nii.gz
      ...
    labels/
      STS2024_Test_Labeled_0001_Mask.nii.gz
      ...
```

The training script accepts either `Images/Masks` or `images/labels` for labeled data.

## 3. Environment Installation

Recommended environment:

- Python 3.10 or 3.11
- CUDA-capable GPU
- PyTorch
- MONAI
- nibabel
- matplotlib
- scikit-image

Create a conda environment:

```bash
conda create -n cafnet-cbct python=3.10 -y
conda activate cafnet-cbct
```

Install PyTorch. Choose the command that matches your CUDA driver from the official PyTorch selector. For example, for CUDA 12.1 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining packages:

```bash
pip install monai nibabel numpy scipy scikit-image matplotlib tqdm
```

Check the environment:

```bash
python -c "import torch, monai, nibabel; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('monai', monai.__version__)"
nvidia-smi
python -m py_compile networks/simam_hier_unet.py networks/cafnet_ssl.py train_cafnet_cbct.py
```

If `torch.cuda.is_available()` is `False`, training can still start on CPU, but 3D CBCT training will be extremely slow.

## 4. Training

Example for the local repository dataset:

```bash
python train_cafnet_cbct.py \
  --root "./data" \
  --outdir "runs/cafnet_dual_teacher" \
  --epochs 110 \
  --val_every 10 \
  --num_classes 49 \
  --roi 160 160 160 \
  --batch_l 1 \
  --batch_u 1 \
  --workers 4 \
  --amp
```

Example for an AutoDL/Linux path:

```bash
python train_cafnet_cbct.py \
  --root "/root/autodl-tmp/3D_CBCT" \
  --outdir "/root/autodl-tmp/CBCT/runs_cafnet_dual_teacher" \
  --epochs 110 \
  --val_every 10 \
  --roi 160 160 160 \
  --batch_l 1 \
  --batch_u 1 \
  --workers 4 \
  --amp
```

Important training parameters:

```text
--roi                 3D crop size. Use 160 160 160 by default; reduce to 128 128 128 if OOM.
--batch_l             labeled batch size. Usually 1 for 3D CBCT.
--batch_u             unlabeled batch size. Usually 1.
--consist_w           max weight for KL consistency loss.
--boundary_w          weight for instance-boundary-aware consistency.
--consist_ramp        ramp-up epochs for unsupervised losses.
--tau_conf            confidence threshold for reliable pseudo-label regions.
--tau_disagree        JS-disagreement threshold between Teacher-A and Teacher-B.
--ema_global          EMA rate for Teacher-A.
--ema_detail          EMA rate for Teacher-B. If omitted, uses --ema.
--amp                 enable mixed precision on CUDA.
```

The log line contains:

```text
sup   supervised DiceCE loss
con   reliability-calibrated KL consistency
bd    boundary consistency
mask  reliable region ratio
conf  mean fused pseudo-label confidence
js    mean teacher disagreement
w     current KL/boundary ramp-up weights
```

Useful tuning rules:

- If `mask` is almost always `0.000`, lower `--tau_conf` or increase `--tau_disagree`.
- If pseudo-label noise is obvious, increase `--tau_conf` or decrease `--tau_disagree`.
- If training is unstable early, increase `--consist_ramp` or lower `--consist_w`.
- If adjacent teeth merge, try increasing `--boundary_w` from `0.5` to `1.0`.
- If CUDA OOM occurs, reduce `--roi`, use `--amp`, reduce `--sw_batch`, or set `--workers 0`.

## 5. Training Outputs

The output directory will contain:

```text
runs/cafnet_dual_teacher/
  best_student.pt
  best_teacher_global.pt
  best_teacher_detail.pt
  predTs/
    *_pred.nii.gz
```

`best_student.pt` is selected by validation Dice. The two teacher checkpoints are saved at the same best epoch.

After training finishes, the script automatically runs batch inference on `dental_CBCT_test_set` using a dual-teacher ensemble and saves NIfTI predictions to `predTs/`.

## 6. Validation and Single-Case Visualization

During training, validation is run every `--val_every` epochs if `dental_CBCT_test_set/images` and `dental_CBCT_test_set/labels` both exist.

For a single case, run:

```bash
python infer_cafnet_case.py \
  --ckpt "runs/cafnet_dual_teacher/best_student.pt" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz" \
  --num_classes 49 \
  --roi 160 160 160 \
  --out "case0001_compare.png" \
  --dpi 600 \
  --amp
```

This command performs sliding-window inference with `best_student.pt` and saves a GT-vs-pred comparison figure.

If you already have a predicted NIfTI from `predTs/`, visualize it directly:

```bash
python infer_cafnet_case.py \
  --pred "runs/cafnet_dual_teacher/predTs/STS2024_Test_Labeled_0001_pred.nii.gz" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz" \
  --out "case0001_predTs_compare.png" \
  --dpi 600
```

## 7. Batch Test Inference

The current main batch inference path is the final stage of `train_cafnet_cbct.py`.

After training, predictions are saved here:

```text
<outdir>/predTs/
```

The batch inference uses:

```text
0.5 * softmax(Teacher-A(global-context view)) +
0.5 * softmax(Teacher-B(local-detail view))
```

This matches the complementary dual-teacher pseudo-labeling design used during training.

## 8. Figure Generation for Paper

High-DPI GT-vs-pred comparison:

```bash
python "viz_cbct_compare.py" \
  --ckpt "runs/cafnet_dual_teacher/best_student.pt" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0050_Mask.nii.gz" \
  --num_classes 49 \
  --roi 160 160 160 \
  --out "viz_compare50.png" \
  --dpi 600 \
  --amp
```

If you want to visualize only selected tooth IDs:

```bash
python "viz_cbct_compare.py" \
  --pred "runs/cafnet_dual_teacher/predTs/STS2024_Test_Labeled_0050_pred.nii.gz" \
  --img "data/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz" \
  --label "data/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0050_Mask.nii.gz" \
  --classes 16 23 37 \
  --class_names Tooth16 Tooth23 Tooth37 \
  --out "viz_selected_teeth.png" \
  --dpi 600
```

## 9. Recommended Experiment Plan

For a clean paper experiment, run these variants:

```text
1. Baseline MONAI UNet / old script
2. CAF-Net backbone only
3. CAF-Net + single mean teacher
4. CAF-Net + complementary dual teacher
5. CAF-Net + dual teacher + reliability mask
6. Full method: CAF-Net + reliability-calibrated dual teacher + boundary consistency
```

Report at least:

```text
Mean Dice
HD95
Per-tooth Dice if possible
Qualitative adjacent-tooth boundary examples
```

For the final method, use `train_cafnet_cbct.py` with the default SSL parameters.

## 10. Common Problems

### CUDA out of memory

Use:

```bash
--roi 128 128 128 --batch_l 1 --batch_u 1 --amp --sw_batch 1
```

If it still OOMs, set:

```bash
--workers 0
```

### No reliable pseudo-label regions

If logs show `mask 0.000` for many iterations:

```bash
--tau_conf 0.5 --tau_disagree 0.08
```

### Too much noisy pseudo-label propagation

Use stricter filtering:

```bash
--tau_conf 0.7 --tau_disagree 0.03
```

### Slow validation

Increase validation interval:

```bash
--val_every 20
```

or reduce sliding-window batch:

```bash
--val_sw_batch 1
```

### Windows command style

The recommended filenames no longer contain Chinese characters, parentheses, or `+`, so normal command-line execution is enough:

```bash
python train_cafnet_cbct.py
python infer_cafnet_case.py
```

## 11. Suggested Final Command

For the full method used in the manuscript:

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

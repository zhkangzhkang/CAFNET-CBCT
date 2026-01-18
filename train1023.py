
# train1023.py
# Semi-supervised 3D CBCT segmentation with MONAI (Mean Teacher) + optional AMP.
# - Labeled:   <root>/Train-Labeled/images/*.nii.gz  and  <root>/Train-Labeled/labels/*.nii.gz
#               (also compatible with Images/Masks as directory names)
# - Unlabeled: <root>/Train-Unlabeled/*.nii.gz
# - Optional Val: <root>/Validation-Public/images + labels (if labels missing, skip validation)
# - Optional Test: <root>/dental_CBCT_test_set/*.nii.gz
#
# Example:
#   python train1023.py --root "D:/3D_CBCT" --num_classes 2 --epochs 200 --roi 160 160 160 --amp
#
# Dependencies: torch (CUDA), monai, nibabel

import os, argparse, math
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,ResampleToMatchd,
    RandAdjustContrastd, RandSpatialCropd, EnsureTyped, Compose, AsDiscrete, SaveImaged,Lambda
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data.utils import pad_list_data_collate


def list_nii(folder):
    return sorted(glob(os.path.join(folder, "*.nii*")))


def build_lists(root):
    """Build labeled/unlabeled/val/test lists. If Validation-Public lacks labels, skip validation and
       treat its files as additional test images."""
    # labeled (allow Images/Masks or images/labels)
    img_l = list_nii(os.path.join(root, "Train-Labeled", "images")) or \
            list_nii(os.path.join(root, "Train-Labeled", "Images"))
    lab_l = list_nii(os.path.join(root, "Train-Labeled", "labels")) or \
            list_nii(os.path.join(root, "Train-Labeled", "Masks"))
    if len(img_l) == 0 or len(img_l) != len(lab_l):
        raise SystemExit("Train-Labeled/images 与 labels 数量不一致，或为空。")

    labeled = [{"image": i, "label": l} for i, l in zip(img_l, lab_l)]

    # unlabeled
    unlabeled = [{"image": p} for p in list_nii(os.path.join(root, "Train-Unlabeled"))]

    # validation (only if both exist and length matches)
    img_v = list_nii(os.path.join(root, "dental_CBCT_test_set", "images"))
    lab_v = list_nii(os.path.join(root, "dental_CBCT_test_set", "labels"))
    valset = [{"image": i, "label": l} for i, l in zip(img_v, lab_v)] if len(img_v)==len(lab_v) and len(img_v)>0 else []

    # test: official test + (if val has no labels) raw files in Validation-Public
    testset = [{"image": p} for p in list_nii(os.path.join(root, "dental_CBCT_test_set"))]
    if len(valset) == 0:
        vp_raw = list_nii(os.path.join(root, "Validation-Public"))
        testset += [{"image": p} for p in vp_raw]

    return labeled, unlabeled, valset, testset


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def update_ema(student, teacher, ema=0.99):
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(ema).add_(p_s.data, alpha=1.0 - ema)


def resize_hwd_to_like(src_hwd, ref_hwd):
    """
    src_hwd:  [H,W,D] tensor  (要被对齐的)
    ref_hwd:  [H,W,D] 或 [1,H,W,D] tensor (参考形状)
    返回:      [H,W,D]，大小与 ref_hwd 一致
    """
    # 取出 ref 的 H,W,D
    if ref_hwd.ndim == 4 and ref_hwd.shape[0] == 1:
        H, W, D = ref_hwd.shape[1], ref_hwd.shape[2], ref_hwd.shape[3]
    elif ref_hwd.ndim == 3:
        H, W, D = ref_hwd.shape
    else:
        raise ValueError(f"ref_hwd shape not supported: {tuple(ref_hwd.shape)}")

    # PyTorch 3D 插值要求输入为 [N,C,D,H,W]，size 为 (D,H,W)
    x = src_hwd.permute(2, 0, 1)           # [D,H,W]
    x = x.unsqueeze(0).unsqueeze(0)        # [1,1,D,H,W]
    x = F.interpolate(x, size=(D, H, W), mode="nearest")
    x = x.squeeze(0).squeeze(0).permute(1, 2, 0)  # 回到 [H,W,D]
    return x

from monai.metrics import HausdorffDistanceMetric


def compute_hd95_mean_per_sample(pred_lbl_1hot: torch.Tensor,
                                 true_lbl: torch.Tensor,
                                 num_classes: int) -> float:
    """
    pred_lbl_1hot: [B, 1, D, H, W] 的类别标签(非 one-hot)，这里传入 B=1
    true_lbl:      [B, 1, D, H, W] 的 GT 标签，B=1
    仅对前景类(1..num_classes-1)且 pred/gt 均有前景的类计算 HD95，最后对有效类求均值。
    """
    # 取出第一个（B=1）
    y_pred_lbl = pred_lbl_1hot[0]   # [1, D, H, W]
    y_true_lbl = true_lbl[0]        # [1, D, H, W]

    # squeeze 掉通道维，转为 [D, H, W]
    y_pred_lbl = y_pred_lbl.squeeze(0)
    y_true_lbl = y_true_lbl.squeeze(0)

    hd_vals = []
    for c in range(1, num_classes):  # 跳过背景0
        pred_c = (y_pred_lbl == c)
        true_c = (y_true_lbl == c)
        if pred_c.any() and true_c.any():
            # compute_percent_hausdorff_distance 期望 [B,1,D,H,W] 或 [1,D,H,W]
            hd_metric = HausdorffDistanceMetric(percentile=95.0, include_background=False, reduction="none")

            # 在循环内使用
            hd = hd_metric(
                pred_c.unsqueeze(0).unsqueeze(0).float(),
                true_c.unsqueeze(0).unsqueeze(0).float()
            )
            hd = float(hd.cpu().numpy()[0])  # 取数值
            # 返回张量，取标量
            hd_vals.append(float(hd))
        # 两边有一边全0则跳过该类

    if len(hd_vals) == 0:
        return float("nan")
    return float(np.mean(hd_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"/root/autodl-tmp/3D_CBCT", help="root folder containing Train-Labeled/ Train-Unlabeled/ ...")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--num_classes", type=int, default=49)
    ap.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0,1.0,1.0])
    ap.add_argument("--batch_l", type=int, default=1, help="batch size for labeled")
    ap.add_argument("--batch_u", type=int, default=1, help="batch size for unlabeled")
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs_cbct_ssl")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--consist_w", type=float, default=1.0, help="max weight for consistency loss")
    ap.add_argument("--consist_ramp", type=int, default=40, help="epochs for ramp-up")
    ap.add_argument("--amp", default=True,action="store_true", help="enable mixed precision (AMP)")
    args = ap.parse_args()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    os.makedirs(args.outdir, exist_ok=True)
    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled, unlabeled, valset, testset = build_lists(args.root)
    print(f"#labeled {len(labeled)}, #unlabeled {len(unlabeled)}, #val {len(valset)}, #test {len(testset)}")

    # -------- Transforms --------
    # Labeled: image+label preprocessing
    base_pre_lab = [
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=args.spacing, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=True),
    ]
    # Unlabeled: image-only preprocessing
    base_pre_img = [
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
    ]

    from monai.transforms import SpatialPadd
    from monai.transforms import DivisiblePadd
    from monai.transforms import SpatialPadd, DivisiblePadd, Lambdad


    tr_labeled = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=args.spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # 几何增强
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2)),
        RandAffined(keys=["image", "label"], prob=0.2,
                    rotate_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest")),

        # ★ 关键：先让尺寸可被16整除，再兜底到ROI大小，然后固定裁剪
        DivisiblePadd(keys=["image", "label"], k=16),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(args.roi)),
        RandSpatialCropd(keys=["image", "label"], roi_size=tuple(args.roi), random_center=True, random_size=False),

        # 强度增强
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),

        # 标签确保为整数（多类）; 若想二分类，取消下一行注释
        # Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.int64)),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])

    tr_unlabeled_teacher = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),

        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),

        DivisiblePadd(keys=["image"], k=16),
        SpatialPadd(keys=["image"], spatial_size=tuple(args.roi)),
        RandSpatialCropd(keys=["image"], roi_size=tuple(args.roi), random_center=True, random_size=False),

        EnsureTyped(keys=["image"]),
    ])

    tr_unlabeled_student = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),

        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),

        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image"], prob=0.2, spatial_axes=(1, 2)),
        RandAffined(keys=["image"], prob=0.2,
                    rotate_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear",)),

        DivisiblePadd(keys=["image"], k=16),
        SpatialPadd(keys=["image"], spatial_size=tuple(args.roi)),
        RandSpatialCropd(keys=["image"], roi_size=tuple(args.roi), random_center=True, random_size=False),

        RandGaussianNoised(keys=["image"], prob=0.15, mean=0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),

        EnsureTyped(keys=["image"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear",)),
        ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"),

        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0., b_max=1., clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])

    # -------- Datasets / Loaders --------
    ds_l = CacheDataset(labeled, transform=tr_labeled, cache_rate=0.0, num_workers=args.workers)
    dl_l = DataLoader(ds_l, batch_size=args.batch_l, shuffle=True, num_workers=args.workers, pin_memory=False)

    ds_u_teacher = CacheDataset(unlabeled, transform=tr_unlabeled_teacher, cache_rate=0.0, num_workers=args.workers)
    ds_u_student = CacheDataset(unlabeled, transform=tr_unlabeled_student, cache_rate=0.0, num_workers=args.workers)
    dl_u_teacher = DataLoader(ds_u_teacher, batch_size=args.batch_u, shuffle=True, num_workers=args.workers, pin_memory=False)
    dl_u_student = DataLoader(ds_u_student, batch_size=args.batch_u, shuffle=True, num_workers=args.workers, pin_memory=False)


    def unlabeled_iter():
        while True:
            for (bt, bs) in zip(dl_u_teacher, dl_u_student):
                yield bt, bs
    ul_it = unlabeled_iter()

    val_loader = None
    if len(valset) > 0:
        ds_v = CacheDataset(valset, transform=val_transforms, cache_rate=0.0, num_workers=args.workers)
        val_loader = DataLoader(ds_v, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    # -------- Models --------
    def make_unet():
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2,2,2,2),
            num_res_units=2,
            norm='INSTANCE'
        )

    student = make_unet().to(device)
    teacher = make_unet().to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    sup_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0., smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    best_dice = 0.0
    best_path = os.path.join(args.outdir, "best_student.pt")
    os.makedirs(args.outdir, exist_ok=True)

    # --- Quick sanity check (放在 for epoch 之前) ---
    student.eval()
    print("[DEBUG] sanity check: fetching one labeled batch...")
    b0 = next(iter(dl_l))
    x0 = b0["image"].to(device)
    print("[DEBUG] labeled batch shape:", tuple(x0.shape))
    with torch.no_grad():
        y0 = student(x0)
    print("[DEBUG] forward ok, logits:", tuple(y0.shape))

    print("[DEBUG] fetching one unlabeled-teacher & student batch...")
    bt = next(iter(dl_u_teacher))
    bs = next(iter(dl_u_student))
    print("[DEBUG] ut/us shapes:", tuple(bt["image"].shape), tuple(bs["image"].shape))
    with torch.no_grad():
        _ = student(bs["image"].to(device))
    print("[DEBUG] unlabeled forward ok")
    student.train()
    # -----------------------------------------------

    for epoch in range(1, args.epochs+1):
        student.train()
        epoch_loss = 0.0

        for i, batch_l in enumerate(dl_l):
            import time
            t0 = time.perf_counter()

            images_l = batch_l["image"].to(device)
            labels_l = batch_l["label"].to(device)

            try:
                batch_t, batch_s = next(ul_it)
            except StopIteration:
                ul_it = unlabeled_iter()
                batch_t, batch_s = next(ul_it)

            images_ut = batch_t["image"].to(device)  # weak aug
            images_us = batch_s["image"].to(device)  # strong aug

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                # supervised branch
                logits_l = student(images_l)
                loss_sup = sup_loss(logits_l, labels_l)

                # teacher prediction (weak aug)
                with torch.no_grad():
                    logits_ut_teacher = teacher(images_ut)
                    probs_ut_teacher = torch.softmax(logits_ut_teacher, dim=1)

                # student prediction (strong aug)
                logits_us_student = student(images_us)
                probs_us_student = torch.softmax(logits_us_student, dim=1)

                # consistency (MSE on probabilities)
                loss_cons = F.mse_loss(probs_us_student, probs_ut_teacher)
                w = args.consist_w * sigmoid_rampup(epoch, args.consist_ramp)
                loss = loss_sup + w * loss_cons

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(student, teacher, ema=args.ema)
            epoch_loss += float(loss.detach().cpu())
            # ✅ 这里加入时间打印
            it_time = time.perf_counter() - t0
            if (i + 1) % 5 == 0:
                print(
                    f"[epoch {epoch:03d}] iter {i + 1} | sup {float(loss_sup):.4f} | cons {float(loss_cons):.4f} | w {w:.3f} | {it_time:.2f}s/it")

        epoch_loss /= max(1, len(dl_l))
        msg = f"Epoch {epoch:03d} | loss {epoch_loss:.4f}"

        # -------- Validation --------
        if val_loader is not None and epoch % args.val_every == 0:
            student.eval()
            dices, hd95s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)  # [B,1,D,H,W]
                    labels = batch["label"].to(device)  # [B,1,D,H,W]

                    # 1) 滑窗推理 -> [B,C,D,H,W]
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        sw_pred = sliding_window_inference(
                            images,
                            roi_size=tuple(args.roi),
                            sw_batch_size = 2,
                            predictor=student,
                            overlap=0.5,
                        )

                    # 2) Dice（不计背景）：argmax -> one-hot
                    y_pred_lbl_list = [AsDiscrete(argmax=True)(p) for p in
                                       decollate_batch(sw_pred)]  # list of [1,D,H,W]
                    y_true_oh_list = [AsDiscrete(to_onehot=args.num_classes)(y) for y in decollate_batch(labels)]
                    y_pred_oh_list = [AsDiscrete(to_onehot=args.num_classes)(p) for p in y_pred_lbl_list]
                    dice_metric.reset()
                    dice_metric(y_pred_oh_list, y_true_oh_list)
                    d = dice_metric.aggregate().item()
                    dices.append(d)

                    # 3) HD95（前景类有效性过滤后求均值）
                    # 说明：val_loader 的 batch_size=1，如为 >1，可逐个样本循环
                    hd_mean = compute_hd95_mean_per_sample(
                        pred_lbl_1hot=torch.stack(y_pred_lbl_list, dim=0),  # 还原成 [B,1,D,H,W]；此处B=1
                        true_lbl=labels,
                        num_classes=args.num_classes
                    )
                    hd95s.append(hd_mean)

            mean_d = float(np.mean(dices)) if dices else 0.0
            mean_h = float(np.nanmean(hd95s)) if hd95s else float("nan")  # 用 nanmean 忽略无有效类的样本
            msg += f" | val Dice {mean_d:.4f} | val HD95 {mean_h:.2f}"
            if mean_d > best_dice:
                best_dice = mean_d
                torch.save(student.state_dict(), best_path)
                msg += "  >> saved best"
        print(msg)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # -------- Inference on test set (EMA teacher) --------
    if len(testset) > 0:
        infer_ds = CacheDataset(testset, transform=test_transforms, cache_rate=0.0, num_workers=args.workers)
        infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        out_dir = os.path.join(args.outdir, "predTs")
        os.makedirs(out_dir, exist_ok=True)
        teacher.eval()
        with torch.no_grad():
            for batch in infer_loader:
                images = batch["image"].to(device)
                meta = batch["image_meta_dict"]
                with torch.cuda.amp.autocast(enabled=args.amp):
                    sw_pred = sliding_window_inference(
                        images, roi_size=tuple(args.roi), sw_batch_size=args.sw_batch, predictor=teacher, overlap=0.5
                    )
                y = torch.argmax(sw_pred, dim=1, keepdim=True)
                saver = SaveImaged(keys=["pred"], output_dir=out_dir, output_postfix="pred", resample=False,
                                   separate_folder=False, print_log=False)
                data = {"pred": y, "pred_meta_dict": meta}
                saver(data)
        print(f"Inference saved to {out_dir}")


if __name__ == "__main__":
    main()

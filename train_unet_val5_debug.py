# train1023_gpu_val.py
# Semi-supervised 3D CBCT segmentation with MONAI (Mean Teacher) + AMP.
# 验证 & 测试阶段：启用 GPU 加速滑窗（sw_device/device=CUDA），支持 sw_batch 可调。

import os, argparse, math
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.networks.blocks.convolutions import ResidualUnit


# ---- multiprocessing safety (avoid ancdata issues) ----
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

# 如追求速度且不强制可复现，可设 benchmark=True、deterministic=False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised, ResampleToMatchd,
    RandAdjustContrastd, RandSpatialCropd, EnsureTyped, Compose, AsDiscrete, SaveImaged,
    SpatialPadd, DivisiblePadd
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


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
    valset = [{"image": i, "label": l} for i, l in zip(img_v, lab_v)] if len(img_v) == len(lab_v) and len(img_v) > 0 else []

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







# ---- 3D SimAM (parameter-free attention) ----
class SimAM3D(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, d, h, w = x.size()
        n = d * h * w - 1
        mu = x.mean(dim=[2, 3, 4], keepdim=True)
        var = ((x - mu) ** 2).sum(dim=[2, 3, 4], keepdim=True) / (n + 1e-6)
        e = (x - mu) / (4 * var + self.e_lambda) + 0.5
        return x * torch.sigmoid(e)


# 递归把 UNet 里的所有 ResidualUnit 后面加一个 SimAM3D
def add_simam_to_unet(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, ResidualUnit):
            # 用 Sequential(原 ResidualUnit, SimAM3D) 替换
            setattr(module, name, nn.Sequential(child, SimAM3D()))
        else:
            add_simam_to_unet(child)

















def compute_hd95_mean_per_sample(pred_lbl_1hot: torch.Tensor,
                                 true_lbl: torch.Tensor,
                                 num_classes: int) -> float:
    """
    pred_lbl_1hot: [B, 1, D, H, W] 的类别标签(非 one-hot)，这里传入 B=1
    true_lbl:      [B, 1, D, H, W] 的 GT 标签，B=1
    仅对前景类(1..num_classes-1)且 pred/gt 均有前景的类计算 HD95，最后对有效类求均值。
    """
    y_pred_lbl = pred_lbl_1hot[0]   # [1, D, H, W]
    y_true_lbl = true_lbl[0]        # [1, D, H, W]

    y_pred_lbl = y_pred_lbl.squeeze(0)  # [D,H,W]
    y_true_lbl = y_true_lbl.squeeze(0)  # [D,H,W]

    hd_vals = []
    for c in range(1, num_classes):  # 跳过背景0
        pred_c = (y_pred_lbl == c)
        true_c = (y_true_lbl == c)
        if pred_c.any() and true_c.any():
            hd_metric = HausdorffDistanceMetric(percentile=95.0, include_background=False, reduction="none")
            hd = hd_metric(
                pred_c.unsqueeze(0).unsqueeze(0).float(),
                true_c.unsqueeze(0).unsqueeze(0).float()
            )
            hd = float(hd.cpu().numpy()[0])
            hd_vals.append(float(hd))

    if len(hd_vals) == 0:
        return float("nan")
    return float(np.mean(hd_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"/root/autodl-tmp/3D_CBCT", help="root folder containing Train-Labeled/ Train-Unlabeled/ ...")
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--val_every", type=int, default=5)  # 每 5 个 epoch 验证一次
    ap.add_argument("--num_classes", type=int, default=49)
    ap.add_argument("--roi", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--batch_l", type=int, default=1, help="batch size for labeled")
    ap.add_argument("--batch_u", type=int, default=1, help="batch size for unlabeled")
    ap.add_argument("--sw_batch", type=int, default=2, help="sliding-window batch size for test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="runs_cbct_Unet")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--consist_w", type=float, default=4.0, help="max weight for consistency loss")
    ap.add_argument("--consist_ramp", type=int, default=60, help="epochs for ramp-up")
    ap.add_argument("--amp", default=False, action="store_true", help="enable mixed precision (AMP)")

    # 新增：验证阶段的 sw_batch & workers（用于更好地吃满 GPU）
    ap.add_argument("--val_sw_batch", type=int, default=4, help="sliding-window batch size for validation")
    ap.add_argument("--val_workers", type=int, default=2, help="num_workers for validation dataloader")

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

        # 先整除，再兜底到 ROI，然后固定裁剪
        DivisiblePadd(keys=["image", "label"], k=16),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(args.roi)),
        RandSpatialCropd(keys=["image", "label"], roi_size=tuple(args.roi), random_center=True, random_size=False),

        # 强度增强
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),

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
    dl_l = DataLoader(
        ds_l, batch_size=args.batch_l, shuffle=True,
        num_workers=min(args.workers, 4), pin_memory=True,
        persistent_workers=False, prefetch_factor=1
    )

    ds_u_teacher = CacheDataset(unlabeled, transform=tr_unlabeled_teacher, cache_rate=0.0, num_workers=args.workers)
    ds_u_student = CacheDataset(unlabeled, transform=tr_unlabeled_student, cache_rate=0.0, num_workers=args.workers)
    dl_u_teacher = DataLoader(
        ds_u_teacher, batch_size=args.batch_u, shuffle=True,
        num_workers=min(args.workers, 4), pin_memory=True,
        persistent_workers=False, prefetch_factor=1
    )
    dl_u_student = DataLoader(
        ds_u_student, batch_size=args.batch_u, shuffle=True,
        num_workers=min(args.workers, 4), pin_memory=True,
        persistent_workers=False, prefetch_factor=1
    )

    def unlabeled_iter():
        while True:
            for (bt, bs) in zip(dl_u_teacher, dl_u_student):
                yield bt, bs
    ul_it = unlabeled_iter()

    val_loader = None
    if len(valset) > 0:
        ds_v = CacheDataset(valset, transform=val_transforms, cache_rate=0.0, num_workers=0)
        val_loader = DataLoader(
            ds_v, batch_size=1, shuffle=False,
            num_workers=args.val_workers, pin_memory=True,  # pin + non_blocking
            persistent_workers=False
        )

    # -------- Models --------
    def make_unet():
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
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

    # 新 AMP 接口（消除弃用警告）
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_dice = 0.0
    best_path = os.path.join(args.outdir, "best_student.pt")
    os.makedirs(args.outdir, exist_ok=True)

    # --- Quick sanity check (放在 for epoch 之前) ---
    student.eval()
    print("[DEBUG] sanity check: fetching one labeled batch...")
    b0 = next(iter(dl_l))
    x0 = b0["image"].to(device, non_blocking=True)
    print("[DEBUG] labeled batch shape:", tuple(x0.shape))
    with torch.no_grad():
        y0 = student(x0)
    print("[DEBUG] forward ok, logits:", tuple(y0.shape))

    print("[DEBUG] fetching one unlabeled-teacher & student batch...")
    bt = next(iter(dl_u_teacher))
    bs = next(iter(dl_u_student))
    print("[DEBUG] ut/us shapes:", tuple(bt["image"].shape), tuple(bs["image"].shape))
    with torch.no_grad():
        _ = student(bs["image"].to(device, non_blocking=True))
    print("[DEBUG] unlabeled forward ok")
    student.train()
    # -----------------------------------------------

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0

        for i, batch_l in enumerate(dl_l):
            import time
            t0 = time.perf_counter()

            images_l = batch_l["image"].to(device, non_blocking=True)
            labels_l = batch_l["label"].to(device, non_blocking=True)

            try:
                batch_t, batch_s = next(ul_it)
            except StopIteration:
                ul_it = unlabeled_iter()
                batch_t, batch_s = next(ul_it)

            images_ut = batch_t["image"].to(device, non_blocking=True)  # weak aug
            images_us = batch_s["image"].to(device, non_blocking=True)  # strong aug

            optimizer.zero_grad(set_to_none=True)

# 让一致性真正起作用

            # with torch.amp.autocast('cuda', enabled=args.amp):
            #     # supervised branch
            #     logits_l = student(images_l)
            #     loss_sup = sup_loss(logits_l, labels_l)
            #
            #     # teacher prediction (weak aug)
            #     with torch.no_grad():
            #         logits_ut_teacher = teacher(images_ut)
            #         probs_ut_teacher = torch.softmax(logits_ut_teacher, dim=1)
            #
            #     # student prediction (strong aug)
            #     logits_us_student = student(images_us)
            #     probs_us_student = torch.softmax(logits_us_student, dim=1)
            #
            #     # consistency (MSE on probabilities)
            #     loss_cons = F.mse_loss(probs_us_student, probs_ut_teacher)
            #     w = args.consist_w * sigmoid_rampup(epoch, args.consist_ramp)
            #     loss = loss_sup + w * loss_cons
            with torch.cuda.amp.autocast(enabled=args.amp):
                # supervised branch
                logits_l = student(images_l)
                loss_sup = sup_loss(logits_l, labels_l)

                # -------- Teacher prediction (weak augmentation) --------
                teacher.eval()
                with torch.no_grad():
                    logits_ut_teacher = teacher(images_ut)
                    probs_ut_teacher = torch.softmax(logits_ut_teacher, dim=1)
                    conf_map, cls_map = torch.max(probs_ut_teacher, dim=1, keepdim=True)  # [B,1,D,H,W]

                # -------- Student prediction (strong augmentation) --------
                logits_us_student = student(images_us)
                probs_us_student = torch.softmax(logits_us_student, dim=1)

                # -------- Foreground mask-based consistency --------
                # 去掉背景通道（通道1..C-1）
                probs_t_fg = probs_ut_teacher[:, 1:, ...]
                probs_s_fg = probs_us_student[:, 1:, ...]
                mask = (conf_map > 0.6).float()  # 高置信mask，可调
                mask_fg = mask

                # KL version (较推荐；数值敏感)
                T = 0.7
                t_logits_T = logits_ut_teacher / T
                s_logits_T = logits_us_student / T
                log_t = torch.log_softmax(t_logits_T, dim=1)[:, 1:, ...]
                log_s = torch.log_softmax(s_logits_T, dim=1)[:, 1:, ...]
                p_t = torch.softmax(t_logits_T, dim=1)[:, 1:, ...]
                kl = torch.sum(p_t * (log_t - log_s), dim=1, keepdim=True)
                loss_cons = torch.sum(kl * mask_fg) / (torch.sum(mask_fg) + 1e-6)

                # 若想先简单验证，也可暂时用 MSE：
                # loss_cons = torch.sum((probs_s_fg - probs_t_fg)**2 * mask_fg) / (torch.sum(mask_fg) + 1e-6)

                # -------- Total loss --------
                w = args.consist_w * sigmoid_rampup(epoch, args.consist_ramp)
                loss = loss_sup + w * loss_cons

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(student, teacher, ema=args.ema)
            epoch_loss += float(loss.detach().cpu())

            it_time = time.perf_counter() - t0
            if (i + 1) % 5 == 0:
                print(
                    f"[epoch {epoch:03d}] iter {i + 1} | "
                    f"sup {loss_sup.detach().item():.4f} | "
                    f"cons {loss_cons.detach().item():.4f} | w {w:.3f} | {it_time:.2f}s/it"
                )

        epoch_loss /= max(1, len(dl_l))
        msg = f"Epoch {epoch:03d} | loss {epoch_loss:.4f}"

        # -------- Validation (每 val_every 个 epoch 执行一次) --------
        if val_loader is not None and epoch % args.val_every == 0:
            student.eval()
            dices, hd95s = [], []
            with torch.inference_mode():  # 比 no_grad 更快更省
                for batch in val_loader:
                    images = batch["image"].to(device, non_blocking=True)  # [B,1,D,H,W]
                    labels = batch["label"].to(device, non_blocking=True)  # [B,1,D,H,W]

                    # 1) GPU 滑窗推理 -> [B,C,D,H,W]
                    with torch.amp.autocast('cuda', enabled=args.amp):
                        sw_pred = sliding_window_inference(
                            images,
                            roi_size=tuple(args.roi),
                            sw_batch_size=args.val_sw_batch,   # ↑ 调大以压满显存
                            predictor=student,
                            overlap=0.5,
                            sw_device=device,                   # 关键：GPU 上聚合
                            device=device                       # 关键：GPU 上搬运
                        )

                    # 2) Dice（不计背景）：argmax -> one-hot
                    y_pred_lbl_list = [AsDiscrete(argmax=True)(p) for p in decollate_batch(sw_pred)]  # list of [1,D,H,W]
                    y_true_oh_list = [AsDiscrete(to_onehot=args.num_classes)(y) for y in decollate_batch(labels)]
                    y_pred_oh_list = [AsDiscrete(to_onehot=args.num_classes)(p) for p in y_pred_lbl_list]
                    dice_metric.reset()
                    dice_metric(y_pred_oh_list, y_true_oh_list)
                    d = dice_metric.aggregate().item()
                    dices.append(d)

                    # 3) HD95（前景类有效性过滤后求均值）
                    hd_mean = compute_hd95_mean_per_sample(
                        pred_lbl_1hot=torch.stack(y_pred_lbl_list, dim=0),  # [B,1,D,H,W]；此处B=1
                        true_lbl=labels,
                        num_classes=args.num_classes
                    )
                    hd95s.append(hd_mean)

            mean_d = float(np.mean(dices)) if dices else 0.0
            mean_h = float(np.nanmean(hd95s)) if hd95s else float("nan")
            msg += f" | val Dice {mean_d:.4f} | val HD95 {mean_h:.2f}"
            if mean_d > best_dice:
                best_dice = mean_d
                torch.save(student.state_dict(), best_path)
                msg += "  >> saved best"

        print(msg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    # -------- Inference on test set (EMA teacher) --------
    if len(testset) > 0:
        infer_ds = CacheDataset(testset, transform=test_transforms, cache_rate=0.0, num_workers=0)
        infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        out_dir = os.path.join(args.outdir, "predTs")
        os.makedirs(out_dir, exist_ok=True)
        teacher.eval()
        with torch.inference_mode():
            for batch in infer_loader:
                images = batch["image"].to(device, non_blocking=True)
                meta = batch["image_meta_dict"]
                with torch.amp.autocast('cuda', enabled=args.amp):
                    sw_pred = sliding_window_inference(
                        images,
                        roi_size=tuple(args.roi),
                        sw_batch_size=args.sw_batch,
                        predictor=teacher,
                        overlap=0.5,
                        sw_device=device,
                        device=device
                    )
                y = torch.argmax(sw_pred, dim=1, keepdim=True)
                saver = SaveImaged(
                    keys=["pred"], output_dir=out_dir, output_postfix="pred",
                    resample=False, separate_folder=False, print_log=False
                )
                data = {"pred": y, "pred_meta_dict": meta}
                saver(data)
        print(f"Inference saved to {out_dir}")


if __name__ == "__main__":
    main()



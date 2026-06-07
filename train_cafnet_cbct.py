# train1023_gpu_val.py
# Semi-supervised 3D CBCT segmentation with MONAI (Mean Teacher) + AMP.
# 验证 & 测试阶段：启用 GPU 加速滑窗（sw_device/device=CUDA），支持 sw_batch 可调。

import os, argparse
from glob import glob
import numpy as np
import torch
from networks.simam_hier_unet import CAFNet3D
from networks.cafnet_ssl import (
    boundary_consistency_loss,
    global_context_view,
    local_detail_view,
    masked_kl_consistency,
    reliability_calibrated_fusion,
    strong_intensity_view,
)


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
    ap.add_argument("--val_every", type=int, default=10)  # 每 5 个 epoch 验证一次
    ap.add_argument("--num_classes", type=int, default=49)
    ap.add_argument("--roi", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--batch_l", type=int, default=1, help="batch size for labeled")
    ap.add_argument("--batch_u", type=int, default=1, help="batch size for unlabeled")
    ap.add_argument("--sw_batch", type=int, default=2, help="sliding-window batch size for test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="runs_cbct_Unet_SCDA_GATE_Fusion_Improve1123")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--ema_global", type=float, default=0.995, help="EMA rate for Teacher-A/global-context teacher")
    ap.add_argument("--ema_detail", type=float, default=None, help="EMA rate for Teacher-B/local-detail teacher; default uses --ema")
    ap.add_argument("--consist_w", type=float, default=4.0, help="max weight for consistency loss")
    ap.add_argument("--consist_ramp", type=int, default=60, help="epochs for ramp-up")
    ap.add_argument("--boundary_w", type=float, default=0.5, help="weight for instance-boundary-aware consistency")
    ap.add_argument("--tau_conf", type=float, default=0.6, help="confidence threshold tau_c for reliable pseudo-label regions")
    ap.add_argument("--tau_disagree", type=float, default=0.05, help="teacher JS disagreement threshold tau_d")
    ap.add_argument("--temperature", type=float, default=0.7, help="temperature for pseudo-label KL consistency")
    ap.add_argument("--global_kernel", type=int, default=5, help="smoothing kernel for Teacher-A global-context view")
    ap.add_argument("--global_mix", type=float, default=0.35, help="mixing ratio for Teacher-A global-context view")
    ap.add_argument("--detail_kernel", type=int, default=3, help="unsharp kernel for Teacher-B local-detail view")
    ap.add_argument("--detail_gain", type=float, default=0.5, help="unsharp gain for Teacher-B local-detail view")
    ap.add_argument("--student_noise_std", type=float, default=0.03, help="Gaussian noise std for student strong view")
    ap.add_argument("--student_gamma", type=float, nargs=2, default=[0.7, 1.3], help="gamma range for student strong view")
    ap.add_argument("--boundary_focus", type=float, default=2.0, help="extra weight on teacher boundary regions")
    ap.add_argument("--amp", default=False, action="store_true", help="enable mixed precision (AMP)")

    # 新增：验证阶段的 sw_batch & workers（用于更好地吃满 GPU）
    ap.add_argument("--val_sw_batch", type=int, default=1, help="sliding-window batch size for validation")
    ap.add_argument("--val_workers", type=int, default=2, help="num_workers for validation dataloader")

    args = ap.parse_args()
    if args.ema_detail is None:
        args.ema_detail = args.ema

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

    tr_unlabeled = Compose([
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
    train_workers = min(args.workers, 4)
    train_loader_kwargs = {
        "num_workers": train_workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    if train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1

    ds_l = CacheDataset(labeled, transform=tr_labeled, cache_rate=0.0, num_workers=args.workers)
    dl_l = DataLoader(
        ds_l, batch_size=args.batch_l, shuffle=True,
        **train_loader_kwargs
    )

    ds_u = CacheDataset(unlabeled, transform=tr_unlabeled, cache_rate=0.0, num_workers=args.workers)
    dl_u = DataLoader(
        ds_u, batch_size=args.batch_u, shuffle=True,
        **train_loader_kwargs
    )

    def unlabeled_iter():
        while True:
            for batch_u in dl_u:
                yield batch_u
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
        # CAF-Net: DSCA encoder blocks + SFG3D hierarchical decoder.
        return CAFNet3D(num_classes=args.num_classes)

    # Student + complementary EMA teachers.
    student = make_unet().to(device)
    teacher_global = make_unet().to(device)
    teacher_detail = make_unet().to(device)
    teacher_global.load_state_dict(student.state_dict(), strict=True)
    teacher_detail.load_state_dict(student.state_dict(), strict=True)
    for teacher_model in (teacher_global, teacher_detail):
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    sup_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0., smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_dice = 0.0
    best_path = os.path.join(args.outdir, "best_student.pt")
    best_teacher_global_path = os.path.join(args.outdir, "best_teacher_global.pt")
    best_teacher_detail_path = os.path.join(args.outdir, "best_teacher_detail.pt")
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

    print("[DEBUG] fetching one aligned unlabeled batch...")
    bu = next(iter(dl_u))
    print("[DEBUG] unlabeled shape:", tuple(bu["image"].shape))
    with torch.no_grad():
        x_u0 = bu["image"].to(device, non_blocking=True)
        _ = teacher_global(global_context_view(x_u0, args.global_kernel, args.global_mix))
        _ = teacher_detail(local_detail_view(x_u0, args.detail_kernel, args.detail_gain))
        _ = student(strong_intensity_view(x_u0, args.student_noise_std, tuple(args.student_gamma)))
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
                batch_u = next(ul_it)
            except StopIteration:
                ul_it = unlabeled_iter()
                batch_u = next(ul_it)

            images_u = batch_u["image"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ========= 1) supervised branch on labeled data =========
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits_l = student(images_l)
                loss_sup = sup_loss(logits_l, labels_l)

            scaler.scale(loss_sup).backward()

            # ========= 2) complementary dual-teacher pseudo-labeling =========
            teacher_global.eval()
            teacher_detail.eval()
            with torch.no_grad():
                images_u_global = global_context_view(images_u, args.global_kernel, args.global_mix)
                images_u_detail = local_detail_view(images_u, args.detail_kernel, args.detail_gain)

                logits_t_global = teacher_global(images_u_global)
                logits_t_detail = teacher_detail(images_u_detail)
                probs_t_global = torch.softmax(logits_t_global, dim=1)
                probs_t_detail = torch.softmax(logits_t_detail, dim=1)

                fused_probs, reliable_mask, conf_map, disagree_map, _ = reliability_calibrated_fusion(
                    probs_t_global,
                    probs_t_detail,
                    tau_conf=args.tau_conf,
                    tau_disagree=args.tau_disagree,
                    foreground_only=True,
                )

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                images_us = strong_intensity_view(
                    images_u,
                    noise_std=args.student_noise_std,
                    gamma_range=tuple(args.student_gamma),
                )
                logits_us_student = student(images_us)
                probs_us_student = torch.softmax(logits_us_student, dim=1)

                loss_cons = masked_kl_consistency(
                    logits_us_student,
                    fused_probs.detach(),
                    reliable_mask.detach(),
                    temperature=args.temperature,
                    ignore_background=True,
                )
                loss_bd = boundary_consistency_loss(
                    probs_us_student,
                    fused_probs.detach(),
                    reliable_mask.detach(),
                    focus_scale=args.boundary_focus,
                    ignore_background=True,
                )

                ramp = sigmoid_rampup(epoch, args.consist_ramp)
                w = args.consist_w * ramp
                wb = args.boundary_w * ramp
                loss_unsup_w = w * loss_cons + wb * loss_bd

            scaler.scale(loss_unsup_w).backward()

            # ========= 3) update student and EMA teachers =========
            scaler.step(optimizer)
            scaler.update()

            update_ema(student, teacher_global, ema=args.ema_global)
            update_ema(student, teacher_detail, ema=args.ema_detail)

            total_loss = loss_sup + loss_unsup_w
            epoch_loss += float(total_loss.detach().cpu())
            reliable_ratio = reliable_mask.detach().mean().item()
            mean_conf = conf_map.detach().mean().item()
            mean_js = disagree_map.detach().mean().item()

            it_time = time.perf_counter() - t0
            if (i + 1) % 5 == 0:
                print(
                    f"[epoch {epoch:03d}] iter {i + 1} | "
                    f"sup {loss_sup.detach().item():.4f} | "
                    f"con {loss_cons.detach().item():.4f} | "
                    f"bd {loss_bd.detach().item():.4f} | "
                    f"mask {reliable_ratio:.3f} | conf {mean_conf:.3f} | js {mean_js:.4f} | "
                    f"w {w:.3f}/{wb:.3f} | {it_time:.2f}s/it"
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
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
                torch.save(teacher_global.state_dict(), best_teacher_global_path)
                torch.save(teacher_detail.state_dict(), best_teacher_detail_path)
                msg += "  >> saved best"

        print(msg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    # -------- Inference on test set (dual EMA teacher ensemble) --------
    if len(testset) > 0:
        infer_ds = CacheDataset(testset, transform=test_transforms, cache_rate=0.0, num_workers=0)
        infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        out_dir = os.path.join(args.outdir, "predTs")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(best_teacher_global_path) and os.path.exists(best_teacher_detail_path):
            teacher_global.load_state_dict(torch.load(best_teacher_global_path, map_location=device), strict=True)
            teacher_detail.load_state_dict(torch.load(best_teacher_detail_path, map_location=device), strict=True)
            print(f"[INFO] Loaded best dual-teacher checkpoints for inference from {args.outdir}")
        elif os.path.exists(best_path):
            student.load_state_dict(torch.load(best_path, map_location=device), strict=True)
            update_ema(student, teacher_global, ema=0.0)
            update_ema(student, teacher_detail, ema=0.0)
            print(f"[INFO] Best teacher checkpoints not found; copied {best_path} into both teachers for inference")
        teacher_global.eval()
        teacher_detail.eval()

        def teacher_ensemble_predictor(x):
            logits_g = teacher_global(global_context_view(x, args.global_kernel, args.global_mix))
            logits_d = teacher_detail(local_detail_view(x, args.detail_kernel, args.detail_gain))
            return 0.5 * (torch.softmax(logits_g, dim=1) + torch.softmax(logits_d, dim=1))

        with torch.inference_mode():
            for batch in infer_loader:
                images = batch["image"].to(device, non_blocking=True)
                meta = batch["image_meta_dict"]
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    sw_pred = sliding_window_inference(
                        images,
                        roi_size=tuple(args.roi),
                        sw_batch_size=args.sw_batch,
                        predictor=teacher_ensemble_predictor,
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



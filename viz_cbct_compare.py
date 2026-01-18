#
# """
# viz_cbct_compare.py
# -------------------
# High-quality visualization for CBCT segmentation:
# - Loads a MONAI UNet checkpoint (student) to predict, OR reads an existing prediction NIfTI.
# - Aligns ground-truth label (if provided) to the preprocessed image space.
# - Exports a crisp comparison figure (GT vs Pred) across axial / sagittal / coronal.
# - Saves at high DPI and uses nearest-neighbor display for masks to avoid blur.
#
# Example:
#   python viz_cbct_compare.py \
#     --ckpt runs_cbct_ssl/best_student.pt \
#     --img  /data/case01.nii.gz \
#     --label /data/labels/case01.nii.gz \
#     --num_classes 49 \
#     --classes 16 23 37 \
#     --class_names Mandible Maxilla Zygomatic \
#     --out fig_cmp_case01.png --dpi 600 --amp
# """
#
# import os, glob, argparse
# import numpy as np
# import nibabel as nib
# import torch
# from monai.transforms import (
#     Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
#     ScaleIntensityRanged, EnsureTyped, ResampleToMatchd, AsDiscrete
# )
# from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet
# import matplotlib.pyplot as plt
#
#
# def make_unet(num_classes: int):
#     return UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=num_classes,
#         channels=(32, 64, 128, 256, 512),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm="INSTANCE",
#     )
#
#
# def build_transforms(spacing):
#     # Process image; resample label to image if provided
#     tr_img = Compose([
#         LoadImaged(keys=["image"]),
#         EnsureChannelFirstd(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
#         ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
#         EnsureTyped(keys=["image"]),
#     ])
#     tr_pair = Compose([
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
#         ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"),
#         ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
#         EnsureTyped(keys=["image", "label"]),
#     ])
#     return tr_img, tr_pair
#
#
# def find_first_nii(path):
#     if os.path.isdir(path):
#         cand = sorted(glob.glob(os.path.join(path, "*.nii"))) + sorted(glob.glob(os.path.join(path, "*.nii.gz")))
#         if not cand:
#             raise SystemExit(f"No NIfTI found under: {path}")
#         return cand[0]
#     return path
#
#
# def pick_indices(mask3d):
#     inds = np.argwhere(mask3d > 0)
#     if inds.size == 0:
#         D, H, W = mask3d.shape
#         return D // 2, H // 2, W // 2
#     z = np.median(inds[:, 0]).astype(int)
#     y = np.median(inds[:, 1]).astype(int)
#     x = np.median(inds[:, 2]).astype(int)
#     return z, y, x
#
#
# def overlay(ax, img2d, mask2d, class_ids, colors, title, alpha=0.35):
#     # crisp rendering
#     ax.imshow(img2d, cmap="gray", interpolation="none", vmin=0.0, vmax=1.0)
#     for cid, color in zip(class_ids, colors):
#         m = (mask2d == cid).astype(np.uint8)
#         if m.max() > 0:
#             ax.imshow(np.ma.masked_where(m == 0, m), alpha=alpha, interpolation="nearest")
#             ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
#     ax.set_title(title, fontsize=11)
#     ax.axis("off")
#
#
# def dice_score(pred, gt, class_ids):
#     eps = 1e-6
#     dices = {}
#     for c in class_ids:
#         p = (pred == c).astype(np.uint8)
#         g = (gt == c).astype(np.uint8)
#         inter = (p & g).sum()
#         s = p.sum() + g.sum()
#         if s == 0:
#             dices[c] = np.nan
#         else:
#             dices[c] = (2.0 * inter) / (s + eps)
#     return dices
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", default="/root/autodl-tmp/CBCT/runs_cbct_Unet/best_student.pt",
#                     help="Path to best_student.pt")
#     ap.add_argument("--img",
#                     default="/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz",
#                     help="Path to input NIfTI image (.nii or .nii.gz)")
#     ap.add_argument("--label", default="/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0001_Mask.nii.gz", help="(Optional) GT label NIfTI")
#     ap.add_argument("--pred", default=None, help="(Optional) Prediction NIfTI to visualize; if absent, run inference")
#     ap.add_argument("--num_classes", type=int, default=49)
#     ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
#     ap.add_argument("--roi", type=int, nargs=3, default=[160, 160, 160])
#     ap.add_argument("--sw_batch", type=int, default=2)
#     ap.add_argument("--classes", type=int, nargs="+", default=[16, 23, 37])
#     ap.add_argument("--class_names", type=str, nargs="+", default=None)
#     ap.add_argument("--out", default="viz_compare01.png")
#     ap.add_argument("--dpi", type=int, default=600)
#     ap.add_argument("--amp", action="store_true")
#     args = ap.parse_args()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     args.img = find_first_nii(args.img)
#
#     # Colors for classes
#     palette = ["#7FBF7F", "#F2CE4E", "#6BB6E8", "#FF8C8C", "#A98DF5", "#FFB570", "#92D7F5"]
#     colors = palette[: len(args.classes)]
#
#     tr_img, tr_pair = build_transforms(args.spacing)
#
#     # load image (+ label if given) in same processed space
#     if args.label:
#         data = {"image": args.img, "label": args.label}
#         d = tr_pair(data)
#         img = d["image"].numpy()[0]
#         gt = d["label"].numpy()[0].astype(np.int16)
#     else:
#         d = tr_img({"image": args.img})
#         img = d["image"].numpy()[0]
#         gt = None
#
#     # get prediction: either load NIfTI and (if label available) resample to image space, or run inference
#     if args.pred:
#         # load pred and align to image with same pipeline as label (Orientation + resample-to-match)
#         tr_pred = Compose([
#             LoadImaged(keys=["pred"]),
#             EnsureChannelFirstd(keys=["pred"]),
#             Orientationd(keys=["pred"], axcodes="RAS"),
#             Spacingd(keys=["pred"], pixdim=args.spacing, mode=("nearest",)),
#             EnsureTyped(keys=["pred"]),
#         ])
#         pd = tr_pred({"pred": args.pred})
#         pred = pd["pred"].numpy()[0].astype(np.int16)
#         # if label present, enforce shape equal to image via simple resize (nearest) when mismatch
#         if pred.shape != img.shape:
#             # naive fix using center crop/pad to match; for clean results, prefer saving pred in same meta as image
#             from monai.transforms import Resize
#             r = Resize(spatial_size=img.shape, mode="nearest")
#             pred = r(pred[None, None]).numpy()[0, 0].astype(np.int16)
#     else:
#         if not args.ckpt:
#             raise SystemExit("Either --pred or --ckpt must be provided.")
#         net = make_unet(args.num_classes).to(device)
#         ckpt = torch.load(args.ckpt, map_location=device)
#         net.load_state_dict(ckpt, strict=True)
#         net.eval()
#         with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=='cuda')):
#             inp = torch.from_numpy(img[None, None]).float().to(device)
#             logits = sliding_window_inference(
#                 inputs=inp,
#                 roi_size=tuple(args.roi),
#                 sw_batch_size=args.sw_batch,
#                 predictor=net,
#                 overlap=0.5,
#                 sw_device=device,
#                 device=device,
#             )
#             pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)
#
#     # choose slices using GT first (if available), else use pred
#     chooser = (gt if gt is not None else pred)
#     zc, yc, xc = pick_indices((chooser > 0).astype(np.uint8))
#
#     # build figure: two rows (GT, Pred) x 3 views
#     fig = plt.figure(figsize=(12, 7))
#     gs = fig.add_gridspec(2, 3, hspace=0.10, wspace=0.02)
#
#     # axial, sagittal, coronal cuts
#     ax_img = img[zc]
#     sg_img = img[:, :, xc].T
#     co_img = img[yc]
#
#     pr_ax = pred[zc]; pr_sg = pred[:, :, xc].T; pr_co = pred[yc]
#     if gt is not None:
#         gt_ax = gt[zc]; gt_sg = gt[:, :, xc].T; gt_co = gt[yc]
#     else:
#         gt_ax = gt_sg = gt_co = np.zeros_like(pr_ax)
#
#     # top row: GT
#     overlay(fig.add_subplot(gs[0, 0]), ax_img, gt_ax, args.classes, colors, "GT: Axial")
#     overlay(fig.add_subplot(gs[0, 1]), sg_img, gt_sg, args.classes, colors, "GT: Sagittal")
#     overlay(fig.add_subplot(gs[0, 2]), co_img, gt_co, args.classes, colors, "GT: Coronal")
#
#     # bottom row: Pred
#     overlay(fig.add_subplot(gs[1, 0]), ax_img, pr_ax, args.classes, colors, "Pred: Axial")
#     overlay(fig.add_subplot(gs[1, 1]), sg_img, pr_sg, args.classes, colors, "Pred: Sagittal")
#     overlay(fig.add_subplot(gs[1, 2]), co_img, pr_co, args.classes, colors, "Pred: Coronal")
#
#     # Dice summary (whole-volume)
#     if gt is not None:
#         dices = dice_score(pred, gt, args.classes)
#         txt = "Dice " + ", ".join([f"{(args.class_names[i] if args.class_names else 'C'+str(c))}: {dices[c]:.3f}" if dices[c]==dices[c] else f"{(args.class_names[i] if args.class_names else 'C'+str(c))}: NA" for i,c in enumerate(args.classes)])
#         fig.suptitle(txt, fontsize=12)
#
#     fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
#     print(f"[OK] Saved -> {args.out}")
#     plt.close(fig)
#
#
# if __name__ == "__main__":
#     main()
#
#
#
"""
viz_cbct_compare.py
-------------------
High-quality visualization for CBCT segmentation:
- Loads a MONAI UNet checkpoint (student) to predict, OR reads an existing prediction NIfTI.
- Aligns ground-truth label (if provided) to the preprocessed image space.
- Exports a crisp comparison figure (GT vs Pred) across axial / sagittal / coronal.
- Saves at high DPI and uses nearest-neighbor display for masks to avoid blur.

Example:
  python viz_cbct_compare.py \
    --ckpt runs_cbct_ssl/best_student.pt \
    --img  /data/case01.nii.gz \
    --label /data/labels/case01.nii.gz \
    --num_classes 49 \
    --out fig_cmp_case01.png --dpi 600 --amp
"""

import os, glob, argparse
import numpy as np
import nibabel as nib
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, ResampleToMatchd, Resize
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import matplotlib.pyplot as plt


def make_unet(num_classes: int):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def build_transforms(spacing):
    # Process image; resample label to image if provided
    tr_img = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])
    tr_pair = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
        ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return tr_img, tr_pair


def find_first_nii(path):
    if os.path.isdir(path):
        cand = sorted(glob.glob(os.path.join(path, "*.nii"))) + sorted(glob.glob(os.path.join(path, "*.nii.gz")))
        if not cand:
            raise SystemExit(f"No NIfTI found under: {path}")
        return cand[0]
    return path


def pick_indices(mask3d):
    inds = np.argwhere(mask3d > 0)
    if inds.size == 0:
        D, H, W = mask3d.shape
        return D // 2, H // 2, W // 2
    z = np.median(inds[:, 0]).astype(int)
    y = np.median(inds[:, 1]).astype(int)
    x = np.median(inds[:, 2]).astype(int)
    return z, y, x


def overlay(ax, img2d, mask2d, class_ids, colors, title, alpha=0.35):
    # crisp rendering
    ax.imshow(img2d, cmap="gray", interpolation="none", vmin=0.0, vmax=1.0)
    for cid, color in zip(class_ids, colors):
        m = (mask2d == cid).astype(np.uint8)
        if m.max() > 0:
            ax.imshow(np.ma.masked_where(m == 0, m), alpha=alpha, interpolation="nearest")
            ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def dice_score(pred, gt, class_ids):
    eps = 1e-6
    dices = {}
    for c in class_ids:
        p = (pred == c).astype(np.uint8)
        g = (gt == c).astype(np.uint8)
        inter = (p & g).sum()
        s = p.sum() + g.sum()
        dices[c] = np.nan if s == 0 else (2.0 * inter) / (s + eps)
    return dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/CBCT/runs_cbct_Unet_SCDA_GATE_Fusion_Improve1123/best_student.pt",
                    help="Path to best_student.pt")
    ap.add_argument("--img",
                    default="/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/images/STS2024_Test_Labeled_0050.nii.gz",
                    help="Path to input NIfTI image (.nii or .nii.gz)")
    ap.add_argument("--label", default="/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/labels/STS2024_Test_Labeled_0050_Mask.nii.gz",
                    help="(Optional) GT label NIfTI")
    ap.add_argument("--pred", default=None,
                    help="(Optional) Prediction NIfTI to visualize; if absent, run inference")
    ap.add_argument("--num_classes", type=int, default=49)
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--roi", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--sw_batch", type=int, default=2)
    # 不再写死类别，允许空；若为空将基于当前病例自动检测
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="IDs to visualize; if omitted, auto-detect from GT or Pred")
    ap.add_argument("--class_names", type=str, nargs="+", default=None,
                    help="Optional names aligned with --classes; if missing, use C<ID>")
    ap.add_argument("--out", default="viz_compare50.png")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.img = find_first_nii(args.img)

    # transforms
    tr_img, tr_pair = build_transforms(args.spacing)

    # load image (+ label if given) in same processed space
    if args.label:
        data = {"image": args.img, "label": args.label}
        d = tr_pair(data)
        img = d["image"].numpy()[0]
        gt = d["label"].numpy()[0].astype(np.int16)
    else:
        d = tr_img({"image": args.img})
        img = d["image"].numpy()[0]
        gt = None

    # get prediction: either load NIfTI and align, or run inference
    if args.pred:
        tr_pred = Compose([
            LoadImaged(keys=["pred"]),
            EnsureChannelFirstd(keys=["pred"]),
            Orientationd(keys=["pred"], axcodes="RAS"),
            Spacingd(keys=["pred"], pixdim=args.spacing, mode=("nearest",)),
            EnsureTyped(keys=["pred"]),
        ])
        pd = tr_pred({"pred": args.pred})
        pred = pd["pred"].numpy()[0].astype(np.int16)
        if pred.shape != img.shape:
            r = Resize(spatial_size=img.shape, mode="nearest")
            pred = r(pred[None, None]).numpy()[0, 0].astype(np.int16)
    else:
        if not args.ckpt:
            raise SystemExit("Either --pred or --ckpt must be provided.")
        net = make_unet(args.num_classes).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(ckpt, strict=True)
        net.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=='cuda')):
            inp = torch.from_numpy(img[None, None]).float().to(device)
            logits = sliding_window_inference(
                inputs=inp,
                roi_size=tuple(args.roi),
                sw_batch_size=args.sw_batch,
                predictor=net,
                overlap=0.5,
                sw_device=device,
                device=device,
            )
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)

    # ---------- 自动检测本病例实际存在的类别 ----------
    if args.classes is None or len(args.classes) == 0:
        present_classes = np.unique(gt) if gt is not None else np.unique(pred)
        # 排除背景（0）
        args.classes = [int(c) for c in present_classes if int(c) != 0]
        print(f"[INFO] Auto-detected classes: {args.classes}")

    # 颜色与类名准备（长度不够就循环/补齐）
    base_palette = ["#7FBF7F", "#F2CE4E", "#6BB6E8", "#FF8C8C", "#A98DF5", "#FFB570", "#92D7F5"]
    palette = (base_palette * ((len(args.classes) + len(base_palette) - 1) // len(base_palette)))[:len(args.classes)]
    colors = palette

    if args.class_names:
        # 若提供了名字，但数量与 classes 不一致，则截断或补齐
        if len(args.class_names) < len(args.classes):
            args.class_names = args.class_names + [f"C{cid}" for cid in args.classes[len(args.class_names):]]
        elif len(args.class_names) > len(args.classes):
            args.class_names = args.class_names[:len(args.classes)]

    # choose slices using GT first (if available), else pred
    chooser = (gt if gt is not None else pred)
    zc, yc, xc = pick_indices((chooser > 0).astype(np.uint8))

    # build figure: two rows (GT, Pred) x 3 views
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, hspace=0.10, wspace=0.02)

    # axial, sagittal, coronal cuts
    ax_img = img[zc]
    sg_img = img[:, :, xc].T
    co_img = img[yc]

    pr_ax = pred[zc]; pr_sg = pred[:, :, xc].T; pr_co = pred[yc]
    if gt is not None:
        gt_ax = gt[zc]; gt_sg = gt[:, :, xc].T; gt_co = gt[yc]
    else:
        gt_ax = gt_sg = gt_co = np.zeros_like(pr_ax)

    # top row: GT
    overlay(fig.add_subplot(gs[0, 0]), ax_img, gt_ax, args.classes, colors, "GT: Axial")
    overlay(fig.add_subplot(gs[0, 1]), sg_img, gt_sg, args.classes, colors, "GT: Sagittal")
    overlay(fig.add_subplot(gs[0, 2]), co_img, gt_co, args.classes, colors, "GT: Coronal")

    # bottom row: Pred
    overlay(fig.add_subplot(gs[1, 0]), ax_img, pr_ax, args.classes, colors, "Pred: Axial")
    overlay(fig.add_subplot(gs[1, 1]), sg_img, pr_sg, args.classes, colors, "Pred: Sagittal")
    overlay(fig.add_subplot(gs[1, 2]), co_img, pr_co, args.classes, colors, "Pred: Coronal")

    # Dice summary (whole-volume)
    if gt is not None:
        dices = dice_score(pred, gt, args.classes)
        def cname(i, c):
            if args.class_names:
                return args.class_names[i]
            return f"C{c}"
        txt = "Dice " + ", ".join(
            [f"{cname(i, c)}: {dices[c]:.3f}" if dices[c]==dices[c] else f"{cname(i, c)}: NA"
             for i, c in enumerate(args.classes)]
        )
        fig.suptitle(txt, fontsize=12)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()


"""
viz_cbct_ssl.py
---------------
Load a trained MONAI UNet (your Mean-Teacher *student* checkpoint), run sliding-window
inference on a CBCT NIfTI volume, and export figure(s) like the preview image:
- three orthogonal slices (axial / sagittal / coronal) with segmentation overlays
- a simple 3D surface render (marching cubes) of selected classes

Usage (example):
  python viz_cbct_ssl.py \
      --ckpt runs_cbct_ssl/best_student.pt \
      --img  /path/to/case.nii.gz \
      --num_classes 49 \
      --spacing 1.0 1.0 1.0 \
      --classes 16 23 37 \
      --class_names Mandible Maxilla Zygomatic \
      --out fig_case01.png

Notes
- "best_student.pt" is saved by your training script in <outdir>/best_student.pt (default outdir=runs_cbct_ssl).
- If you also want to visualize teacher predictions saved by the training script,
  point --img to the raw image and --pred to the corresponding *_pred.nii.gz (optional).
"""

import os, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.apps.auto3dseg.bundle_gen import default_algo_zip
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import matplotlib.pyplot as plt

# 3D render
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def build_preprocess(spacing):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])


def overlay_slice(ax, img2d, mask2d, class_ids, colors, alpha=0.35):
    ax.imshow(img2d, cmap="gray", interpolation="nearest")
    for cid, color in zip(class_ids, colors):
        m = (mask2d == cid).astype(float)
        if m.max() > 0:
            ax.imshow(np.ma.masked_where(m == 0, m), cmap=None, alpha=alpha)
            # apply a solid color via RGBA overlay
            ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
    ax.axis("off")


def pick_indices(mask3d):
    # choose center of mass of foreground to cut slices through an interesting region
    inds = np.argwhere(mask3d > 0)
    if inds.size == 0:
        D, H, W = mask3d.shape
        return D // 2, H // 2, W // 2
    z = np.median(inds[:, 0]).astype(int)
    y = np.median(inds[:, 1]).astype(int)
    x = np.median(inds[:, 2]).astype(int)
    return z, y, x


def render_3d(ax, mask, class_ids, colors, step=2):
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")
    for cid, color in zip(class_ids, colors):
        vol = (mask == cid).astype(np.uint8)
        if vol.sum() == 0:
            continue
        # marching cubes expects zyx order; skimage uses (z, y, x)
        try:
            verts, faces, _, _ = marching_cubes(vol[::step, ::step, ::step], level=0.5)
        except Exception:
            continue
        mesh = Poly3DCollection(verts[faces], alpha=0.5)
        mesh.set_edgecolor("k")
        mesh.set_linewidth(0.1)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, vol.shape[2]//step)
        ax.set_ylim(0, vol.shape[1]//step)
        ax.set_zlim(0, vol.shape[0]//step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/CBCT/runs_cbct_ssl/best_student.pt",help="Path to best_student.pt")
    ap.add_argument("--img", default="/root/autodl-tmp/3D_CBCT/dental_CBCT_test_set/images/STS2024_Test_Labeled_0001.nii.gz", help="Path to input NIfTI image (.nii or .nii.gz)")
    ap.add_argument("--pred", default=None, help="(Optional) direct path to a predicted mask NIfTI to visualize")
    ap.add_argument("--num_classes", type=int, default=49)
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--roi", type=int, nargs=3, default=[192, 192, 192])
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--classes", type=int, nargs="+", default=[16, 23, 37], help="class ids to visualize")
    ap.add_argument("--class_names", type=str, nargs="+", default=None, help="names for legend (same length as --classes)")
    ap.add_argument("--out", default="viz_cbct.png")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Colors to roughly match demo (you can tweak):
    palette = ["#7FBF7F", "#F2CE4E", "#6BB6E8", "#FF8C8C", "#A98DF5"]
    colors = palette[:len(args.classes)]

    # 1) Get prediction: either load provided pred or run model
    if args.pred and os.path.exists(args.pred):
        pred_nii = nib.load(args.pred)
        pred = np.asarray(pred_nii.get_fdata()).astype(np.int16)
        img_nii = nib.load(args.img); img = img_nii.get_fdata().astype(np.float32)
    else:
        # preprocess & infer
        x = {"image": args.img}
        prep = build_preprocess(args.spacing)
        x = prep(x)
        img = x["image"].numpy()[0]  # [D,H,W]
        # build & load net
        net = make_unet(args.num_classes).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(ckpt, strict=True)
        net.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
            # [1,1,D,H,W]
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
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]  # [D,H,W]

    # 2) Choose cut planes
    zc, yc, xc = pick_indices((pred > 0).astype(np.uint8))
    axial    = (img[zc],    pred[zc])
    sagittal = (img[:, :, xc], pred[:, :, xc])
    coronal  = (img[:, yc, :], pred[:, yc, :])

    # 3) Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.05, wspace=0.02)

    ax0 = fig.add_subplot(gs[0, 0]); overlay_slice(ax0, axial[0], axial[1], args.classes, colors)
    ax1 = fig.add_subplot(gs[0, 1]); overlay_slice(ax1, sagittal[0].T, sagittal[1].T, args.classes, colors)
    ax2 = fig.add_subplot(gs[0, 2]); overlay_slice(ax2, coronal[0], coronal[1], args.classes, colors)

    ax3 = fig.add_subplot(gs[1, :], projection="3d"); render_3d(ax3, pred, args.classes, colors, step=2)

    # optional legend
    if args.class_names and len(args.class_names) == len(args.classes):
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, edgecolor="k", label=n, alpha=0.5) for c, n in zip(colors, args.class_names)]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)

    plt.tight_layout()
    fig.savefig(args.out, dpi=220)
    print(f"[OK] Saved figure -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F


def _odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def global_context_view(x, kernel_size=5, mix=0.35):
    """Weak global-context view for Teacher-A; keeps the crop spatially aligned."""
    kernel_size = _odd_kernel_size(kernel_size)
    if kernel_size <= 1 or mix <= 0:
        return x
    smooth = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.clamp((1.0 - mix) * x + mix * smooth, 0.0, 1.0)


def local_detail_view(x, kernel_size=3, gain=0.5):
    """Weak local-detail view for Teacher-B; uses unsharp masking without geometry changes."""
    kernel_size = _odd_kernel_size(kernel_size)
    if kernel_size <= 1 or gain <= 0:
        return x
    smooth = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.clamp(x + gain * (x - smooth), 0.0, 1.0)


def strong_intensity_view(x, noise_std=0.03, gamma_range=(0.7, 1.3)):
    """Student strong view; intensity-only so pseudo-labels remain voxel-aligned."""
    view = torch.clamp(x, 0.0, 1.0)
    if gamma_range is not None:
        gamma_min, gamma_max = float(gamma_range[0]), float(gamma_range[1])
        if gamma_max > 0 and abs(gamma_max - gamma_min) > 1e-6:
            gamma = torch.empty(
                (view.shape[0], 1, 1, 1, 1),
                dtype=view.dtype,
                device=view.device,
            ).uniform_(gamma_min, gamma_max)
            view = torch.clamp(view, min=0.0).pow(gamma)
    if noise_std > 0:
        view = view + torch.randn_like(view) * float(noise_std)
    return torch.clamp(view, 0.0, 1.0)


def js_divergence(probs_a, probs_b, eps=1e-6):
    """Per-voxel Jensen-Shannon divergence between two class-probability volumes."""
    probs_a = probs_a.clamp_min(eps)
    probs_b = probs_b.clamp_min(eps)
    mean_probs = 0.5 * (probs_a + probs_b)
    js = 0.5 * (
        torch.sum(probs_a * (probs_a.log() - mean_probs.log()), dim=1, keepdim=True)
        + torch.sum(probs_b * (probs_b.log() - mean_probs.log()), dim=1, keepdim=True)
    )
    return js


def reliability_calibrated_fusion(
    probs_a,
    probs_b,
    tau_conf=0.6,
    tau_disagree=0.05,
    foreground_only=True,
):
    """
    Fuse two teachers and build M_i = 1(C_i > tau_c) * 1(D_i < tau_d).

    Returns fused probabilities, reliable mask, confidence map, JS map, and pseudo labels.
    """
    fused_probs = 0.5 * (probs_a + probs_b)
    conf_map, pseudo_label = torch.max(fused_probs, dim=1, keepdim=True)
    disagree_map = js_divergence(probs_a, probs_b)

    reliable_mask = (conf_map > float(tau_conf)) & (disagree_map < float(tau_disagree))
    if foreground_only:
        reliable_mask = reliable_mask & (pseudo_label > 0)
    return fused_probs, reliable_mask.float(), conf_map, disagree_map, pseudo_label


def masked_kl_consistency(
    student_logits,
    target_probs,
    reliable_mask,
    temperature=0.7,
    ignore_background=True,
    eps=1e-6,
):
    """KL(P_f || P_s) over reliability-calibrated voxels."""
    student_log_probs = F.log_softmax(student_logits / float(temperature), dim=1)
    target_probs = target_probs.clamp_min(eps)
    target_log_probs = target_probs.log()

    if ignore_background and student_log_probs.shape[1] > 1:
        student_log_probs = student_log_probs[:, 1:, ...]
        target_probs = target_probs[:, 1:, ...]
        target_log_probs = target_log_probs[:, 1:, ...]

    voxel_kl = torch.sum(target_probs * (target_log_probs - student_log_probs), dim=1, keepdim=True)
    return torch.sum(voxel_kl * reliable_mask) / (torch.sum(reliable_mask) + eps)


def soft_boundary_map(probs):
    """B(P) = |grad_x P| + |grad_y P| + |grad_z P| for 3D probability volumes."""
    grad_z = torch.abs(probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :])
    grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))

    grad_y = torch.abs(probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :])
    grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))

    grad_x = torch.abs(probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1])
    grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
    return grad_x + grad_y + grad_z


def boundary_consistency_loss(
    student_probs,
    target_probs,
    reliable_mask,
    focus_scale=2.0,
    ignore_background=True,
    eps=1e-6,
):
    """Reliability-weighted instance-boundary consistency."""
    if ignore_background and student_probs.shape[1] > 1:
        student_probs = student_probs[:, 1:, ...]
        target_probs = target_probs[:, 1:, ...]

    student_boundary = soft_boundary_map(student_probs)
    target_boundary = soft_boundary_map(target_probs).detach()
    boundary_l1 = torch.sum(torch.abs(student_boundary - target_boundary), dim=1, keepdim=True)

    weights = reliable_mask
    if focus_scale > 0:
        boundary_strength = torch.sum(target_boundary, dim=1, keepdim=True)
        max_strength = boundary_strength.amax(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
        weights = weights * (1.0 + float(focus_scale) * boundary_strength / max_strength)

    return torch.sum(boundary_l1 * weights) / (torch.sum(weights) + eps)

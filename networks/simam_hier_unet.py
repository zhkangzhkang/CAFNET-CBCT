# networks/simam_hier_unet.py
# 自定义 3D SimAM + ECA + SCDA Block + Hierarchical Decoder UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFG3D(nn.Module):
    """
    极简跳连融合模块（轻量版）：
    - 输入: skip, up  [B, C, D, H, W]，通道 C 相同
    - 输出: fused     [B, C, D, H, W]
    - 思路: 通道注意力 + 门控加权  fused = w * skip + (1-w) * up
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 先把 skip 和 up 拼在一起做一次 1x1x1 融合
        self.conv_cat = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False)
        # 通道注意力（类似 SE，但只做一次，非常省）
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                    # B,C,1,1,1
            nn.Conv3d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, skip, up):
        # skip, up: [B, C, D, H, W]
        x = torch.cat([skip, up], dim=1)        # [B, 2C, D, H, W]
        x = self.conv_cat(x)                    # [B, C, D, H, W]
        w = self.fc(x)                          # [B, C, 1, 1, 1]
        fused = w * skip + (1.0 - w) * up       # 通道门控融合
        return fused





# ---------------- SimAM 3D（parameter-free voxel attention）----------------
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


# ---------------- ECA 3D（高效通道注意力：GAP + 1D Conv）----------------
class ECA3D(nn.Module):
    """
    Efficient Channel Attention 的 3D 版本：
    GAP 得到 [B, C, 1, 1, 1] -> reshape 为 [B, 1, C] 上做 1D Conv -> Sigmoid -> 通道权重
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, D, H, W]
        y = self.gap(x)                  # [B, C, 1, 1, 1]
        y = y.view(y.size(0), 1, -1)     # [B, 1, C]
        y = self.conv1d(y)               # [B, 1, C]
        y = self.sigmoid(y)              # [B, 1, C]
        y = y.view(x.size(0), x.size(1), 1, 1, 1)  # [B, C, 1, 1, 1]
        return x * y


# ---------------- SCDA3DBlock：Conv + 残差 + 并联 SimAM/ECA + 加权融合 ----------------
class DSCA_Block(nn.Module):
    """
    Spatial-Channel Dual Attention (SCDA) 3D Block

    结构（简要）：
      x
      ├─ Conv3d → IN → PReLU → Conv3d → IN ────────────┐
      │                                               │
      ├───────────────(1x1 Conv 可选)─────────────────┘  ← residual
      ↓
      base
      ├─ SimAM3D ────────────────────────┐
      └─ ECA3D  ────────────────────────┤
                  α · f_sim + β · f_eca ─┘ → PReLU → out
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 双 3x3x3 CBR
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_ch, affine=True)

        # SimAM + ECA 分支
        self.simam = SimAM3D()
        self.eca = ECA3D(out_ch)

        # 可学习融合权重（初始化为 0.5 / 0.5）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # 输出激活
        self.out_act = nn.PReLU()

        # 通道不一致时的残差 1x1 Conv
        if in_ch != out_ch:
            self.res_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x

        # 基础卷积主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差对齐
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        base = out + identity  # [B, C, D, H, W]

        # 并联空间/通道注意力
        f_sim = self.simam(base)
        f_eca = self.eca(base)

        # 加权融合（SCDA）
        out = self.alpha * f_sim + self.beta * f_eca
        out = self.out_act(out)
        return out


# ---------------- Encoder / Decoder Blocks ----------------
class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DSCA_Block(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x




# class UpBlock3D(nn.Module):
#     """
#     Decoder 上采样块 + 轻量 GatedSkip3D 跳连融合：
#       b_{l+1} --up--> up_feat
#       skip_l --------┐
#                      ├─ GatedSkip3D(skip_l, up_feat) = fused
#       up_feat -------┘
#       concat([fused, up_feat]) → SCDA3DBlock → out
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # 上采样后用 1x1x1 对通道对齐
#         self.up_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
#         # 新增：轻量跳连融合模块
#         self.skip_fuse = GatedSkip3D(out_ch)
#         # concat 之后通道翻倍 → 走你原来的 SCDA3DBlock
#         self.conv = SCDA3DBlock(2 * out_ch, out_ch)
#
#     def forward(self, x, skip):
#         # x: 上一解码层输出   skip: 对应 encoder 特征
#         x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
#         up_feat = self.up_conv(x)                     # [B, out_ch, D, H, W]
#
#         fused = self.skip_fuse(skip, up_feat)         # [B, out_ch, D, H, W]
#
#         x = torch.cat([fused, up_feat], dim=1)        # [B, 2*out_ch, D, H, W]
#         x = self.conv(x)                              # SCDA3DBlock 内还有 SimAM3D/ECA3D
#         return x
#


class UpBlock3D(nn.Module):
    """
    Decoder 上采样块 + 轻量 GatedSkip3D 跳连融合 + Hierarchical Add：

      更深层 decoder 输出 b_{l+1}:
        1) 上采样 + 1x1x1 Conv 得到 up_feat
        2) 与当前 encoder skip 逐元素相加:  skip_h = skip_l + up_feat
        3) GatedSkip3D(skip_h, up_feat) 自适应融合
        4) concat([fused, up_feat]) → SCDA3DBlock → out

      对应公式：
        base_l = E_l + Up(D_{l+1})
        D_l    = SCDA(  [ GatedSkip(base_l, Up(D_{l+1})), Up(D_{l+1}) ]  )
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 上采样后用 1x1x1 对通道对齐
        self.up_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        # 轻量跳连融合模块（门控）
        self.skip_fuse = SFG3D(out_ch)
        # concat 之后通道翻倍 → 再用 SCDA3DBlock
        self.conv = DSCA_Block(2 * out_ch, out_ch)

    def forward(self, x, skip):
        """
        x    : 上一解码层输出 D_{l+1}
        skip : 当前编码层输出 E_l（与 x 对应的 skip 连接）
        """
        # 1) 上采样 deeper decoder 特征
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        up_feat = self.up_conv(x)                     # [B, out_ch, D, H, W]

        # 2) Hierarchical Add：encoder skip + 上采样的 deeper decoder
        skip_h = skip + up_feat                       # 逐元素相加，hierarchical decoder

        # 3) 门控跳连融合
        fused = self.skip_fuse(skip_h, up_feat)       # [B, out_ch, D, H, W]

        # 4) concat → SCDA3DBlock
        x = torch.cat([fused, up_feat], dim=1)        # [B, 2*out_ch, D, H, W]
        x = self.conv(x)
        return x




# ---------------- 主干网络：SimAM + ECA + SCDA + Hierarchical Fusion UNet ----------------
class SimAMHierUNet3D(nn.Module):
    """
    结构说明：
    - Encoder: 4 层下采样 + Bottleneck（均采用 SCDA3DBlock，内置 SimAM + ECA）
    - Decoder: 4 层上采样（UpBlock3D，仍然保持原始绿色 skip 连接）
    - 解码端层级融合（Hierarchical Fusion Head）：
        取 dec1 / dec2 / dec3 三个尺度特征，上采样到最高分辨率后 concat，再预测 logits
    """

    def __init__(self, num_classes=49, in_channels=1, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = DSCA_Block(in_channels, c1)   # -> 1/1
        self.enc2 = DownBlock3D(c1, c2)            # -> 1/2
        self.enc3 = DownBlock3D(c2, c3)            # -> 1/4
        self.enc4 = DownBlock3D(c3, c4)            # -> 1/8
        self.bottleneck = DownBlock3D(c4, c5)      # -> 1/16

        # Decoder
        self.dec4 = UpBlock3D(c5, c4)              # 1/16 -> 1/8
        self.dec3 = UpBlock3D(c4, c3)              # 1/8  -> 1/4
        self.dec2 = UpBlock3D(c3, c2)              # 1/4  -> 1/2
        self.dec1 = UpBlock3D(c2, c1)              # 1/2  -> 1/1

        # Hierarchical Fusion Head
        # dec1: [B, c1, D, H, W]
        # dec2: [B, c2, D/2, H/2, W/2]
        # dec3: [B, c3, D/4, H/4, W/4]
        fusion_in_ch = c1 + c2 + c3
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(fusion_in_ch, fusion_in_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(fusion_in_ch, affine=True),
            nn.PReLU(),
            nn.Conv3d(fusion_in_ch, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # 1/1
        e2 = self.enc2(e1)         # 1/2
        e3 = self.enc3(e2)         # 1/4
        e4 = self.enc4(e3)         # 1/8
        b  = self.bottleneck(e4)   # 1/16

        # Decoder
        d4 = self.dec4(b, e4)      # 1/8
        d3 = self.dec3(d4, e3)     # 1/4
        d2 = self.dec2(d3, e2)     # 1/2
        d1 = self.dec1(d2, e1)     # 1/1

        # Hierarchical multi-scale fusion
        d2_up = F.interpolate(d2, size=d1.shape[2:], mode="trilinear", align_corners=False)
        d3_up = F.interpolate(d3, size=d1.shape[2:], mode="trilinear", align_corners=False)

        fusion_feat = torch.cat([d1, d2_up, d3_up], dim=1)
        logits = self.fusion_conv(fusion_feat)  # [B, num_classes, D, H, W]
        return logits


# Paper-facing name: Anatomy-Aware CAF-Net backbone.
CAFNet3D = SimAMHierUNet3D

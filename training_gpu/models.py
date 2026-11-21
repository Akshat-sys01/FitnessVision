"""
FitnessVision PyTorch Pose Estimation Architecture
MobileNetV2 feature extractor + Multi-scale FPN Decoder with skip connections.
Outputs 16 joint heatmaps at 64x64 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FitnessVisionPoseNet(nn.Module):
    """
    Encoder: Pretrained MobileNetV2 features
      Stage 1: features[0..3]   -> 64x64, 24 channels
      Stage 2: features[4..6]   -> 32x32, 32 channels
      Stage 3: features[7..13]  -> 16x16, 96 channels
      Stage 4: features[14..18] -> 8x8, 320 channels (or 1280 with last conv)
    Decoder: Progressive upsampling with skip connections + refinement
    Head: 1x1 Conv to 16 channels
    """
    def __init__(self, num_joints=16, pretrained=True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)

        self.stage1 = nn.Sequential(*base.features[:4])    # -> [B, 24, 64, 64]
        self.stage2 = nn.Sequential(*base.features[4:7])   # -> [B, 32, 32, 32]
        self.stage3 = nn.Sequential(*base.features[7:14])  # -> [B, 96, 16, 16]
        self.stage4 = nn.Sequential(*base.features[14:])   # -> [B, 1280, 8, 8]

        # Decoder blocks
        self.dec_bottleneck = ConvBlock(1280, 256)

        # 8x8 -> 16x16
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            ConvBlock(256 + 96, 192),
            ConvBlock(192, 192)
        )

        # 16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            ConvBlock(192 + 32, 128),
            ConvBlock(128, 128)
        )

        # 32x32 -> 64x64
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            ConvBlock(128 + 24, 96),
            ConvBlock(96, 96)
        )

        # Heatmap prediction head
        self.final_conv = nn.Conv2d(96, num_joints, kernel_size=1)

    def forward(self, x):
        # x is [B, 3, 256, 256]
        f1 = self.stage1(x)   # [B, 24, 64, 64]
        f2 = self.stage2(f1)  # [B, 32, 32, 32]
        f3 = self.stage3(f2)  # [B, 96, 16, 16]
        f4 = self.stage4(f3)  # [B, 1280, 8, 8]

        x = self.dec_bottleneck(f4)      # [B, 256, 8, 8]

        x = self.up1(x)                  # [B, 256, 16, 16]
        x = torch.cat([x, f3], dim=1)    # [B, 256+96, 16, 16]
        x = self.dec1(x)                 # [B, 192, 16, 16]

        x = self.up2(x)                  # [B, 192, 32, 32]
        x = torch.cat([x, f2], dim=1)    # [B, 192+32, 32, 32]
        x = self.dec2(x)                 # [B, 128, 32, 32]

        x = self.up3(x)                  # [B, 128, 64, 64]
        x = torch.cat([x, f1], dim=1)    # [B, 128+24, 64, 64]
        x = self.dec3(x)                 # [B, 96, 64, 64]

        heatmaps = self.final_conv(x)    # [B, 16, 64, 64]
        return heatmaps


class MaskedHeatmapMSELoss(nn.Module):
    """
    Calculates MSE loss only on joints that are marked as visible/present.
    targets: [B, 16, 64, 64]
    preds:   [B, 16, 64, 64]
    masks:   [B, 16] (1 for present, 0 for ignored)
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, masks):
        # Broadcast masks to [B, 16, 1, 1]
        weight = masks.unsqueeze(-1).unsqueeze(-1)
        squared_diff = (preds - targets) ** 2 * weight
        # Mean across spatial dims per joint
        loss_per_joint = squared_diff.sum(dim=(2, 3)) / (64.0 * 64.0)
        # Weight by number of visible joints
        total_visible = torch.clamp(masks.sum(), min=1.0)
        loss = loss_per_joint.sum() / total_visible
        return loss

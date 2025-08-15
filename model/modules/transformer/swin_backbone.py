# -*-coding:utf8-*-
"""
SwinStage2 backbone wrapper (timm) for SuperPoint-style detectors.

- Uses Swin-Tiny (default) from timm, created with a FIXED img_size.
- Taps Stage-2 (index=1) => stride-8 features (H/8 x W/8).
- Handles grayscale by repeating to 3 channels.
- Optional ImageNet mean/std normalization.
- Strict asserts so you never silently train at the wrong stride.

Usage:
    from model.modules.transformer.swin_backbone import SwinStage2
    bb = SwinStage2(
        name="swin_tiny_patch4_window7_224",
        pretrained=True,
        norm_in=True,
        out_index=1,              # 0->s4, 1->s8, 2->s16, 3->s32 (Swin)
        img_size=(480, 640),      # <-- set the resolution you will ALWAYS use
        enforce_stride8=True,
        require_hw_div8=True,
    )
    f = bb(x)  # x: (B,1|3,H,W) with H,W == img_size; returns (B,C≈192,H/8,W/8)
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class SwinStage2(nn.Module):
    def __init__(
        self,
        name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        norm_in: bool = True,
        out_index: int = 1,                  # 0->s4, 1->s8, 2->s16, 3->s32
        img_size: Tuple[int, int] = (120, 160),
        enforce_stride8: bool = True,        # assert effective stride is 8x
        require_hw_div8: bool = True,        # guard inputs are multiples of 8
    ) -> None:
        super().__init__()
        self.norm_in = norm_in
        self.enforce_stride8 = enforce_stride8
        self.require_hw_div8 = require_hw_div8
        self.img_size = tuple(img_size)

        # buffers for normalization (move with .to(device))
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        # Create timm model with a FIXED img_size so window masks match runtime
        # NOTE: For Swin in timm, out_indices: 0->s4, 1->s8, 2->s16, 3->s32
        self.m = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,),
            img_size=self.img_size,
        )

        # Channel dim and nominal reduction reported by timm
        self.out_ch = self.m.feature_info.channels()[-1]
        self.reduction = self.m.feature_info.reduction()[-1]  # expect 8 at stage-2

        if self.enforce_stride8:
            assert self.reduction == 8, (
                f"[SwinStage2] out_index={out_index} reports reduction={self.reduction}, "
                f"expected 8 (stage-2). Use out_index=1 for stride-8 on Swin."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1|3,H,W) with H,W exactly == self.img_size
        returns: feature map (B, C=self.out_ch, H/8, W/8)
        """
        assert x.ndim == 4 and x.shape[2:] == self.img_size, (
            f"[SwinStage2] Input size must equal img_size={self.img_size}. "
            f"Got {tuple(x.shape[2:])}. Resize your images accordingly."
        )
        B, C, H, W = x.shape

        if self.require_hw_div8:
            assert (H % 8 == 0) and (W % 8 == 0), (
                f"[SwinStage2] Input must be divisible by 8; got H={H}, W={W}."
            )

        # grayscale -> RGB
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # ImageNet normalization (optional)
        if self.norm_in:
            x = (x - self.mean) / self.std

        # ---- get features from timm ----
        f = self.m(x)[-1]  # could be NHWC or NCHW depending on timm/version/env

        # ---- normalize to NCHW if needed ----
        if f.shape[1] != self.out_ch and f.shape[-1] == self.out_ch:
            f = f.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

        # ---- stride check (now safe) ----
        Hc, Wc = f.shape[-2:]
        h_ratio, w_ratio = H / float(Hc), W / float(Wc)
        assert abs(h_ratio - 8.0) < 1e-3 and abs(w_ratio - 8.0) < 1e-3, (
            f"[SwinStage2] Feature stride check failed: H/Hc={h_ratio:.3f}, "
            f"W/Wc={w_ratio:.3f} (expected 8)."
        )

        return f


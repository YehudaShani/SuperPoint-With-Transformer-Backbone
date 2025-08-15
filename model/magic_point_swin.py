# -*-coding:utf8-*-
"""
MagicPointSwin: Swin-Tiny backbone + SuperPoint-style detector head (65-channel).

- Mirrors model/magic_point.py API so existing train/eval/loss code works.
- Backbone: SwinStage2 (stride-8 features) from model.modules.transformer.swin_backbone
- Head: DetectorHead (produces 65-channel logits on H/8 x W/8, pixel-unshuffles to heatmap)

Expected config keys (minimal):
model:
  name: magicpoint_swin
  grid_size: 8
  det_thresh: 0.001
  nms: 4
  topk: -1
  backbone:
    swin:
      name: swin_tiny_patch4_window7_224
      pretrained: true
      normalize: true
      img_size: [120, 160]     # H, W (Synthetic); e.g., [480, 640] for HPatches eval
  det_head:
    feat_in_dim: 192           # Swin-T stage-2 channels (default auto)

Outputs dict:
  'logits': B x 65 x H/8 x W/8
  'prob'  : B x H x W
  'prob_nms': B x H x W    (NMS+threshold applied if nms>0)
  'pred'  : 1-D tensor of scores >= det_thresh (after NMS), concatenated per-batch
  (Optional convenience, if you enable below:)
  'kpts'  : list of (Nb_i x 2) tensors of (x,y) after NMS+threshold
  'scores': list of (Nb_i) tensors of scores after NMS+threshold
"""

from typing import Any, Dict, Tuple
import torch
from solver.nms import box_nms
from model.modules.transformer.swin_backbone import SwinStage2
from model.modules.cnn.cnn_heads import DetectorHead


class MagicPointSwin(torch.nn.Module):
    def __init__(self,
                 config: Dict[str, Any],
                 input_channel: int = 1,
                 grid_size: int = 8,
                 using_bn: bool = True,
                 device: str = 'cpu'):
        super(MagicPointSwin, self).__init__()
        # --- detector meta ---
        self.grid_size  = int(config.get('grid_size', grid_size))
        self.nms        = config['nms']               # int radius in pixels (e.g., 4) or 0/None
        self.det_thresh = float(config['det_thresh']) # score threshold for keeping points
        self.topk       = int(config['topk'])         # -1 means unlimited in repo convention

        # --- backbone cfg ---
        bb_root   = config.get('backbone', {})
        bb_cfg    = bb_root.get('swin', {}) if isinstance(bb_root, dict) else {}
        swin_name = bb_cfg.get('name', 'swin_tiny_patch4_window7_224')
        pretrained = bool(bb_cfg.get('pretrained', True))
        norm_in    = bool(bb_cfg.get('normalize', True))
        # default to Synthetic Shapes size; pass [480,640] etc. for eval configs
        img_size = bb_cfg.get('img_size', [120, 160])
        if isinstance(img_size, tuple):
            img_hw = img_size
        else:
            assert isinstance(img_size, (list, tuple)) and len(img_size) == 2, \
                "backbone.swin.img_size must be [H, W]"
            img_hw = (int(img_size[0]), int(img_size[1]))


        self.backbone = SwinStage2(
            name=swin_name,
            pretrained=pretrained,
            norm_in=norm_in,
            out_index=1,            # Swin stage-2 => stride-8
            img_size=img_hw
        )

        bb_cfg = config.get('backbone', {}).get('swin', {})
        freeze_bb = bool(bb_cfg.get('freeze_backbone', False))

        if freeze_bb:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # keep backbone deterministic (disables DropPath, etc.)
            self.backbone.eval()

        self._freeze_backbone = freeze_bb  # remember for forward()

        # --- head cfg ---
        head_cfg = config.get('det_head', {}) if isinstance(config.get('det_head', {}), dict) else {}
        # default to backbone channels (Swin-T stage-2 -> 192)
        feat_in = int(head_cfg.get('feat_in_dim', self.backbone.out_ch))

        # Optional 1x1 neck to adapt backbone channels -> head input channels
        if feat_in != self.backbone.out_ch:
            self.neck = torch.nn.Conv2d(self.backbone.out_ch, feat_in, kernel_size=1, bias=True)
        else:
            self.neck = torch.nn.Identity()

        self.detector_head = DetectorHead(
            input_channel=feat_in,
            grid_size=self.grid_size,
            using_bn=using_bn
        )

    def forward(self, x: Any) -> Dict[str, torch.Tensor]:
        """
        x: tensor N x 1 x H x W  (grayscale)  OR dict with key 'img' -> tensor
        returns: dict with keys described in the module docstring.
        """
        img = x['img'] if isinstance(x, dict) else x  # keep parity with repo style
        # Backbone: (B, Cb, H/8, W/8)
        if getattr(self, "_freeze_backbone", False):
            with torch.no_grad():
                feat_map = self.backbone(img)
        else:
            feat_map = self.backbone(img)

        feat_map = self.neck(feat_map)
        # Head: produces {'logits': (B,65,H/8,W/8), 'prob': (B,H,W)}
        outputs = self.detector_head(feat_map)

        prob = outputs['prob']  # B x H x W

        # Apply NMS if enabled (radius > 0)
        if isinstance(self.nms, (int, float)) and self.nms > 0:
            prob_nms = []
            det_thresh = float(self.det_thresh if self.det_thresh is not None else 0.0)
            keep_k = -1 if (self.topk is None or int(self.topk) < 0) else int(self.topk)

            for p in prob:  # p: H x W
                pn = box_nms(
                    p.unsqueeze(0).contiguous(),  # [1,H,W]
                    int(self.nms),  # nms radius (pixels)
                    det_thresh,  # min_prob (threshold)
                    keep_k  # keep_top_k (-1 for unlimited)
                ).squeeze(0)
                prob_nms.append(pn)

            prob = torch.stack(prob_nms, dim=0)
        # else: leave prob as-is (no NMS)

        # Attach post-NMS prob
        outputs.setdefault('prob_nms', prob)

        # Keep a simple 'pred' vector of scores >= thresh (concat over batch)
        # (matches common usage pattern; downstream code usually consumes prob_nms)
        mask = (prob >= self.det_thresh)
        if mask.any():
            outputs.setdefault('pred', prob[mask])
        else:
            outputs.setdefault('pred', torch.empty(0, device=prob.device, dtype=prob.dtype))

        # (Optional) also expose coords + scores per image (uncomment if you need them)
        # kpts_list, scores_list = [], []
        # B, H, W = prob.shape
        # for b in range(B):
        #     ys, xs = torch.nonzero(prob[b] >= self.det_thresh, as_tuple=True)
        #     scores = prob[b, ys, xs]
        #     kpts = torch.stack([xs.float(), ys.float()], dim=1)  # (N, 2) as (x, y)
        #     kpts_list.append(kpts)
        #     scores_list.append(scores)
        # outputs['kpts'] = kpts_list
        # outputs['scores'] = scores_list

        return outputs

import torch
from model.magic_point_swin import MagicPointSwin

cfg = {
  "nms":4, "det_thresh":1e-3, "topk":1000, "grid_size":8,
  "backbone":{"swin":{"name":"swin_tiny_patch4_window7_224","pretrained":True,"normalize":True,"img_size":[120,160]}},
  "det_head":{"feat_in_dim":192}
}
m = MagicPointSwin(cfg).eval()
x = torch.randn(1,1,120,160)
with torch.no_grad(): out = m(x)
print("logits:", out["logits"].shape, "  # expect (1,65,15,20)")
print("prob  :", out["prob"].shape,   "  # expect (1,120,160)")
print("prob_nms:", out["prob_nms"].shape)

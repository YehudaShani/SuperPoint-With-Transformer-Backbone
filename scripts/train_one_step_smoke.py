import torch
from model.magic_point_swin import MagicPointSwin

cfg = {"nms":4,"det_thresh":1e-3,"topk":1000,"grid_size":8,
       "backbone":{"swin":{"name":"swin_tiny_patch4_window7_224","pretrained":True,"normalize":True,"img_size":[120,160],"freeze_backbone":True}},
       "det_head":{"feat_in_dim":192}}
m = MagicPointSwin(cfg).train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)

x = torch.randn(8,1,120,160)
out = m(x)
loss = 1.0 - out["prob"].mean()   # placeholder loss: just to test backward
opt.zero_grad(set_to_none=True)
loss.backward(); opt.step()
print("loss:", float(loss))
print("OK: backward/step completed with frozen backbone.")

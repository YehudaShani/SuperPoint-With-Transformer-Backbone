import torch, timm, platform
from model.magic_point_swin import MagicPointSwin

print("Python  :", platform.python_version())
print("PyTorch :", torch.__version__)
print("CUDA?   :", torch.cuda.is_available(), " | device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("timm    :", timm.__version__)

cfg = {
  "nms":4, "det_thresh":1e-3, "topk":1000, "grid_size":8,
  "backbone":{"swin":{"name":"swin_tiny_patch4_window7_224","pretrained":True,"normalize":True,"img_size":[120,160],"freeze_backbone":True}},
  "det_head":{"feat_in_dim":192}
}
m = MagicPointSwin(cfg)
trainable = [(n, p.numel()) for n,p in m.named_parameters() if p.requires_grad]
frozen    = [(n, p.numel()) for n,p in m.named_parameters() if not p.requires_grad]
print("# trainable params :", sum(n for _,n in trainable))
print("# frozen params    :", sum(n for _,n in frozen))
print("any backbone trainable? ->", any(n.startswith("backbone") for n,_ in trainable))
print("sample trainable names :", [n for n,_ in trainable[:5]])

# PYTHONPATH=. python scripts/check_env.py
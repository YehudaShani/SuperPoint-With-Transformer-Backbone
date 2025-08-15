import cv2, torch, numpy as np
from model.magic_point_swin import MagicPointSwin

cfg = {
  "nms":4, "det_thresh":1e-3, "topk":500, "grid_size":8,
  "backbone":{"swin":{"name":"swin_tiny_patch4_window7_224","pretrained":True,"normalize":True,"img_size":[120,160]}},
  "det_head":{"feat_in_dim":192}
}
img_path = "/home/yonim/SuperPoint_Drive/data/synthetic_shapes/draw_cube/images/training/60.png"  # or any grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None: raise SystemExit(f"missing image: {img_path}")
img = cv2.resize(img, (160,120), interpolation=cv2.INTER_AREA)

t = torch.from_numpy(img)[None,None].float()/255.0
m = MagicPointSwin(cfg).eval()
with torch.no_grad(): out = m(t)
prob = out["prob_nms"][0].cpu().numpy()
ys, xs = np.where(prob >= cfg["det_thresh"])
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x,y in zip(xs,ys): cv2.circle(vis,(int(x),int(y)),1,(0,255,0),-1)
cv2.imwrite("vis_swin_kpts_120x160.png", vis)
print("wrote vis_swin_kpts_120x160.png | #kpts:", len(xs))

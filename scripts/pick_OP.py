import os, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from dataset.synthetic_shapes import SyntheticShapes
from model.magic_point_swin import MagicPointSwin
from train import do_eval

cfg_path = "config/magic_point_syn_train_swin_160x120.yaml"
ckpt     = "export/mg_syn_swin_6_0.195.pth"
with open(cfg_path,'r') as f: cfg = yaml.safe_load(f)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_ds = SyntheticShapes(cfg['data'], task=['test'], device='cpu')
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=test_ds.batch_collator, num_workers=4, pin_memory=True)

grid = {
  "det_thresh":[0.003, 0.005, 0.01, 0.02],
  "nms":[0,2,4,6],
  "topk":[500,1000,2000]
}

best = None
for th in grid["det_thresh"]:
  for n in grid["nms"]:
    for k in grid["topk"]:
      cfg2 = yaml.safe_load(open(cfg_path))  # fresh copy
      cfg2['model']['det_thresh'] = float(th)
      cfg2['model']['nms'] = int(n)
      cfg2['model']['topk'] = int(k)
      model = MagicPointSwin(cfg2['model']).to(device).eval()
      model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
      loss, P, R, fp_img, gt_img = do_eval(model, test_loader, cfg2, device)
      print(f"th={th:.3g} nms={n} topk={k} -> loss={loss:.3f} P={P:.3f} R={R:.3f} FP/img={fp_img:.1f}")
      score = P * R  # simple F-like proxy
      if (best is None) or (score > best[0]):
        best = (score, th, n, k, loss, P, R, fp_img)
print("\nBEST:", best)

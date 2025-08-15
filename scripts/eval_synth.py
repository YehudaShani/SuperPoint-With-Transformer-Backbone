# scripts/eval_synth.py
import os, yaml, torch
from torch.utils.data import DataLoader
from dataset.synthetic_shapes import SyntheticShapes
from model.magic_point_swin import MagicPointSwin
from train import do_eval, move_batch_to_device  # re-use your functions

cfg_path = "config/magic_point_syn_train_swin_160x120.yaml"
ckpt     = "export/mg_syn_swin_6_0.195.pth"

with open(cfg_path,'r') as f: cfg = yaml.safe_load(f)
cfg['solver']['test_batch_size'] = 64  # faster eval
cfg['model']['det_thresh'] = 0.1  # faster eval


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_ds = SyntheticShapes(cfg['data'], task=['test'], device='cpu')
test_loader = DataLoader(test_ds, batch_size=cfg['solver']['test_batch_size'],
                         shuffle=False, collate_fn=test_ds.batch_collator, num_workers=4, pin_memory=True)

model = MagicPointSwin(cfg['model']).to(device).eval()
sd = torch.load(ckpt, map_location='cpu')
model.load_state_dict(sd, strict=False)

loss, P, R, fp_img, gt_img = do_eval(model, test_loader, cfg, device)
print(f"[Synthetic/Test] loss={loss:.3f}  P={P:.3f}  R={R:.3f}  FP/img={fp_img:.1f}  GT/img={gt_img:.1f}")


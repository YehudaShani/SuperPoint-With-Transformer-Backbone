import numpy as np, torch, cv2
from model.magic_point_swin import MagicPointSwin
import yaml
cfg = yaml.safe_load(open("config/magic_point_syn_train_swin_160x120.yaml"))
m = MagicPointSwin(cfg['model']).eval()
m.load_state_dict(torch.load("export/mg_syn_swin_6_0.195.pth", map_location='cpu'), strict=False)
x = np.zeros((1,1,120,160), np.float32)  # pure black
with torch.no_grad():
    out = m(torch.from_numpy(x))
print("max prob on blank:", float(out['prob'].max()))

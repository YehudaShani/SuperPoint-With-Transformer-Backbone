import numpy as np
from model.magic_point import MagicPoint
import yaml, pathlib, torch
import cv2, numpy as np

# ---------- Load Config & Model ----------
cfg = yaml.safe_load(open('config/magic_point_syn_train.yaml'))['model']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_path  = pathlib.Path('sample.jpg')
ckpt_path = pathlib.Path('export/mg_syn_19_0.085.pth')

net = MagicPoint(cfg).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
net.load_state_dict(state_dict)
net.eval()

# ---------- Load Image ----------
g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
if g is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
g = cv2.resize(g, (320, 240))
t = torch.from_numpy(g / 255.).float()[None, None].to(device)

# ---------- Forward Pass ----------
with torch.no_grad():
    out = net(t)
    print("Model output keys:", out.keys())
    print("prob_nms shape:", out['prob_nms'].shape)

    prob_nms = out['prob_nms'][0].cpu().numpy()  # ✅ correct for shape [1, H, W]

ys, xs = np.where(prob_nms > 0.015)

print(f"Found {len(xs)} keypoints")

overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
for x, y in zip(xs, ys):
    cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)

# ---------- Save Result ----------
out_path = img_path.with_suffix('.magicpoint.png')
cv2.imwrite(str(out_path), overlay)
print(f'✅ Saved: {out_path}')


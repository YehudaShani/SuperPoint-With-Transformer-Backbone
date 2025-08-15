# scripts/infer_dir_magicpoint.py
# Detect keypoints on all images in a folder, always working at 120x160.
# Saves 120x160 overlays with drawn keypoints.
#
# Example:
# export PYTHONPATH=.
# python scripts/infer_dir_magicpoint.py \
#   --img_dir /content/drive/MyDrive/SuperPoint-Pytorch/data/Euroc/mav0/cam0/images_last2000 \
#   --ckpt export/finetune_swin/best.pth \
#   --out_dir /content/drive/MyDrive/SuperPoint-Pytorch/data/Euroc/mav0/cam0/images_last2000_kps \
#   --det_thresh 0.05 --nms 4 --topk 1000

import os, glob, argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.magic_point_swin import MagicPointSwin

# Fixed working size
FIXED_H = 120
FIXED_W = 160

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(img_dir, exts=("*.jpg","*.jpeg","*.png","*.bmp")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    paths.sort()
    return paths

def draw_kpts(overlay_bgr, kpts_xy, radius=2, thickness=1, color=(0,255,0)):
    for (x, y) in kpts_xy:
        cv2.circle(overlay_bgr, (int(x), int(y)), radius, color, thickness, cv2.LINE_AA)
    return overlay_bgr

@torch.no_grad()
def nms_peaks_from_prob(prob_t, det_thresh, nms_radius, topk):
    """
    prob_t: (H,W) float tensor on device, values in [0,1]
    returns: pts (N,2 int32), scores (N,)
    """
    if nms_radius > 0:
        r = int(nms_radius); k = 2*r + 1
        p4 = prob_t.unsqueeze(0).unsqueeze(0)                 # 1x1xH×W
        pooled = F.max_pool2d(p4, kernel_size=k, stride=1, padding=r)
        mask = (p4 == pooled) & (p4 >= float(det_thresh))
        ys, xs = torch.nonzero(mask[0,0], as_tuple=True)
        scores = prob_t[ys, xs]
    else:
        mask = prob_t >= float(det_thresh)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        scores = prob_t[ys, xs]

    if ys.numel() == 0:
        return np.empty((0,2), np.int32), np.empty((0,), np.float32)

    if topk is not None and int(topk) > 0 and scores.numel() > topk:
        scores, idx = torch.topk(scores, int(topk))
        ys, xs = ys[idx], xs[idx]

    pts = torch.stack([xs, ys], 1).to(torch.int32).cpu().numpy()
    scores = scores.float().cpu().numpy().astype(np.float32)
    return pts, scores

def build_model(device, H=FIXED_H, W=FIXED_W, feat_dim=192, det_thresh=0.05, nms=4, topk=1000):
    cfg = {
        "nms": nms,
        "det_thresh": det_thresh,
        "topk": topk,
        "grid_size": 8,
        "backbone": {"swin": {
            "name": "swin_tiny_patch4_window7_224",
            "pretrained": False,
            "normalize": True,
            "img_size": [H, W],  # IMPORTANT: fixed 120x160 pipeline
        }},
        "det_head": {"feat_in_dim": feat_dim},
    }
    model = MagicPointSwin(cfg).to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser("Infer dir (fixed 120x160): detect kpts and save 120x160 overlays")
    ap.add_argument("--img_dir", required=True, help="input images folder")
    ap.add_argument("--ckpt",    required=True, help="path to .pth (state_dict or {'model': state_dict})")
    ap.add_argument("--out_dir", required=True, help="where to save 120x160 overlays")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--det_thresh", type=float, default=0.05)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp")
    ap.add_argument("--save_prob", action="store_true", help="also save 120x160 prob heatmaps")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")

    # Build model
    model = build_model(device, H=FIXED_H, W=FIXED_W,
                        det_thresh=args.det_thresh, nms=args.nms, topk=args.topk)
    # Load checkpoint (support plain or wrapped)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    missing = model.load_state_dict(state, strict=False)
    print("Loaded ckpt. Missing:", len(missing.missing_keys), "Unexpected:", len(missing.unexpected_keys))
    model.eval()

    # Collect images
    exts = tuple(f"*.{e.strip()}" for e in args.exts.split(","))
    paths = list_images(args.img_dir, exts)
    if not paths:
        raise RuntimeError(f"No images in {args.img_dir} with exts={args.exts}")
    print(f"Found {len(paths)} frames. Processing at fixed {FIXED_W}x{FIXED_H} …")

    for p in tqdm(paths, ncols=100):
        stem = os.path.splitext(os.path.basename(p))[0]

        # Read → downsample immediately to 120x160 and STAY there
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        net_img = cv2.resize(gray, (FIXED_W, FIXED_H), interpolation=cv2.INTER_AREA)

        x = torch.from_numpy(net_img.astype(np.float32)/255.0)[None, None].to(device)  # 1x1x120x160

        with torch.no_grad():
            out = model(x)
            prob_t = out["prob"][0]  # (H,W) tensor on device in [0,1]

        # NMS @ fixed res
        pts, scores = nms_peaks_from_prob(prob_t, args.det_thresh, args.nms, args.topk)

        # Make 120x160 overlay & (optional) prob heatmap — both saved at 120x160
        overlay = cv2.cvtColor(net_img, cv2.COLOR_GRAY2BGR)
        overlay = draw_kpts(overlay, pts, radius=2, thickness=1, color=(0,255,0))
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_kps.jpg"), overlay)

        if args.save_prob:
            prob_u8 = np.clip(prob_t.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out_dir, f"{stem}_prob.jpg"), prob_u8)

    print(f"Done. 120x160 overlays saved to: {args.out_dir}")

if __name__ == "__main__":
    main()

# %env PYTHONPATH=.
# !python scripts/homographic_adaptation.py \
#   --img_dir /content/drive/MyDrive/SuperPoint-Pytorch/data/real/val2017 \
#   --ckpt /content/drive/MyDrive/SuperPoint-Pytorch/export/mg_syn_swin_15_0.176.pth \
#   --out_dir /content/drive/MyDrive/SuperPoint-Pytorch/runs/ha_out \
#   --H 120 --W 160 \
#   --num_aug 32 --agg mean \
#   --det_thresh 0.015 --nms 4 --topk 1000 \
#   --valid_margin -1 --erode_border -1 --save_viz --save_npys --seed 0
#
# Homographic Adaptation (HA) for MagicPoint/MagicPointSwin.
# Changes:
# - Adds a rectangular border crop (--erode_border). We zero-out agg_prob within
#   that border and also drop any points there before saving overlays.
# - Default --erode_border = --nms (if -1), so out-of-the-box it matches your NMS radius.

import os, argparse, glob, random, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.magic_point_swin import MagicPointSwin

# -------------------------- Small utils --------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def imread_gray(path, size_hw):
    H, W = size_hw
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if (img.shape[0], img.shape[1]) != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img

def to_tensor_gray_u8(img_u8):
    ten = torch.from_numpy(img_u8.astype(np.float32)/255.0)[None, None]  # 1×1×H×W
    return ten

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(img_dir, exts=("*.jpg","*.jpeg","*.png","*.bmp")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    paths.sort()
    return paths

# ----- Homography sampling -----

def sample_homography(H, W,
                      max_angle_deg=15.0,
                      scale_amplitude=0.2,
                      trans_amplitude=0.1,
                      persp_amplitude=0.001):
    """Small random sim+perspective around image center."""
    cx, cy = W/2.0, H/2.0
    ang = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], np.float32)
    s = 1.0 + np.random.uniform(-scale_amplitude, scale_amplitude)
    S = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    tx = np.random.uniform(-trans_amplitude, trans_amplitude) * W
    ty = np.random.uniform(-trans_amplitude, trans_amplitude) * H
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float32)
    px = np.random.uniform(-persp_amplitude, persp_amplitude)
    py = np.random.uniform(-persp_amplitude, persp_amplitude)
    P = np.array([[1,0,0],[0,1,0],[px,py,1]], np.float32)
    C1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
    C2 = np.array([[1,0, cx],[0,1, cy],[0,0,1]], np.float32)
    H_A2B = C2 @ (P @ (T @ (R @ (S @ C1))))
    return H_A2B.astype(np.float32)

# ----- NMS on prob map (same as infer_one style) -----

@torch.no_grad()
def nms_peaks_from_prob(prob_t, det_thresh, nms_radius, topk):
    """
    prob_t: torch.Tensor (H,W) on any device, values in [0,1]
    return: pts (N,2 int32), scores (N,)
    """
    if nms_radius > 0:
        r = int(nms_radius); k = 2*r + 1
        p4 = prob_t.unsqueeze(0).unsqueeze(0)                # 1×1×H×W
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

# -------------------------- Core HA per image --------------------------

@torch.no_grad()
def homographic_adaptation_one(img_u8, model, size_hw, num_aug=32, agg="mean",
                               det_thresh=0.015, nms_radius=4, topk=1000,
                               valid_margin=-1, erode_border=-1,
                               max_angle_deg=15.0, scale_amp=0.2, trans_amp=0.1, persp_amp=0.001,
                               device=torch.device("cpu")):
    """
    Returns:
      agg_prob   (H,W) float32 in [0,1], with rectangular border zeroed if erode_border>0
      pts, scores after NMS on agg_prob
    """
    H, W = size_hw

    # storage for aggregation
    if agg == "max":
        agg_prob = np.zeros((H, W), np.float32)
    else:  # mean (weighted)
        num = np.zeros((H, W), np.float32)
        den = np.zeros((H, W), np.float32)

    # margin for warped-valid suppression
    erode_r = valid_margin if valid_margin >= 0 else max(int(nms_radius), 1)
    kernel = np.ones((2*erode_r+1, 2*erode_r+1), np.uint8) if erode_r > 0 else None

    # rectangular border crop (applied after aggregation)
    rb = erode_border if erode_border >= 0 else int(nms_radius)

    # always include identity
    Hs = [np.eye(3, dtype=np.float32)]
    for _ in range(num_aug):
        Hs.append(sample_homography(H, W, max_angle_deg, scale_amp, trans_amp, persp_amp))

    # loop
    for H_A2B in Hs:
        # warp image A->B
        imgB = cv2.warpPerspective(img_u8, H_A2B, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # valid mask for B (then back to A using inv)
        maskA = np.ones_like(img_u8, np.uint8) * 255
        maskB = cv2.warpPerspective(maskA, H_A2B, (W, H),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # run model on B
        xB = to_tensor_gray_u8(imgB).to(device)
        outB = model(xB)                         # dict with 'prob' (B,H,W)
        probB = outB["prob"][0].float().cpu().numpy()  # (H,W) np.float32

        # warp prob back to A with inverse homography
        H_B2A = np.linalg.inv(H_A2B)
        probA = cv2.warpPerspective(probB, H_B2A, (W, H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wB = (maskB > 0).astype(np.uint8)
        wA = cv2.warpPerspective(wB, H_B2A, (W, H),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

        # erode the warped-valid area to kill borders introduced by warps
        if erode_r > 0:
            wA = cv2.erode(wA.astype(np.uint8), kernel, 1).astype(bool)

        if agg == "max":
            agg_prob[wA] = np.maximum(agg_prob[wA], probA[wA])
        else:
            num[wA] += probA[wA]
            den[wA] += 1.0

    if agg == "max":
        agg_prob = agg_prob
    else:
        den = np.maximum(den, 1e-6)
        agg_prob = (num / den).astype(np.float32)

    # rectangular border zero-out to forbid edge peaks in overlays
    if rb > 0:
        agg_prob[:rb, :] = 0
        agg_prob[-rb:, :] = 0
        agg_prob[:, :rb] = 0
        agg_prob[:, -rb:] = 0

    # NMS over the aggregated RAW map (scores from agg_prob)
    prob_t = torch.from_numpy(agg_prob)
    pts, scores = nms_peaks_from_prob(prob_t, det_thresh, nms_radius, topk)

    # redundant safety: drop any remaining points within rb pixels of the border
    if rb > 0 and len(pts) > 0:
        keep = (pts[:,0] >= rb) & (pts[:,0] < W - rb) & (pts[:,1] >= rb) & (pts[:,1] < H - rb)
        pts, scores = pts[keep], scores[keep]

    return agg_prob, pts, scores

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser("Homographic Adaptation for MagicPointSwin")
    ap.add_argument("--img_dir", required=True, help="folder with unlabeled real images")
    ap.add_argument("--ckpt",    required=True, help="path to .pth checkpoint")
    ap.add_argument("--out_dir", default="runs/ha_out", help="where to save pseudo labels")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)

    # HA settings
    ap.add_argument("--num_aug", type=int, default=32, help="random homographies (in addition to identity)")
    ap.add_argument("--agg", choices=["mean","max"], default="mean")

    # detector postproc for final pseudo-kpts
    ap.add_argument("--det_thresh", type=float, default=0.015)
    ap.add_argument("--nms", type=int, default=4, help="radius px; 0 disables NMS")
    ap.add_argument("--topk", type=int, default=1000)

    # homography ranges
    ap.add_argument("--max_angle_deg", type=float, default=15.0)
    ap.add_argument("--scale_amp",     type=float, default=0.2)
    ap.add_argument("--trans_amp",     type=float, default=0.1)
    ap.add_argument("--persp_amp",     type=float, default=0.001)

    # border controls
    ap.add_argument("--valid_margin", type=int, default=-1,
                    help="erode warped valid mask by this radius (default: use --nms)")
    ap.add_argument("--erode_border", type=int, default=-1,
                    help="rectangular border crop in pixels for agg_prob & points (default: use --nms)")

    # I/O & misc
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp")
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--save_npys", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")

    # ---- Build model ----
    cfg = {
        "nms": args.nms,
        "det_thresh": args.det_thresh,
        "topk": args.topk,
        "grid_size": 8,
        "backbone": {"swin": {
            "name": "swin_tiny_patch4_window7_224",
            "pretrained": True,
            "normalize": True,
            "img_size": [args.H, args.W],
        }},
        "det_head": {"feat_in_dim": 192},
    }
    model = MagicPointSwin(cfg).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # ---- Images ----
    exts = tuple(f"*.{e.strip()}" for e in args.exts.split(","))
    paths = list_images(args.img_dir, exts)
    if not paths:
        raise RuntimeError(f"No images found in {args.img_dir} with exts={args.exts}")

    print(f"Found {len(paths)} images. Writing to: {args.out_dir}")

    H, W = args.H, args.W
    t0 = time.time()
    for p in tqdm(paths, ncols=100):
        name = os.path.splitext(os.path.basename(p))[0]
        out_base = os.path.join(args.out_dir, name)

        # read & HA
        imgA = imread_gray(p, (H, W))
        agg_prob, pts, scores = homographic_adaptation_one(
            imgA, model, (H, W),
            num_aug=args.num_aug, agg=args.agg,
            det_thresh=args.det_thresh, nms_radius=args.nms, topk=args.topk,
            valid_margin=args.valid_margin, erode_border=args.erode_border,
            max_angle_deg=args.max_angle_deg, scale_amp=args.scale_amp,
            trans_amp=args.trans_amp, persp_amp=args.persp_amp,
            device=device
        )

        # save npys (prob + kpts)
        if args.save_npys:
            np.save(out_base + "_agg_prob.npy", agg_prob)
            np.save(out_base + "_kpts.npy", pts.astype(np.int32))
            np.save(out_base + "_scores.npy", scores.astype(np.float32))

        # viz
        if args.save_viz:
            # heatmap jpg (0..255)
            hm = np.clip(agg_prob * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(out_base + "_agg_prob.jpg", hm)

            # overlay (points already filtered by erode_border)
            vis = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
            for (x, y) in pts:
                cv2.circle(vis, (int(x), int(y)), 2, (0,255,0), 1, cv2.LINE_AA)
            cv2.imwrite(out_base + "_overlay.jpg", vis)

    dt = time.time() - t0
    print(f"Done HA on {len(paths)} images in {dt:.1f}s")
    print(f"Pseudo-labels saved in {args.out_dir}")
    print("Use these *_agg_prob.npy or *_kpts.npy to fine-tune your detector on real images.")

if __name__ == "__main__":
    main()

# scripts/homography_consistency_dir.py
#
# Evaluate homography consistency over a directory of images.
# Metrics per image:
#   - rep_min  = Nc / min(Na, Nb_valid)
#   - rep_sym  = 2*Nc / (Na + Nb_valid)
#   - mre_px   = mean reprojection error (px)
#   - border_fp_rate = 1 - Nb_valid / max(1, Nb_raw)
 # scripts/homography_consistency_dir.py
#
# Evaluate homography consistency for a directory of images.
# Works with either a Swin-based or CNN-based MagicPoint by using --arch {swin, cnn}.
#
# Example:
# export PYTHONPATH=.
# python scripts/homography_consistency_dir.py \
#   --arch swin \
#   --img_dir /path/to/images \
#   --ckpt export/mg_syn_swin_15_0.176.pth \
#   --csv_out runs/homo_dir/swin.csv \
#   --H 120 --W 160 --det_thresh 0.015 --nms 4 --topk 1000 \
#   --px_thresh 3.0 --valid_margin -1 --trials 1 --seed 123
#
# And for a CNN MagicPoint checkpoint:
# python scripts/homography_consistency_dir.py \
#   --arch cnn \
#   --img_dir /path/to/images \
#   --ckpt export/mg_syn_cnn_xxx.pth \
#   --csv_out runs/homo_dir/cnn.csv \
#   --H 120 --W 160 --det_thresh 0.015 --nms 4 --topk 1000 \
#   --px_thresh 3.0 --valid_margin -1 --trials 1 --seed 123
#
# Notes:
# - We use local-max NMS ONLY to pick locations; scores always come from the RAW prob map.
# - Border-FP rate is computed on B **before** filtering (detections landing outside
#   the eroded warped-valid mask). We still filter B by the mask before matching.

import os, csv, glob, math, random, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm


# -------------------------- Utilities --------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    d = p if os.path.splitext(p)[1] == "" else os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def list_images(img_dir, exts=("*.jpg","*.jpeg","*.png","*.bmp")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    paths.sort()
    return paths

def load_image_gray(path, size_hw):
    H, W = size_hw
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if (img.shape[0], img.shape[1]) != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy((img.astype(np.float32)/255.0))[None,None,...]  # 1×1×H×W
    return img, ten

def rand_homography(H, W,
                    max_angle_deg=15,
                    scale_amplitude=0.2,
                    trans_amplitude=0.1,
                    persp_amplitude=0.001):
    """Small random similarity + mild perspective around image center."""
    cx, cy = W/2.0, H/2.0
    ang = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, 0],
                  [sa,  ca, 0],
                  [ 0,   0, 1]], dtype=np.float32)

    s = 1.0 + np.random.uniform(-scale_amplitude, scale_amplitude)
    S = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]], dtype=np.float32)

    tx = np.random.uniform(-trans_amplitude, trans_amplitude) * W
    ty = np.random.uniform(-trans_amplitude, trans_amplitude) * H
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float32)

    px = np.random.uniform(-persp_amplitude, persp_amplitude)
    py = np.random.uniform(-persp_amplitude, persp_amplitude)
    P = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [px, py, 1]], dtype=np.float32)

    C1 = np.array([[1, 0, -cx],[0, 1, -cy],[0, 0, 1]], dtype=np.float32)
    C2 = np.array([[1, 0,  cx],[0, 1,  cy],[0, 0, 1]], dtype=np.float32)

    Hmat = C2 @ (P @ (T @ (R @ (S @ C1))))
    return Hmat.astype(np.float32)

def warp_image_and_mask(img_u8, H_A2B, size_hw):
    H, W = size_hw
    imgB = cv2.warpPerspective(img_u8, H_A2B, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    maskA = np.ones_like(img_u8, dtype=np.uint8)*255
    maskB = cv2.warpPerspective(maskA, H_A2B, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    validB = (maskB > 0)
    return imgB, validB

def nms_peaks_from_prob(prob_t, det_thresh, nms_radius, topk):
    """
    Simple local-max NMS on the RAW map:
    - pick locations using max-pooling equality & threshold
    - scores are always pulled from the RAW map (not the pooled map)
    Inputs:
      prob_t: (H,W) torch float tensor on any device, in [0,1]
    Returns:
      pts: (N,2) numpy int32 [x,y]
      scores: (N,) numpy float32
    """
    if nms_radius > 0:
        r = int(nms_radius); k = 2*r + 1
        p4 = prob_t.unsqueeze(0).unsqueeze(0)       # 1×1×H×W
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

def nn_pairs_under_H(ptsA, ptsB, H_A2B, H, W, px_thresh=3.0, validB=None):
    """One-way NN from A→B projections under H, with optional valid mask filter."""
    if len(ptsA) == 0 or len(ptsB) == 0:
        return np.empty((0,2), np.int32), np.empty((0,2), np.int32), np.array([])

    ones = np.ones((len(ptsA),1), dtype=np.float32)
    ph = np.hstack([ptsA.astype(np.float32), ones])    # N×3 in A
    ph2 = (H_A2B @ ph.T).T
    ph2 = ph2[:, :2] / (ph2[:, 2:3] + 1e-8)            # projected to B

    # inside B bounds
    inb = (ph2[:,0]>=0)&(ph2[:,0]<W)&(ph2[:,1]>=0)&(ph2[:,1]<H)
    if validB is not None:
        valid_idx = ph2[inb].round().astype(int)
        inb_sub = validB[valid_idx[:,1].clip(0,H-1), valid_idx[:,0].clip(0,W-1)]
        tmp = np.zeros_like(inb); tmp[np.where(inb)[0]] = inb_sub
        inb = tmp.astype(bool)

    A_keep = ptsA[inb]
    A_proj = ph2[inb]
    if len(A_keep) == 0:
        return np.empty((0,2), np.int32), np.empty((0,2), np.int32), np.array([])

    # NN search
    d2 = (ptsB[None,:,0]-A_proj[:,None,0])**2 + (ptsB[None,:,1]-A_proj[:,None,1])**2
    nn_idx  = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(A_proj)), nn_idx])
    mask = nn_dist <= float(px_thresh)

    A_sel = A_keep[mask].astype(np.int32)
    B_sel = ptsB[nn_idx[mask]].astype(np.int32)
    e_pix = nn_dist[mask]
    return A_sel, B_sel, e_pix

def draw_kpts(overlay_bgr, kpts_xy, color=(0,255,0), radius=2, thickness=1):
    for (x, y) in kpts_xy:
        cv2.circle(overlay_bgr, (int(x), int(y)), radius, color, thickness, cv2.LINE_AA)
    return overlay_bgr

def draw_matches_side_by_side(imgA_u8, imgB_u8, pairsA, pairsB, max_draw=200):
    H, W = imgA_u8.shape
    canvas = np.zeros((H, W*2, 3), np.uint8)
    canvas[:, :W] = cv2.cvtColor(imgA_u8, cv2.COLOR_GRAY2BGR)
    canvas[:,  W:] = cv2.cvtColor(imgB_u8, cv2.COLOR_GRAY2BGR)
    n = min(max_draw, len(pairsA))
    for i in range(n):
        x1,y1 = map(int, pairsA[i]); x2,y2 = map(int, pairsB[i])
        cv2.circle(canvas, (x1, y1), 2, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x2+W, y2), 2, (0,0,255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x1,y1), (x2+W,y2), (0,255,0), 1, cv2.LINE_AA)
    return canvas


# -------------------------- Model factory --------------------------

def build_model(arch, H, W, det_thresh, nms, topk, device):
    """
    Returns a model that exposes out['prob'] with shape (B,H,W) in [0,1].
    Modify the CNN import if your repo uses a different class/module name.
    """
    if arch == "swin":
        from model.magic_point_swin import MagicPointSwin
        cfg = {
            "nms": nms, "det_thresh": det_thresh, "topk": topk,
            "grid_size": 8,
            "backbone": {"swin": {
                "name": "swin_tiny_patch4_window7_224",
                "pretrained": False, "normalize": True,
                "img_size": [H, W],
            }},
            "det_head": {"feat_in_dim": 192},
        }
        model = MagicPointSwin(cfg)
    elif arch == "cnn":
        from model.magic_point import MagicPoint
        # IMPORTANT: VGGBackboneBN expects a 'vgg' sub-dict with 'channels'
        # The common SuperPoint-like layout uses 8 convs with 64/128 widths,
        # yielding 128 channels at the output used by the detector head.
        vgg_cfg = {
            "channels": [64, 64, 64, 64, 128, 128, 128, 128],  # <-- required
            # Optional/harmless extras if your backbone uses them:
            "normalize": True,
            "img_size": [H, W],
        }
        cfg = {
            "nms": nms, "det_thresh": det_thresh, "topk": topk,
            "grid_size": 8,
            "backbone": {"vgg": vgg_cfg},
            "det_head": {"feat_in_dim": 128},  # match last channel above
        }
        model = MagicPoint(cfg)

    else:
        raise ValueError(f"Unknown --arch '{arch}'. Use 'swin' or 'cnn'.")

    model.to(device).eval()
    return model


# -------------------------- Main loop per image --------------------------

@torch.no_grad()
def evaluate_one_image(img_path, model, device, args, trial_seed=None, viz_dir=None):
    H, W = args.H, args.W

    # deterministic per-trial RNG if provided
    if trial_seed is not None:
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)

    # Load A and create B via random H
    imgA_u8, xA = load_image_gray(img_path, (H, W))
    H_A2B = rand_homography(H, W,
                            max_angle_deg=args.max_angle_deg,
                            scale_amplitude=args.scale_amp,
                            trans_amplitude=args.trans_amp,
                            persp_amplitude=args.persp_amp)
    imgB_u8, validB = warp_image_and_mask(imgA_u8, H_A2B, (H, W))

    # Erode valid border (default radius = NMS radius if --valid_margin < 0)
    erode_r = args.valid_margin if args.valid_margin >= 0 else max(int(args.nms), 1)
    kernel = np.ones((2*erode_r+1, 2*erode_r+1), np.uint8) if erode_r > 0 else None
    if kernel is not None:
        validB = cv2.erode(validB.astype(np.uint8), kernel, 1).astype(bool)

    # Run detector
    xA = xA.to(device)
    xB = torch.from_numpy((imgB_u8.astype(np.float32)/255.0))[None,None].to(device)

    with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
        outA = model(xA)
        outB = model(xB)

    # RAW prob maps
    probA_t = outA["prob"][0].float()  # (H,W)
    probB_t = outB["prob"][0].float()  # (H,W)

    # NMS → keypoints (locations from NMS, scores from RAW)
    ptsA, scA = nms_peaks_from_prob(probA_t, args.det_thresh, args.nms, args.topk)
    ptsB_raw, scB_raw = nms_peaks_from_prob(probB_t, args.det_thresh, args.nms, args.topk)

    # Border FP rate on B BEFORE filtering
    Nb_raw = len(ptsB_raw)
    if Nb_raw > 0:
        keep_mask_border = validB[ptsB_raw[:,1].clip(0,H-1), ptsB_raw[:,0].clip(0,W-1)]
        border_fp_count = int((~keep_mask_border).sum())
        border_fp_rate = border_fp_count / float(Nb_raw)
    else:
        border_fp_count = 0
        border_fp_rate = 0.0

    # Filter B by valid mask for matching
    if Nb_raw > 0:
        keep_mask = keep_mask_border
        ptsB = ptsB_raw[keep_mask]
        scB  = scB_raw[keep_mask]
    else:
        ptsB = np.empty((0,2), np.int32); scB = np.empty((0,), np.float32)

    # A→B matches under H
    A_sel, B_sel, e_pix = nn_pairs_under_H(
        ptsA, ptsB, H_A2B, H, W, px_thresh=args.px_thresh, validB=validB
    )

    # Metrics
    Na = len(ptsA)
    Nb = len(ptsB)
    M  = len(A_sel)
    rep_min = M / max(1, min(Na, Nb))
    rep_sym = M / max(1, 0.5*(Na + Nb))
    mean_err = float(e_pix.mean()) if len(e_pix) else float("nan")

    # Optional visualization
    if viz_dir:
        base = os.path.splitext(os.path.basename(img_path))[0]
        visA = draw_kpts(cv2.cvtColor(imgA_u8, cv2.COLOR_GRAY2BGR), ptsA, (0,255,0))
        visB = draw_kpts(cv2.cvtColor(imgB_u8, cv2.COLOR_GRAY2BGR), ptsB_raw, (0,255,0))
        cv2.imwrite(os.path.join(viz_dir, f"{base}_A.jpg"), visA)
        cv2.imwrite(os.path.join(viz_dir, f"{base}_B.jpg"), visB)
        side = draw_matches_side_by_side(imgA_u8, imgB_u8, A_sel, B_sel, max_draw=args.viz_max)
        cv2.imwrite(os.path.join(viz_dir, f"{base}_matches.jpg"), side)

    return {
        "img": os.path.basename(img_path),
        "Na": Na, "Nb_raw": Nb_raw, "Nb": Nb, "M": M,
        "rep_min": rep_min, "rep_sym": rep_sym, "mre": mean_err,
        "border_fp_rate": border_fp_rate, "border_fp_count": border_fp_count
    }


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser("Homography consistency over a directory (Swin/CNN)")
    ap.add_argument("--arch", choices=["swin","cnn"], default="swin",
                    help="which backbone to evaluate")
    ap.add_argument("--img_dir", required=True, help="folder with images")
    ap.add_argument("--ckpt",    required=True, help="path to model .pth")
    ap.add_argument("--csv_out", required=True, help="where to write the CSV")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)

    # detector / NMS
    ap.add_argument("--det_thresh", type=float, default=0.015)
    ap.add_argument("--nms", type=int, default=4, help="radius px; 0 disables NMS")
    ap.add_argument("--topk", type=int, default=1000)

    # homography & matching
    ap.add_argument("--px_thresh", type=float, default=3.0)
    ap.add_argument("--max_angle_deg", type=float, default=15.0)
    ap.add_argument("--scale_amp",     type=float, default=0.2)
    ap.add_argument("--trans_amp",     type=float, default=0.1)
    ap.add_argument("--persp_amp",     type=float, default=0.001)

    # border FP control
    ap.add_argument("--valid_margin", type=int, default=-1,
                    help="erode warped valid mask by this radius (default: use --nms)")

    # multi-trials per image
    ap.add_argument("--trials", type=int, default=1, help="# random homos per image (avg metrics)")
    ap.add_argument("--seed", type=int, default=123)

    # optional viz
    ap.add_argument("--viz_dir", default="", help="save per-image visualizations here")
    ap.add_argument("--viz_max", type=int, default=200)

    # image extensions
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp")
    args = ap.parse_args()

    set_seed(args.seed)
    paths = list_images(args.img_dir, tuple(f"*.{e.strip()}" for e in args.exts.split(",")))
    if not paths:
        raise RuntimeError(f"No images found in {args.img_dir} (exts={args.exts})")

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")

    # Build model & load weights
    model = build_model(args.arch, args.H, args.W, args.det_thresh, args.nms, args.topk, device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    missing = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint. Missing:", len(missing.missing_keys), "Unexpected:", len(missing.unexpected_keys))
    model.to(device).eval()

    # I/O
    ensure_dir(args.csv_out)
    if args.viz_dir:
        ensure_dir(args.viz_dir)

    # CSV header
    header = ["img", "Na", "Nb_raw", "Nb", "M", "rep_min", "rep_sym", "mre",
              "border_fp_rate", "border_fp_count"]
    rows = []
    sums = {k: 0.0 for k in header if k not in ["img"]}

    print(f"Evaluating {len(paths)} images (trials per image = {args.trials}) ...")
    for idx, p in enumerate(tqdm(paths, ncols=100)):
        # Average over trials (different random homographies)
        acc = {k: 0.0 for k in sums.keys()}
        for t in range(args.trials):
            trial_seed = args.seed + (idx * 131 + t * 977)  # stable per-image-per-trial seed
            res = evaluate_one_image(
                p, model, device, args,
                trial_seed=trial_seed,
                viz_dir=(args.viz_dir if (args.viz_dir and t == 0) else None)  # viz only once
            )
            for k in acc.keys():
                acc[k] += float(res[k])

        for k in acc.keys():
            acc[k] /= float(args.trials)

        row = [os.path.basename(p)] + [acc[k] for k in header if k not in ["img"]]
        rows.append(row)
        for k in acc.keys():
            sums[k] += acc[k]

    # Write CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

        # Summary row (mean over images)
        n = float(len(paths))
        mean_vals = [("MEAN",)]
        for k in header[1:]:
            mean_vals[0] += (sums[k] / n,)
        w.writerow([])
        w.writerow(["MEAN"] + [f"{v:.6f}" for v in mean_vals[0][1:]])

    # Print summary
    n = float(len(paths))
    print("\n=== Summary (mean over images) ===")
    print(f"Na:       {sums['Na']/n:.2f}")
    print(f"Nb_raw:   {sums['Nb_raw']/n:.2f}")
    print(f"Nb:       {sums['Nb']/n:.2f}")
    print(f"M:        {sums['M']/n:.2f}")
    print(f"Rep(min): {sums['rep_min']/n:.4f}")
    print(f"Rep(sym): {sums['rep_sym']/n:.4f}")
    print(f"MRE(px):  {sums['mre']/n:.3f}")
    print(f"Border FP rate: {(sums['border_fp_rate']/n):.4f}")

    print(f"\nCSV saved → {args.csv_out}")
    if args.viz_dir:
        print(f"Visualizations in → {args.viz_dir}")


if __name__ == "__main__":
    main()

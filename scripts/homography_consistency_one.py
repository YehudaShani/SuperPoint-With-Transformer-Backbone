# scripts/homography_consistency_one.py
#
# Example:
#   export PYTHONPATH=.
#   python scripts/homography_consistency_one.py \
#     --img /content/drive/MyDrive/SuperPoint-Pytorch/sample.jpg \
#     --ckpt /content/drive/MyDrive/SuperPoint-Pytorch/export/mg_syn_swin_15_0.176.pth \
#     --out_prefix runs/homo_one \
#     --H 120 --W 160 \
#     --det_thresh 0.015 --nms 4 --topk 30 \
#     --px_thresh 3.0 --viz_max 20 --seed 123
#
# Notes:
# - NMS is identical to scripts/infer_one.py (local max via max-pool on RAW prob).
# - Scores are always taken from the RAW prob map.
# - Border masking is OFF by default here for apples-to-apples counts with infer_one.py.
#   If you want it, pass --valid_margin R (e.g., R=4).

import os, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from model.magic_point_swin import MagicPointSwin


# -------------------------- Utils --------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_image_gray(path, size_hw=(120, 160)):
    H, W = size_hw
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    ten = torch.from_numpy(img_f)[None, None, ...]  # 1×1×H×W
    return img, ten

def rand_homography(H, W,
                    max_angle_deg=15,
                    scale_amplitude=0.2,
                    trans_amplitude=0.1,
                    persp_amplitude=0.001):
    cx, cy = W / 2.0, H / 2.0
    ang = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, 0],
                  [sa,  ca, 0],
                  [ 0,   0, 1]], np.float32)
    s = 1.0 + np.random.uniform(-scale_amplitude, scale_amplitude)
    S = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]], np.float32)
    tx = np.random.uniform(-trans_amplitude, trans_amplitude) * W
    ty = np.random.uniform(-trans_amplitude, trans_amplitude) * H
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], np.float32)
    px = np.random.uniform(-persp_amplitude, persp_amplitude)
    py = np.random.uniform(-persp_amplitude, persp_amplitude)
    P = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [px, py, 1]], np.float32)
    C1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,   1]], np.float32)
    C2 = np.array([[1, 0,  cx],
                   [0, 1,  cy],
                   [0, 0,   1]], np.float32)
    return (C2 @ (P @ (T @ (R @ (S @ C1))))).astype(np.float32)

def warp_image_and_mask(img_u8, H_A2B, size_hw):
    H, W = size_hw
    imgB = cv2.warpPerspective(img_u8, H_A2B, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    maskA = np.ones_like(img_u8, np.uint8) * 255
    maskB = cv2.warpPerspective(maskA, H_A2B, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    validB = (maskB > 0)
    return imgB, validB

def draw_kpts(overlay_bgr, kpts_xy, color=(0, 255, 0), radius=2, thickness=1):
    for (x, y) in kpts_xy:
        cv2.circle(overlay_bgr, (int(x), int(y)), radius, color, thickness, cv2.LINE_AA)
    return overlay_bgr

def nn_pairs_under_H(ptsA, ptsB, H_A2B, H, W, px_thresh=3.0, validB=None):
    """
    For each A keypoint, project by H_A2B and take NN in B; accept if within px_thresh.
    If validB (bool mask) is provided, reject projections that land in invalid areas.
    """
    if len(ptsA) == 0 or len(ptsB) == 0:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32), np.array([])

    ones = np.ones((len(ptsA), 1), np.float32)
    ph = np.hstack([ptsA.astype(np.float32), ones])  # N×3
    ph2 = (H_A2B @ ph.T).T
    ph2 = ph2[:, :2] / (ph2[:, 2:3] + 1e-8)         # projected A→B
    inb = (ph2[:, 0] >= 0) & (ph2[:, 0] < W) & (ph2[:, 1] >= 0) & (ph2[:, 1] < H)

    if validB is not None:
        valid_idx = ph2[inb].round().astype(int)
        inb_sub = validB[valid_idx[:, 1], valid_idx[:, 0]]
        tmp = np.zeros_like(inb); tmp[np.where(inb)[0]] = inb_sub
        inb = tmp.astype(bool)

    A_keep = ptsA[inb]
    A_proj = ph2[inb]
    if len(A_keep) == 0:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32), np.array([])

    d2 = (ptsB[None, :, 0] - A_proj[:, None, 0])**2 + (ptsB[None, :, 1] - A_proj[:, None, 1])**2
    nn_idx  = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(A_proj)), nn_idx])
    mask = nn_dist <= px_thresh

    A_sel = A_keep[mask].astype(np.int32)
    B_sel = ptsB[nn_idx[mask]].astype(np.int32)
    e_pix = nn_dist[mask]
    return A_sel, B_sel, e_pix

def draw_matches_side_by_side(imgA_u8, imgB_u8, pairsA, pairsB, max_draw=200):
    H, W = imgA_u8.shape
    canvas = np.zeros((H, W * 2, 3), np.uint8)
    canvas[:, :W] = cv2.cvtColor(imgA_u8, cv2.COLOR_GRAY2BGR)
    canvas[:,  W:] = cv2.cvtColor(imgB_u8, cv2.COLOR_GRAY2BGR)
    n = min(max_draw, len(pairsA))
    for i in range(n):
        x1, y1 = map(int, pairsA[i]); x2, y2 = map(int, pairsB[i])
        cv2.circle(canvas, (x1, y1), 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x2 + W, y2), 2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x2 + W, y2), (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


# ----- NMS identical to infer_one.py -----

def logits_to_points(out_dict, det_thresh, topk, nms_used=True, nms_radius=4):
    """
    Use local-max NMS on the RAW prob map (if enabled), and take scores from RAW map.
    This mirrors scripts/infer_one.py.
    Returns:
      pts: (N,2) int32 (x,y)
      scores: (N,) float32
    """
    prob_raw_t = out_dict["prob"][0]  # (H, W) torch tensor

    if nms_used and nms_radius > 0:
        r = int(nms_radius)
        k = 2 * r + 1
        p4 = prob_raw_t.unsqueeze(0).unsqueeze(0)  # 1×1×H×W
        pooled = F.max_pool2d(p4, kernel_size=k, stride=1, padding=r)
        peaks_mask = (p4 == pooled) & (p4 >= float(det_thresh))
        ys, xs = torch.nonzero(peaks_mask.squeeze(0).squeeze(0), as_tuple=True)
        scores_t = prob_raw_t[ys, xs]
    else:
        mask = prob_raw_t >= float(det_thresh)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        scores_t = prob_raw_t[ys, xs]

    # Top-K on raw scores
    ys_np = ys.cpu().numpy()
    xs_np = xs.cpu().numpy()
    scores = scores_t.float().cpu().numpy()

    if scores.size:
        order = np.argsort(-scores)
        if topk is not None and int(topk) > 0:
            order = order[:min(int(topk), len(order))]
        xs_np, ys_np, scores = xs_np[order], ys_np[order], scores[order]
        pts = np.stack([xs_np, ys_np], 1).astype(np.int32)
    else:
        pts = np.empty((0, 2), np.int32)
        scores = np.empty((0,), np.float32)

    return pts, scores


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser("Homography consistency on one image (+matches viz)")
    ap.add_argument("--img",  required=True, help="path to grayscale image (any size)")
    ap.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    ap.add_argument("--out_prefix", default="runs/homo_one", help="prefix for outputs")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)
    # detector runtime
    ap.add_argument("--det_thresh", type=float, default=0.015)
    ap.add_argument("--nms", type=int, default=4, help="radius in px; 0 disables NMS")
    ap.add_argument("--topk", type=int, default=1000)
    # homography
    ap.add_argument("--max_angle_deg", type=float, default=15.0)
    ap.add_argument("--scale_amp", type=float, default=0.2)
    ap.add_argument("--trans_amp", type=float, default=0.1)
    ap.add_argument("--persp_amp", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    # matching/vis
    ap.add_argument("--px_thresh", type=float, default=3.0, help="NN acceptance threshold (pixels)")
    ap.add_argument("--viz_max", type=int, default=200)
    # border FP control (OFF by default to match infer_one counts)
    ap.add_argument("--valid_margin", type=int, default=-1,
                    help=">=0: erode warped valid mask by this radius and filter B keypoints; -1: disable")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # ---- Build model (match infer_one.py cfg) ----
    cfg = {
        "nms": args.nms,
        "det_thresh": args.det_thresh,
        "topk": args.topk,
        "grid_size": 8,
        "backbone": {"swin": {
            "name": "swin_tiny_patch4_window7_224",
            "pretrained": False,
            "normalize": True
        }},
        "det_head": {"feat_in_dim": 192},
    }
    model = MagicPointSwin(cfg).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if ("model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # ---- Prepare images ----
    imgA_u8, xA = load_image_gray(args.img, size_hw=(args.H, args.W))
    H_A2B = rand_homography(args.H, args.W,
                            max_angle_deg=args.max_angle_deg,
                            scale_amplitude=args.scale_amp,
                            trans_amplitude=args.trans_amp,
                            persp_amplitude=args.persp_amp)
    imgB_u8, validB = warp_image_and_mask(imgA_u8, H_A2B, (args.H, args.W))

    # Optional: erode & use valid mask to drop border FPs in B
    if args.valid_margin is not None and int(args.valid_margin) >= 0:
        r = int(args.valid_margin)
        if r > 0:
            k = 2 * r + 1
            validB = cv2.erode(validB.astype(np.uint8), np.ones((k, k), np.uint8), 1).astype(bool)
    else:
        validB = None  # keep counts directly comparable to infer_one.py

    xA = xA.to(device)
    xB = torch.from_numpy((imgB_u8.astype(np.float32) / 255.0))[None, None].to(device)

    # ---- Inference ----
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        outA = model(xA)
        outB = model(xB)

    # NMS + scores identical to infer_one.py
    ptsA, scA = logits_to_points(outA, args.det_thresh, args.topk, nms_used=(args.nms > 0), nms_radius=args.nms)
    ptsB, scB = logits_to_points(outB, args.det_thresh, args.topk, nms_used=(args.nms > 0), nms_radius=args.nms)

    # Optionally filter B by valid mask (if enabled)
    if validB is not None and len(ptsB) > 0:
        keep = validB[ptsB[:, 1].clip(0, args.H - 1), ptsB[:, 0].clip(0, args.W - 1)]
        ptsB, scB = ptsB[keep], scB[keep]

    # ---- Matches A->B under H ----
    A_sel, B_sel, e_pix = nn_pairs_under_H(
        ptsA, ptsB, H_A2B, args.H, args.W, px_thresh=args.px_thresh, validB=validB
    )

    # ---- Simple metrics ----
    Na, Nb = len(ptsA), len(ptsB)
    M = len(A_sel)
    rep_min = M / max(1, min(Na, Nb))
    rep_sym = M / max(1, 0.5 * (Na + Nb))
    mean_err = float(e_pix.mean()) if len(e_pix) else float('nan')

    print(f"\n=== Homography consistency ===")
    print(f"Detected A: {Na}, B: {Nb}, Matches: {M}")
    print(f"Repeatability (min): {rep_min:.3f}")
    print(f"Repeatability (sym): {rep_sym:.3f}")
    print(f"Mean reprojection error: {mean_err:.2f} px")

    # ---- Visualizations ----
    visA = draw_kpts(cv2.cvtColor(imgA_u8, cv2.COLOR_GRAY2BGR), ptsA, (0, 255, 0))
    visB = draw_kpts(cv2.cvtColor(imgB_u8, cv2.COLOR_GRAY2BGR), ptsB, (0, 255, 0))
    cv2.imwrite(args.out_prefix + "_A.jpg", visA)
    cv2.imwrite(args.out_prefix + "_B.jpg", visB)

    vis_matches = draw_matches_side_by_side(imgA_u8, imgB_u8, A_sel, B_sel, max_draw=args.viz_max)
    cv2.imwrite(args.out_prefix + "_matches.jpg", vis_matches)

    # ---- Save raw data for debugging/reuse ----
    np.save(args.out_prefix + "_H_A2B.npy", H_A2B)
    np.save(args.out_prefix + "_ptsA.npy", ptsA)
    np.save(args.out_prefix + "_ptsB.npy", ptsB)
    np.save(args.out_prefix + "_pairsA.npy", A_sel)
    np.save(args.out_prefix + "_pairsB.npy", B_sel)
    np.save(args.out_prefix + "_err.npy", e_pix)

    print(f"\nSaved:")
    print(f"  {args.out_prefix}_A.jpg        (keypoints on A)")
    print(f"  {args.out_prefix}_B.jpg        (keypoints on B)")
    print(f"  {args.out_prefix}_matches.jpg  (side-by-side matches)")
    print(f"  + NPYs: H, pts, pairs, errors\n")


if __name__ == "__main__":
    main()

# scripts/track_seq_lk_magicpoint_sxs_firstonly.py
# Detect ONCE (frame 0) with MagicPoint(Swin) -> track the same points with LK.
# - frames are read from --img_dir, sorted lexicographically
# - only the first --num_frames are used (default 200)
# - every frame is resized to 120x160 and kept that way
# - output: 120x320 video showing (prev | current) + lines for tracked points

import os, glob, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from model.magic_point_swin import MagicPointSwin

FIXED_H, FIXED_W = 120, 160  # always use 120x160

def ensure_dir_for_file(p):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)

def list_images_sorted(img_dir, exts=("*.png","*.jpg","*.jpeg","*.bmp")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(img_dir, e))
    paths.sort()
    return paths

def random_colors(N, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(map(int, c)) for c in rng.uniform(0,255,size=(N,3)).astype(np.uint8)]

@torch.no_grad()
def nms_peaks_from_prob(prob_t, det_thresh, nms_radius, topk):
    """Local-max NMS over a prob map (H,W) tensor in [0,1]."""
    if nms_radius > 0:
        r = int(nms_radius); k = 2*r + 1
        p4 = prob_t.unsqueeze(0).unsqueeze(0)  # 1x1xH×W
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
        "nms": nms, "det_thresh": det_thresh, "topk": topk, "grid_size": 8,
        "backbone": {"swin": {
            "name":"swin_tiny_patch4_window7_224",
            "pretrained": False, "normalize": True, "img_size":[H, W],
        }},
        "det_head": {"feat_in_dim": feat_dim},
    }
    model = MagicPointSwin(cfg).to(device).eval()
    return model

@torch.no_grad()
def detect_magicpoint(gray_u8, model, device, det_thresh, nms, topk):
    x = torch.from_numpy(gray_u8.astype(np.float32)/255.0)[None,None].to(device)
    out = model(x)
    prob = out["prob"][0]
    pts, scores = nms_peaks_from_prob(prob, det_thresh, nms, topk)
    return pts, scores

def main():
    ap = argparse.ArgumentParser("Track ONLY the first-frame keypoints (SxS video, fixed 120x160).")
    ap.add_argument("--img_dir", required=True, help="folder of frames (sorted lexicographically)")
    ap.add_argument("--out_video", required=True, help="output .mp4 path")
    ap.add_argument("--ckpt", required=True, help="MagicPointSwin .pth")
    ap.add_argument("--device", default="cuda:0")

    # detection (first frame only)
    ap.add_argument("--det_thresh", type=float, default=0.02)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--topk", type=int, default=50)

    # LK params
    ap.add_argument("--win_size", type=int, default=21)
    ap.add_argument("--max_level", type=int, default=3)
    ap.add_argument("--fb_check", action="store_true", help="optional forward-backward check")
    ap.add_argument("--fb_thresh", type=float, default=2.0)

    # sequence control
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--fps", type=float, default=5.0)

    args = ap.parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")

    # Load detector
    model = build_model(device, det_thresh=args.det_thresh, nms=args.nms, topk=args.topk)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    missing = model.load_state_dict(state, strict=False)
    print("Loaded ckpt. Missing:", len(missing.missing_keys), "Unexpected:", len(missing.unexpected_keys))
    model.eval()

    # Frames
    paths = list_images_sorted(args.img_dir)
    if not paths:
        raise RuntimeError(f"No images found in {args.img_dir}")
    paths = paths[:max(0, args.num_frames)]
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 frames, found {len(paths)}")
    print(f"Using {len(paths)} frames from: {args.img_dir}")

    # Video writer (panel: 120 x 320)
    ensure_dir_for_file(args.out_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (FIXED_W*2, FIXED_H), True)

    # LK setup
    termcrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    win = (int(args.win_size), int(args.win_size))

    # --- First frame: detect ONCE ---
    prev_bgr = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if prev_bgr is None:
        writer.release(); raise FileNotFoundError(paths[0])
    prev_small = cv2.resize(prev_bgr, (FIXED_W, FIXED_H), interpolation=cv2.INTER_AREA)
    prev_gray  = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    init_pts_np, _ = detect_magicpoint(prev_gray, model, device, args.det_thresh, args.nms, args.topk)
    N0 = len(init_pts_np)
    if N0 == 0:
        writer.release()
        raise RuntimeError("No keypoints found on the first frame. Try lowering --det_thresh or raising --topk.")

    print(f"First frame keypoints: {N0}")
    colors = random_colors(N0, seed=0)

    # Track state: full arrays (N0), plus a validity mask
    prev_pts_full = init_pts_np.astype(np.float32)            # (N0,2)
    valid = np.ones((N0,), dtype=bool)

    # Iterate
    for fi in range(1, len(paths)):
        curr_bgr = cv2.imread(paths[fi], cv2.IMREAD_COLOR)
        if curr_bgr is None:
            print(f"⚠️ Skipping unreadable frame: {paths[fi]}")
            continue
        curr_small = cv2.resize(curr_bgr, (FIXED_W, FIXED_H), interpolation=cv2.INTER_AREA)
        curr_gray  = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        # Run LK ONLY on currently-valid tracks
        if not np.any(valid):
            print(f"All tracks lost by frame {fi}.")
            break

        prev_active = prev_pts_full[valid].reshape(-1,1,2)
        next_active, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_active, None,
            winSize=win, maxLevel=args.max_level, criteria=termcrit
        )
        st = (st.reshape(-1) > 0) if st is not None else np.zeros((prev_active.shape[0],), bool)

        # Optional forward-backward consistency
        if args.fb_check and next_active is not None and np.any(st):
            back_active, st_b, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_gray, next_active, None,
                winSize=win, maxLevel=args.max_level, criteria=termcrit
            )
            fb_err = np.linalg.norm(prev_active - back_active, axis=2).reshape(-1)
            st = st & (fb_err < float(args.fb_thresh))

        # Scatter back to full set
        curr_pts_full = prev_pts_full.copy()
        valid_idx = np.where(valid)[0]
        curr_pts_full[valid_idx[st]] = next_active[st].reshape(-1,2)
        # tracks that failed become invalid forever (no re-detect)
        valid[valid_idx[~st]] = False

        # Build side-by-side panel and draw ONLY the surviving tracks
        panel = np.zeros((FIXED_H, FIXED_W*2, 3), np.uint8)
        panel[:, :FIXED_W] = prev_small
        panel[:, FIXED_W:] = curr_small

        survivors = np.where(valid)[0]
        for i in survivors:
            c = colors[i]
            x0, y0 = map(int, prev_pts_full[i])
            x1, y1 = map(int, curr_pts_full[i])
            cv2.circle(panel, (x0, y0), 2, c, -1, cv2.LINE_AA)               # left
            cv2.circle(panel, (x1 + FIXED_W, y1), 2, c, -1, cv2.LINE_AA)     # right
            cv2.line(panel, (x0, y0), (x1 + FIXED_W, y1), c, 1, cv2.LINE_AA)

        writer.write(panel)

        # advance
        prev_small = curr_small
        prev_gray  = curr_gray
        prev_pts_full = curr_pts_full

    writer.release()
    print(f"Done. Saved → {args.out_video}")

if __name__ == "__main__":
    main()

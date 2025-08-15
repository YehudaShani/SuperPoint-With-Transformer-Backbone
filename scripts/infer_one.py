# scripts/infer_one.py
import os, argparse, cv2, numpy as np, torch
from model.magic_point_swin import MagicPointSwin
from solver.nms import box_nms


def load_image_gray(path, size_hw=(120,160)):
    H, W = size_hw
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # (W,H)!
    img_f = img.astype(np.float32) / 255.0                        # 0..1
    ten = torch.from_numpy(img_f)[None, None, ...]                # 1×1×H×W
    return img, ten

def draw_kpts(overlay_bgr, kpts_xy, radius=2, thickness=1):
    for (x, y) in kpts_xy:
        cv2.circle(overlay_bgr, (int(x), int(y)), radius, (0,255,0), thickness)
    return overlay_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  required=True, help="path to .jpg")
    ap.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    ap.add_argument("--out",  default="out_vis.jpg", help="overlay output")
    ap.add_argument("--out_prob", default="out_prob.jpg", help="probability heatmap")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--det_thresh", type=float, default=0.015)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # Build model (same head dim as we trained: 192 for Swin-T stage2)
    cfg = {
        "nms": args.nms, "det_thresh": args.det_thresh, "topk": args.topk,
        "grid_size": 8,
        "backbone": {"swin": {"name":"swin_tiny_patch4_window7_224","pretrained":False,"normalize":True}},
        "det_head": {"feat_in_dim": 192},
    }
    model = MagicPointSwin(cfg).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)  # strict=False to be forgiving
    model.to(device).eval()

    # Image
    raw_u8, x = load_image_gray(args.img, size_hw=(args.H, args.W))
    x = x.to(device)
    
    # --- Inference ---
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
        out = model(x)  # dict; we will ignore out['prob_nms'] and use RAW

    prob_raw_t = out["prob"][0]           # (H,W) tensor on device
    H, W = prob_raw_t.shape

    # Save RAW prob for viz
    prob_raw = prob_raw_t.float().cpu().numpy()
    np.save(os.path.splitext(args.out_prob)[0] + ".npy", prob_raw)
    cv2.imwrite(args.out_prob, np.clip(prob_raw*255.0, 0, 255).astype(np.uint8))

    # --- Simple local-max NMS on RAW map (radius = args.nms pixels) ---
    if args.nms > 0:
        import torch.nn.functional as F
        r = int(args.nms)
        k = 2*r + 1
        p4 = prob_raw_t.unsqueeze(0).unsqueeze(0)        # 1×1×H×W
        pooled = F.max_pool2d(p4, kernel_size=k, stride=1, padding=r)
        # peak = equals local max AND above threshold
        peaks_mask = (p4 == pooled) & (p4 >= float(args.det_thresh))
        ys, xs = torch.nonzero(peaks_mask.squeeze(0).squeeze(0), as_tuple=True)
        scores_t = prob_raw_t[ys, xs]
    else:
        mask = prob_raw_t >= float(args.det_thresh)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        scores_t = prob_raw_t[ys, xs]

    # Top-K and numpy conversion
    ys, xs, scores = ys.cpu().numpy(), xs.cpu().numpy(), scores_t.float().cpu().numpy()
    if scores.size:
        order = np.argsort(-scores)
        if args.topk > 0:
            order = order[:min(args.topk, len(order))]
        xs, ys, scores = xs[order], ys[order], scores[order]
        pts = np.stack([xs, ys], 1).astype(np.int32)
    else:
        pts = np.empty((0, 2), np.int32)
        scores = np.empty((0,), np.float32)



    # --- print Top-K coordinates WITH score when NMS is disabled ---
    if  len(pts) > 0:
        print(f"Top-{len(pts)} coordinates (x, y, score) in {args.W}x{args.H}:")
        for i, ((x_i, y_i), s_i) in enumerate(zip(pts, scores)):
            print(f"{i:4d}: ({int(x_i):3d}, {int(y_i):3d})  {s_i:.4f}")

    # save overlay
    vis = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
    vis = draw_kpts(vis, pts, radius=2, thickness=1)
    cv2.imwrite(args.out, vis)

    # save probability heatmap (uint8)
    # prob_u8 = np.clip(prob_raw_t*255.0, 0, 255).astype(np.uint8)
    # cv2.imwrite(args.out_prob, prob_u8)

    # optional: also save coords
    np.save(os.path.splitext(args.out)[0] + "_kpts.npy", pts.astype(np.int32))
    print(f"✔ Saved: {args.out} (overlay), {args.out_prob} (prob), and *_kpts.npy")
    print(f"Detected {len(pts)} keypoints ≥ {args.det_thresh}")

if __name__ == "__main__":
    main()

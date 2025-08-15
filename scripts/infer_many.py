# %env PYTHONPATH=.
# python scripts/infer_many.py \
#   --in_root data/synthetic_shapes/test/images \
#   --ckpt export/mg_syn_swin_6_0.195.pth \
#   --out_dir runs/vis_syn \
#   --det_thresh 0.01 --nms 4 --topk 1000 --num 50 --save_np

# scripts/infer_many.py
import os, glob, argparse, csv
import cv2, numpy as np, torch
from tqdm import tqdm

from model.magic_point_swin import MagicPointSwin

def load_image_gray(path, size_hw=(120,160)):
    H, W = size_hw
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # (W,H)!
    img_f = img.astype(np.float32) / 255.0
    ten = torch.from_numpy(img_f)[None, None, ...]  # 1×1×H×W
    return img, ten

def draw_kpts(overlay_bgr, kpts_xy, radius=2, thickness=1, color=(0,255,0)):
    for (x, y) in kpts_xy:
        cv2.circle(overlay_bgr, (int(x), int(y)), radius, color, thickness)
    return overlay_bgr

def collect_images(in_root, exts=("*.png","*.jpg","*.jpeg","*.bmp"), recursive=False):
    pats = []
    for e in exts:
        if recursive:
            pats += glob.glob(os.path.join(in_root, "**", e), recursive=True)
        else:
            pats += glob.glob(os.path.join(in_root, e))
    return sorted(pats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="folder with images")
    ap.add_argument("--ckpt",     required=True, help="path to model .pth")
    ap.add_argument("--out_dir",  required=True, help="output folder")
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)
    ap.add_argument("--det_thresh", type=float, default=0.015)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--num",  type=int, default=-1, help="limit #images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--save_np", action="store_true", help="save .npy prob & kpts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # --- build model once ---
    cfg = {
        "nms": args.nms, "det_thresh": args.det_thresh, "topk": args.topk,
        "grid_size": 8,
        "backbone": {"swin": {"name":"swin_tiny_patch4_window7_224","pretrained":False,"normalize":True, "img_size":[args.H,args.W]}},
        "det_head": {"feat_in_dim": 192},
    }
    model = MagicPointSwin(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # --- gather images ---
    ims = collect_images(args.in_root, recursive=args.recursive)
    if args.num > 0:
        ims = ims[:args.num]
    if not ims:
        print("No images found.")
        return

    # CSV summary
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image","num_kpts","det_thresh","nms","topk"])

        pbar = tqdm(ims, ncols=100)
        for ipath in pbar:
            pbar.set_description(os.path.basename(ipath))
            try:
                raw_u8, x = load_image_gray(ipath, size_hw=(args.H, args.W))
            except FileNotFoundError:
                continue
            x = x.to(device, non_blocking=True)

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
                out = model(x)

            # RAW probability map for scores & heatmap
            prob_raw = out["prob"][0].float().cpu().numpy()    # H×W

            # If NMS ran, use it ONLY to pick peak locations; scores from RAW
            if args.nms > 0 and "prob_nms" in out:
                peaks = out["prob_nms"][0].float().cpu().numpy()
                mask  = peaks > 0
                ys, xs = np.where(mask)
            else:
                ys, xs = np.where(prob_raw >= args.det_thresh)

            if len(xs) == 0:
                pts = np.empty((0,2), np.int32)
                scores = np.empty((0,), np.float32)
            else:
                scores = prob_raw[ys, xs]
                order = np.argsort(-scores)
                if args.topk > 0:
                    order = order[:min(args.topk, len(order))]
                xs, ys, scores = xs[order], ys[order], scores[order]
                pts = np.stack([xs, ys], 1).astype(np.int32)

            # Save overlay & prob heatmap
            base = os.path.splitext(os.path.basename(ipath))[0]
            out_vis  = os.path.join(args.out_dir, f"{base}_overlay.jpg")
            out_prob = os.path.join(args.out_dir, f"{base}_prob.jpg")

            vis = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
            vis = draw_kpts(vis, pts, radius=2, thickness=1)
            cv2.imwrite(out_vis, vis)

            prob_u8 = np.clip(prob_raw * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(out_prob, prob_u8)

            if args.save_np:
                np.save(os.path.join(args.out_dir, f"{base}_prob.npy"), prob_raw.astype(np.float32))
                np.save(os.path.join(args.out_dir, f"{base}_kpts.npy"), pts.astype(np.int32))
                np.save(os.path.join(args.out_dir, f"{base}_scores.npy"), scores.astype(np.float32))

            # per-image CSV with coords+scores
            per_csv = os.path.join(args.out_dir, f"{base}_kpts.csv")
            with open(per_csv, "w", newline="") as f2:
                w2 = csv.writer(f2)
                w2.writerow(["x","y","score"])
                for (x_i, y_i), s_i in zip(pts, scores):
                    w2.writerow([int(x_i), int(y_i), float(s_i)])

            writer.writerow([ipath, len(pts), args.det_thresh, args.nms, args.topk])

    print(f"Done. Results in: {args.out_dir}")
    print(f"- Per-image: *_overlay.jpg, *_prob.jpg, *_kpts.csv (+*.npy if --save_np)")
    print(f"- Summary CSV: {summary_csv}")

if __name__ == "__main__":
    main()

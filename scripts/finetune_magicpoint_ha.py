# scripts/finetune_magicpoint_ha.py
# Fine-tune MagicPointSwin on HA pseudo-labels with a 2-stage LR schedule.
#
# Example:
# export PYTHONPATH=.
# python scripts/finetune_magicpoint_ha.py \
#   --img_dir /content/drive/MyDrive/SuperPoint-Pytorch/data/Euroc/mav0/cam0/images_last2000 \
#   --lab_dir /content/drive/MyDrive/SuperPoint-Pytorch/runs/ha_out \
#   --ckpt export/mg_syn_swin_15_0.176.pth \
#   --out_dir export/finetune_swin \
#   --log_dir runs/ft_logs \
#   --H 120 --W 160 \
#   --epochs1 10 --epochs2 10 --lr1 1e-4 --lr2 1e-5

import os, glob, argparse, csv, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.magic_point_swin import MagicPointSwin

# -------------- Dataset --------------

class RealHADataset(Dataset):
    """
    Expects:
      - img_dir: folder with real images (jpg/png/...)
      - lab_dir: same stems containing *_agg_prob.npy (preferred) or *_kpts.npy
    If only *_kpts.npy exists, creates a binary target map with small dilation.
    """
    def __init__(self, img_dir, lab_dir, size_hw=(120,160),
                 exts=("*.jpg","*.jpeg","*.png","*.bmp"),
                 use_gauss=False, pt_radius=1):
        self.H, self.W = size_hw
        self.img_paths = []
        for e in exts:
            self.img_paths += sorted(glob.glob(os.path.join(img_dir, e)))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {img_dir}")
        self.lab_dir = lab_dir
        self.use_gauss = use_gauss
        self.pt_radius = int(pt_radius)

    def _read_img(self, p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        if (img.shape[0], img.shape[1]) != (self.H, self.W):
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return img

    def _make_binary_from_kpts(self, stem):
        # load *_kpts.npy
        kpts_path = os.path.join(self.lab_dir, stem + "_kpts.npy")
        if not os.path.exists(kpts_path):
            return None
        pts = np.load(kpts_path)  # (N,2) int
        tgt = np.zeros((self.H, self.W), np.float32)
        if pts.size:
            xs = np.clip(pts[:,0], 0, self.W-1)
            ys = np.clip(pts[:,1], 0, self.H-1)
            tgt[ys, xs] = 1.0
            if self.pt_radius > 0:
                k = 2*self.pt_radius+1
                kernel = np.ones((k,k), np.uint8)
                tgt = cv2.dilate((tgt>0).astype(np.uint8), kernel, 1).astype(np.float32)
        if self.use_gauss:
            tgt = cv2.GaussianBlur(tgt, (0,0), sigmaX=0.6)
            tgt = np.clip(tgt, 0, 1)
        return tgt

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        p = self.img_paths[i]
        stem = os.path.splitext(os.path.basename(p))[0]
        img = self._read_img(p).astype(np.float32) / 255.0  # HxW in [0,1]

        # prefer *_agg_prob.npy
        prob_path = os.path.join(self.lab_dir, stem + "_agg_prob.npy")
        if os.path.exists(prob_path):
            tgt = np.load(prob_path).astype(np.float32)
            if tgt.shape != (self.H, self.W):
                tgt = cv2.resize(tgt, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            tgt = np.clip(tgt, 0.0, 1.0)
        else:
            tgt = self._make_binary_from_kpts(stem)
            if tgt is None:
                # fallback: skip to next (assumes most have labels)
                return self[(i+1) % len(self)]

        x = torch.from_numpy(img)[None, ...]        # 1xHxW
        y = torch.from_numpy(tgt)[None, ...]        # 1xHxW
        meta = {"path": p, "stem": stem}
        return {"img": x, "target": y, "meta": meta}

# -------------- Utils --------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_model(H, W, det_thresh=0.015, nms=4, topk=1000, feat_dim=192, device="cuda:0"):
    cfg = {
        "nms": nms, "det_thresh": det_thresh, "topk": topk,
        "grid_size": 8,
        "backbone": {"swin": {
            "name": "swin_tiny_patch4_window7_224",
            "pretrained": False,
            "normalize": True,
            "img_size": [H, W],
        }},
        "det_head": {"feat_in_dim": feat_dim},
    }
    model = MagicPointSwin(cfg)
    model.to(device)
    return model

# -------------- Training helpers --------------

def run_phase(model, dl, optim, device, writer, csv_path,
              start_epoch, num_epochs, global_step):
    """
    Runs num_epochs and logs per-batch loss.
    Returns updated (global_step, mean_epoch_losses)
    """
    mean_epoch_losses = []
    for e in range(num_epochs):
        epoch_idx = start_epoch + e
        running = []
        for batch in dl:
            x = batch["img"].to(device, non_blocking=True)        # Bx1xHxW
            y = batch["target"].to(device, non_blocking=True)     # Bx1xHxW

            optim.zero_grad(set_to_none=True)
            out = model(x)                       # dict with 'prob' in [0,1], BxHxW
            prob = out["prob"].unsqueeze(1)      # Bx1xHxW
            loss = F.binary_cross_entropy(
                prob.clamp(1e-6, 1 - 1e-6), y, reduction="mean"
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running.append(loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [time.strftime("%Y-%m-%d %H:%M:%S"), epoch_idx, global_step, f"{loss.item():.6f}"]
                )
            global_step += 1

        epoch_loss = float(np.mean(running)) if running else 0.0
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch_idx)
        mean_epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch_idx}: mean train loss = {epoch_loss:.4f}")
    return global_step, mean_epoch_losses

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser("Fine-tune MagicPointSwin on HA pseudo-labels (2-stage LR)")
    ap.add_argument("--img_dir", required=True, help="folder with real images")
    ap.add_argument("--lab_dir", required=True, help="folder with *_agg_prob.npy / *_kpts.npy")
    ap.add_argument("--ckpt",    required=True, help="init checkpoint (.pth)")
    ap.add_argument("--out_dir", default="export/finetune_swin", help="where to save checkpoints")
    ap.add_argument("--log_dir", default="runs/ft_logs", help="TensorBoard + CSV logs")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--H", type=int, default=120)
    ap.add_argument("--W", type=int, default=160)

    # 2-stage schedule
    ap.add_argument("--epochs1", type=int, default=10, help="epochs in stage 1")
    ap.add_argument("--epochs2", type=int, default=10, help="epochs in stage 2")
    ap.add_argument("--lr1", type=float, default=1e-4, help="LR in stage 1")
    ap.add_argument("--lr2", type=float, default=None, help="LR in stage 2 (default: 0.1*lr1)")

    # loader/regularization
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_gauss", action="store_true", help="gaussian blur for kpts -> map")
    ap.add_argument("--pt_radius", type=int, default=1, help="dilate radius for kpts maps")
    args = ap.parse_args()

    if args.lr2 is None:
        args.lr2 = args.lr1 * 0.1

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    ensure_dir(args.out_dir); ensure_dir(args.log_dir)

    # Data
    ds = RealHADataset(args.img_dir, args.lab_dir, size_hw=(args.H, args.W),
                       use_gauss=args.use_gauss, pt_radius=args.pt_radius)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=(args.num_workers > 0))

    # Model
    model = build_model(args.H, args.W, device=device)
    # Load init weights (either plain state_dict or dict with 'model')
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    missing = model.load_state_dict(state, strict=False)
    print("Loaded ckpt. Missing:", len(missing.missing_keys), "Unexpected:", len(missing.unexpected_keys))

    # Optimizer (start with lr1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr1, weight_decay=args.weight_decay)

    # Logging
    writer = SummaryWriter(log_dir=args.log_dir)
    csv_path = os.path.join(args.log_dir, "train_batch_loss.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["time", "epoch", "global_step", "batch_loss"])

    global_step = 0
    best_loss = float("inf")
    best_snapshot = None
    start_epoch_global = 0

    model.train()

    # -------- Stage 1 --------
    print(f"\n=== Stage 1: epochs={args.epochs1}, lr={args.lr1:g} ===")
    global_step, losses1 = run_phase(model, dl, optim, device, writer, csv_path,
                                     start_epoch=start_epoch_global, num_epochs=args.epochs1,
                                     global_step=global_step)
    # Save after stage 1
    ep1_last = start_epoch_global + args.epochs1 - 1
    last_path = os.path.join(args.out_dir, f"last_stage1_e{ep1_last}.pth")
    torch.save({
        "epoch": ep1_last,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "global_step": global_step,
        "stage": 1
    }, last_path)
    mean1 = float(np.mean(losses1)) if losses1 else 0.0
    if mean1 < best_loss:
        best_loss = mean1
        best_snapshot = {
            "epoch": ep1_last,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "global_step": global_step,
            "best_loss": best_loss,
            "stage": 1
        }

    # -------- Stage 2 (lower LR) --------
    print(f"\n=== Stage 2: epochs={args.epochs2}, lr={args.lr2:g} ===")
    for pg in optim.param_groups:
        pg["lr"] = args.lr2
    start_epoch_global += args.epochs1

    global_step, losses2 = run_phase(model, dl, optim, device, writer, csv_path,
                                     start_epoch=start_epoch_global, num_epochs=args.epochs2,
                                     global_step=global_step)
    # Save after stage 2
    ep2_last = start_epoch_global + args.epochs2 - 1
    last_path = os.path.join(args.out_dir, f"last_stage2_e{ep2_last}.pth")
    torch.save({
        "epoch": ep2_last,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "global_step": global_step,
        "stage": 2
    }, last_path)

    mean2 = float(np.mean(losses2)) if losses2 else 0.0
    if mean2 < best_loss:
        best_loss = mean2
        best_snapshot = {
            "epoch": ep2_last,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "global_step": global_step,
            "best_loss": best_loss,
            "stage": 2
        }

    # Save overall best
    if best_snapshot is not None:
        best_path = os.path.join(args.out_dir, "best.pth")
        torch.save(best_snapshot, best_path)
        print(f"\nBest snapshot saved → {best_path} (mean loss {best_loss:.6f}, stage {best_snapshot['stage']})")

    writer.close()
    print(f"\nDone. Logs → {args.log_dir} | ckpts → {args.out_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# scripts/bench_inference.py
import argparse, time, statistics as stats
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=120, help="input height")
    p.add_argument("--w", type=int, default=160, help="input width")
    p.add_argument("--batch", type=int, default=1, help="batch size")
    p.add_argument("--iters", type=int, default=200, help="timed iterations")
    p.add_argument("--warmup", type=int, default=30, help="warmup iterations")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--amp", action="store_true", help="enable autocast on CUDA")
    p.add_argument("--backbone-only", action="store_true", help="benchmark Swin backbone only")
    p.add_argument("--freeze-backbone", action="store_true", help="(no effect on eval speed, but mirrors cfg)")
    args = p.parse_args()

    dev = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device
    device = torch.device(dev)
    torch.set_grad_enabled(False)

    H, W = args.h, args.w
    B = args.batch
    img = torch.randn(B, 1, H, W, device=device)

    if args.backbone_only:
        from model.modules.transformer.swin_backbone import SwinStage2
        model = SwinStage2(img_size=(H, W)).to(device).eval()
        name = f"SwinStage2({H}x{W})"
        def forward(x):
            return model(x)
    else:
        from model.magic_point_swin import MagicPointSwin
        cfg = {
            "nms": 4, "det_thresh": 1e-3, "topk": 1000, "grid_size": 8,
            "backbone": {"swin": {
                "name": "swin_tiny_patch4_window7_224",
                "pretrained": True, "normalize": True,
                "img_size": [H, W],
                "freeze_backbone": args.freeze_backbone
            }},
            "det_head": {"feat_in_dim": 192}
        }
        model = MagicPointSwin(cfg).to(device).eval()
        name = f"MagicPointSwin({H}x{W})"
        def forward(x):
            return model(x)

    # warmup
    for _ in range(args.warmup):
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type=="cuda")):
            _ = forward(img)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # benchmark
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.iters):
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type=="cuda")):
            _ = forward(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)  # ms

    mean_ms   = sum(times) / len(times)
    med_ms    = stats.median(times)
    p90_ms    = stats.quantiles(times, n=10)[8] if len(times) >= 10 else max(times)
    thr_img_s = (B * 1000.0) / mean_ms
    peak_mem  = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type=="cuda" else 0.0

    print(f"\n=== Benchmark: {name} ===")
    print(f"device     : {device.type.upper()}{' ('+torch.cuda.get_device_name(0)+')' if device.type=='cuda' else ''}")
    print(f"batch      : {B} | size: {H}x{W} | amp: {args.amp}")
    print(f"iters      : {args.iters} (warmup {args.warmup})")
    print(f"latency ms : mean {mean_ms:.2f} | median {med_ms:.2f} | p90 {p90_ms:.2f}")
    print(f"throughput : {thr_img_s:.1f} img/s")
    if device.type == "cuda":
        print(f"peak mem   : {peak_mem:.1f} MB")

if __name__ == "__main__":
    main()

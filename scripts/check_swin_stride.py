import torch, timm

H, W = 120, 160
name = "swin_tiny_patch4_window7_224"
patch = 4

# find deepest valid stage i where H and W divisible by patch*(2**i)
valid = []
for i in range(4):  # stages 0..3 -> strides 4,8,16,32
    stride = patch * (2 ** i)
    if H % stride == 0 and W % stride == 0:
        valid.append(i)
out_idx = tuple(valid)           # for 120x160 -> (0,1)

m = timm.create_model(name, pretrained=True, features_only=True,
                      out_indices=out_idx, img_size=(H, W))
print("reductions:", m.feature_info.reduction())
print("channels :", m.feature_info.channels())

x = torch.randn(1,3,H,W)
with torch.no_grad():
    feats = m(x)

ch = m.feature_info.channels()
for i, f in enumerate(feats):
    # normalize to NCHW if timm returns NHWC
    if f.shape[1] != ch[i] and f.shape[-1] == ch[i]:
        f = f.permute(0,3,1,2).contiguous()
    Hc, Wc = f.shape[-2:]
    print(f"stage {out_idx[i]}: {tuple(f.shape)} | stride≈ {H/Hc:.3f}×{W/Wc:.3f}")

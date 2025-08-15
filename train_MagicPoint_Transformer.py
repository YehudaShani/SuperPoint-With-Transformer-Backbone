# -*-coding:utf8-*-
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF noise if present

import time
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---- Speed knobs ----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---- Project imports ----
from utils.tensor_op import pixel_shuffle_inv
from solver.loss import loss_func
from model.magic_point import MagicPoint
from model.superpoint_bn import SuperPointBNNet
from model.magic_point_swin import MagicPointSwin

# (kept for completeness; not used in this script)
model_dict_map = {
    'conv3b.weight':'backbone.block3_2.0.weight',
    'conv4b.bias':'backbone.block4_2.0.bias',
    'conv4b.weight':'backbone.block4_2.0.weight',
    'conv1b.bias':'backbone.block1_2.0.bias',
    'conv3a.bias':'backbone.block3_1.0.bias',
    'conv1b.weight':'backbone.block1_2.0.weight',
    'conv2b.weight':'backbone.block2_2.0.weight',
    'convDa.bias':'descriptor_head.convDa.bias',
    'conv1a.weight':'backbone.block1_1.0.weight',
    'convDa.weight':'descriptor_head.convDa.weight',
    'conv4a.bias':'backbone.block4_1.0.bias',
    'conv2a.bias':'backbone.block2_1.0.bias',
    'conv2a.weight':'backbone.block2_1.0.weight',
    'convPb.weight':'detector_head.convPb.weight',
    'convPa.bias':'detector_head.convPa.bias',
    'convPa.weight':'detector_head.convPa.weight',
    'conv2b.bias':'backbone.block2_2.0.bias',
    'conv1a.bias':'backbone.block1_1.0.bias',
    'convDb.weight':'descriptor_head.convDb.weight',
    'conv3a.weight':'backbone.block3_1.0.weight',
    'conv4a.weight':'backbone.block4_1.0.weight',
    'convPb.bias':'detector_head.convPb.bias',
    'convDb.bias':'descriptor_head.convDb.bias',
    'conv3b.bias':'backbone.block3_2.0.bias'
}

def move_batch_to_device(batch, device):
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        elif isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            t = [move(v) for v in x]
            return type(x)(t)
        return x
    return move(batch)

def train_eval(model, dataloader, config, run_name):
    device = next(model.parameters()).device
    model.train()

    # ---- knobs from config ----
    base_lr    = float(config['solver']['base_lr'])
    weightdec  = float(config['solver'].get('weight_decay', 0.0))
    epochs     = int(config['solver']['epoch'])
    grid_size  = int(config.get('grid_size', 8))
    log_every  = int(config['solver'].get('log_every', 100))
    skip_init  = bool(config['solver'].get('skip_initial_eval', False))

    # ---- optimizer only on trainable params ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weightdec)

    # ---- AMP scaler (new API to remove deprecation) ----
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    global_step = 0

    # ---- optional initial eval ----
    if not skip_init:
        print("\n🔍 Running initial evaluation on test set before training...")
        model.eval()
        with torch.no_grad():
            init_loss, init_precision, init_recall, init_fp_img, init_gt_img = do_eval(
                model, dataloader['test'], config, device
            )
        writer.add_scalar("val/init_loss",      init_loss,     global_step)
        writer.add_scalar("val/init_precision", init_precision, global_step)
        writer.add_scalar("val/init_recall",    init_recall,    global_step)
        writer.add_scalar("val/init_fp_img",    init_fp_img,    global_step)
        writer.add_scalar("val/init_gt_img",    init_gt_img,    global_step)
        print(f"[Initial Eval] LR {optimizer.param_groups[0]['lr']:.2e} "
              f"P {init_precision:.3f} R {init_recall:.3f} "
              f"FP/img {init_fp_img:.1f} GT/img {init_gt_img:.1f} "
              f"Loss {init_loss:.3f}")
    else:
        print("\n⏭️  Skipping initial eval (solver.skip_initial_eval=true)")

    # ---- training loop ----
    for epoch in range(epochs):
        model.train()
        mean_loss = []
        pbar = tqdm(enumerate(dataloader['train']), total=len(dataloader['train']), ncols=100)

        for i, data in pbar:
            data = move_batch_to_device(data, device)
            if i == 0:
                print(f"\n🌀 Epoch {epoch} start | device={device}")

            start = time.perf_counter()

            # forward + loss under AMP
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                raw_outputs = model(data['raw'])  # detector-only path (MagicPoint/MagicPointSwin)
                prob = raw_outputs
                desc = prob_warp = desc_warp = None

                loss = loss_func(
                    config['solver'], data, prob, desc, prob_warp, desc_warp, device, "TRAIN"
                )

            # backward + step
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            mean_loss.append(loss.item())

            # quick metrics
            with torch.no_grad():
                score_map = prob.get('prob_nms', prob['prob'])
                det_thresh = float(getattr(model, 'det_thresh',
                                    config['model'].get('det_thresh', 0.015)))
                pred = score_map >= det_thresh

                pred_cell = pixel_shuffle_inv(pred.float().unsqueeze(1), grid_size).sum(1) > 0
                gt_cell   = pixel_shuffle_inv(
                                data['raw']['kpts_map'].unsqueeze(1).float(), grid_size
                            ).sum(1) > 0

                TP = (pred_cell & gt_cell).sum().item()
                FP = (pred_cell & ~gt_cell).sum().item()
                FN = (~pred_cell & gt_cell).sum().item()
                precision = TP / (TP + FP + 1e-9)
                recall    = TP / (TP + FN + 1e-9)
                fp_img    = FP / pred_cell.shape[0]
                avg_gt_kpts = gt_cell.sum(dim=[1, 2]).float().mean().item()

            iter_ms = (time.perf_counter() - start) * 1000.0
            writer.add_scalar("train/loss",      loss.item(), global_step)
            writer.add_scalar("train/precision", precision,   global_step)
            writer.add_scalar("train/recall",    recall,      global_step)
            writer.add_scalar("train/fp_img",    fp_img,      global_step)
            writer.add_scalar("train/time_ms",   iter_ms,     global_step)
            global_step += 1

            if (i % log_every) == 0:
                pbar.set_postfix({
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "Loss": f"{np.mean(mean_loss):.3f}",
                    "P": f"{precision:.3f}",
                    "R": f"{recall:.3f}",
                    "FP/img": f"{fp_img:.1f}",
                    "GT/img": f"{avg_gt_kpts:.1f}",
                    "ms": f"{iter_ms:.1f}"
                })
                mean_loss = []

        # ---- end epoch: eval & save ----
        model.eval()
        with torch.no_grad():
            eval_loss, eval_precision, eval_recall, eval_fp_img, eval_gt_img = do_eval(
                model, dataloader["test"], config, device
            )
        writer.add_scalar("val/loss",      eval_loss,     global_step)
        writer.add_scalar("val/precision", eval_precision, global_step)
        writer.add_scalar("val/recall",    eval_recall,    global_step)
        writer.add_scalar("val/fp_img",    eval_fp_img,    global_step)
        writer.add_scalar("val/gt_img",    eval_gt_img,    global_step)

        filename = f"{config['solver']['model_name']}_{epoch}_{eval_loss:.3f}.pth"
        save_path = os.path.join(config['solver']['save_dir'], filename)
        os.makedirs(config['solver']['save_dir'], exist_ok=True)
        torch.save(model.state_dict(), save_path)

        print(f"Eval Epoch [{epoch}/{epochs}] "
              f"LR {optimizer.param_groups[0]['lr']:.2e} "
              f"P {eval_precision:.3f} R {eval_recall:.3f} "
              f"FP/img {eval_fp_img:.1f} GT/img {eval_gt_img:.1f} "
              f"Eval loss {eval_loss:.3f} | saved → {save_path}")

    writer.close()

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    _eval_loss = []
    TP_total = FP_total = FN_total = GT_total = 0
    N_images = 0
    grid_size = int(config.get('grid_size', 8))
    detector_only = str(config['model']['name']).lower() in ('magicpoint', 'magicpoint_swin')
    max_b = int(config['solver'].get('eval_max_batches', -1))

    for ind, data in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
        data = move_batch_to_device(data, device)

        # AMP for faster eval
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            raw_outputs = model(data['raw'])

        if detector_only or ('warp' not in data) or (data.get('warp') is None):
            prob = raw_outputs
            desc = prob_warp = desc_warp = None
        else:
            warp_outputs = model(data['warp'])
            prob, desc, prob_warp, desc_warp = (
                raw_outputs['det_info'], raw_outputs['desc_info'],
                warp_outputs['det_info'], warp_outputs['desc_info']
            )

        loss = loss_func(config['solver'], data, prob, desc, prob_warp, desc_warp, device, "EVAL")
        _eval_loss.append(loss.item())

        det_thresh = float(getattr(model, 'det_thresh', config['model'].get('det_thresh', 0.015)))
        score_map = prob['prob_nms'] if 'prob_nms' in prob else prob['prob']
        pred = score_map >= det_thresh

        pred_cell = pixel_shuffle_inv(pred.float().unsqueeze(1), grid_size).sum(1) > 0
        gt_cell   = pixel_shuffle_inv(data['raw']['kpts_map'].unsqueeze(1).float(), grid_size).sum(1) > 0

        TP_total += (pred_cell & gt_cell).sum().item()
        FP_total += (pred_cell & ~gt_cell).sum().item()
        FN_total += (~pred_cell & gt_cell).sum().item()
        GT_total += gt_cell.sum().item()
        N_images += gt_cell.shape[0]

        if max_b > 0 and (ind + 1) >= max_b:
            break

    precision = TP_total / (TP_total + FP_total + 1e-9)
    recall    = TP_total / (TP_total + FN_total + 1e-9)
    fp_img    = FP_total / max(1, N_images)
    gt_img    = GT_total / max(1, N_images)
    loss_mean = float(np.mean(_eval_loss)) if _eval_loss else 0.0
    return loss_mean, precision, recall, fp_img, gt_img

if __name__ == '__main__':
    # Use 'spawn' so CUDA context is not created in forked workers
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to YAML config")
    parser.add_argument("--run_name", default="default_run",
                        help="sub-directory under runs/ for TensorBoard logs")
    args = parser.parse_args()

    config_file = args.config
    assert os.path.exists(config_file), f"Config not found: {config_file}"
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    os.makedirs(config['solver']['save_dir'], exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ---- Datasets / Dataloaders ----
    name = str(config['data']['name']).lower()
    if name == 'coco':
        from dataset.coco import COCODataset
        # IMPORTANT: keep datasets on CPU, move to GPU in main process only
        datasets = {
            'train': COCODataset(config['data'], is_train=True,  device='cpu'),
            'test':  COCODataset(config['data'], is_train=False, device='cpu')
        }
        data_loaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=config['solver']['train_batch_size'],
                shuffle=True,
                collate_fn=datasets['train'].batch_collator,
                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=config['solver']['test_batch_size'],
                shuffle=False,
                collate_fn=datasets['test'].batch_collator,
                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
            )
        }

    elif name == 'synthetic':
        from dataset.synthetic_shapes import SyntheticShapes
        # IMPORTANT: keep datasets on CPU, move to GPU in main process only
        datasets = {
            'train': SyntheticShapes(config['data'], task=['training', 'validation'], device='cpu'),
            'test':  SyntheticShapes(config['data'], task=['test'],                     device='cpu')
        }
        data_loaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=config['solver']['train_batch_size'],
                shuffle=True,
                collate_fn=datasets['train'].batch_collator,
                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=config['solver']['test_batch_size'],
                shuffle=False,  # eval doesn't need shuffle
                collate_fn=datasets['test'].batch_collator,
                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
            )
        }
    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    # ---- Model ----
    mname = str(config['model']['name']).lower()
    if mname == 'superpoint':
        model = SuperPointBNNet(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif mname == 'magicpoint':
        model = MagicPoint(config['model'], device=device)
    elif mname == 'magicpoint_swin':
        model = MagicPointSwin(config['model'], device=device)
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    # ---- Pretrained (optional) ----
    ckpt_path = str(config['model'].get('pretrained_model', 'none'))
    if ckpt_path and ckpt_path.lower() != 'none' and os.path.exists(ckpt_path):
        pre_model_dict = torch.load(ckpt_path, map_location='cpu')
        model_dict = model.state_dict()
        for k, v in pre_model_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        print(f"↪️ loaded checkpoint: {ckpt_path}")
    else:
        print("↪️ no checkpoint loaded.")

    model.to(device)
    print("Model on:", next(model.parameters()).device)

    # ---- Train ----
    train_eval(model, data_loaders, config, args.run_name)
    print('Done')


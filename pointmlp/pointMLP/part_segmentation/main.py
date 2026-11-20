"""
main.py — PointMLP / PointNet++ training & evaluation for tree point cloud segmentation.

python main.py --eval --exp_name scratch-partial-revised-wce  --model pointnet2 --data_root ../data/partial_revised

python main.py --save_preds --exp_name scratch-combined --model pointMLP --data_root ../data/combined --num_points 8192
    
python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised \
            --epochs 400 \
            --batch_size 8 \
            --num_points 16384 \
            --wandb_project pointcept


python main.py --model pointnet2 \
            --data_root ../data/combined \
            --exp_name scratch-combined \
            --epochs 800 \
            --batch_size 8 \
            --num_points 8192 \
            --wandb_project pointcept \
            --resume

# With focal loss:
python main.py --model pointnet2 \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-focal \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss focal --focal_gamma 2.0 --focal_alpha 0.892 0.3778 0.891 0.8392 \ ;

python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-focal \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss focal --focal_gamma 2.0 --focal_alpha 0.892 0.3778 0.891 0.8392

# With weighted cross entropy:

python main.py --model pointnet2 \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-focal \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss focal --focal_gamma 2.0 --focal_alpha 0.892 0.3778 0.891 0.8392 ;

python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-focal \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss focal --focal_gamma 2.0 --focal_alpha 0.892 0.3778 0.891 0.8392 ;


python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-wce \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss weighted_ce --class_weights 1.4104 0.2448 1.3975 0.9473 ;

python main.py --model pointnet2 \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-wce \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss weighted_ce --class_weights 1.4104 0.2448 1.3975 0.9473 ;
"""

from __future__ import print_function
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import model as models
from tree_dataset import TreeNPYDataset, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from util.util import IOStream

_HAS_LOVASZ = False

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    print("[Warning] wandb not found; logging disabled. pip install wandb")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args, device):
    """
    Returns (nn.DataParallel model, log_softmax_input: bool).
    log_softmax_input=True  → model output is log-softmax; use F.nll_loss.
    log_softmax_input=False → raw logits; use F.cross_entropy.
    """
    if args.model.lower() in ("pointmlp", "pointmlp1"):
        base = models.__dict__[args.model](NUM_CLASSES).to(device)
        base.apply(weight_init)
        model = PointMLPSegWrapper(base, NUM_CLASSES).to(device)
        log_softmax = False

    elif args.model.lower() == "pointnet2":
        from models.pointnet2_seg import PointNet2SegWrapper
        model = PointNet2SegWrapper(NUM_CLASSES).to(device)
        log_softmax = True

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return nn.DataParallel(model), log_softmax


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, ignore_index=-1):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, target, log_softmax_input=False):
        valid  = target != self.ignore_index
        inputs = inputs[valid]
        target = target[valid]

        if len(target) == 0:
            return inputs.sum() * 0.0

        num_classes = inputs.size(1)
        target_ohe  = F.one_hot(target, num_classes).float()  # (N, C)

        if log_softmax_input:
            # inputs are already log(softmax(x)) -> convert to probabilities
            probs = inputs.exp().clamp(min=1e-7, max=1.0 - 1e-7)
            bce   = F.binary_cross_entropy(probs, target_ohe, reduction="none")
        else:
            # inputs are raw logits
            probs = torch.sigmoid(inputs)
            bce   = F.binary_cross_entropy_with_logits(inputs, target_ohe, reduction="none")

        one_minus_pt = (1 - probs) * target_ohe + probs * (1 - target_ohe)
        focal_weight = one_minus_pt.pow(self.gamma)

        if isinstance(self.alpha, (list, tuple)):
            alpha_t = inputs.new_tensor(self.alpha)
            alpha_w = alpha_t * target_ohe + (1 - alpha_t) * (1 - target_ohe)
        else:
            alpha_w = self.alpha * target_ohe + (1 - self.alpha) * (1 - target_ohe)

        return (alpha_w * focal_weight * bce).mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer(
            "weight",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, inputs, target, log_softmax_input=False):
        weight = self.weight.to(inputs.device)
        if log_softmax_input:
            return F.nll_loss(inputs, target, weight=weight, ignore_index=self.ignore_index)
        return F.cross_entropy(inputs, target, weight=weight, ignore_index=self.ignore_index)


def build_loss(args):
    if args.loss == "ce":
        def loss_fn(inputs, target, log_softmax_input=False):
            if log_softmax_input:
                return F.nll_loss(inputs, target, ignore_index=IGNORE_INDEX)
            return F.cross_entropy(inputs, target, ignore_index=IGNORE_INDEX)

    elif args.loss == "weighted_ce":
        wce = WeightedCrossEntropyLoss(args.class_weights, ignore_index=IGNORE_INDEX)
        def loss_fn(inputs, target, log_softmax_input=False):
            return wce(inputs, target, log_softmax_input=log_softmax_input)

    elif args.loss == "focal":
        alpha = args.focal_alpha[0] if len(args.focal_alpha) == 1 else args.focal_alpha
        fl = FocalLoss(gamma=args.focal_gamma, alpha=alpha, ignore_index=IGNORE_INDEX)
        def loss_fn(inputs, target, log_softmax_input=False):
            return fl(inputs, target, log_softmax_input=log_softmax_input)

    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    return loss_fn


def segmentation_loss(logits, target, loss_fn, log_softmax_input=False):
    B, C, N     = logits.shape
    flat_logits = logits.permute(0, 2, 1).reshape(-1, C)  # (B*N, C)
    flat_target = target.reshape(-1)                       # (B*N,)

    loss = loss_fn(flat_logits, flat_target, log_softmax_input=log_softmax_input)
    return loss, loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_per_class(pred, target, num_classes, ignore_index=-1):
    iou    = np.full(num_classes, np.nan, dtype=np.float64)
    mask   = (target != ignore_index)
    pred   = pred[mask].cpu().numpy()
    target = target[mask].cpu().numpy()
    for cls in range(num_classes):
        tp    = np.sum((pred == cls) & (target == cls))
        fp    = np.sum((pred == cls) & (target != cls))
        fn    = np.sum((pred != cls) & (target == cls))
        denom = tp + fp + fn
        if denom > 0:
            iou[cls] = tp / denom
    return iou


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class PointMLPSegWrapper(nn.Module):
    def __init__(self, base_model, num_seg_classes):
        super().__init__()
        self.base        = base_model
        self.num_classes = num_seg_classes

    def forward(self, points, feat):
        """points: (B,3,N)  feat: (B,6,N)  →  (B,C,N)"""
        B         = points.shape[0]
        norm_plt  = feat[:, 3:6, :]
        cls_label = torch.zeros(B, 16, device=points.device)
        out       = self.base(points, norm_plt, cls_label)
        if out.ndim == 3 and out.shape[1] != self.num_classes and out.shape[2] == self.num_classes:
            out = out.permute(0, 2, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Init helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)


def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def _save(model, opt, epoch, metrics, path):
    torch.save({
        "model":     (model.module.state_dict()
                      if isinstance(model, nn.DataParallel)
                      else model.state_dict()),
        "optimizer": opt.state_dict(),
        "epoch":     epoch,
        **metrics,
    }, path)

def save_predictions(args, io, split="val"):
    """
    Run inference on the full split and save per-point predictions to disk.
    For each scene writes:
        <scene_dir>/pred.npy        int64 (N,)  predicted class per point
        <scene_dir>/pred_prob.npy   float32 (N, C) class probabilities
    """
    device  = torch.device("cuda" if args.cuda else "cpu")
    loss_fn = build_loss(args)

    dataset = TreeNPYDataset(
        data_root=args.data_root, split=split,
        voxel_size=args.voxel_size, max_points=args.max_points,
        num_points=args.num_points, augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, drop_last=False,
    )

    model, log_softmax = build_model(args, device)
    ckpt_path = f"checkpoints/{args.exp_name}/best_mIoU_model.pth"
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    if list(state.keys())[0].startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.load_state_dict(state)
    model.eval()
    io.cprint(f"[save_predictions] Loaded: {ckpt_path}")

    with torch.no_grad():
        for idx, (points, feat, target) in enumerate(
                tqdm(loader, desc="Saving predictions")):

            N_total  = points.shape[1]
            points   = points.cuda(non_blocking=True)
            feat     = feat.cuda(non_blocking=True)

            # sliding-window inference (same as val_epoch)
            scene_logits = torch.zeros(1, NUM_CLASSES, N_total, device=points.device)
            scene_counts = torch.zeros(N_total, device=points.device)
            evaluated    = torch.zeros(N_total, dtype=torch.bool, device=points.device)
            pts_gpu      = points[0]

            while not evaluated.all():
                uneval_idx = torch.nonzero(~evaluated, as_tuple=False).squeeze(1)
                center     = pts_gpu[uneval_idx[0]]
                dists      = torch.sum((pts_gpu - center) ** 2, dim=1)

                if N_total <= args.num_points:
                    idx_crop  = torch.arange(N_total, device=points.device)
                    pad_idx   = torch.randint(0, N_total,
                                              (args.num_points - N_total,),
                                              device=points.device)
                    chunk_idx = torch.cat([idx_crop, pad_idx])
                else:
                    _, chunk_idx = torch.topk(dists, args.num_points, largest=False)

                crop_pts  = points[:, chunk_idx, :].transpose(2, 1)
                crop_feat = feat[:,   chunk_idx, :].transpose(2, 1)
                logits    = model(crop_pts, crop_feat)

                if N_total <= args.num_points:
                    scene_logits[0].index_add_(1, idx_crop, logits[0, :, :N_total])
                    scene_counts.index_add_(0, idx_crop,
                                            torch.ones(N_total, device=points.device))
                    evaluated[:] = True
                else:
                    scene_logits[0].index_add_(1, chunk_idx, logits[0])
                    scene_counts.index_add_(0, chunk_idx,
                                            torch.ones(args.num_points,
                                                       device=points.device))
                    evaluated[chunk_idx] = True

            scene_logits = scene_logits / scene_counts.clamp(min=1.0).view(
                1, 1, N_total)

            probs = torch.softmax(scene_logits[0], dim=0)   # (C, N)
            pred  = scene_logits[0].argmax(dim=0)           # (N,)

            scene_dir  = dataset.get_scene_dir(idx)
            scene_name = os.path.basename(scene_dir)
            out_dir    = os.path.join("checkpoints", args.exp_name, "predictions",
                                    split, scene_name)
            os.makedirs(out_dir, exist_ok=True)

            np.save(os.path.join(out_dir, "pred.npy"),
                    pred.cpu().numpy().astype(np.int64))
            np.save(os.path.join(out_dir, "pred_prob.npy"),
                    probs.cpu().numpy().astype(np.float32).T)  # (N, C)
            np.save(os.path.join(out_dir, "coord_processed.npy"),
                    points[0].cpu().numpy().astype(np.float32))  # (N, 3)

            io.cprint(f"  Saved: {out_dir}")

    io.cprint(f"[save_predictions] Done. Predictions written to each scene folder.")

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(args, io):
    device = torch.device("cuda" if args.cuda else "cpu")

    loss_fn = build_loss(args)
    io.cprint(f"Loss: {args.loss}"
              + (f"  gamma={args.focal_gamma}  alpha={args.focal_alpha}"
                 if args.loss == "focal" else "")
              + (f"  weights={args.class_weights}"
                 if args.loss == "weighted_ce" else ""))

    train_data = TreeNPYDataset(
        data_root=args.data_root, split="train",
        voxel_size=args.voxel_size, max_points=args.max_points,
        num_points=args.num_points,
    )
    val_data = TreeNPYDataset(
        data_root=args.data_root, split="val",
        voxel_size=args.voxel_size, max_points=args.max_points,
        num_points=args.num_points, augment=False,
    )
    io.cprint(f"Train samples: {len(train_data)}  |  Val samples: {len(val_data)}")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False,
        num_workers=args.workers, drop_last=False, pin_memory=True,
    )

    model, log_softmax = build_model(args, device)
    io.cprint(f"Using {torch.cuda.device_count()} GPU(s)  |  log_softmax={log_softmax}")

    if args.resume:
        ckpt_path = f"checkpoints/{args.exp_name}/best_mIoU_model.pth"
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")["model"]
            if not list(state.keys())[0].startswith("module."):
                state = {"module." + k: v for k, v in state.items()}
            model.load_state_dict(state)
            io.cprint(f"Resumed from {ckpt_path}")
        else:
            io.cprint(f"[Resume] {ckpt_path} not found — training from scratch.")

    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100,
                        momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.05, anneal_strategy="cos",
            div_factor=10.0, final_div_factor=1000.0,
        )
        step_every_batch = True
    elif args.scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr / 100)
        step_every_batch = False
    else:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=0.5)
        step_every_batch = False

    scaler = GradScaler(enabled=args.amp)

    if _HAS_WANDB and not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name,
                   config=vars(args), resume="allow",
                   id=args.wandb_id if args.wandb_id else None)
        wandb.watch(model, log="gradients", log_freq=100)

    best_mIoU = 0.0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            train_loader, model, opt, scheduler, scaler,
            epoch, step_every_batch, loss_fn, log_softmax, io, args,
        )
        val_metrics = val_epoch(
            val_loader, model, epoch, args.num_points, loss_fn, log_softmax, io,
        )

        log_dict = {
            "epoch":          epoch + 1,
            "lr":             opt.param_groups[0]["lr"],
            "train/loss":     train_metrics["loss"],
            "train/mIoU":     train_metrics["mIoU"],
            "train/acc":      train_metrics["acc"],
            "val/loss":       val_metrics["loss"],
            "val/mIoU":       val_metrics["mIoU"],
            "val/mAcc":       val_metrics["mAcc"],
            "val/allAcc":     val_metrics["allAcc"],
        }
        for i, name in enumerate(CLASS_NAMES):
            log_dict[f"val/iou_{name}"] = float(val_metrics["per_class_iou"][i]) \
                if not np.isnan(val_metrics["per_class_iou"][i]) else 0.0
            log_dict[f"val/acc_{name}"] = float(val_metrics["per_class_acc"][i]) \
                if not np.isnan(val_metrics["per_class_acc"][i]) else 0.0

        if _HAS_WANDB and not args.no_wandb:
            wandb.log(log_dict)

        io.cprint(
            f"Epoch {epoch+1:4d} | "
            f"train_loss={train_metrics['loss']:.4f} train_mIoU={train_metrics['mIoU']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['mIoU']:.4f} "
            f"val_mAcc={val_metrics['mAcc']:.4f} val_allAcc={val_metrics['allAcc']:.4f}"
        )

        if val_metrics["mIoU"] > best_mIoU:
            best_mIoU = val_metrics["mIoU"]
            io.cprint(f"  ↑ New best mIoU: {best_mIoU:.4f}")
            _save(model, opt, epoch, {"mIoU": best_mIoU},
                  f"checkpoints/{args.exp_name}/best_mIoU_model.pth")
            
        # in the epoch loop, after the best-model save:
        
        _save(model, opt, epoch, {"mIoU": val_metrics["mIoU"]},
            f"checkpoints/{args.exp_name}/model_last.pth")

        if not step_every_batch:
            scheduler.step()

    _save(model, opt, args.epochs - 1, {"mIoU": best_mIoU},
          f"checkpoints/{args.exp_name}/model_last.pth")
    io.cprint(f"Training complete. Best val mIoU: {best_mIoU:.4f}")

    if _HAS_WANDB and not args.no_wandb:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Train epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(loader, model, opt, scheduler, scaler,
                epoch, step_every_batch, loss_fn, log_softmax, io, args):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for points, feat, target in tqdm(loader, desc=f"Train {epoch+1}", leave=False):
        B      = points.shape[0]
        points = points.transpose(2, 1).cuda(non_blocking=True)
        feat   = feat.transpose(2, 1).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        opt.zero_grad()
        with autocast(enabled=args.amp):
            logits      = model(points, feat)
            loss, _     = segmentation_loss(logits, target, loss_fn, log_softmax)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step_every_batch:
            scheduler.step()

        total_loss += loss.item() * B

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            all_preds.append(pred.reshape(-1).cpu())
            all_targets.append(target.reshape(-1).cpu())

    n           = len(loader.dataset)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    iou_per_cls = compute_iou_per_class(all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX)
    valid_iou   = iou_per_cls[~np.isnan(iou_per_cls)]
    mIoU        = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0
    mask        = all_targets != IGNORE_INDEX
    acc         = float((all_preds[mask] == all_targets[mask]).float().mean())

    return {
        "loss":          total_loss / n,
        "mIoU":          mIoU,
        "acc":           acc,
        "per_class_iou": iou_per_cls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Val epoch — full-scene sliding-window inference
# ─────────────────────────────────────────────────────────────────────────────

def val_epoch(loader, model, epoch, num_points, loss_fn, log_softmax, io):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for points, feat, target in tqdm(loader, desc=f"Val   {epoch+1}", leave=False):
            N_total = points.shape[1]
            points  = points.cuda(non_blocking=True)
            feat    = feat.cuda(non_blocking=True)
            target  = target.cuda(non_blocking=True)

            scene_logits = torch.zeros(1, NUM_CLASSES, N_total, device=points.device)
            scene_counts = torch.zeros(N_total, device=points.device)
            evaluated    = torch.zeros(N_total, dtype=torch.bool, device=points.device)
            pts_gpu      = points[0]

            while not evaluated.all():
                uneval_idx = torch.nonzero(~evaluated, as_tuple=False).squeeze(1)
                center     = pts_gpu[uneval_idx[0]]
                dists      = torch.sum((pts_gpu - center) ** 2, dim=1)

                if N_total <= num_points:
                    idx       = torch.arange(N_total, device=points.device)
                    pad_idx   = torch.randint(0, N_total, (num_points - N_total,),
                                              device=points.device)
                    chunk_idx = torch.cat([idx, pad_idx])
                else:
                    _, chunk_idx = torch.topk(dists, num_points, largest=False)

                crop_pts  = points[:, chunk_idx, :].transpose(2, 1)
                crop_feat = feat[:,   chunk_idx, :].transpose(2, 1)
                logits    = model(crop_pts, crop_feat)

                if N_total <= num_points:
                    scene_logits[0].index_add_(1, idx, logits[0, :, :N_total])
                    scene_counts.index_add_(0, idx,
                                            torch.ones(N_total, device=points.device))
                    evaluated[:] = True
                else:
                    scene_logits[0].index_add_(1, chunk_idx, logits[0])
                    scene_counts.index_add_(0, chunk_idx,
                                            torch.ones(num_points, device=points.device))
                    evaluated[chunk_idx] = True

            scene_logits = scene_logits / scene_counts.clamp(min=1.0).view(1, 1, N_total)
            loss, _      = segmentation_loss(scene_logits, target, loss_fn, log_softmax)
            total_loss  += loss.item()

            pred = scene_logits.argmax(dim=1)
            all_preds.append(pred.reshape(-1).cpu())
            all_targets.append(target.reshape(-1).cpu())

    n           = len(loader.dataset)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    iou_per_cls = compute_iou_per_class(all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX)
    valid_iou   = iou_per_cls[~np.isnan(iou_per_cls)]
    mIoU        = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0
    mask        = all_targets != IGNORE_INDEX
    all_acc     = float((all_preds[mask] == all_targets[mask]).float().mean())

    # per-class accuracy as fixed-length array (nan = class absent)
    per_class_acc = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for c in range(NUM_CLASSES):
        cls_mask = all_targets == c
        if cls_mask.sum() > 0:
            per_class_acc[c] = float(
                (all_preds[cls_mask] == all_targets[cls_mask]).float().mean())
    m_acc = float(np.nanmean(per_class_acc))

    for i, name in enumerate(CLASS_NAMES):
        iou_str = f"{iou_per_cls[i]:.4f}" if not np.isnan(iou_per_cls[i]) else "N/A"
        acc_str = f"{per_class_acc[i]:.4f}" if not np.isnan(per_class_acc[i]) else "N/A"
        io.cprint(f"  [{name:8s}]  IoU: {iou_str}   Acc: {acc_str}")

    return {
        "loss":          total_loss / n,
        "mIoU":          mIoU,
        "mAcc":          m_acc,
        "allAcc":        all_acc,
        "per_class_iou": iou_per_cls,
        "per_class_acc": per_class_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone eval
# ─────────────────────────────────────────────────────────────────────────────

def test(args, io):
    device  = torch.device("cuda" if args.cuda else "cpu")
    loss_fn = build_loss(args)

    test_data = TreeNPYDataset(
        data_root=args.data_root, split="val",
        voxel_size=args.voxel_size, max_points=args.max_points,
        num_points=args.num_points, augment=False,
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False,
        num_workers=args.workers, drop_last=False,
    )

    model, log_softmax = build_model(args, device)

    ckpt_path = f"checkpoints/{args.exp_name}/best_mIoU_model.pth"
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    if list(state.keys())[0].startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.load_state_dict(state)
    io.cprint(f"Loaded checkpoint: {ckpt_path}")

    metrics = val_epoch(test_loader, model, epoch=0,
                        num_points=args.num_points,
                        loss_fn=loss_fn, log_softmax=log_softmax, io=io)

    io.cprint(
        f"\nTest results — "
        f"mIoU: {metrics['mIoU']:.4f}  "
        f"mAcc: {metrics['mAcc']:.4f}  "
        f"allAcc: {metrics['allAcc']:.4f}"
    )
    io.cprint(f"\n{'Class':<12} {'IoU':>8} {'Acc':>8}")
    io.cprint("-" * 30)
    for i, name in enumerate(CLASS_NAMES):
        iou = metrics["per_class_iou"][i]
        acc = metrics["per_class_acc"][i]
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "     N/A"
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "     N/A"
        io.cprint(f"  {name:<12} {iou_str:>8} {acc_str:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point cloud segmentation — tree dataset")

    parser.add_argument("--model",     type=str, default="pointMLP")
    parser.add_argument("--exp_name",  type=str, default="tree_run1")

    parser.add_argument("--data_root",  type=str,   default="../data/combined")
    parser.add_argument("--num_points", type=int,   default=8192)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--max_points", type=int,   default=81920)

    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--test_batch_size", type=int,   default=2,
                        help="Unused — val always runs at batch_size=1")
    parser.add_argument("--epochs",   type=int,   default=800)
    parser.add_argument("--lr",       type=float, default=0.006)
    parser.add_argument("--use_sgd",  action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str,  default="onecycle",
                        choices=["onecycle", "cos", "step"])
    parser.add_argument("--step",    type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--amp",     action="store_true")
    parser.add_argument("--resume",  action="store_true")

    parser.add_argument("--save_preds", action="store_true",
                    help="Run inference and save pred.npy + pred_prob.npy to each scene folder")
    parser.add_argument("--save_preds_split", type=str, default="val",
                        choices=["train", "val"],
                        help="Which split to save predictions for (default: val)")

    # ── Loss selection ────────────────────────────────────────────────────────
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "weighted_ce", "focal"],
                        help=(
                            "ce          — standard cross-entropy (default)\n"
                            "weighted_ce — cross-entropy with per-class weights "
                                          "(set via --class_weights)\n"
                            "focal       — focal loss (set via --focal_gamma / --focal_alpha)"
                        ))

    # Focal loss
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (focusing parameter). "
                             "0=CE, 1-2=moderate focus, >3=aggressive. Default: 2.0")
    parser.add_argument("--focal_alpha", type=float, nargs='+', default=[0.5],
                        help="Focal loss alpha. Single float or 3 values for per-class: --focal_alpha 0.88 0.20 0.91. "
                             "0.5=neutral, >0.5=up-weight target class. Default: 0.5")

    # Weighted CE
    # Weighted CE
    parser.add_argument("--class_weights", type=float, nargs=NUM_CLASSES,
                        default=[1.0] * NUM_CLASSES,
                        help=f"Per-class weights for weighted_ce loss ({NUM_CLASSES} values).")

    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="pointcept")
    parser.add_argument("--wandb_id",      type=str, default=None,
                        help="W&B run ID to resume a specific crashed run "
                             "(e.g. --wandb_id a4rkwdbt)")
    parser.add_argument("--no_wandb",      action="store_true")

    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    args.exp_name = f"{args.model}_{args.exp_name}"
    args.cuda     = not args.no_cuda and torch.cuda.is_available()

    _init_(args)

    log_suffix = "test" if args.eval else "train"
    io = IOStream(f"checkpoints/{args.exp_name}/{args.exp_name}_{log_suffix}.log")
    io.cprint(str(args))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)

    io.cprint("Using GPU" if args.cuda else "Using CPU")

    if args.save_preds:
        save_predictions(args, io, split=args.save_preds_split)
    elif args.eval:
        test(args, io)
    else:
        train(args, io)
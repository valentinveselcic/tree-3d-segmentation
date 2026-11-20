import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from tree_dataset import TreeNPYDataset
from main import build_model

"""
# 1. For PointMLP
python benchmark_mlp.py --model pointMLP --data_root ../data/partial_revised --num_points 16384

# 2. For PointNet++
python benchmark_mlp.py --model pointnet2 --data_root ../data/partial_revised --num_points 16384 --checkpoint pointMLP/part_segmentation/checkpoints/pointnet2_scratch-partial-revised/best_mIoU_model.pth
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PointMLP & PointNet++")
    parser.add_argument("--model", type=str, required=True, choices=["pointMLP", "pointnet2"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--num_warmup", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=100)
    return parser.parse_args()


def run_sliding_window_scene(model, points, feat, num_points):
    N_total = points.shape[1]
    evaluated = torch.zeros(N_total, dtype=torch.bool, device=points.device)
    pts_gpu = points[0]

    while not evaluated.all():
        uneval_idx = torch.nonzero(~evaluated, as_tuple=False).squeeze(1)
        center = pts_gpu[uneval_idx[0]]
        dists = torch.sum((pts_gpu - center) ** 2, dim=1)

        if N_total <= num_points:
            idx = torch.arange(N_total, device=points.device)
            pad_idx = torch.randint(0, N_total, (num_points - N_total,), device=points.device)
            chunk_idx = torch.cat([idx, pad_idx])
        else:
            _, chunk_idx = torch.topk(dists, num_points, largest=False)

        crop_pts = points[:, chunk_idx, :].transpose(2, 1).contiguous()
        crop_feat = feat[:, chunk_idx, :].transpose(2, 1).contiguous()
        _ = model(crop_pts, crop_feat)

        if N_total <= num_points:
            evaluated[:] = True
        else:
            evaluated[chunk_idx] = True


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_model(args, device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Ucitan checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    val_ds = TreeNPYDataset(
        data_root=args.data_root,
        split="val",
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    cached_scenes = []
    cached_patches = []

    for points, feat, _ in val_loader:
        N_total = points.shape[1]
        cached_scenes.append((points, feat))

        if N_total >= args.num_points:
            idx = torch.arange(args.num_points)
        else:
            idx = torch.cat([torch.arange(N_total), torch.randint(0, N_total, (args.num_points - N_total,))])

        patch_pts = points[:, idx, :].transpose(2, 1).contiguous()   # (1, 3, 16384)
        patch_feat = feat[:, idx, :].transpose(2, 1).contiguous()    # (1, 6, 16384)
        cached_patches.append((patch_pts, patch_feat))

        if len(cached_scenes) >= max(args.num_warmup, args.num_runs):
            break

    print(f"\n==========================================")
    print(f"Model:              {args.model}")
    print(f"Parameter num:    {total_params:.2f} M")
    print(f"Patch size:    {args.num_points} tocaka")
    print(f"Loaded validation scenes:  {len(cached_scenes)}")
    print(f"==========================================")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for i in range(args.num_warmup):
            p_pts, p_feat = cached_patches[i % len(cached_patches)]
            _ = model(p_pts.cuda(non_blocking=True), p_feat.cuda(non_blocking=True))
        torch.cuda.synchronize()

        patch_latencies = []
        for i in range(args.num_runs):
            p_pts, p_feat = cached_patches[i % len(cached_patches)]
            p_pts = p_pts.cuda(non_blocking=True)
            p_feat = p_feat.cuda(non_blocking=True)
            torch.cuda.synchronize()

            starter.record()
            _ = model(p_pts, p_feat)
            ender.record()

            torch.cuda.synchronize()
            patch_latencies.append(starter.elapsed_time(ender))

    patch_med = np.median(patch_latencies)
    patch_mean = np.mean(patch_latencies)
    patch_std = np.std(patch_latencies)
    patch_fps = 1000.0 / patch_med


    with torch.inference_mode():
        for i in range(args.num_warmup):
            s_pts, s_feat = cached_scenes[i % len(cached_scenes)]
            run_sliding_window_scene(
                model, s_pts.cuda(non_blocking=True), s_feat.cuda(non_blocking=True), args.num_points
            )
        torch.cuda.synchronize()

        scene_latencies = []
        for i in range(args.num_runs):
            s_pts, s_feat = cached_scenes[i % len(cached_scenes)]
            s_pts = s_pts.cuda(non_blocking=True)
            s_feat = s_feat.cuda(non_blocking=True)
            torch.cuda.synchronize()

            starter.record()
            run_sliding_window_scene(model, s_pts, s_feat, args.num_points)
            ender.record()

            torch.cuda.synchronize()
            scene_latencies.append(starter.elapsed_time(ender))

    scene_med = np.median(scene_latencies)
    scene_mean = np.mean(scene_latencies)
    scene_std = np.std(scene_latencies)
    scene_fps = 1000.0 / scene_med

    print(f"\n--- RESULTS ({args.model}) ---")
    print(f"Parameters (M):                     {total_params:.2f} M")
    print(f"[1 Patch = {args.num_points} pts] Median:   {patch_med:.2f} ms ({patch_fps:.1f} FPS) | Mean: {patch_mean:.2f} ± {patch_std:.2f} ms")
    print(f"[Full Scene SW]          Median:   {scene_med:.2f} ms ({scene_fps:.1f} FPS) | Mean: {scene_mean:.2f} ± {scene_std:.2f} ms")
    print(f"------------------------------------------\n")


if __name__ == "__main__":
    main()
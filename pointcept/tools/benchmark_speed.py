import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Pointcept Benchmark Latency & Params")
    parser.add_argument("--config-file", required=True, help="Putanja do config.py datoteke")
    parser.add_argument("--checkpoint", default=None, help="Opcionalno: putanja do .pth modela")
    parser.add_argument("--num-warmup", type=int, default=20, help="Broj prolaza za zagrijavanje")
    parser.add_argument("--num-runs", type=int, default=100, help="Broj prolaza za mjerenje")
    return parser.parse_args()


def move_to_cuda(data_dict):
    cuda_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            cuda_dict[k] = v.cuda(non_blocking=True)
        else:
            cuda_dict[k] = v
    return cuda_dict


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)

    # 1. Izgradnja modela i prebacivanje na GPU
    model = build_model(cfg.model).cuda()
    model.eval()

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Uklanjanje 'module.' prefiksa ako je trenirano sa DDP
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    # 2. Izračun broja parametara (u milionima)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. Učitavanje validacijskog skupa (koristimo cfg.data.val jer predstavlja 1 prolaz bez TTA)
    val_dataset = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=point_collate_fn,
        pin_memory=True,
    )

    # Prebacujemo sve uzorke u RAM listu kako ne bismo mjerili brzinu diska (I/O)
    cached_batches = []
    for batch in val_loader:
        cached_batches.append(batch)
        if len(cached_batches) >= max(args.num_warmup, args.num_runs):
            break

    print(f"\n==========================================")
    print(f"Model: {cfg.model.backbone.type}")
    print(f"Broj parametara: {total_params:.2f} M")
    print(f"Učitano validacijskih scena: {len(cached_batches)}")
    print(f"==========================================")

    # 4. Faza zagrijavanja GPU-a (Warm-up)
    print(f"Zagrijavanje GPU-a ({args.num_warmup} iteracija)...")
    with torch.inference_mode():
        for i in range(args.num_warmup):
            batch = cached_batches[i % len(cached_batches)]
            input_dict = move_to_cuda(batch)
            _ = model(input_dict)
        torch.cuda.synchronize()

    # 5. Mjerenje latencije pomoću CUDA Event tajmera
    print(f"Mjerenje latencije kroz {args.num_runs} prolaza...")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.inference_mode():
        for i in range(args.num_runs):
            batch = cached_batches[i % len(cached_batches)]
            input_dict = move_to_cuda(batch)
            torch.cuda.synchronize()

            starter.record()
            _ = model(input_dict)
            ender.record()

            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))  # Vrijeme u ms

    latencies = np.array(latencies)
    median_ms = np.median(latencies)
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    fps = 1000.0 / median_ms

    print(f"\n--- REZULTATI ({cfg.model.backbone.type}) ---")
    print(f"Parametri (M):      {total_params:.2f} M")
    print(f"Medijan latencije:  {median_ms:.2f} ms")
    print(f"Srednja latencija:  {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Brzina (FPS):       {fps:.1f} FPS")
    print(f"------------------------------------------\n")


if __name__ == "__main__":
    main()
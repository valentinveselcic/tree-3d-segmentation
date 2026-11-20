"""
tree_dataset.py — NPY dataset for tree point cloud segmentation.

Directory layout:
    <data_root>/
        train/
            scene_001/
                coord.npy       float32  (N, 3)
                color.npy       float32  (N, 3)  range [0,1] or [0,255]
                segment.npy     int64    (N,)     0=trunk 1=branch 2=wire -1=ignore
            ...
        val/
            ...

Returns per sample:
    points : (N, 3)  float32   XYZ (centred)
    feat   : (N, 6)  float32   XYZ + RGB (normalized)
    target : (N,)    int64     class labels
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES  = ["trunk", "branch", "wire", "grass"]
NUM_CLASSES  = 4
IGNORE_INDEX = -1


# ─────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────

def random_rotate_z(xyz, angle_range=(-np.pi, np.pi)):
    angle = np.random.uniform(*angle_range)
    c, s  = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ R.T


def random_scale(xyz, scale_range=(0.9, 1.1)):
    return xyz * np.random.uniform(*scale_range)


def random_flip(xyz, p=0.5):
    if np.random.rand() < p:
        xyz = xyz.copy()
        xyz[:, 0] *= -1
    return xyz


def random_jitter(xyz, sigma=0.005, clip=0.02):
    noise = np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
    return xyz + noise.astype(np.float32)


def random_dropout(xyz, rgb, labels, ratio=0.2, apply_prob=0.2):
    if np.random.rand() < apply_prob:
        n    = xyz.shape[0]
        keep = int(n * (1 - np.random.uniform(0, ratio)))
        idx  = np.random.choice(n, keep, replace=False)
        return xyz[idx], rgb[idx], labels[idx]
    return xyz, rgb, labels


def chromatic_jitter(rgb, p=0.95, std=0.05):
    if np.random.rand() < p:
        rgb = np.clip(rgb + np.random.randn(*rgb.shape).astype(np.float32) * std,
                      0.0, 1.0)
    return rgb


def chromatic_translation(rgb, p=0.95, ratio=0.05):
    if np.random.rand() < p:
        shift = (np.random.rand(3) * 2 - 1).astype(np.float32) * ratio
        rgb   = np.clip(rgb + shift, 0.0, 1.0)
    return rgb


def chromatic_auto_contrast(rgb, p=0.2):
    if np.random.rand() < p:
        lo    = rgb.min(axis=0)
        hi    = rgb.max(axis=0)
        scale = 1.0 / np.maximum(hi - lo, 1e-6)
        rgb   = (rgb - lo) * scale
    return rgb.astype(np.float32)


# ─────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────

def voxel_downsample(xyz, rgb, labels, voxel_size=0.02):
    """
    Voxel grid subsample: one point (the first encountered) per voxel.

    Uses a dict-based approach to avoid int64 overflow on large scenes.
    """
    inv_vs  = 1.0 / voxel_size
    vox_idx = np.floor(xyz * inv_vs).astype(np.int64)

    seen = {}
    sel  = []
    for i, key in enumerate(map(tuple, vox_idx)):
        if key not in seen:
            seen[key] = True
            sel.append(i)

    sel = np.asarray(sel, dtype=np.int64)
    return xyz[sel], rgb[sel], labels[sel]


def sphere_crop(xyz, rgb, labels, max_points=81920, random_center=True):
    """
    Keep the `max_points` nearest points to a chosen centre.
    Training: random centre.  Validation: scene centroid (deterministic).
    """
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb, labels

    if random_center:
        center_idx = np.random.randint(n)
        center     = xyz[center_idx]
    else:
        center = xyz.mean(axis=0)

    dists = np.sum((xyz - center) ** 2, axis=1)
    idx   = np.argsort(dists)[:max_points]
    return xyz[idx], rgb[idx], labels[idx]


def center_shift(xyz, apply_z=True):
    xyz = xyz.copy()
    xyz[:, 0] -= xyz[:, 0].mean()
    xyz[:, 1] -= xyz[:, 1].mean()
    if apply_z:
        xyz[:, 2] -= xyz[:, 2].min()
    return xyz


def fixed_sample(xyz, rgb, labels, num_points, random=True):
    """
    Subsample or upsample to exactly `num_points`.
    Training: random.  Validation: deterministic (evenly spaced).
    """
    n = xyz.shape[0]
    if n >= num_points:
        if random:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.linspace(0, n - 1, num_points, dtype=int)
    else:
        idx = np.random.choice(n, num_points, replace=True)  # upsample
    return xyz[idx], rgb[idx], labels[idx]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TreeNPYDataset(Dataset):
    """
    Parameters
    ----------
    data_root  : root dir containing train/ and val/ sub-dirs
    split      : 'train' or 'val'
    voxel_size : voxel grid resolution (metres); 0 = disabled
    max_points : sphere-crop size before fixed subsample
    augment    : None → auto (True for train, False for val)
    num_points : fixed points per sample fed to the network
                 Val uses the FULL voxel-downsampled scene (no crop,
                 no fixed subsample) so the sliding-window loop in
                 val_epoch can cover every point exactly once.
                 Set num_points=0 to keep this full-scene behaviour.
    """

    def __init__(
        self,
        data_root:  str,
        split:      str   = "train",
        voxel_size: float = 0.02,
        max_points: int   = 81920,
        augment:    bool  = None,
        num_points: int   = 8192,
    ):
        self.split      = split
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.augment    = augment if augment is not None else (split == "train")
        self.num_points = num_points

        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        coord_files = sorted(
            glob.glob(os.path.join(split_dir, "**", "*coord.npy"), recursive=True)
        )
        if not coord_files:
            raise FileNotFoundError(
                f"No '*coord.npy' files found under {split_dir}")

        self.data_paths = []
        for coord_path in coord_files:
            prefix    = coord_path.replace("coord.npy", "")
            color_p   = prefix + "color.npy"
            segment_p = prefix + "segment.npy"
            if not (os.path.exists(color_p) and os.path.exists(segment_p)):
                raise FileNotFoundError(
                    f"Missing color.npy or segment.npy alongside {coord_path}")
            self.data_paths.append((coord_path, color_p, segment_p))

        print(f"[TreeNPYDataset] {split}: {len(self.data_paths)} scenes found.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        coord_p, color_p, segment_p = self.data_paths[idx]

        xyz    = np.load(coord_p).astype(np.float32)
        rgb    = np.load(color_p).astype(np.float32)
        labels = np.load(segment_p).astype(np.int64)

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        # ── Voxel downsample ────────────────────────────────────────────
        if self.voxel_size > 0:
            xyz, rgb, labels = voxel_downsample(xyz, rgb, labels, self.voxel_size)

        # ── Augmentations (train only) ──────────────────────────────────
        if self.augment:
            xyz, rgb, labels = random_dropout(xyz, rgb, labels,
                                              ratio=0.2, apply_prob=0.2)
            xyz = random_rotate_z(xyz)
            for ax in (0, 1):
                a = np.random.uniform(-np.pi / 64, np.pi / 64)
                c, s = np.cos(a), np.sin(a)
                R = (np.array([[1, 0, 0], [0, c, -s], [0, s,  c]], np.float32)
                     if ax == 0 else
                     np.array([[c, 0, s], [0, 1,  0], [-s, 0, c]], np.float32))
                xyz = xyz @ R.T
            xyz = random_scale(xyz)
            xyz = random_flip(xyz)
            xyz = random_jitter(xyz)
            rgb = chromatic_auto_contrast(rgb)
            rgb = chromatic_translation(rgb)
            rgb = chromatic_jitter(rgb)

            # ── Sphere crop (train: random centre) ──────────────────────
            xyz, rgb, labels = sphere_crop(xyz, rgb, labels,
                                           self.max_points, random_center=True)

            # ── First centre shift (with z) ──────────────────────────────
            xyz = center_shift(xyz, apply_z=True)

            # ── Fixed-size subsample ─────────────────────────────────────
            if self.num_points > 0:
                xyz, rgb, labels = fixed_sample(xyz, rgb, labels,
                                                self.num_points, random=True)
        else:
            # ── Val: deterministic centre shift only, no crop/subsample ──
            # The full voxel-downsampled scene is returned so val_epoch's
            # sliding-window loop can evaluate every point.
            xyz = center_shift(xyz, apply_z=True)

        # ── Final centre (xy only) ──────────────────────────────────────
        xyz = center_shift(xyz, apply_z=False)

        features = np.concatenate([xyz, rgb], axis=1).astype(np.float32)

        return (
            torch.from_numpy(xyz),      # (N, 3)
            torch.from_numpy(features), # (N, 6)
            torch.from_numpy(labels),   # (N,)
        )

    def get_scene_dir(self, idx):
        """Return the folder path for sample idx — used to save predictions."""
        coord_p = self.data_paths[idx][0]
        return os.path.dirname(coord_p)
#!/usr/bin/env python3
"""
Visualizes segmentation and boundary metrics.
Merges primary segmentation metrics with optional baseline extra metrics (PointNet++, PointMLP).
Uses meters (m) as the primary unit and includes subplot titles while omitting the main suptitle.

Usage:
    python visualize_metrics.py --in_dir processed_metrics --out_dir processed_metrics/plots
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual encoding across all 6 architectures
ARCH_PALETTE = {
    "Octformer": "#1f77b4",   # Blue
    "PTv2": "#ff7f0e",        # Orange
    "PTv3": "#2ca02c",        # Green
    "SpUNet": "#d62728",      # Red
    "PointNet++": "#9467bd",  # Purple
    "PointMLP": "#8c564b",    # Brown
}

LOSS_DASHES = {
    "CE": "",            # Solid line
    "WCE": (4, 2),       # Dashed line
    "Focal": (1, 1.5),   # Dotted line
}

LOSS_MARKERS = {
    "CE": "o",
    "WCE": "s",
    "Focal": "^",
}

# Meter thresholds and formatted tick labels
TICKS_M = [0.00, 0.01, 0.02, 0.03, 0.05]
TICK_LABELS_M = ["0.00", "0.01", "0.02", "0.03", "0.05"]


def prepare_continuous_tau_df(df_seg: pd.DataFrame) -> pd.DataFrame:
    """
    Injects standard (tau=0.0m) rows into both 'dont_care' and 'slack'
    methods so line plots start cleanly from tau = 0.00 m.
    Converts tau to meters.
    """
    df = df_seg.copy()
    if "tau_m" not in df.columns:
        df["tau_m"] = (df["tau_cm"] / 100.0).round(4)

    std_df = df[df["method"] == "standard"].copy()
    
    dc_zero = std_df.copy()
    dc_zero["method"] = "dont_care"
    
    slack_zero = std_df.copy()
    slack_zero["method"] = "slack"
    
    non_std = df[df["method"].isin(["dont_care", "slack"])].copy()
    combined = pd.concat([dc_zero, slack_zero, non_std], ignore_index=True)
    combined.sort_values(by=["architecture", "loss", "method", "tau_m"], inplace=True)
    return combined


def plot_baseline_bars(df_combined_std: pd.DataFrame, out_dir: Path):
    """
    Plot 1: Standard (tau = 0.00 m) comparison across all architectures and loss functions.
    Shows individual subplot titles for each class and metric.
    """
    if df_combined_std.empty:
        return

    arch_order = ["Octformer", "PTv2", "PTv3", "SpUNet", "PointNet++", "PointMLP"]
    present_archs = [a for a in arch_order if a in df_combined_std["architecture"].unique()]

    metrics = [
        ("mIoU", "Overall mIoU"),
        ("trunk_iou", "Trunk IoU"),
        ("branch_iou", "Branch IoU"),
        ("wire_iou", "Wire IoU"),
        ("grass_iou", "Grass IoU")
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharey=False)

    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=df_combined_std,
            x="architecture",
            y=metric,
            hue="loss",
            order=present_archs,
            hue_order=["CE", "WCE", "Focal"],
            palette="Set2",
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("IoU" if metric == "mIoU" else "")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Dynamic zoom on y-axis
        min_val = df_combined_std[metric].min()
        ax.set_ylim(max(0.0, min_val - 0.05), 1.005)

        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Loss", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"01_baseline_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tau_response_curves(df_cont: pd.DataFrame, out_dir: Path):
    """
    Plot 2: Tau Response Curves in meters with individual subplot titles.
    Rows: Method 1 (Don't-care) vs Method 2 (Slack)
    Cols: mIoU, Trunk IoU, Branch IoU
    """
    methods = [("dont_care", "Don't-Care Margin"), ("slack", "Slack Relaxation")]
    metrics = [
        ("mIoU", "Overall mIoU"),
        ("trunk_iou", "Trunk IoU"),
        ("branch_iou", "Branch IoU")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)

    for row_idx, (m_key, m_label) in enumerate(methods):
        sub_df = df_cont[df_cont["method"] == m_key]
        for col_idx, (metric_col, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            sns.lineplot(
                data=sub_df,
                x="tau_m",
                y=metric_col,
                hue="architecture",
                style="loss",
                palette=ARCH_PALETTE,
                dashes=LOSS_DASHES,
                markers=LOSS_MARKERS,
                markersize=7,
                linewidth=2.0,
                ax=ax
            )
            ax.set_title(f"{m_label} — {metric_title}", fontsize=11, fontweight="bold")
            ax.set_xticks(TICKS_M)
            ax.set_xticklabels(TICK_LABELS_M)
            ax.set_xlabel("Tolerance τ (m)" if row_idx == 1 else "")
            ax.set_ylabel("IoU")
            ax.grid(True, linestyle="--", alpha=0.5)

            if row_idx == 0 and col_idx == 2:
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"02_tau_response_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_divergence(df_cont: pd.DataFrame, out_dir: Path):
    """
    Plot 3: Direct comparison showing difference (Slack - Don't-Care) on Trunk and mIoU with subplot titles.
    """
    pivot = df_cont.pivot_table(
        index=["model_name", "architecture", "loss", "tau_m"],
        columns="method",
        values=["mIoU", "trunk_iou"]
    ).reset_index()

    # Flatten column MultiIndex
    pivot.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in pivot.columns]

    pivot["diff_trunk_iou"] = (pivot["trunk_iou_slack"] - pivot["trunk_iou_dont_care"]) * 100.0
    pivot["diff_mIoU"] = (pivot["mIoU_slack"] - pivot["mIoU_dont_care"]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    targets = [
        ("diff_trunk_iou", "Trunk IoU Difference (Slack − Don't-Care) [pp]"),
        ("diff_mIoU", "mIoU Difference (Slack − Don't-Care) [pp]")
    ]

    for ax, (col, title) in zip(axes, targets):
        sns.lineplot(
            data=pivot,
            x="tau_m",
            y=col,
            hue="architecture",
            style="loss",
            palette=ARCH_PALETTE,
            dashes=LOSS_DASHES,
            markers=LOSS_MARKERS,
            markersize=7,
            linewidth=2.0,
            ax=ax
        )
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(TICKS_M)
        ax.set_xticklabels(TICK_LABELS_M)
        ax.set_xlabel("Tolerance τ (m)", fontsize=11)
        ax.set_ylabel("Difference (Percentage Points)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        if ax == axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"03_method_divergence_trunk.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_metrics(df_bf1: pd.DataFrame, out_dir: Path):
    """
    Plot 4: Boundary F1, Precision, and Recall across Theta in meters with individual subplot titles.
    """
    if df_bf1.empty:
        return

    df = df_bf1.copy()
    if "theta_m" not in df.columns:
        df["theta_m"] = (df["theta_cm"] / 100.0).round(4)

    metrics = [
        ("bf1", "Boundary F1-Score"),
        ("precision", "Boundary Precision"),
        ("recall", "Boundary Recall")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        sns.lineplot(
            data=df,
            x="theta_m",
            y=col,
            hue="architecture",
            style="loss",
            palette=ARCH_PALETTE,
            dashes=LOSS_DASHES,
            markers=LOSS_MARKERS,
            markersize=7,
            linewidth=2.0,
            ax=ax
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(TICKS_M)
        ax.set_xticklabels(TICK_LABELS_M)
        ax.set_xlabel("Boundary Tolerance θ (m)", fontsize=11)
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.5)

        if idx < 2:
            ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"04_boundary_f1_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation and boundary metrics.")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="processed_metrics",
        help="Directory containing CSV files (default: processed_metrics)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="processed_metrics/plots",
        help="Directory to save generated plots (default: processed_metrics/plots)"
    )
    args = parser.parse_args()

    in_path = Path(args.in_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seg_file = in_path / "segmentation_metrics.csv"
    bf1_file = in_path / "boundary_metrics.csv"
    extra_file = in_path / "baseline_extra_metrics.csv"

    if not seg_file.exists():
        print(f"[ERROR] Required file {seg_file} not found. Run extract_metrics.py first.")
        return

    df_seg = pd.read_csv(seg_file)
    df_bf1 = pd.read_csv(bf1_file) if bf1_file.exists() else pd.DataFrame()

    # Merge baseline standard metrics for Plot 1
    std_primary = df_seg[df_seg["method"] == "standard"].copy()
    if extra_file.exists():
        df_extra = pd.read_csv(extra_file)
        df_combined_std = pd.concat([std_primary, df_extra], ignore_index=True)
        print(f"[INFO] Loaded {len(df_extra)} extra baseline runs from {extra_file.name}")
    else:
        df_combined_std = std_primary
        print("[INFO] No extra baseline file found; plotting primary models only.")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    print("Generating Plot 1: Baseline Comparison Bars (All Architectures)...")
    plot_baseline_bars(df_combined_std, out_path)

    print("Generating Plot 2: Tau Response Curves (Don't-Care vs Slack in meters)...")
    df_cont = prepare_continuous_tau_df(df_seg)
    plot_tau_response_curves(df_cont, out_path)

    print("Generating Plot 3: Method Divergence on Trunk & mIoU...")
    plot_method_divergence(df_cont, out_path)

    if not df_bf1.empty:
        print("Generating Plot 4: Boundary F1, Precision & Recall Curves in meters...")
        plot_boundary_metrics(df_bf1, out_path)

    print(f"\n[DONE] All plots saved in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
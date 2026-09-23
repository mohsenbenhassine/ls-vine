"""
src/figures.py

Figure generation for LS-Vine benchmark results.

Contains:
    - fig_boxplots   : AIC boxplots per scenario
    - fig_violin     : DepRec violin plot across methods
    - fig_heatmap    : Heatmap of mean metric per (method, scenario)
    - fig_timing     : Bar chart of mean training time per method
    - generate_all_figures : generate all standard figures

Usage
-----
from src.figures import generate_all_figures
generate_all_figures(df_results)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .benchmark import METHODS


# =============================================================================
# Colour palette
# =============================================================================
PALETTE = {
    "LS-Vine":          "#2a78d6",
    "AE-Vine-Selected": "#8e44ad",
    "PCA-Vine":         "#e67e22",
    "ICA-Vine":         "#27ae60",
    "FA-Vine":          "#9b59b6",
    "KPCA-Vine":        "#f39c12",
    "AE-Vine":          "#c0392b",
    "VAE-Vine":         "#16a085",
    "WAE-Vine":         "#d35400",
    "InfoVAE-Vine":     "#2980b9",
    "Vine-Direct":      "#2c3e50",
    "Vine-Truncated":   "#7f8c8d",
}

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Boxplots of AIC per scenario
# =============================================================================
def fig_boxplots(df: pd.DataFrame):
    """AIC boxplots per scenario (one panel per scenario)."""
    scenarios = df["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(5 * len(scenarios), 5),
                             sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        sub = df[df["scenario"] == sc]
        order = [m for m in METHODS if m in sub["method"].unique()]
        sns.boxplot(data=sub, x="method", y="aic", order=order,
                    palette=PALETTE, ax=ax, width=0.6)
        ax.set_title(sc, fontweight="bold", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("AIC (lower = better)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("AIC distribution per method (lower = better)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplots_aic.png", dpi=130, bbox_inches="tight")
    plt.show()


# =============================================================================
# Violin plot of DepRec
# =============================================================================
def fig_violin(df: pd.DataFrame):
    """DepRec violin plot across all scenarios."""
    sub = df[df["dep_rec"].notna() & df["method"].isin(METHODS)]
    fig, ax = plt.subplots(figsize=(12, 5))
    order = [m for m in METHODS if m in sub["method"].unique()]
    sns.violinplot(data=sub, x="method", y="dep_rec", order=order,
                   palette=PALETTE, ax=ax, inner="box")
    ax.set_title("DepRec distribution (lower = better)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("DepRec (Frobenius ||dtau||)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "violin_deprec.png", dpi=130, bbox_inches="tight")
    plt.show()


# =============================================================================
# Heatmap of mean metric per (method, scenario)
# =============================================================================
def fig_heatmap(df: pd.DataFrame, metric="aic"):
    """Heatmap: mean metric per (method, scenario)."""
    pivot = df.groupby(["method", "scenario"])[metric].mean().unstack()
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                    max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(f"Mean {metric.upper()} per method x scenario",
                 fontweight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("")
    plt.tight_layout()
    fn = FIG_DIR / f"heatmap_{metric}.png"
    plt.savefig(fn, dpi=130, bbox_inches="tight")
    plt.show()


# =============================================================================
# Bar chart of training time
# =============================================================================
def fig_timing(df: pd.DataFrame):
    """Bar chart: mean training time per method."""
    agg = df.groupby("method")["t_train"].mean().reindex(METHODS).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(m, "gray") for m in agg.index]
    ax.bar(agg.index, agg.values, color=colors, edgecolor="white")
    ax.set_ylabel("Mean training time (s)")
    ax.set_title("Training time comparison", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "timing.png", dpi=130, bbox_inches="tight")
    plt.show()


# =============================================================================
# Generate all figures
# =============================================================================
def generate_all_figures(df: pd.DataFrame):
    """Generate all standard figures."""
    print("\n-- Generating figures --")
    fig_boxplots(df)
    fig_violin(df)
    fig_heatmap(df, "aic")
    fig_heatmap(df, "dep_rec")
    fig_timing(df)
    print(f"\nAll figures saved to {FIG_DIR}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import sys

    results_file = Path("results/results.csv")
    if not results_file.exists():
        print(f"ERROR: {results_file} not found.")
        print("Run `python src/benchmark.py` first.")
        sys.exit(1)

    df = pd.read_csv(results_file)
    generate_all_figures(df)

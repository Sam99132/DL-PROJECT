"""
run_ablations.py - Automated CLI for executing all Phase 2 Ablation Studies.

Runs the 4 ablation studies, computes validation RMSE/MAE/WAPE, and generates
a multi-panel publication bar chart comparing configurations.
"""

import sys
import os
import json
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.dataset import M5FeaturePipeline
from src.models.ablation import M5AblationRunner


def plot_ablation_comparison(ablation_results: dict, output_path: Path):
    """Renders a 4-panel publication bar chart of ablation RMSEs."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    # 1. Embeddings
    ax1 = axes[0, 0]
    res1 = ablation_results["ablation_1_embeddings"]
    names1 = [r["configuration"] for r in res1]
    rmses1 = [r["val_rmse"] for r in res1]
    bars1 = ax1.bar(names1, rmses1, color=["#1f77b4", "#aec7e8"], edgecolor="black", width=0.5)
    ax1.set_title("Ablation 1: Entity Embeddings Representation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation RMSE (lower is better)", fontsize=10)
    ax1.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Regularization
    ax2 = axes[0, 1]
    res2 = ablation_results["ablation_2_regularization"]
    names2 = [r["configuration"].split("(")[0].strip() for r in res2]
    rmses2 = [r["val_rmse"] for r in res2]
    bars2 = ax2.bar(names2, rmses2, color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"], edgecolor="black", width=0.6)
    ax2.set_title("Ablation 2: Regularization Mechanisms", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation RMSE", fontsize=10)
    ax2.tick_params(axis="x", rotation=15)
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Architecture
    ax3 = axes[1, 0]
    res3 = ablation_results["ablation_3_architecture"]
    names3 = [r["configuration"].split("(")[0].strip() for r in res3]
    rmses3 = [r["val_rmse"] for r in res3]
    bars3 = ax3.bar(names3, rmses3, color=["#3182bd", "#6baed6", "#9ecae1"], edgecolor="black", width=0.5)
    ax3.set_title("Ablation 3: Depth & Width Scaling", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Validation RMSE", fontsize=10)
    ax3.bar_label(bars3, fmt="%.4f", padding=3, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Loss Functions
    ax4 = axes[1, 1]
    res4 = ablation_results["ablation_4_loss_functions"]
    names4 = [r["configuration"].split("(")[0].strip() for r in res4]
    rmses4 = [r["val_rmse"] for r in res4]
    bars4 = ax4.bar(names4, rmses4, color=["#31a354", "#74c476", "#a1d99b"], edgecolor="black", width=0.5)
    ax4.set_title("Ablation 4: Objective Loss Functions", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Validation RMSE", fontsize=10)
    ax4.bar_label(bars4, fmt="%.4f", padding=3, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Phase 2 Deep Learning Ablation Studies: Empirical Evaluation", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved ablation comparison figure to: {output_path}")


def main():
    print("=" * 70)
    print("🔬 PHASE 2: RUNNING SCIENTIFIC ABLATION EXPERIMENTS")
    print("=" * 70)

    pipeline = M5FeaturePipeline(
        raw_dir=str(PROJECT_ROOT / "data/raw"),
        processed_dir=str(PROJECT_ROOT / "data/processed"),
        n_series=150,  # Fast and representative for 4 ablations
        random_state=42,
    )

    train_dataset, val_dataset, cat_dims, cont_cols = pipeline.build_tabular_dataset(
        start_day=1600,
        end_day=1913,
        val_horizon=28,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    runner = M5AblationRunner(
        cat_dims=cat_dims,
        num_cont_features=len(cont_cols),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs_per_trial=4,
        output_dir=str(PROJECT_ROOT / "reports"),
    )

    results = runner.run_all_ablations()

    # Generate figure
    fig_path = PROJECT_ROOT / "reports/figures/12_ablation_performance_comparison.png"
    plot_ablation_comparison(results, fig_path)


if __name__ == "__main__":
    main()

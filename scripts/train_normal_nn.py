"""
train_normal_nn.py - Master Training Pipeline for the Normal Neural Network.

Trains the NormalDemandMLP on the M5 Inventory Demand Dataset,
evaluates on the 28-day forecast horizon, and saves publication-grade figures and metrics.
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

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import M5FeaturePipeline
from src.models.feedforward_nn import NormalDemandMLP
from src.models.trainer import DemandTrainer


def plot_training_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Generates publication-grade figures for training curves and predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Training & Validation Convergence Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2.0, label="Training Loss (Huber)")
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", linewidth=2.0, label="Validation Loss (Huber)")
    ax1.set_title("Normal Neural Network: Learning Curves & Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Huber Loss (delta=1.0)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, history["val_rmse"], "^-", color="#2ca02c", linewidth=2.0, label="Validation RMSE")
    ax2.plot(epochs, history["val_mae"], "v--", color="#d62728", linewidth=2.0, label="Validation MAE")
    ax2.set_title("Validation Error Trajectory across Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Error Metric (Units)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig1_path = output_dir / "10_normal_nn_training_curves.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved training curves to: {fig1_path}")

    # Figure 2: Actual vs. Predicted Demand Distribution & Error Residuals
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot sample 100 validation points
    sample_indices = np.arange(min(120, len(y_true)))
    ax3.plot(sample_indices, y_true[:120], color="#1f77b4", label=r"Ground Truth Demand ($y$)", linewidth=1.8, alpha=0.85)
    ax3.plot(sample_indices, y_pred[:120], color="#d62728", linestyle="--", label=r"Normal NN Forecast ($\hat{y}$)", linewidth=1.8)
    ax3.set_title("Actual vs. Predicted Daily Demand (Sample Trajectory)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Observation Index", fontsize=11)
    ax3.set_ylabel("Unit Sales Demand", fontsize=11)
    ax3.legend(frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Residuals distribution
    residuals = y_true - y_pred
    ax4.hist(residuals, bins=40, color="#6baed6", edgecolor="black", alpha=0.75, density=True)
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Bias Line")
    ax4.set_title(f"Prediction Residual Distribution (Mean: {np.mean(residuals):.3f})", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"Residual ($y - \hat{y}$)", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    ax4.legend(frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig2_path = output_dir / "11_normal_nn_actual_vs_predicted.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved prediction analysis to: {fig2_path}")


def main():
    print("=" * 70)
    print("🧠 PHASE 2: NORMAL NEURAL NETWORK DEMAND FORECASTER TRAINING")
    print("=" * 70)

    # 1. Feature Engineering and Data Extraction
    pipeline = M5FeaturePipeline(
        raw_dir=str(PROJECT_ROOT / "data/raw"),
        processed_dir=str(PROJECT_ROOT / "data/processed"),
        n_series=200,  # Stratified cross-department sample
        random_state=42,
    )

    train_dataset, val_dataset, cat_dims, cont_cols = pipeline.build_tabular_dataset(
        start_day=1500,
        end_day=1913,
        val_horizon=28,
    )

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print("\n[MODEL] Initializing NormalDemandMLP architecture...")
    print(f"  Categorical Dimensions: {cat_dims}")
    print(f"  Continuous Features ({len(cont_cols)}): {cont_cols}")

    model = NormalDemandMLP(
        cat_dims=cat_dims,
        num_cont_features=len(cont_cols),
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.20,
        use_batch_norm=True,
        activation="gelu",
        use_softplus_output=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # 2. Training Loop
    trainer = DemandTrainer(
        model=model,
        criterion=nn.HuberLoss(delta=1.0),
        lr=1e-3,
        weight_decay=1e-4,
        clip_grad_norm=1.0,
        checkpoint_dir=str(PROJECT_ROOT / "reports/checkpoints"),
    )

    train_results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=12,
        early_stopping_patience=4,
        verbose=True,
    )

    # 3. Output Metrics & Save Report
    print("\n" + "=" * 70)
    print("📊 FINAL VALIDATION PERFORMANCE SCORECARD")
    print("=" * 70)
    metrics = train_results["final_metrics"]
    for k, v in metrics.items():
        print(f"  {k.upper():12s}: {v}")

    # Plot & Save Visuals
    figures_dir = PROJECT_ROOT / "reports/figures"
    plot_training_results(
        history=train_results["history"],
        y_true=train_results["y_true"],
        y_pred=train_results["y_pred"],
        output_dir=figures_dir,
    )

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "reports/normal_nn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model_name": "NormalDemandMLP",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": [256, 128, 64, 32],
            "regularization": {"dropout": 0.20, "weight_decay": 1e-4, "batch_norm": True},
            "training_time_seconds": train_results["total_time_seconds"],
            "epochs_trained": train_results["epochs_trained"],
            "metrics": metrics,
        }, f, indent=2)
    print(f"[METRICS] Saved metrics to: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

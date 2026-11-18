"""
kalman_visualization.py - Publication-Grade Visualizations for Kalman Filter Forecasting.

DL-PROJECT - Team Contribution: Person 3 (Kalman Filter)

Generates 300 DPI publication figures:
  Fig 10: kalman_aggregate_tracking.png
          System-level daily actual demand vs Kalman point forecast + 95% uncertainty envelope.
  Fig 11: kalman_sku_trajectories.png
          Six representative SKU trajectories with 95% and 80% confidence bands.
  Fig 12: kalman_model_comparison.png
          Four-panel benchmark: MAE, RMSE, WAPE comparisons and category-level WAPE breakdown.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Import shared palette from project visualization module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import PALETTE


class M5KalmanVisualizer:
    """
    Renders publication-grade diagnostic plots for Kalman Filter per-SKU forecasts.

    Args:
        figures_dir: Output directory for saved figure PNGs.
    """

    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: System-Level Aggregate Tracking
    # ------------------------------------------------------------------

    def plot_aggregate_tracking(
        self,
        results_df:   pd.DataFrame,
        metrics_meta: Dict[str, Any],
        filename:     str = "10_kalman_aggregate_tracking.png",
    ) -> Path:
        """
        Plots daily aggregate actual demand vs Kalman Filter point forecast
        with an analytical 95% uncertainty envelope (variance summed across series).
        Lower panel shows daily absolute error residuals.

        Note on aggregation:
            If Y_i are independent, Var(sum Y_i) = sum Var(Y_i).
            => Aggregate std = sqrt(sum sigma_i^2)
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        daily = df.groupby("date").agg(
            demand       =("demand",          "sum"),
            pred_kalman  =("pred_kalman",     "sum"),
            std_agg      =("std_kalman",      lambda s: np.sqrt(np.sum(s ** 2))),
            abs_error    =("abs_error_kalman","sum"),
        ).reset_index()

        daily["lower_95_agg"] = np.maximum(0.0, daily["pred_kalman"] - 1.95996 * daily["std_agg"])
        daily["upper_95_agg"] = daily["pred_kalman"] + 1.95996 * daily["std_agg"]

        kf_meta = metrics_meta["overall_metrics"]["kalman_filter"]
        pi_95   = kf_meta["prediction_intervals"]["ci_95"]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.2]}
        )
        fig.patch.set_facecolor(PALETTE["bg_light"])

        # ---- Upper panel: forecast tracking ----
        ax1.set_facecolor(PALETTE["bg_light"])
        ax1.plot(
            daily["date"], daily["demand"],
            marker="o", markersize=5, linewidth=2.4,
            color=PALETTE["primary"], label="Actual Total Demand", zorder=4,
        )
        ax1.plot(
            daily["date"], daily["pred_kalman"],
            marker="s", markersize=5, linewidth=2.4,
            color=PALETTE["accent3"], linestyle="--",
            label="Kalman Filter Point Forecast", zorder=3,
        )
        ax1.fill_between(
            daily["date"],
            daily["lower_95_agg"], daily["upper_95_agg"],
            color=PALETTE["accent3"], alpha=0.18,
            label=f"Aggregate 95% Uncertainty Envelope  (Coverage: {pi_95['coverage']*100:.1f}%)",
            zorder=2,
        )
        ax1.set_title(
            "System-Level Daily Demand Tracking — State-Space Kalman Filter\n"
            f"MAE: {kf_meta['mae']:.4f} | RMSE: {kf_meta['rmse']:.4f} | "
            f"WAPE: {kf_meta['wape']*100:.2f}%",
            fontsize=13, fontweight="bold", pad=10, color=PALETTE["primary"],
        )
        ax1.set_ylabel("Total Units Sold / Day", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        ax1.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", fontsize=10)

        # ---- Lower panel: residual absolute error ----
        ax2.set_facecolor(PALETTE["bg_light"])
        ax2.bar(
            daily["date"], daily["abs_error"],
            width=0.6, color=PALETTE["accent3"], alpha=0.75,
            edgecolor=PALETTE["primary"], linewidth=0.6,
            label="|y - ŷ| per Day",
        )
        mean_ae = daily["abs_error"].mean()
        ax2.axhline(
            mean_ae, color=PALETTE["accent1"], linestyle=":", linewidth=1.8,
            label=f"Mean Abs Error ({mean_ae:.1f} units)",
        )
        ax2.set_xlabel("Forecast Date (Daily)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Abs Error", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        ax2.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        out = self.figures_dir / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return out

    # ------------------------------------------------------------------
    # Figure 2: SKU-Level Trajectories
    # ------------------------------------------------------------------

    def plot_sku_trajectories(
        self,
        results_df: pd.DataFrame,
        train_df:   pd.DataFrame,
        filename:   str = "11_kalman_sku_trajectories.png",
    ) -> Path:
        """
        Plots 6 representative SKU trajectories across categories, showing:
        - Last 21 days of training history (context window)
        - Forecast origin divider
        - Actual validation demand
        - Kalman point forecast
        - 95% and 80% confidence bands
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        tr = train_df.copy()
        tr["date"] = pd.to_datetime(tr["date"])

        # Pick 2 SKUs per category for diversity
        sample_skus: List[str] = []
        for cat in sorted(df["cat_id"].unique()):
            cat_ids = df[df["cat_id"] == cat]["id"].unique()
            # Prefer moderate-volume series (sorted by mean demand)
            vols = {
                sid: df[df["id"] == sid]["demand"].mean()
                for sid in cat_ids
            }
            sorted_ids = sorted(vols, key=vols.get)
            # Pick one low-volume and one mid-volume
            lo  = sorted_ids[len(sorted_ids) // 4]
            mid = sorted_ids[len(sorted_ids) // 2]
            sample_skus.extend([lo, mid])
            if len(sample_skus) >= 6:
                break

        sample_skus = sample_skus[:6]
        if len(sample_skus) < 6:
            extra = [s for s in df["id"].unique() if s not in sample_skus]
            sample_skus.extend(extra[:6 - len(sample_skus)])

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
        fig.patch.set_facecolor(PALETTE["bg_light"])
        axes_flat = axes.flatten()

        origin_date = df["date"].min()
        hist_start  = origin_date - pd.Timedelta(days=21)

        for i, sid in enumerate(sample_skus):
            ax    = axes_flat[i]
            ax.set_facecolor(PALETTE["bg_light"])
            s_val = df[df["id"] == sid].sort_values("date")
            s_tr  = tr[(tr["id"] == sid) & (tr["date"] >= hist_start)].sort_values("date")

            # Training context
            ax.plot(
                s_tr["date"], s_tr["demand"],
                color="#64748b", linewidth=1.5, marker=".", markersize=4,
                label="Training Context (Last 21d)",
            )
            # Forecast origin
            ax.axvline(
                origin_date, color=PALETTE["accent1"],
                linestyle="--", linewidth=1.6, alpha=0.8, label="Forecast Origin T",
            )
            # Actuals
            ax.plot(
                s_val["date"], s_val["demand"],
                color=PALETTE["primary"], linewidth=2.0, marker="o", markersize=4.5,
                label="Actual Demand",
            )
            # Kalman point forecast
            ax.plot(
                s_val["date"], s_val["pred_kalman"],
                color=PALETTE["accent3"], linewidth=2.0, linestyle="--", marker="s", markersize=4.5,
                label="Kalman Forecast",
            )
            # Confidence bands
            ax.fill_between(
                s_val["date"], s_val["lower_95"], s_val["upper_95"],
                color=PALETTE["accent3"], alpha=0.15, label="95% CI",
            )
            ax.fill_between(
                s_val["date"], s_val["lower_80"], s_val["upper_80"],
                color=PALETTE["accent3"], alpha=0.28, label="80% CI",
            )

            cat_id   = s_val["cat_id"].iloc[0]
            store_id = s_val["store_id"].iloc[0]
            mae_sku  = float(np.mean(s_val["abs_error_kalman"]))
            sum_act  = float(np.sum(s_val["demand"]))
            wape_sku = (float(np.sum(s_val["abs_error_kalman"])) / sum_act * 100) if sum_act > 0 else 0.0

            clean_sid = sid.replace("_validation", "")
            ax.set_title(
                f"{clean_sid}  ({cat_id} @ {store_id})\n"
                f"MAE: {mae_sku:.2f} | WAPE: {wape_sku:.1f}%",
                fontsize=10.5, fontweight="bold", color=PALETTE["primary"],
            )
            ax.set_ylabel("Units Sold", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["grid"])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=8)
            if i == 0:
                ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.5)

        fig.suptitle(
            "Kalman Filter SKU Forecast Trajectories with Dynamic Uncertainty Bands\n"
            "28-Day Out-of-Sample Horizon | Historical Context: Last 21 Training Days",
            fontsize=13, fontweight="bold", y=1.002, color=PALETTE["primary"],
        )
        plt.tight_layout()
        out = self.figures_dir / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return out

    # ------------------------------------------------------------------
    # Figure 3: Model Comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(
        self,
        kalman_meta:    Dict[str, Any],
        nn_metrics_path: Optional[Path] = None,
        filename:       str = "12_kalman_model_comparison.png",
    ) -> Path:
        """
        Four-panel comparison chart:
          Panel 1: MAE   — Kalman vs Normal NN (and Seasonal-Naive if available)
          Panel 2: RMSE
          Panel 3: WAPE
          Panel 4: Category-level WAPE breakdown
        """
        kf = kalman_meta["overall_metrics"]["kalman_filter"]
        cat_data = kalman_meta.get("category_breakdown", {})

        # Load NN metrics if available
        nn_mae, nn_rmse, nn_wape = None, None, None
        if nn_metrics_path and Path(nn_metrics_path).exists():
            try:
                with open(nn_metrics_path) as f:
                    nn_data   = json.load(f)
                nn_mae  = nn_data.get("test_mae",  nn_data.get("mae"))
                nn_rmse = nn_data.get("test_rmse", nn_data.get("rmse"))
                nn_wape = nn_data.get("test_wape", nn_data.get("wape"))
            except Exception:
                pass

        # Build model lists
        models = ["Kalman Filter\n(This Work)"]
        maes   = [kf["mae"]]
        rmses  = [kf["rmse"]]
        wapes  = [kf["wape"] * 100]
        colors = [PALETTE["accent3"]]

        if nn_mae is not None:
            models.insert(0, "Normal NN\n(Team P2)")
            maes.insert(0,  nn_mae)
            rmses.insert(0, nn_rmse)
            wapes.insert(0, nn_wape * 100 if nn_wape < 5 else nn_wape)
            colors.insert(0, PALETTE["secondary"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor(PALETTE["bg_light"])
        (ax1, ax2), (ax3, ax4) = axes

        def _bar_panel(ax, values, title, ylabel, fmt=".4f"):
            bars = ax.bar(models, values, color=colors, width=0.45,
                          edgecolor=PALETTE["primary"], linewidth=1.2)
            ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["primary"])
            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["grid"], axis="y")
            ax.set_facecolor(PALETTE["bg_light"])
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, h * 1.02,
                    f"{h:{fmt}}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=PALETTE["primary"],
                )
            ax.set_ylim(0, max(values) * 1.25)

        _bar_panel(ax1, maes,  "Mean Absolute Error (MAE - Lower is Better)",  "MAE (Units)", ".4f")
        _bar_panel(ax2, rmses, "Root Mean Squared Error (RMSE - Lower is Better)", "RMSE (Units)", ".4f")
        _bar_panel(ax3, wapes, "Weighted Abs % Error (WAPE - Lower is Better)", "WAPE (%)", ".2f")

        # Category-level WAPE breakdown
        categories = sorted(cat_data.keys())
        x     = np.arange(len(categories))
        kf_w  = [cat_data[c]["wape"] * 100 for c in categories]

        ax4.set_facecolor(PALETTE["bg_light"])
        bars4 = ax4.bar(x, kf_w, color=PALETTE["accent3"], width=0.45,
                        edgecolor=PALETTE["primary"], linewidth=1.2, label="Kalman Filter")
        for bar in bars4:
            h = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0, h * 1.02,
                f"{h:.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=PALETTE["primary"],
            )
        ax4.set_title("Category-Level WAPE Breakdown (%)", fontsize=11,
                      fontweight="bold", color=PALETTE["primary"])
        ax4.set_ylabel("WAPE (%)", fontsize=10, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4, color=PALETTE["grid"], axis="y")
        ax4.set_ylim(0, max(kf_w) * 1.3 if kf_w else 1)
        ax4.legend(loc="upper right", frameon=True, fontsize=9)

        fig.suptitle(
            "Kalman Filter Benchmark — MAE · RMSE · WAPE & Category Breakdown\n"
            "M5 Walmart Dataset | 28-Day Out-of-Sample Forecast Horizon",
            fontsize=13, fontweight="bold", y=1.004, color=PALETTE["primary"],
        )
        plt.tight_layout()
        out = self.figures_dir / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return out

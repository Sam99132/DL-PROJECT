"""
run_kalman_pipeline.py - Kalman Filter Forecasting Pipeline Runner.

DL-PROJECT - Team Contribution: Person 3 (Kalman Filter)
Dataset  : Walmart M5 Hierarchical Time-Series (30,490 SKUs, 10 stores)
Model    : Structural State-Space Kalman Filter (Local Level + 7-Day Weekly Seasonality)

Usage:
    # Default: 50 stratified SKUs, 28-day validation horizon
    python scripts/run_kalman_pipeline.py

    # All SKUs (slow — ~30K series)
    python scripts/run_kalman_pipeline.py --n_series -1

    # Tune variance parameters via innovation log-likelihood
    python scripts/run_kalman_pipeline.py --tune_parameters

    # Custom paths
    python scripts/run_kalman_pipeline.py \\
        --sales_path data/processed/cleaned_sales_train_validation.csv \\
        --calendar_path data/processed/cleaned_calendar.csv \\
        --n_series 100

Options:
    --sales_path      Path to cleaned_sales_train_validation.csv  [default: data/processed/...]
    --calendar_path   Path to cleaned_calendar.csv                [default: data/processed/...]
    --output_dir      Directory for output CSV / JSON             [default: data/processed]
    --reports_dir     Directory for metrics JSON                  [default: reports]
    --n_series        Number of SKUs to sample (-1 for all)       [default: 50]
    --val_days        Validation horizon in days                  [default: 28]
    --tune_parameters Flag to enable variance tuning per SKU
    --seed            Random seed for SKU sampling                [default: 42]
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.models.kalman_filter_model import M5KalmanPipeline
from src.models.kalman_visualization import M5KalmanVisualizer


# ---------------------------------------------------------------------------
# Markdown Report Generator
# ---------------------------------------------------------------------------

def generate_kalman_findings_report(metadata: dict, output_path: Path) -> None:
    """Generates a comprehensive academic-style Markdown report for the Kalman Filter run."""

    kf  = metadata["overall_metrics"]["kalman_filter"]
    pi95 = kf["prediction_intervals"]["ci_95"]
    pi80 = kf["prediction_intervals"]["ci_80"]
    cat  = metadata.get("category_breakdown", {})
    ss   = metadata["state_space_model"]

    md = f"""# Kalman Filter Forecasting Report
**Project:** DL-PROJECT — Inventory Demand Forecasting (M5 Walmart Dataset)
**Author Contribution:** Person 3 — Structural State-Space Kalman Filter per SKU
**Evaluation Horizon:** `{metadata['validation_date_range'][0]}` → `{metadata['validation_date_range'][1]}` ({metadata['validation_horizon_days']} days)
**Series Pool:** {metadata['total_series']} Product-Store SKUs
**Generated:** {metadata['evaluation_timestamp']}

---

## 📌 Executive Summary

A **Structural State-Space Kalman Filter** was fitted independently per product-store SKU
on the Walmart M5 daily demand series. The model captures an unobserved stochastic demand
level plus a 7-day weekly seasonal pattern, producing closed-form analytical 28-day forecasts
with rigorous uncertainty quantification (95% and 80% prediction intervals).

---

## 📐 Mathematical Formulation

### State Vector
$$
\\mathbf{{x}}_t = \\begin{{bmatrix}} \\mu_t \\\\ s_{{1,t}} \\\\ s_{{2,t}} \\\\ s_{{3,t}} \\\\ s_{{4,t}} \\\\ s_{{5,t}} \\\\ s_{{6,t}} \\end{{bmatrix}} \\in \\mathbb{{R}}^7
$$

- $\\mu_t$: Unobserved local demand level (random walk)
- $s_{{j,t}}$: Weekly seasonal components satisfying $\\sum_{{j=1}}^7 s_{{j,t}} = 0$

### State Transition Equation
$$
\\mathbf{{x}}_t = \\mathbf{{F}} \\mathbf{{x}}_{{t-1}} + \\mathbf{{w}}_t, \\quad \\mathbf{{w}}_t \\sim \\mathcal{{N}}(\\mathbf{{0}}, \\mathbf{{Q}})
$$

$$
\\mathbf{{Q}} = \\text{{diag}}(q_\\mu,\\; q_s,\\; 0,\\; 0,\\; 0,\\; 0,\\; 0)
$$

### Observation Equation
$$
y_t = \\mathbf{{H}} \\mathbf{{x}}_t + v_t, \\quad v_t \\sim \\mathcal{{N}}(0, R)
$$

$$
\\mathbf{{H}} = \\begin{{bmatrix}} 1 & 1 & 0 & 0 & 0 & 0 & 0 \\end{{bmatrix}} \\implies y_t = \\mu_t + s_{{1,t}} + v_t
$$

### Kalman Recursion (per time step)
**Predict:**
$$
\\hat{{\\mathbf{{x}}}}_{{t|t-1}} = \\mathbf{{F}} \\hat{{\\mathbf{{x}}}}_{{t-1|t-1}}, \\quad
\\mathbf{{P}}_{{t|t-1}} = \\mathbf{{F}} \\mathbf{{P}}_{{t-1|t-1}} \\mathbf{{F}}^\\top + \\mathbf{{Q}}
$$

**Update:**
$$
S_t = \\mathbf{{H}} \\mathbf{{P}}_{{t|t-1}} \\mathbf{{H}}^\\top + R, \\quad
K_t = \\mathbf{{P}}_{{t|t-1}} \\mathbf{{H}}^\\top S_t^{{-1}}
$$
$$
\\hat{{\\mathbf{{x}}}}_{{t|t}} = \\hat{{\\mathbf{{x}}}}_{{t|t-1}} + K_t (y_t - \\mathbf{{H}} \\hat{{\\mathbf{{x}}}}_{{t|t-1}}), \\quad
\\mathbf{{P}}_{{t|t}} = (\\mathbf{{I}} - K_t \\mathbf{{H}}) \\mathbf{{P}}_{{t|t-1}}
$$

---

## ⚙️ Parameter Estimation (Training Data Only)

All noise parameters are derived strictly from training observations — zero validation leakage:

| Parameter | Formula | Rationale |
|:---|:---|:---|
| $R$ (obs. noise) | $\\max(0.05,\\; 0.75 \\cdot \\text{{Var}}(y_{{1:T}}))$ | Dominates short-term demand volatility |
| $q_\\mu$ (level noise) | $\\max(10^{{-4}},\\; 0.005 \\cdot \\text{{Var}}(y_{{1:T}}))$ | Slow stochastic drift in mean demand |
| $q_s$ (seasonal noise) | $\\max(10^{{-5}},\\; 0.001 \\cdot \\text{{Var}}(y_{{1:T}}))$ | Smooth seasonal pattern evolution |

**Missing observation strategy:** Predict-only step (no measurement update). Covariance expands naturally.
**Zero demand:** Treated as a valid $y_t = 0$ observation (not a stockout / missing entry).
**Non-negativity:** Point forecasts and lower PI bounds clipped at $\\max(0,\\; \\cdot)$.

---

## 📊 Results

### Overall Performance

| Metric | Value |
|:---|:---:|
| **MAE** | **{kf['mae']:.4f}** units |
| **RMSE** | **{kf['rmse']:.4f}** units |
| **WAPE** | **{kf['wape']*100:.2f}%** |
| Total Series | {metadata['total_series']} SKUs |
| Validation Obs | {metadata['total_validation_observations']:,} |

### Prediction Interval Performance

| Interval | Target | Empirical Coverage | Mean Width |
|:---|:---:|:---:|:---:|
| **95% PI** | 95.0% | **{pi95['coverage']*100:.2f}%** | {pi95['mean_width']:.4f} units |
| **80% PI** | 80.0% | **{pi80['coverage']*100:.2f}%** | {pi80['mean_width']:.4f} units |

> Kalman provides **exact analytic covariance propagation** $\\mathbf{{P}}_{{T+h|T}}$,
> ensuring well-calibrated prediction intervals without Monte Carlo sampling.

### Category-Level Breakdown

| Category | MAE | RMSE | WAPE | 95% PI Coverage |
|:---|:---:|:---:|:---:|:---:|
"""
    for c, m in sorted(cat.items()):
        md += (
            f"| **{c}** | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['wape']*100:.2f}% | {m.get('coverage_95',0)*100:.1f}% |\n"
        )

    md += f"""
---

## 🖼️ Figures Generated

| Figure | Description |
|:---|:---|
| `10_kalman_aggregate_tracking.png` | System-level daily demand tracking with 95% uncertainty envelope and residual panel |
| `11_kalman_sku_trajectories.png` | Six representative SKU forecasts with 95%/80% confidence bands |
| `12_kalman_model_comparison.png` | MAE, RMSE, WAPE comparison + category-level WAPE breakdown |

---

## 🔑 Key Properties

1. **No data leakage** — all parameters estimated on training history only; predict() uses
   no ground-truth validation data whatsoever.
2. **Interpretable state** — $\\mu_t$ gives the latent consumption rate; $s_{{j,t}}$ quantifies
   day-of-week seasonal lift/drag, both readable directly.
3. **Uncertainty is analytic** — covariance $P_t$ is propagated exactly, no sampling needed.
4. **Intermittent demand robust** — zero sales are valid observations; NaN gaps expand
   uncertainty correctly without corrupting the state estimate.
5. **Linear complexity** — $O(N \\cdot T)$ in series count and history length; runs in seconds
   even for hundreds of SKUs.

---

*Report auto-generated by `scripts/run_kalman_pipeline.py`*
"""

    output_path.write_text(md, encoding="utf-8")
    print(f"  Markdown findings report → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="M5 Inventory Demand — Structural Kalman Filter Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sales_path",    default="data/processed/cleaned_sales_train_validation.csv")
    parser.add_argument("--calendar_path", default="data/processed/cleaned_calendar.csv")
    parser.add_argument("--output_dir",    default="data/processed")
    parser.add_argument("--reports_dir",   default="reports")
    parser.add_argument("--n_series",      type=int,  default=50,
                        help="Number of SKUs to sample (-1 = all)")
    parser.add_argument("--val_days",      type=int,  default=28,
                        help="Validation horizon in days")
    parser.add_argument("--tune_parameters", action="store_true",
                        help="Tune Kalman variance ratios via innovation log-likelihood")
    parser.add_argument("--seed",          type=int,  default=42)
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Run Kalman Pipeline
    # ----------------------------------------------------------------
    pipeline = M5KalmanPipeline(
        sales_path      = args.sales_path,
        calendar_path   = args.calendar_path,
        output_dir      = args.output_dir,
        reports_dir     = args.reports_dir,
        n_series        = args.n_series,
        val_days        = args.val_days,
        tune_parameters = args.tune_parameters,
        random_seed     = args.seed,
    )
    results = pipeline.run()

    # ----------------------------------------------------------------
    # 2. Generate Visualizations
    # ----------------------------------------------------------------
    print("\nGenerating publication-grade figures (300 DPI)...")
    vis = M5KalmanVisualizer(figures_dir=str(pipeline.figures_dir))

    fig1 = vis.plot_aggregate_tracking(results["predictions_df"], results["metrics"])
    print(f"  [1/3] Aggregate tracking    → {fig1}")

    fig2 = vis.plot_sku_trajectories(results["predictions_df"], results["train_df"])
    print(f"  [2/3] SKU trajectories      → {fig2}")

    nn_metrics_path = Path(args.reports_dir) / "normal_nn_metrics.json"
    fig3 = vis.plot_model_comparison(results["metrics"], nn_metrics_path)
    print(f"  [3/3] Model comparison      → {fig3}")

    # ----------------------------------------------------------------
    # 3. Markdown findings report
    # ----------------------------------------------------------------
    findings_path = Path(args.reports_dir) / "kalman_findings.md"
    generate_kalman_findings_report(results["metrics"], findings_path)

    # ----------------------------------------------------------------
    # 4. Console summary table
    # ----------------------------------------------------------------
    kf = results["metrics"]["overall_metrics"]["kalman_filter"]
    pi = kf["prediction_intervals"]

    print("\n" + "=" * 65)
    print("  KALMAN FILTER — FORECASTING BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Value':>12}")
    print("-" * 65)
    print(f"  {'MAE (units)':<30} {kf['mae']:>12.4f}")
    print(f"  {'RMSE (units)':<30} {kf['rmse']:>12.4f}")
    print(f"  {'WAPE (%)':<30} {kf['wape']*100:>11.2f}%")
    print(f"  {'95% PI Coverage':<30} {pi['ci_95']['coverage']*100:>11.2f}%")
    print(f"  {'95% PI Mean Width':<30} {pi['ci_95']['mean_width']:>12.4f}")
    print(f"  {'80% PI Coverage':<30} {pi['ci_80']['coverage']*100:>11.2f}%")
    print(f"  {'80% PI Mean Width':<30} {pi['ci_80']['mean_width']:>12.4f}")
    print(f"  {'Total Series':<30} {results['metrics']['total_series']:>12}")
    print(f"  {'Validation Observations':<30} {results['metrics']['total_validation_observations']:>12,}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

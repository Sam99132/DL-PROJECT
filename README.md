# Inventory Demand Forecasting: Data Cleaning & Exploratory Data Analysis (EDA)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Walmart M5](https://img.shields.io/badge/Dataset-Walmart%20M5-orange.svg)](https://www.kaggle.com/c/m5-forecasting-accuracy)
[![Pipeline Status: Validated](https://img.shields.io/badge/Pipeline-Validated-success.svg)](#)

A high-performance, mathematically rigorous **Data Cleaning**, **Quality Audit**, and **Exploratory Data Analysis (EDA)** pipeline for the Walmart M5 Hierarchical Time-Series Dataset (30,490 SKUs across 10 stores in California, Texas, and Wisconsin).

This module establishes the empirical foundation, data integrity, and theoretical motivation for downstream time-series forecasting using **State-Space Kalman Filters** and **Global DeepAR Probabilistic Models**.

---

## 📌 Executive Summary & Key Insights

1. **Severe Intermittent & Lumpy Sparsity:** Over **90.84%** of Walmart's SKUs fall into either the **Intermittent (72.55%)** or **Lumpy (18.29%)** demand quadrants under the Syntetos-Boylan / Croston classification ($ADI \ge 1.32$). Standard Gaussian autoregressive models fail; state-space filtering and zero-inflated distributions are mathematically essential.
2. **Active vs. Pre-Release Zeros:** Raw sales exhibit **68.20% zero sales**, but **12.38 million observations** are pre-release leading zeros before product shelf introduction. Filtering these reveals a true active shelf zero rate of **59.62%**.
3. **SNAP Policy Demand Shocks:** Supplemental Nutrition Assistance Program (SNAP) disbursements trigger massive, statistically significant demand surges: **+7.56% in CA**, **+11.05% in TX**, and **+21.31% in WI**, primarily concentrated in the `FOODS` department.
4. **Deterministic Holiday Closures:** Total chain sales drop to near-zero ($11$ units chain-wide) strictly on **December 25 (Christmas Day)**, representing store closures rather than demand collapse.
5. **Stationarity & Autocorrelation:** The raw system-wide demand series is non-stationary ($p = 0.501$), but first-differencing achieves complete stationarity ($p < 10^{-15}$). Autocorrelation shows sharp harmonic weekly spikes ($r_7 = 0.81$, $r_{14} = 0.74$, $r_{28} = 0.82$).
6. **Kalman State-Space Tracking:** Demonstrates how a 1D/2D local linear trend Kalman filter separates the latent demand velocity from noisy, sporadic discrete point observations while providing exact uncertainty bounds ($\pm 1.96\sigma$).

---

## 📂 Project Architecture & Directory Layout

```text
DL-PROJECT/
├── data/
│   ├── raw/                              # Linked to raw/ M5 competition CSVs
│   │   ├── calendar.csv                  # 1,969 daily calendar & SNAP records
│   │   ├── sell_prices.csv               # 6.84M weekly SKU price records
│   │   ├── sales_train_validation.csv    # 30,490 SKUs × 1,913 days
│   │   ├── sales_train_evaluation.csv    # 30,490 SKUs × 1,941 days
│   │   └── sample_submission.csv         # Official submission template
│   └── processed/                        # 1-to-1 Cleaned production datasets
│       ├── cleaned_calendar.csv          # Dates, cyclics, event flags, Christmas
│       ├── cleaned_sell_prices.csv       # Regular price baseline, discount %
│       ├── cleaned_sales_train_validation.csv  # Non-negative, clipped, verified
│       ├── cleaned_sales_train_evaluation.csv  # Full 1,941-day evaluated matrix
│       └── cleaned_sample_submission.csv       # Verified submission schema
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # Interactive step-by-step cleaning notebook
│   └── 02_exploratory_data_analysis.ipynb# Interactive end-to-end EDA notebook
├── reports/
│   ├── data_quality_report.md            # Markdown Data Quality Audit Report
│   ├── data_quality_report.json          # Machine-readable quality metrics
│   ├── eda_findings.md                   # Comprehensive academic EDA report
│   ├── eda_metrics.json                  # Machine-readable statistical metrics
│   └── figures/                          # 9 Publication-Grade 300 DPI Figures
│       ├── 01_hierarchical_sales_overview.png
│       ├── 02_intermittent_croston_matrix.png
│       ├── 03_zero_demand_distribution.png
│       ├── 04_seasonality_and_calendar_effects.png
│       ├── 05_holiday_and_snap_impact.png
│       ├── 06_price_elasticity_analysis.png
│       ├── 07_store_level_comparison.png
│       ├── 08_acf_pacf_stationarity.png
│       └── 09_kalman_filter_motivation.png
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Memory-optimized downcasting loaders
│   ├── data_cleaning.py                  # 1-to-1 Automated Data Cleaner
│   ├── data_quality.py                   # Data Quality & Integrity Auditor
│   ├── eda_analysis.py                   # Statistical, Croston, & State-Space Engine
│   ├── visualization.py                  # Publication-grade visual generator
│   └── run_eda_pipeline.py               # Master end-to-end orchestration pipeline
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Ignored artifacts & temporary files
└── README.md                             # Project documentation
```

---

## ⚙️ Data Cleaning Pipeline (1-to-1 Mapping)

The data cleaning pipeline (`src/data_cleaning.py`) executes a strict **1-to-1 transformation** from `data/raw/` to `data/processed/`:

| Raw File | Cleaned Output | Key Transformations Applied |
| :--- | :--- | :--- |
| `calendar.csv` | `cleaned_calendar.csv` | Date parsing; cyclic trigonometric encodings ($\sin/\cos$ for day-of-week and month); binary event flags; Christmas closure indicators; missing event imputation (`'None'`). |
| `sell_prices.csv` | `cleaned_sell_prices.csv` | Regular price baseline calculation ($\max$ price per item-store); discount percentage calculation ($(\text{reg} - \text{sell})/\text{reg}$); promotional flag (`discount_pct > 0.05`). |
| `sales_train_validation.csv` | `cleaned_sales_train_validation.csv` | Audit 30,490 series across 1,913 days; non-negativity verification; clipping lower at 0; fill any missing entries. |
| `sales_train_evaluation.csv` | `cleaned_sales_train_evaluation.csv` | Extended evaluation matrix (1,941 days); validated non-negativity and integrity. |
| `sample_submission.csv` | `cleaned_sample_submission.csv` | Validated 60,980 rows; filled forecast nulls with $0.0$; schema verification. |

Run the cleaner directly:
```bash
python3 src/data_cleaning.py
```

---

## 📊 Data Quality & Integrity Scorecard

Evaluated by `src/data_quality.py` across **58,327,370 unit-day observations**:

| Quality Dimension | Evaluated Metric | Value | Verdict |
| :--- | :--- | :--- | :--- |
| **Completeness** | Sales Matrix Nulls | 0 / 58,327,370 (0.00%) | **EXCELLENT** |
| **Validity** | Non-Negative Sales | 100% (No negative values) | **VALID** |
| **Price Sanity** | Minimum Sell Price | \$0.01 (No $\le 0$ prices) | **VALID** |
| **Uniqueness** | SKU-Store Pairs | 30,490 unique series | **EXCELLENT** |
| **Continuity** | Temporal Coverage | 1,969 unbroken calendar days | **UNBROKEN** |
| **Referential Integrity**| Keys Across Tables | 100% matched items and stores | **CONFIRMED** |

---

## 🔬 Exploratory Data Analysis & Statistical Modeling

### 1. Syntetos-Boylan & Croston Intermittent Demand Categorization
Retail items are partitioned along two axes:
- **Average Demand Interval (ADI):** Mean time between positive orders ($\alpha = 1.32$).
- **Squared Coefficient of Variation ($CV^2$):** Variance of non-zero quantities over mean squared ($\beta = 0.49$).

$$\text{ADI} = \frac{T}{N_{\text{active}}}, \quad \text{CV}^2 = \left(\frac{\sigma_{\text{non-zero}}}{\mu_{\text{non-zero}}}\right)^2$$

| Demand Category | Diagnostic Condition | SKU Count | Percentage | Forecasting Method |
| :--- | :--- | :--- | :--- | :--- |
| **Intermittent** | $\text{ADI} \ge 1.32, CV^2 < 0.49$ | **22,121** | **72.55%** | Croston / SBA / Kalman Filters |
| **Lumpy** | $\text{ADI} \ge 1.32, CV^2 \ge 0.49$ | **5,577** | **18.29%** | State-Space / DeepAR Probabilistic |
| **Smooth** | $\text{ADI} < 1.32, CV^2 < 0.49$ | **1,912** | **6.27%** | GBDT / Linear Baselines |
| **Erratic** | $\text{ADI} < 1.32, CV^2 \ge 0.49$ | **880** | **2.89%** | Global Neural Networks |

### 2. SNAP Policy & Seasonality Effects
- **SNAP Uplift:** Scheduled monthly food assistance disbursements cause surges of **+7.56% (CA)**, **+11.05% (TX)**, and **+21.31% (WI)**.
- **Weekly Seasonality:** Peak purchasing occurs on Saturday (index: 1.25) and Sunday (index: 1.21), while Tuesday and Wednesday represent trough days (index: 0.88).
- **Christmas Anomaly:** Store-wide closure on December 25 drops chain sales to near-zero ($11$ units), requiring explicit holiday masking in state-space transition equations.

### 3. Stationarity & Autocorrelation
- **ADF Raw Series:** $t = -1.5654$ ($p = 0.5010$), non-stationary.
- **ADF First Difference ($\Delta y_t$):** $t = -23.4596$ ($p < 10^{-15}$), stationary.
- **Harmonic ACF Peaks:** Multi-lag weekly periodicity at lag 7 ($r = 0.81$), lag 14 ($r = 0.74$), lag 21 ($r = 0.72$), and lag 28 ($r = 0.82$).

### 4. Kalman Filter State-Space Tracking
Daily demand $z_t$ is modeled as a noisy realization of a latent state $x_t$:
$$\begin{aligned}
x_t &= F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
z_t &= H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
\end{aligned}$$
The Kalman filter dynamically estimates the underlying consumption rate while filtering out sporadic observation zeros, maintaining error covariance $P_t$ for dynamic safety stock calculation.

---

## 🖼️ Generated Publication Figures (`reports/figures/`)

All figures are rendered at **300 DPI** using modern high-contrast styling:

| Figure File | Description |
| :--- | :--- |
| `01_hierarchical_sales_overview.png` | System-wide 5-year sales trajectory, 30-day moving average, Christmas closures, and State/Category breakdowns. |
| `02_intermittent_croston_matrix.png` | Scatter plot and donut distribution across Smooth, Intermittent, Erratic, and Lumpy quadrants. |
| `03_zero_demand_distribution.png` | Kernel density comparison of raw zero rates vs. active shelf zero rates. |
| `04_seasonality_and_calendar_effects.png` | Normalized daily demand indices for day-of-week and month-of-year. |
| `05_holiday_and_snap_impact.png` | State-wise SNAP demand uplift (%) and top 8 holiday event sales spikes. |
| `06_price_elasticity_analysis.png` | Category-level price distributions and department price dispersions. |
| `07_store_level_comparison.png` | Total sales volume and category composition across all 10 Walmart stores. |
| `08_acf_pacf_stationarity.png` | Autocorrelation function bar chart (lags 1-28) and ADF stationarity scorecard. |
| `09_kalman_filter_motivation.png` | Latent state tracking vs. noisy intermittent sales with $95\%$ confidence intervals. |

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Sam99132/DL-PROJECT.git
cd DL-PROJECT
pip install -r requirements.txt
```

### 2. Run Data Cleaning
Transform raw files into cleaned datasets:
```bash
python3 src/data_cleaning.py
```

### 3. Run End-to-End EDA & Quality Pipeline
Execute data quality audit, compute all statistical aggregations, generate 9 figures, and export findings:
```bash
python3 src/run_eda_pipeline.py
```

### 4. Interactive Jupyter Notebooks
Launch Jupyter to explore the step-by-step interactive workflows:
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
jupyter notebook notebooks/02_exploratory_data_analysis.ipynb
```

---

## 📄 Documentation Reports

- **Data Quality Audit Report:** [`reports/data_quality_report.md`](file:///Users/dakarapuhemamdharnath/Desktop/DL-PROJECT/reports/data_quality_report.md)
- **EDA & Theoretical Findings Report:** [`reports/eda_findings.md`](file:///Users/dakarapuhemamdharnath/Desktop/DL-PROJECT/reports/eda_findings.md)
- **Quality Metrics (JSON):** [`reports/data_quality_report.json`](file:///Users/dakarapuhemamdharnath/Desktop/DL-PROJECT/reports/data_quality_report.json)
- **Statistical Metrics (JSON):** [`reports/eda_metrics.json`](file:///Users/dakarapuhemamdharnath/Desktop/DL-PROJECT/reports/eda_metrics.json)


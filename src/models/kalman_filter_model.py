"""
kalman_filter_model.py - Structural State-Space Kalman Filtering per SKU for M5 Demand.

DL-PROJECT - Team Contribution: Person 3 (Kalman Filter)
Dataset: Walmart M5 Hierarchical Time-Series Forecasting

This module implements:
1. SKUKalmanForecaster: A modular, univariate structural time-series model
   (Local Level + 7-Day Weekly Seasonality) fitted independently per product-store series.
2. M5KalmanPipeline: End-to-end pipeline that fits separate filters per SKU,
   forecasts 28-day out-of-sample validation horizon with analytical uncertainty bounds,
   evaluates MAE, RMSE, WAPE, and prediction interval coverage/width, and exports results.

State-Space Mathematical Formulation:
--------------------------------------
State Vector x_t in R^7:
    x_t = [mu_t, s_{1,t}, s_{2,t}, s_{3,t}, s_{4,t}, s_{5,t}, s_{6,t}]^T
    where:
    - mu_t         : Unobserved local demand level (random walk / stochastic drift).
    - s_{1,t}...s_{6,t}: Unobserved weekly seasonal components satisfying
      the zero-sum constraint: sum_{j=1}^{7} s_{j,t} = 0.

State Transition Equation:
    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
    F = [[1,  0,  0,  0,  0,  0,  0],
         [0, -1, -1, -1, -1, -1, -1],
         [0,  1,  0,  0,  0,  0,  0],
         [0,  0,  1,  0,  0,  0,  0],
         [0,  0,  0,  1,  0,  0,  0],
         [0,  0,  0,  0,  1,  0,  0],
         [0,  0,  0,  0,  0,  1,  0]]
    Q = diag(q_mu, q_s, 0, 0, 0, 0, 0)

Observation Equation:
    y_t = H x_t + v_t,   v_t ~ N(0, R)
    H = [1, 1, 0, 0, 0, 0, 0]
    => y_t = mu_t + s_{1,t} + v_t

Missing Observation Handling:
    When y_t is NaN, the filter performs the time update (predict step) only,
    skipping the measurement update. Uncertainty P_t properly expands.
    Zero demand (y_t = 0) is treated as a valid observation of zero sales.

Non-Negative Demand Handling:
    Point forecasts and lower prediction interval bounds are clipped at 0.0,
    reflecting physical retail inventory constraints:
        y_hat      = max(0.0, H x_{T+h|T})
        lower_95   = max(0.0, y_hat_raw - 1.96 * sigma)
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("KalmanFilter")


# ===========================================================================
# Metric Helpers
# ===========================================================================

def compute_forecasting_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and WAPE (Weighted Absolute Percentage Error).
    WAPE avoids division-by-zero when total actual demand is 0.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("Cannot evaluate metrics on empty arrays.")

    abs_err = np.abs(y_t - y_p)
    mae  = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    sum_actual  = float(np.sum(y_t))
    sum_abs_err = float(np.sum(abs_err))
    wape = sum_abs_err / sum_actual if sum_actual > 0 else 0.0

    return {
        "mae":          round(mae,  4),
        "rmse":         round(rmse, 4),
        "wape":         round(wape, 4),
        "sum_actual":   round(sum_actual,  2),
        "sum_abs_err":  round(sum_abs_err, 2),
        "count":        len(y_t),
    }


def compute_prediction_interval_metrics(
    y_true:      Union[np.ndarray, pd.Series],
    lower_bound: Union[np.ndarray, pd.Series],
    upper_bound: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Computes empirical prediction interval evaluation metrics:
    1. Coverage  : Proportion of ground-truth observations in [lower, upper].
    2. Mean Width: Average spread (upper - lower) measuring forecast sharpness.
    """
    y_t = np.asarray(y_true,      dtype=np.float64)
    l_b = np.asarray(lower_bound, dtype=np.float64)
    u_b = np.asarray(upper_bound, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("Cannot evaluate intervals on empty arrays.")

    inside   = (y_t >= l_b) & (y_t <= u_b)
    coverage = float(np.mean(inside))
    width    = float(np.mean(u_b - l_b))

    return {
        "coverage":     round(coverage, 4),
        "mean_width":   round(width,    4),
        "inside_count": int(np.sum(inside)),
        "total_count":  len(y_t),
    }


# ===========================================================================
# SKUKalmanForecaster
# ===========================================================================

class SKUKalmanForecaster:
    """
    Univariate Structural Time-Series Kalman Filter for a single SKU.

    Implements a Local Level + 7-Day Weekly Seasonality state-space model:
        State dimension: 7
        State: [mu, s_1, s_2, s_3, s_4, s_5, s_6]

    Parameters are estimated strictly from training data to prevent leakage.

    Args:
        q_level_ratio  : Fraction of empirical variance for level process noise.
        q_season_ratio : Fraction of empirical variance for seasonal process noise.
        r_ratio        : Fraction of empirical variance for observation noise.
        tune_parameters: If True, selects best variance ratios via training
                         innovation log-likelihood grid search.
    """

    def __init__(
        self,
        q_level_ratio:  float = 0.005,
        q_season_ratio: float = 0.001,
        r_ratio:        float = 0.75,
        tune_parameters: bool = False,
    ):
        self.q_level_ratio  = q_level_ratio
        self.q_season_ratio = q_season_ratio
        self.r_ratio        = r_ratio
        self.tune_parameters = tune_parameters

        self.dim_x        = 7
        self.season_length = 7

        # ----------------------------------------------------------------
        # Measurement matrix H: shape (1, 7)
        # y_t = mu_t + s_{1,t} + v_t
        # ----------------------------------------------------------------
        self.H = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        # ----------------------------------------------------------------
        # Transition matrix F: shape (7, 7)
        # Row 0: mu_t  = mu_{t-1}                    (level random walk)
        # Row 1: s_{1} = -(s_1+...+s_6)_{t-1}        (sum-to-zero seasonal)
        # Row 2..6: s_{j,t} = s_{j-1,t-1}            (shift register)
        # ----------------------------------------------------------------
        self.F = np.zeros((7, 7), dtype=np.float64)
        self.F[0, 0]   = 1.0           # Level random walk
        self.F[1, 1:7] = -1.0          # Seasonal sum constraint
        for i in range(2, 7):
            self.F[i, i - 1] = 1.0    # Shift register

        # Fitted attributes (populated after fit())
        self.R:    Optional[float]     = None
        self.q_mu: Optional[float]     = None
        self.q_s:  Optional[float]     = None
        self.Q:    Optional[np.ndarray] = None
        self.x_post: Optional[np.ndarray] = None   # Posterior state at origin T (7,)
        self.P_post: Optional[np.ndarray] = None   # Posterior covariance at T   (7,7)
        self.log_likelihood_:    Optional[float]  = None
        self.training_series_id: Optional[str]   = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_Q(self, q_mu: float, q_s: float) -> np.ndarray:
        """Constructs the 7x7 process noise covariance matrix Q."""
        Q = np.zeros((7, 7), dtype=np.float64)
        Q[0, 0] = q_mu
        Q[1, 1] = q_s
        return Q

    def _evaluate_log_likelihood(
        self,
        y:    np.ndarray,
        q_mu: float,
        q_s:  float,
        R:    float,
    ) -> float:
        """
        Computes per-step Gaussian innovation log-likelihood on training data:
            LL = -0.5 * mean_t [ ln(2*pi*S_t) + v_t^2 / S_t ]
        Evaluated after a 28-day burn-in period.
        """
        Q   = self._build_Q(q_mu, q_s)
        var_y = float(np.var(y)) if len(y) > 0 else 1.0
        x   = np.zeros(7, dtype=np.float64)
        x[0] = float(np.mean(y[:min(len(y), 28)]))
        P   = np.eye(7, dtype=np.float64) * max(1.0, var_y)

        total_ll   = 0.0
        valid_steps = 0
        burn_in    = min(28, len(y) // 4)

        for t, yt in enumerate(y):
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + Q

            if np.isnan(yt):
                x, P = x_pred, P_pred
                continue

            S     = float((self.H @ P_pred @ self.H.T)[0, 0] + R)
            innov = float(yt - (self.H @ x_pred)[0])

            if t >= burn_in:
                total_ll   += -0.5 * (np.log(2.0 * np.pi * max(S, 1e-9)) + (innov ** 2) / max(S, 1e-9))
                valid_steps += 1

            K = (P_pred @ self.H.T) / max(S, 1e-9)
            x = x_pred + K.flatten() * innov
            P = (np.eye(7) - K @ self.H) @ P_pred

        return total_ll / max(1, valid_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_series: Union[np.ndarray, pd.Series],
        series_id:    Optional[str]        = None,
        dates:        Optional[pd.Series]  = None,
    ) -> "SKUKalmanForecaster":
        """
        Fits the Kalman Filter sequentially over the training demand series.

        Procedure:
          1. Estimate noise variances from training data only (no validation leakage).
          2. Optionally tune variance ratios via innovation log-likelihood grid search.
          3. Run forward filtering pass to obtain posterior x_{T|T}, P_{T|T}.

        Args:
            train_series: Array of historical daily demand units.
            series_id   : Optional identifier for logging.
            dates       : Optional date index aligned to train_series.

        Returns:
            self: Fitted SKUKalmanForecaster.
        """
        self.training_series_id = series_id

        y = np.asarray(train_series, dtype=np.float64)
        if len(y) == 0:
            raise ValueError(f"Training series for {series_id} is empty.")

        # Empirical variance from training (excluding NaNs)
        valid_y = y[~np.isnan(y)]
        var_y   = float(np.var(valid_y)) if len(valid_y) > 1 else 1.0
        if var_y < 1e-4:
            var_y = 1.0

        # ------------------------------------------------------------------
        # Parameter selection
        # ------------------------------------------------------------------
        if self.tune_parameters and len(valid_y) >= 56:
            candidate_qls = [0.001, 0.005, 0.01, 0.02]
            candidate_qss = [0.0005, 0.001, 0.005]
            best_ll = -np.inf
            best_ql = self.q_level_ratio
            best_qs = self.q_season_ratio
            base_R  = max(0.05, self.r_ratio * var_y)

            for ql in candidate_qls:
                for qs in candidate_qss:
                    ll = self._evaluate_log_likelihood(y, ql * var_y, qs * var_y, base_R)
                    if ll > best_ll:
                        best_ll, best_ql, best_qs = ll, ql, qs

            self.q_mu  = max(1e-4, best_ql * var_y)
            self.q_s   = max(1e-5, best_qs * var_y)
            self.R     = base_R
            self.log_likelihood_ = best_ll
        else:
            self.R    = max(0.05, self.r_ratio        * var_y)
            self.q_mu = max(1e-4, self.q_level_ratio  * var_y)
            self.q_s  = max(1e-5, self.q_season_ratio * var_y)

        self.Q = self._build_Q(self.q_mu, self.q_s)

        # ------------------------------------------------------------------
        # State / Covariance initialisation
        # ------------------------------------------------------------------
        # Initial level = sample mean of first 28 (or fewer) valid observations
        init_level = float(np.mean(valid_y[:min(len(valid_y), 28)])) if len(valid_y) > 0 else 0.0
        x = np.zeros(7, dtype=np.float64)
        x[0] = init_level

        # Diffuse initialisation: uncertainty scaled to empirical variance
        P = np.eye(7, dtype=np.float64) * var_y

        # ------------------------------------------------------------------
        # Forward Filtering Pass
        # ------------------------------------------------------------------
        for yt in y:
            # 1. Time Update (Predict)
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # 2. Missing Observation: skip measurement update
            if np.isnan(yt):
                x, P = x_pred, P_pred
                continue

            # 3. Measurement Update
            y_pred = float((self.H @ x_pred)[0])
            innov  = yt - y_pred                            # Innovation v_t

            S = float((self.H @ P_pred @ self.H.T)[0, 0] + self.R)  # Innovation covariance
            K = (P_pred @ self.H.T) / max(S, 1e-9)         # Kalman Gain (7,1)

            x = x_pred + K.flatten() * innov               # State update
            P = (np.eye(7) - K @ self.H) @ P_pred          # Covariance update

            # Enforce numerical symmetry
            P = 0.5 * (P + P.T)

        # Store posterior distribution at forecast origin T
        self.x_post = x.copy()
        self.P_post = P.copy()
        return self

    def predict(self, horizon: int = 28) -> Dict[str, np.ndarray]:
        """
        Generates multi-step out-of-sample forecasts for h = 1 ... horizon
        WITHOUT consuming any validation data.

        State propagation (pure open-loop):
            x_{T+h|T}  = F^h x_{T|T}
            P_{T+h|T}  = F P_{T+h-1|T} F^T + Q
            y_hat_raw  = H x_{T+h|T}
            sigma^2    = H P_{T+h|T} H^T + R

        Non-negative clipping for retail demand:
            point      = max(0, y_hat_raw)
            lower_95   = max(0, y_hat_raw - 1.96 * sigma)
            upper_95   = max(0, y_hat_raw + 1.96 * sigma)

        Args:
            horizon: Number of forecast steps (default 28 days).

        Returns:
            Dict with keys:
              'point'    : Clipped point forecasts (horizon,)
              'raw_point': Unclipped point forecasts
              'std'      : Forecast standard deviations
              'lower_95' : Lower 95% PI bound (clipped at 0)
              'upper_95' : Upper 95% PI bound
              'lower_80' : Lower 80% PI bound (clipped at 0)
              'upper_80' : Upper 80% PI bound
        """
        if self.x_post is None or self.P_post is None:
            raise RuntimeError("Call fit() before predict().")

        z_95 = 1.95996   # Normal quantile for 95% PI
        z_80 = 1.28155   # Normal quantile for 80% PI

        preds_point, preds_raw, preds_std = [], [], []
        lower_95, upper_95 = [], []
        lower_80, upper_80 = [], []

        x_h = self.x_post.copy()
        P_h = self.P_post.copy()

        for _ in range(horizon):
            # Propagate state and covariance forward one step
            x_h = self.F @ x_h
            P_h = self.F @ P_h @ self.F.T + self.Q
            P_h = 0.5 * (P_h + P_h.T)   # Symmetry

            # Point forecast and total predictive variance
            y_raw    = float((self.H @ x_h)[0])
            var_pred = float((self.H @ P_h @ self.H.T)[0, 0] + self.R)
            std_pred = float(np.sqrt(max(0.0, var_pred)))

            # Non-negative clipping
            point_clipped = max(0.0, y_raw)
            l95 = max(0.0, y_raw - z_95 * std_pred)
            u95 = max(0.0, y_raw + z_95 * std_pred)
            l80 = max(0.0, y_raw - z_80 * std_pred)
            u80 = max(0.0, y_raw + z_80 * std_pred)

            preds_point.append(point_clipped)
            preds_raw.append(y_raw)
            preds_std.append(std_pred)
            lower_95.append(l95);  upper_95.append(u95)
            lower_80.append(l80);  upper_80.append(u80)

        return {
            "point":     np.array(preds_point, dtype=np.float64),
            "raw_point": np.array(preds_raw,   dtype=np.float64),
            "std":       np.array(preds_std,   dtype=np.float64),
            "lower_95":  np.array(lower_95,    dtype=np.float64),
            "upper_95":  np.array(upper_95,    dtype=np.float64),
            "lower_80":  np.array(lower_80,    dtype=np.float64),
            "upper_80":  np.array(upper_80,    dtype=np.float64),
        }


# ===========================================================================
# M5KalmanPipeline
# ===========================================================================

class M5KalmanPipeline:
    """
    End-to-End Orchestrator for Structural Kalman Filtering per SKU on M5 data.

    Workflow:
      1. Loads & reshapes cleaned M5 sales CSVs into long format train/val splits.
      2. Fits a dedicated SKUKalmanForecaster for each product-store time series.
      3. Produces 28-day out-of-sample multi-step forecasts with analytical uncertainty.
      4. Evaluates MAE, RMSE, WAPE and 95% / 80% interval coverage + width.
      5. Saves kalman_predictions.csv and kalman_metrics.json.

    Args:
        sales_path       : Path to cleaned_sales_train_validation.csv (wide format).
        calendar_path    : Path to cleaned_calendar.csv.
        output_dir       : Directory for output CSV / JSON.
        reports_dir      : Directory for metrics JSON.
        n_series         : Number of SKUs to sample (-1 = all).
        val_days         : Validation horizon in days (default 28).
        tune_parameters  : Whether to tune Kalman variance params per SKU.
        random_seed      : Seed for reproducible SKU sampling.
    """

    def __init__(
        self,
        sales_path:      str  = "data/processed/cleaned_sales_train_validation.csv",
        calendar_path:   str  = "data/processed/cleaned_calendar.csv",
        output_dir:      str  = "data/processed",
        reports_dir:     str  = "reports",
        n_series:        int  = 50,
        val_days:        int  = 28,
        tune_parameters: bool = False,
        random_seed:     int  = 42,
    ):
        self.sales_path      = Path(sales_path)
        self.calendar_path   = Path(calendar_path)
        self.output_dir      = Path(output_dir)
        self.reports_dir     = Path(reports_dir)
        self.figures_dir     = self.reports_dir / "figures"
        self.n_series        = n_series
        self.val_days        = val_days
        self.tune_parameters = tune_parameters
        self.random_seed     = random_seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _load_and_reshape(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads wide-format M5 sales CSV and calendar, then:
          - Melts to long format (id, date, demand, item_id, dept_id, cat_id, store_id, state_id)
          - Splits into train / val based on the last val_days days in the calendar.

        Returns:
            (train_df, val_df): Long-format DataFrames.
        """
        logger.info(f"Loading sales data from {self.sales_path}...")
        sales = pd.read_csv(self.sales_path)
        logger.info(f"  Loaded {len(sales):,} rows × {len(sales.columns)} cols.")

        logger.info(f"Loading calendar from {self.calendar_path}...")
        cal = pd.read_csv(self.calendar_path)

        # Build d-column → date mapping from calendar
        # Calendar has 'd' column like 'd_1', 'd_2', ... and 'date' column
        if "d" not in cal.columns and "date" in cal.columns:
            cal = cal.reset_index(drop=True)
            cal["d"] = ["d_" + str(i + 1) for i in range(len(cal))]

        d_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))

        # Identify d-columns in sales
        meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols    = [c for c in sales.columns if c.startswith("d_")]
        d_cols_sorted = sorted(d_cols, key=lambda x: int(x.split("_")[1]))

        # Optionally sub-sample SKUs
        if self.n_series > 0 and self.n_series < len(sales):
            rng = np.random.default_rng(self.random_seed)
            # Stratified sample across categories
            sampled_rows = []
            for cat in sales["cat_id"].unique():
                cat_rows = sales[sales["cat_id"] == cat]
                n_cat    = max(1, int(round(self.n_series * len(cat_rows) / len(sales))))
                idx      = rng.choice(len(cat_rows), size=min(n_cat, len(cat_rows)), replace=False)
                sampled_rows.append(cat_rows.iloc[idx])
            sales = pd.concat(sampled_rows, ignore_index=True)
            # Trim to exactly n_series
            if len(sales) > self.n_series:
                sales = sales.sample(n=self.n_series, random_state=self.random_seed).reset_index(drop=True)
            logger.info(f"  Sampled {len(sales)} SKUs for fitting.")

        # Melt to long format
        logger.info("Melting to long format...")
        long_df = sales.melt(
            id_vars=meta_cols,
            value_vars=d_cols_sorted,
            var_name="d",
            value_name="demand",
        )
        long_df["date"]   = long_df["d"].map(d_to_date)
        long_df["demand"] = pd.to_numeric(long_df["demand"], errors="coerce").fillna(0.0)
        long_df           = long_df.dropna(subset=["date"]).sort_values(["id", "date"]).reset_index(drop=True)

        # Train / val split on last val_days calendar dates
        all_dates  = sorted(long_df["date"].unique())
        val_dates  = set(all_dates[-self.val_days:])
        train_df   = long_df[~long_df["date"].isin(val_dates)].reset_index(drop=True)
        val_df     = long_df[long_df["date"].isin(val_dates)].reset_index(drop=True)

        logger.info(
            f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows "
            f"({len(val_df['id'].unique())} series × {self.val_days} days)"
        )
        return train_df, val_df

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Executes full per-SKU Kalman filter forecasting workflow.

        Returns:
            Dict with keys: 'metrics', 'predictions_df', 'models'
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("  M5 INVENTORY DEMAND FORECASTING - STRUCTURAL KALMAN FILTER")
        logger.info("=" * 70)

        # 1. Load & reshape
        train_df, val_df = self._load_and_reshape()

        unique_series = val_df["id"].unique()
        n_series      = len(unique_series)
        val_horizon   = val_df.groupby("id")["date"].count().iloc[0]
        val_min_date  = val_df["date"].min().strftime("%Y-%m-%d")
        val_max_date  = val_df["date"].max().strftime("%Y-%m-%d")

        logger.info(
            f"Dataset ready: {n_series} series | {len(train_df):,} train rows | "
            f"{len(val_df):,} val rows ({val_horizon} days: {val_min_date} → {val_max_date})"
        )

        # 2. Fit individual Kalman Filter per SKU + 28-day forecast
        logger.info(f"Fitting SKUKalmanForecaster across {n_series} series...")
        predictions_by_series: Dict[str, Dict] = {}
        fitted_models:         Dict[str, SKUKalmanForecaster] = {}

        for idx, sid in enumerate(unique_series, 1):
            s_train = train_df[train_df["id"] == sid].sort_values("date")
            s_val   = val_df[val_df["id"]   == sid].sort_values("date")

            if len(s_train) == 0:
                logger.warning(f"  No training data for {sid}, skipping.")
                continue

            forecaster = SKUKalmanForecaster(
                q_level_ratio  = 0.005,
                q_season_ratio = 0.001,
                r_ratio        = 0.75,
                tune_parameters= self.tune_parameters,
            )
            forecaster.fit(
                train_series = s_train["demand"].values,
                series_id    = sid,
                dates        = s_train["date"],
            )

            fc = forecaster.predict(horizon=len(s_val))
            fc["dates"]   = s_val["date"].values
            fc["actuals"] = s_val["demand"].values

            predictions_by_series[sid] = fc
            fitted_models[sid]         = forecaster

            if idx % 10 == 0 or idx == n_series:
                logger.info(f"  Processed {idx}/{n_series} | last: {sid}")

        # 3. Assemble results DataFrame
        logger.info("Assembling prediction records...")
        pred_records = []
        for sid in unique_series:
            if sid not in predictions_by_series:
                continue
            s_val = val_df[val_df["id"] == sid].sort_values("date")
            fc    = predictions_by_series[sid]

            for h in range(len(s_val)):
                row     = s_val.iloc[h]
                y_act   = float(row["demand"])
                y_pred  = float(fc["point"][h])
                y_std   = float(fc["std"][h])
                err     = y_pred - y_act
                abs_err = abs(err)

                pred_records.append({
                    "id":              sid,
                    "item_id":         row["item_id"],
                    "dept_id":         row["dept_id"],
                    "cat_id":          row["cat_id"],
                    "store_id":        row["store_id"],
                    "state_id":        row["state_id"],
                    "date":            row["date"].strftime("%Y-%m-%d"),
                    "demand":          y_act,
                    "pred_kalman":     y_pred,
                    "std_kalman":      y_std,
                    "lower_95":        float(fc["lower_95"][h]),
                    "upper_95":        float(fc["upper_95"][h]),
                    "lower_80":        float(fc["lower_80"][h]),
                    "upper_80":        float(fc["upper_80"][h]),
                    "error_kalman":    err,
                    "abs_error_kalman": abs_err,
                })

        results_df = pd.DataFrame(pred_records)
        output_csv = self.output_dir / "kalman_predictions.csv"
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions → {output_csv} ({len(results_df):,} rows)")

        # 4. Compute evaluation metrics
        y_true = results_df["demand"].values
        y_pred = results_df["pred_kalman"].values
        l_95   = results_df["lower_95"].values
        u_95   = results_df["upper_95"].values
        l_80   = results_df["lower_80"].values
        u_80   = results_df["upper_80"].values

        overall_metrics = compute_forecasting_metrics(y_true, y_pred)
        pi_95_metrics   = compute_prediction_interval_metrics(y_true, l_95, u_95)
        pi_80_metrics   = compute_prediction_interval_metrics(y_true, l_80, u_80)

        logger.info("-" * 70)
        logger.info(
            f"  OVERALL → MAE: {overall_metrics['mae']:.4f} | "
            f"RMSE: {overall_metrics['rmse']:.4f} | "
            f"WAPE: {overall_metrics['wape'] * 100:.2f}%"
        )
        logger.info(
            f"  95% PI → Coverage: {pi_95_metrics['coverage'] * 100:.2f}% "
            f"(Width: {pi_95_metrics['mean_width']:.4f}) | "
            f"80% PI → Coverage: {pi_80_metrics['coverage'] * 100:.2f}% "
            f"(Width: {pi_80_metrics['mean_width']:.4f})"
        )
        logger.info("-" * 70)

        # 5. Category and store breakdowns
        cat_breakdown   = {}
        for cat in results_df["cat_id"].unique():
            sub    = results_df[results_df["cat_id"] == cat]
            sub_m  = compute_forecasting_metrics(sub["demand"].values, sub["pred_kalman"].values)
            sub_pi = compute_prediction_interval_metrics(sub["demand"].values, sub["lower_95"].values, sub["upper_95"].values)
            sub_m["coverage_95"]  = sub_pi["coverage"]
            sub_m["mean_width_95"] = sub_pi["mean_width"]
            cat_breakdown[cat]    = sub_m

        store_breakdown = {}
        for store in results_df["store_id"].unique():
            sub    = results_df[results_df["store_id"] == store]
            sub_m  = compute_forecasting_metrics(sub["demand"].values, sub["pred_kalman"].values)
            sub_pi = compute_prediction_interval_metrics(sub["demand"].values, sub["lower_95"].values, sub["upper_95"].values)
            sub_m["coverage_95"]  = sub_pi["coverage"]
            sub_m["mean_width_95"] = sub_pi["mean_width"]
            store_breakdown[store] = sub_m

        # 6. Compile and save JSON metrics report
        metrics_payload = {
            "evaluation_timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_horizon_days":       int(val_horizon),
            "validation_date_range":         [val_min_date, val_max_date],
            "total_series":                  int(n_series),
            "total_validation_observations": int(len(results_df)),
            "state_space_model": {
                "architecture":                   "Local Level + Weekly Seasonality (Basic Structural Model)",
                "state_dimension":                7,
                "seasonality_period":             7,
                "observation_noise_ratio":        0.75,
                "level_process_noise_ratio":      0.005,
                "seasonal_process_noise_ratio":   0.001,
                "non_negative_clipping":          True,
                "missing_observation_strategy":   "predict_only_skip_update",
                "intermittent_zero_handling":     "preserved_as_valid_demand_observation",
            },
            "overall_metrics": {
                "kalman_filter": {
                    "mae":          overall_metrics["mae"],
                    "rmse":         overall_metrics["rmse"],
                    "wape":         overall_metrics["wape"],
                    "sum_actual":   overall_metrics["sum_actual"],
                    "sum_abs_err":  overall_metrics["sum_abs_err"],
                    "count":        overall_metrics["count"],
                    "prediction_intervals": {
                        "ci_95": pi_95_metrics,
                        "ci_80": pi_80_metrics,
                    },
                },
            },
            "category_breakdown": cat_breakdown,
            "store_breakdown":    store_breakdown,
        }

        output_json = self.reports_dir / "kalman_metrics.json"
        with open(output_json, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info(f"Saved metrics report → {output_json}")

        elapsed = time.time() - start_time
        logger.info(f"Kalman pipeline completed in {elapsed:.2f}s")

        return {
            "metrics":        metrics_payload,
            "predictions_df": results_df,
            "models":         fitted_models,
            "train_df":       train_df,
        }

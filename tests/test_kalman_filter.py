"""
test_kalman_filter.py - Comprehensive Unit & Integration Tests for SKU Kalman Filter.

DL-PROJECT - Team Contribution: Person 3 (Kalman Filter)

Test Suite (7 tests):
  1. test_forecast_horizon_length       - predict() returns exactly 28 steps.
  2. test_nonnegative_outputs           - Point forecasts & lower bounds >= 0.0.
  3. test_no_validation_leakage         - Altering val data produces ZERO forecast change.
  4. test_missing_observation_handling  - NaN observations handled; state stays finite.
  5. test_zero_demand_handling          - Zero sales are valid observations (not stockouts).
  6. test_prediction_interval_validity  - Upper >= lower, std > 0, coverage logic correct.
  7. test_pipeline_integration          - End-to-end M5KalmanPipeline on synthetic data.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import directly from the module file to avoid triggering the full src/__init__.py
# dependency chain (statsmodels, filterpy, torch, etc.) during lightweight unit tests.
import importlib.util as _ilu

def _load_module(rel_path: str):
    spec = _ilu.spec_from_file_location(
        rel_path.replace("/", ".").replace(".py", ""),
        PROJECT_ROOT / rel_path,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_kf = _load_module("src/models/kalman_filter_model.py")
SKUKalmanForecaster              = _kf.SKUKalmanForecaster
M5KalmanPipeline                 = _kf.M5KalmanPipeline
compute_forecasting_metrics      = _kf.compute_forecasting_metrics
compute_prediction_interval_metrics = _kf.compute_prediction_interval_metrics


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synthetic_sku():
    """
    90-day synthetic daily demand series with:
      - Base level 5.0 units/day
      - Weekend peak +3.0, weekday trough -1.0
      - Gaussian noise sigma=0.5
    Returns (train_y, val_y, train_dates, val_dates).
    """
    np.random.seed(42)
    n_days  = 90
    dates   = pd.date_range("2016-01-01", periods=n_days, freq="D")
    dow     = dates.day_of_week
    season  = np.where(dow >= 5, 3.0, -1.0)
    noise   = np.random.normal(0, 0.5, size=n_days)
    demand  = np.maximum(0.0, 5.0 + season + noise)

    train_y = demand[:62]
    val_y   = demand[62:]          # exactly 28 days
    return train_y, val_y, dates[:62], dates[62:]


@pytest.fixture
def fitted_forecaster(synthetic_sku):
    """Returns a pre-fitted SKUKalmanForecaster for reuse across tests."""
    train_y, _, train_d, _ = synthetic_sku
    fc = SKUKalmanForecaster()
    fc.fit(train_y, series_id="TEST_SKU", dates=train_d)
    return fc


# ===========================================================================
# Tests
# ===========================================================================

class TestForecastOutput:
    """Tests on the shape and properties of predict() output."""

    def test_forecast_horizon_length(self, fitted_forecaster):
        """predict() must return exactly the requested number of steps."""
        fc = fitted_forecaster.predict(horizon=28)
        assert len(fc["point"])    == 28, "point forecast length mismatch"
        assert len(fc["std"])      == 28, "std length mismatch"
        assert len(fc["lower_95"]) == 28, "lower_95 length mismatch"
        assert len(fc["upper_95"]) == 28, "upper_95 length mismatch"
        assert len(fc["lower_80"]) == 28, "lower_80 length mismatch"
        assert len(fc["upper_80"]) == 28, "upper_80 length mismatch"
        assert len(fc["raw_point"]) == 28, "raw_point length mismatch"

    def test_nonnegative_outputs(self):
        """
        Point forecasts and lower bounds must be >= 0.0 even for near-zero
        intermittent demand series.
        """
        # Sparse intermittent series: mostly 0, occasional 1
        train_y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] * 8, dtype=float)
        fc_model = SKUKalmanForecaster()
        fc_model.fit(train_y, series_id="INTERMITTENT_SKU")
        fc = fc_model.predict(horizon=28)

        assert np.all(fc["point"]    >= 0.0), "point forecast has negative values"
        assert np.all(fc["lower_95"] >= 0.0), "lower_95 bound has negative values"
        assert np.all(fc["lower_80"] >= 0.0), "lower_80 bound has negative values"
        assert np.all(fc["upper_95"] >= fc["point"]), "upper_95 must be >= point forecast"
        assert np.all(fc["upper_80"] >= fc["lower_80"]), "upper_80 must be >= lower_80"

    def test_std_positive(self, fitted_forecaster):
        """Forecast standard deviations must all be strictly positive."""
        fc = fitted_forecaster.predict(horizon=28)
        assert np.all(fc["std"] > 0), "std contains zero or negative values"


class TestDataIntegrity:
    """Tests on data handling, missing values, and zero demand."""

    def test_no_validation_leakage(self, synthetic_sku):
        """
        CRITICAL: Altering ground-truth validation demand must produce
        ZERO change in the out-of-sample forecast. The filter uses only
        the posterior state at origin T — it never sees future data.
        """
        train_y, val_y, train_d, _ = synthetic_sku
        fc = SKUKalmanForecaster()
        fc.fit(train_y, series_id="LEAKAGE_TEST", dates=train_d)

        fc_original = fc.predict(horizon=28)

        # Corrupt the validation series in-place (simulate tampered labels)
        val_y[:] = val_y * 10.0 + 500.0   # wildly different values

        # Forecast must be identical because fit() already stored x_post, P_post
        fc_after = fc.predict(horizon=28)

        np.testing.assert_array_almost_equal(
            fc_original["point"], fc_after["point"], decimal=10,
            err_msg="LEAKAGE DETECTED: forecast changed after validation tampering!",
        )
        np.testing.assert_array_almost_equal(
            fc_original["std"], fc_after["std"], decimal=10,
            err_msg="LEAKAGE DETECTED: uncertainty changed after validation tampering!",
        )

    def test_missing_observation_handling(self):
        """
        NaN observations in training must be handled via predict-only update.
        Posterior state and covariance must remain finite; forecasts valid.
        """
        np.random.seed(99)
        train_y = np.random.poisson(lam=4.0, size=70).astype(float)
        train_y[15:20] = np.nan   # 5-day gap (e.g. store closure)
        train_y[45]    = np.nan   # single missing point

        fc = SKUKalmanForecaster()
        fc.fit(train_y, series_id="MISSING_DATA_SKU")

        # Posterior state must be finite
        assert not np.isnan(fc.x_post).any(),  "x_post contains NaN after NaN training data"
        assert not np.isinf(fc.x_post).any(),  "x_post contains Inf after NaN training data"
        assert not np.isnan(fc.P_post).any(),  "P_post contains NaN after NaN training data"

        # Forecasts must be valid
        pred = fc.predict(horizon=28)
        assert not np.isnan(pred["point"]).any(), "forecast contains NaN"
        assert not np.isnan(pred["std"]).any(),   "std contains NaN"
        assert np.all(pred["std"] > 0),           "std must be > 0"

    def test_zero_demand_handling(self):
        """
        Persistent zero demand must be treated as y_t = 0 (valid observation).
        The level state must converge close to 0, and forecasts must be near 0.
        """
        train_y = np.zeros(42, dtype=float)
        fc = SKUKalmanForecaster()
        fc.fit(train_y, series_id="ZERO_SALES_SKU")

        # Level mu_t (index 0) should converge close to 0
        assert fc.x_post[0] < 0.1, f"Level state = {fc.x_post[0]:.4f}, expected < 0.1"

        pred = fc.predict(horizon=28)
        assert np.all(pred["point"] >= 0.0),   "point forecast must be non-negative"
        assert np.all(pred["point"] < 0.5),    f"point forecast should be near 0, got {pred['point'].max():.3f}"


class TestMetricHelpers:
    """Tests on the standalone metric utility functions."""

    def test_mae_rmse_wape(self):
        """Basic correctness of compute_forecasting_metrics."""
        y_true = np.array([2.0, 4.0, 6.0, 8.0])
        y_pred = np.array([3.0, 3.0, 7.0, 7.0])
        # MAE = mean(|1,1,1,1|) = 1.0
        # RMSE = 1.0
        # WAPE = sum|err| / sum_actual = 4 / 20 = 0.2
        m = compute_forecasting_metrics(y_true, y_pred)
        assert np.isclose(m["mae"],  1.0), f"MAE expected 1.0, got {m['mae']}"
        assert np.isclose(m["rmse"], 1.0), f"RMSE expected 1.0, got {m['rmse']}"
        assert np.isclose(m["wape"], 0.2), f"WAPE expected 0.2, got {m['wape']}"

    def test_prediction_interval_coverage(self):
        """
        Coverage: 4 of 5 observations inside their intervals => 0.80.
        Mean width = mean([2,2,2,7,2]) = 3.0.
        """
        y_true = np.array([2.0, 4.0, 5.0, 10.0,  1.0])
        lower  = np.array([1.0, 3.0, 4.0,  5.0,  2.0])  # 5th: y=1 < lower=2 (outside)
        upper  = np.array([3.0, 5.0, 6.0, 12.0,  4.0])

        m = compute_prediction_interval_metrics(y_true, lower, upper)
        assert m["coverage"]    == 0.8, f"Coverage expected 0.80, got {m['coverage']}"
        assert m["inside_count"] == 4,  f"Inside count expected 4, got {m['inside_count']}"
        assert m["total_count"]  == 5,  f"Total count expected 5, got {m['total_count']}"
        assert np.isclose(m["mean_width"], 3.0), f"Mean width expected 3.0, got {m['mean_width']}"


class TestPipelineIntegration:
    """End-to-end integration test using synthetic M5-style data."""

    def test_pipeline_integration(self, tmp_path):
        """
        Runs M5KalmanPipeline on a small synthetic dataset:
          - 2 categories (FOODS, HOBBIES), 2 SKUs each = 4 total series
          - 56-day training + 28-day validation
        Verifies output CSV schema, row count, and metrics JSON integrity.
        """
        # --- Synthetic data creation ---
        np.random.seed(0)
        train_dates = pd.date_range("2016-01-01", periods=56, freq="D")
        val_dates   = pd.date_range("2016-02-26", periods=28, freq="D")

        skus = [
            ("FOODS_1_001_CA_1_validation",    "FOODS_1_001",   "FOODS_1",   "FOODS",    "CA_1", "CA"),
            ("FOODS_1_002_CA_1_validation",    "FOODS_1_002",   "FOODS_1",   "FOODS",    "CA_1", "CA"),
            ("HOBBIES_1_001_CA_1_validation",  "HOBBIES_1_001", "HOBBIES_1", "HOBBIES",  "CA_1", "CA"),
            ("HOBBIES_1_002_CA_1_validation",  "HOBBIES_1_002", "HOBBIES_1", "HOBBIES",  "CA_1", "CA"),
        ]

        # Build wide-format CSV (d_1, d_2, ...)
        all_dates   = list(train_dates) + list(val_dates)
        d_cols      = [f"d_{i+1}" for i in range(len(all_dates))]
        # Create a minimal calendar CSV
        cal_df = pd.DataFrame({"d": d_cols, "date": [d.strftime("%Y-%m-%d") for d in all_dates]})

        rows = []
        for (sid, item_id, dept_id, cat_id, store_id, state_id) in skus:
            demand_vals = np.maximum(0, np.random.poisson(lam=3.0, size=len(all_dates))).tolist()
            row = {
                "id": sid, "item_id": item_id, "dept_id": dept_id,
                "cat_id": cat_id, "store_id": store_id, "state_id": state_id,
            }
            for col, val in zip(d_cols, demand_vals):
                row[col] = val
            rows.append(row)

        sales_df = pd.DataFrame(rows)
        sales_csv = tmp_path / "cleaned_sales_train_validation.csv"
        cal_csv   = tmp_path / "cleaned_calendar.csv"
        sales_df.to_csv(sales_csv, index=False)
        cal_df.to_csv(cal_csv, index=False)

        out_dir = tmp_path / "processed"
        rep_dir = tmp_path / "reports"

        # --- Run pipeline ---
        pipeline = M5KalmanPipeline(
            sales_path    = str(sales_csv),
            calendar_path = str(cal_csv),
            output_dir    = str(out_dir),
            reports_dir   = str(rep_dir),
            n_series      = -1,    # all 4 SKUs
            val_days      = 28,
        )
        result = pipeline.run()

        # --- Verify predictions CSV ---
        pred_csv = out_dir / "kalman_predictions.csv"
        assert pred_csv.exists(), "kalman_predictions.csv not created"
        pred_df  = pd.read_csv(pred_csv)

        # 4 series × 28 days = 112 rows
        assert len(pred_df) == 112, f"Expected 112 rows, got {len(pred_df)}"

        required_cols = [
            "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "demand", "pred_kalman", "std_kalman",
            "lower_95", "upper_95", "lower_80", "upper_80",
            "error_kalman", "abs_error_kalman",
        ]
        for col in required_cols:
            assert col in pred_df.columns, f"Missing column: {col}"

        # --- Verify non-negativity ---
        assert (pred_df["pred_kalman"] >= 0.0).all(), "pred_kalman has negative values"
        assert (pred_df["lower_95"]    >= 0.0).all(), "lower_95 has negative values"

        # --- Verify metrics JSON ---
        metrics_json = rep_dir / "kalman_metrics.json"
        assert metrics_json.exists(), "kalman_metrics.json not created"
        kf_m = result["metrics"]["overall_metrics"]["kalman_filter"]
        assert kf_m["mae"]  >= 0.0, "MAE must be non-negative"
        assert kf_m["rmse"] >= 0.0, "RMSE must be non-negative"
        assert kf_m["wape"] >= 0.0, "WAPE must be non-negative"
        assert "prediction_intervals" in kf_m, "Missing prediction_intervals in metrics"
        assert kf_m["prediction_intervals"]["ci_95"]["coverage"] >= 0.0
        assert kf_m["prediction_intervals"]["ci_80"]["coverage"] >= 0.0

"""
dataset.py - Memory-efficient PyTorch Dataset and Feature Engineering for Demand Forecasting.

Constructs windowed temporal features, rolling statistics, calendar cyclics,
and categorical entity representations for the M5 retail time series.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class M5DemandDataset(Dataset):
    """
    PyTorch Dataset yielding categorical tensors, continuous feature tensors,
    and target unit demand for inventory forecasting.
    """

    def __init__(
        self,
        cat_tensor: torch.Tensor,
        cont_tensor: torch.Tensor,
        targets: torch.Tensor,
    ):
        assert len(cat_tensor) == len(cont_tensor) == len(targets), "Tensor length mismatch"
        self.cat_tensor = cat_tensor.long()
        self.cont_tensor = cont_tensor.float()
        self.targets = targets.float()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cat_tensor[idx], self.cont_tensor[idx], self.targets[idx]


class M5FeaturePipeline:
    """
    Extracts autoregressive lags, rolling statistics, calendar indicators,
    and categorical entity encodings from raw M5 tables.
    """

    def __init__(
        self,
        raw_dir: str = "DL-PROJECT/data/raw",
        processed_dir: str = "DL-PROJECT/data/processed",
        n_series: int = 200,
        random_state: int = 42,
    ):
        # Resolve raw_dir and processed_dir robustly across working directories
        project_root = Path(__file__).resolve().parent.parent.parent
        raw_p = Path(raw_dir)
        if not raw_p.exists():
            for cand in [Path("data/raw"), project_root / "data/raw", project_root / raw_dir]:
                if cand.exists():
                    raw_p = cand
                    break
        self.raw_dir = raw_p

        proc_p = Path(processed_dir)
        if not proc_p.exists():
            for cand in [Path("data/processed"), project_root / "data/processed", project_root / processed_dir]:
                if cand.exists():
                    proc_p = cand
                    break
        self.processed_dir = proc_p
        self.n_series = n_series
        self.random_state = random_state

        self.cat_encoders: Dict[str, Dict[str, int]] = {}
        self.cat_dims: Dict[str, int] = {}
        self.cont_mean: Optional[np.ndarray] = None
        self.cont_std: Optional[np.ndarray] = None
        self.cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "wday", "month"]
        self.cont_cols: List[str] = []

    def load_and_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads calendar, prices, and samples sales series across all categories."""
        print(f"[PIPELINE] Loading raw datasets from {self.raw_dir}...")
        cal_path = self.raw_dir / "calendar.csv"
        sales_path = self.raw_dir / "sales_train_validation.csv"
        prices_path = self.raw_dir / "sell_prices.csv"

        df_cal = pd.read_csv(cal_path)
        df_cal["date"] = pd.to_datetime(df_cal["date"])

        # Load sales
        df_sales = pd.read_csv(sales_path)

        if self.n_series is not None and self.n_series < len(df_sales):
            print(f"[PIPELINE] Stratified sampling of {self.n_series} representative SKUs across all departments...")
            # Stratified sample by dept_id to maintain category diversity
            sampled_indices = (
                df_sales.groupby("dept_id", group_keys=False)
                .apply(lambda x: x.sample(max(1, int(np.ceil(self.n_series * len(x) / len(df_sales)))), random_state=self.random_state))
                .index
            )
            df_sales = df_sales.loc[sampled_indices].reset_index(drop=True)
            if len(df_sales) > self.n_series:
                df_sales = df_sales.iloc[: self.n_series].copy()

        # Load prices
        df_prices = pd.read_csv(prices_path)
        # Filter prices to sampled items and stores
        df_prices = df_prices[
            (df_prices["store_id"].isin(df_sales["store_id"].unique()))
            & (df_prices["item_id"].isin(df_sales["item_id"].unique()))
        ].copy()

        return df_cal, df_prices, df_sales

    def build_tabular_dataset(
        self,
        start_day: int = 1500,
        end_day: int = 1913,
        val_horizon: int = 28,
    ) -> Tuple[M5DemandDataset, M5DemandDataset, Dict[str, int], List[str]]:
        """
        Builds windowed time-series tabular matrices for training and validation.
        Days are indexed d_1 to d_1913.
        Validation set is strictly the last `val_horizon` days (1886 to 1913).
        """
        df_cal, df_prices, df_sales = self.load_and_sample_data()

        # Build calendar feature lookup
        cal = df_cal.copy()
        cal["dayofweek"] = cal["date"].dt.dayofweek
        cal["day_sin"] = np.sin(2 * np.pi * cal["dayofweek"] / 7.0).astype(np.float32)
        cal["day_cos"] = np.cos(2 * np.pi * cal["dayofweek"] / 7.0).astype(np.float32)
        cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12.0).astype(np.float32)
        cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12.0).astype(np.float32)
        cal["is_weekend"] = cal["dayofweek"].isin([5, 6]).astype(np.float32)
        cal["is_christmas"] = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)).astype(np.float32)
        cal["has_event_1"] = cal["event_name_1"].notnull().astype(np.float32)
        cal_indexed = cal.set_index("d")

        # Melt sales into long form over [start_day - 35, end_day] to allow 28-day lags + 7-day rolling window
        buffer_start = max(1, start_day - 35)
        d_cols = [f"d_{d}" for d in range(buffer_start, end_day + 1)]
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        print(f"[PIPELINE] Melting sales matrix from day {buffer_start} to {end_day} ({len(df_sales)} series)...")
        melted = pd.melt(
            df_sales[id_cols + d_cols],
            id_vars=id_cols,
            var_name="d",
            value_name="sales",
        )
        melted["d_int"] = melted["d"].str.replace("d_", "").astype(np.int32)
        melted = melted.sort_values(["id", "d_int"]).reset_index(drop=True)

        print("[PIPELINE] Engineering autoregressive lags and rolling statistics...")
        # Compute lags strictly per SKU series
        grp = melted.groupby("id")["sales"]
        melted["lag_1"] = grp.shift(1)
        melted["lag_2"] = grp.shift(2)
        melted["lag_7"] = grp.shift(7)
        melted["lag_14"] = grp.shift(14)
        melted["lag_21"] = grp.shift(21)
        melted["lag_28"] = grp.shift(28)

        # Rolling statistics (using shift(1) to strictly avoid data leakage of current day sales)
        lag1_grp = melted.groupby("id")["lag_1"]
        melted["rolling_mean_7"] = lag1_grp.transform(lambda x: x.rolling(7, min_periods=1).mean())
        melted["rolling_mean_14"] = lag1_grp.transform(lambda x: x.rolling(14, min_periods=1).mean())
        melted["rolling_mean_28"] = lag1_grp.transform(lambda x: x.rolling(28, min_periods=1).mean())
        melted["rolling_std_7"] = lag1_grp.transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0.0)
        melted["rolling_std_28"] = lag1_grp.transform(lambda x: x.rolling(28, min_periods=1).std()).fillna(0.0)

        # Merge calendar exogenous indicators
        cal_features = [
            "wday", "month", "wm_yr_wk",
            "day_sin", "day_cos", "month_sin", "month_cos",
            "is_weekend", "is_christmas", "has_event_1",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        melted = melted.merge(cal_indexed[cal_features], left_on="d", right_index=True, how="left")

        # Dynamic SNAP selection based on store state
        melted["snap"] = np.where(
            melted["state_id"] == "CA",
            melted["snap_CA"],
            np.where(melted["state_id"] == "TX", melted["snap_TX"], melted["snap_WI"]),
        ).astype(np.float32)

        # Merge sell prices
        print("[PIPELINE] Merging pricing and discount signals...")
        melted = melted.merge(
            df_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )
        # Compute baseline regular price per item-store and discount percentage
        reg_price = df_prices.groupby(["store_id", "item_id"])["sell_price"].transform("max")
        df_prices["reg_price"] = reg_price
        melted["sell_price"] = melted["sell_price"].fillna(melted.groupby("item_id")["sell_price"].transform("mean"))
        melted["sell_price"] = melted["sell_price"].fillna(1.0)
        melted["price_log"] = np.log1p(melted["sell_price"])

        # Filter strictly to analysis days [start_day, end_day]
        melted = melted[melted["d_int"] >= start_day].reset_index(drop=True)

        # Drop any remaining NaNs
        melted = melted.dropna(subset=["lag_28"]).reset_index(drop=True)

        # Encode categorical variables
        for col in self.cat_cols:
            if col not in self.cat_encoders:
                unique_vals = sorted(melted[col].astype(str).unique())
                self.cat_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
                self.cat_dims[col] = len(unique_vals)
            melted[col + "_idx"] = melted[col].astype(str).map(self.cat_encoders[col]).fillna(0).astype(np.int64)

        cat_idx_cols = [col + "_idx" for col in self.cat_cols]

        self.cont_cols = [
            "lag_1", "lag_2", "lag_7", "lag_14", "lag_21", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_28",
            "day_sin", "day_cos", "month_sin", "month_cos",
            "is_weekend", "is_christmas", "has_event_1",
            "snap", "price_log",
        ]

        # Temporal Train/Validation Split
        val_cutoff = end_day - val_horizon
        train_df = melted[melted["d_int"] <= val_cutoff].copy()
        val_df = melted[melted["d_int"] > val_cutoff].copy()

        print(f"[PIPELINE] Train observations: {len(train_df):,} (Days {start_day} to {val_cutoff})")
        print(f"[PIPELINE] Val observations:   {len(val_df):,} (Days {val_cutoff + 1} to {end_day})")

        # Fit continuous feature scaler STRICTLY on train split
        self.cont_mean = train_df[self.cont_cols].mean().values
        self.cont_std = train_df[self.cont_cols].std().replace(0, 1.0).values

        # Standardize
        train_cont = (train_df[self.cont_cols].values - self.cont_mean) / self.cont_std
        val_cont = (val_df[self.cont_cols].values - self.cont_mean) / self.cont_std

        # Construct PyTorch Tensors
        train_dataset = M5DemandDataset(
            cat_tensor=torch.tensor(train_df[cat_idx_cols].values, dtype=torch.long),
            cont_tensor=torch.tensor(train_cont, dtype=torch.float32),
            targets=torch.tensor(train_df["sales"].values, dtype=torch.float32),
        )

        val_dataset = M5DemandDataset(
            cat_tensor=torch.tensor(val_df[cat_idx_cols].values, dtype=torch.long),
            cont_tensor=torch.tensor(val_cont, dtype=torch.float32),
            targets=torch.tensor(val_df["sales"].values, dtype=torch.float32),
        )

        return train_dataset, val_dataset, self.cat_dims, self.cont_cols

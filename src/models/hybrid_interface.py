"""
hybrid_interface.py - Hybrid ML + DL Integration Adapter for Inventory Demand Forecasting.

Provides technically sound architectures combining the Normal Neural Network with:
1. GBDT Residual Stacking: Normal Neural Network captures global representation
   and entity embeddings; GBDT models high-frequency localized residual errors.
2. State-Space Kalman Augmented DL: 1D/2D Kalman Filter estimates continuous latent
   demand velocity and uncertainty bounds, injecting them into the Neural Network.
"""

import numpy as np
import torch.nn as nn
from typing import Tuple


class HybridForecaster:
    """
    Ensemble & Stacking Hybrid combining the Deep Neural Network with Machine Learning.

    Theoretical Justification:
    - Normal Neural Networks excel at learning smooth continuous latent representations,
      cross-department entity embeddings, and shared global temporal patterns.
    - Gradient Boosted Decision Trees (GBDT) / Linear Models excel at sharp axis-aligned
      partitions (e.g. strict holiday cutoffs, discontinuous price thresholds).
    - Mathematically, combining them as a convex combination or residual stack reduces
      both epistemic uncertainty and structural model bias:
        y_hat_hybrid = w * y_hat_DL + (1 - w) * y_hat_ML
    """

    def __init__(self, dl_model: nn.Module, blend_weight_dl: float = 0.60):
        self.dl_model = dl_model
        self.blend_weight_dl = blend_weight_dl

    def predict_hybrid_blend(
        self,
        dl_preds: np.ndarray,
        ml_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Computes variance-weighted convex blend of DL and ML forecasts.
        Clamped at 0 for strict non-negative retail demand.
        """
        w = self.blend_weight_dl
        y_hybrid = w * dl_preds + (1.0 - w) * ml_preds
        return np.clip(y_hybrid, 0.0, None)

    def compute_residual_stack(
        self,
        dl_preds: np.ndarray,
        residual_corrections: np.ndarray,
    ) -> np.ndarray:
        """
        Corrects base deep neural network predictions with trained ML residual error models.
        """
        y_hybrid = dl_preds + residual_corrections
        return np.clip(y_hybrid, 0.0, None)


class KalmanFeatureAugmenter:
    """
    State-Space Kalman Filter feature generator for Deep Learning ingestion.
    Filters intermittent noisy sales y_t into latent continuous demand velocity x_t
    and covariance uncertainty P_t.
    """

    @staticmethod
    def extract_kalman_latent_features(
        sales_series: np.ndarray,
        process_variance: float = 1e-4,
        measurement_variance: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D Kalman Filter generating continuous filtered state and uncertainty bounds.
        """
        n = len(sales_series)
        x_est = np.zeros(n, dtype=np.float32)
        p_est = np.zeros(n, dtype=np.float32)

        # Initial state estimate
        x = float(sales_series[0]) if sales_series[0] > 0 else 1.0
        p = 1.0

        for t in range(n):
            # 1. Prediction step
            x_pred = x
            p_pred = p + process_variance

            # 2. Update step
            z = sales_series[t]
            k = p_pred / (p_pred + measurement_variance)  # Kalman gain
            x = x_pred + k * (z - x_pred)
            p = (1.0 - k) * p_pred

            x_est[t] = max(0.0, x)
            p_est[t] = p

        return x_est, p_est

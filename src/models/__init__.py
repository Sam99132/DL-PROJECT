"""
src/models - Deep Learning models, datasets, trainers, ablations, and Kalman filter
             for Inventory Demand Forecasting on the Walmart M5 dataset.
"""

from src.models.feedforward_nn import (
    NormalDemandMLP,
    ResidualDemandMLP,
    ResidualDemandBlock,
    TweedieDevianceLoss,
)
from src.models.dataset import (
    M5DemandDataset,
    M5FeaturePipeline,
)
from src.models.trainer import (
    DemandTrainer,
    calculate_metrics,
)
from src.models.ablation import M5AblationRunner
from src.models.hybrid_interface import HybridForecaster, KalmanFeatureAugmenter
from src.models.kalman_filter_model import (
    SKUKalmanForecaster,
    M5KalmanPipeline,
    compute_forecasting_metrics,
    compute_prediction_interval_metrics,
)

__all__ = [
    # Neural Network models
    "NormalDemandMLP",
    "ResidualDemandMLP",
    "ResidualDemandBlock",
    "TweedieDevianceLoss",
    # Dataset & training utilities
    "M5DemandDataset",
    "M5FeaturePipeline",
    "DemandTrainer",
    "calculate_metrics",
    "M5AblationRunner",
    # Hybrid & interface
    "HybridForecaster",
    "KalmanFeatureAugmenter",
    # Kalman Filter (Person 3 contribution)
    "SKUKalmanForecaster",
    "M5KalmanPipeline",
    "compute_forecasting_metrics",
    "compute_prediction_interval_metrics",
]

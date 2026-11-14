"""
src/models - Deep Learning models, datasets, trainers, and ablations for Inventory Demand Forecasting.
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

__all__ = [
    "NormalDemandMLP",
    "ResidualDemandMLP",
    "ResidualDemandBlock",
    "TweedieDevianceLoss",
    "M5DemandDataset",
    "M5FeaturePipeline",
    "DemandTrainer",
    "calculate_metrics",
    "M5AblationRunner",
    "HybridForecaster",
    "KalmanFeatureAugmenter",
]

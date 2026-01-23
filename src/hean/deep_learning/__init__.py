"""
Deep Learning module for HEAN trading system.

Provides:
- Temporal Fusion Transformer (TFT)
- N-BEATS forecaster
- Multi-horizon price forecasting
- Training pipelines
"""

from hean.deep_learning.deep_forecaster import (
    DeepForecaster,
    TFTConfig,
    ForecastResult,
)

__all__ = [
    "DeepForecaster",
    "TFTConfig",
    "ForecastResult",
]

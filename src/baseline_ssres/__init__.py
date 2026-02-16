"""Baseline self-supervised denoising strategies: Noise2Noise, Noise2Void, Noise2Self.

Standalone package -- works independently or as an ssres plugin.
Install ssres to enable automatic plugin discovery via entry points.
"""

from .base import BaseBlindSpotStrategy, BaseSelfSupervisedStrategy
from .config import (
    DataDimensionality,
    MaskParams,
    N2NParams,
    N2SParams,
    NormalizationParams,
    PartitionScheme,
    StrategyOutput,
)
from .noise2noise import Noise2NoiseStrategy, Noise2NoiseValidationStrategy
from .noise2self import Noise2SelfStrategy, Noise2SelfValidationStrategy
from .noise2void import Noise2VoidStrategy, Noise2VoidValidationStrategy

__all__ = [
    # Config
    "DataDimensionality",
    "MaskParams",
    "N2NParams",
    "N2SParams",
    "NormalizationParams",
    "PartitionScheme",
    "StrategyOutput",
    # Base classes
    "BaseBlindSpotStrategy",
    "BaseSelfSupervisedStrategy",
    # Noise2Noise
    "Noise2NoiseStrategy",
    "Noise2NoiseValidationStrategy",
    # Noise2Void
    "Noise2VoidStrategy",
    "Noise2VoidValidationStrategy",
    # Noise2Self
    "Noise2SelfStrategy",
    "Noise2SelfValidationStrategy",
    # Plugin
    "register",
]


def register():
    """Register baseline strategies as an ssres plugin.

    Called automatically by ssres plugin discovery via entry points.
    Returns None if ssres is not installed (standalone usage).
    """
    try:
        from core.plugin_discovery import PluginContribution, PluginManifest
    except ImportError:
        return None

    return PluginManifest(
        name="baseline",
        version="0.1.0",
        description="Baseline self-supervised strategies (N2N, N2V, N2S)",
        contributions=[
            PluginContribution(
                category="strategy",
                name="noise2noise",
                factory=lambda: Noise2NoiseStrategy,
            ),
            PluginContribution(
                category="strategy",
                name="noise2void",
                factory=lambda: Noise2VoidStrategy,
            ),
            PluginContribution(
                category="strategy",
                name="noise2self",
                factory=lambda: Noise2SelfStrategy,
            ),
        ],
    )

"""Noise2Void (N2V) strategy for self-supervised denoising.

Reference:
    - Krull et al. "Noise2Void" CVPR 2019
    - Hock et al. "N2V2" ECCV 2022
"""

import numpy as np
from torch import Tensor

from .base import BaseBlindSpotStrategy
from .config import DataDimensionality, MaskParams, NormalizationParams, StrategyOutput


class Noise2VoidStrategy(BaseBlindSpotStrategy):
    """Noise2Void blind-spot strategy.

    Masks random pixels and trains the network to predict them from
    surrounding context.
    """

    def __init__(
        self,
        mask_params: MaskParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        is_training: bool = True,
        dimensionality: DataDimensionality | str = DataDimensionality.AUTO,
    ):
        super().__init__(mask_params, normalization, seed, is_training, dimensionality)

    @property
    def name(self) -> str:
        return "noise2void"

    def sample(self, patch: Tensor | np.ndarray, **kwargs) -> StrategyOutput:
        """Generate training data with blind-spot masking."""
        patch = self._to_tensor(patch)
        ground_truth = patch.clone()

        spatial_shape = self._get_spatial_shape(patch)
        mask = self._generate_mask(spatial_shape)
        masked_input = self._replace_masked_pixels(patch, mask)
        output_mask = self._expand_mask_to_patch(mask, patch)

        return StrategyOutput(
            input=masked_input,
            target=ground_truth.clone(),
            ground_truth=ground_truth,
            mask=output_mask,
        )

    def create_validation_variant(self) -> "Noise2VoidStrategy":
        """Create validation variant without masking."""
        return Noise2VoidValidationStrategy(
            mask_params=self._mask_params,
            normalization=self._normalization,
            seed=self._seed,
            dimensionality=self._dimensionality,
        )


class Noise2VoidValidationStrategy(Noise2VoidStrategy):
    """Validation variant of Noise2VoidStrategy.

    Uses no masking for clean evaluation on all pixels.
    """

    def __init__(self, **kwargs):
        kwargs["is_training"] = False
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "noise2void_validation"

    @property
    def requires_mask(self) -> bool:
        """Validation variant doesn't require mask."""
        return False

    def sample(self, patch: Tensor | np.ndarray, **kwargs) -> StrategyOutput:
        """Return full prediction (no masking) for validation."""
        patch = self._to_tensor(patch)

        return StrategyOutput(
            input=patch.clone(),
            target=patch.clone(),
            ground_truth=patch.clone(),
            mask=None,
        )

    def create_validation_variant(self) -> "Noise2VoidValidationStrategy":
        """Return self (already validation variant)."""
        return self


__all__ = [
    "Noise2VoidStrategy",
    "Noise2VoidValidationStrategy",
]

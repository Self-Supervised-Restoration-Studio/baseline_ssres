"""Noise2Noise strategy for temporal self-supervised denoising.

Reference:
    Lehtinen et al., "Noise2Noise: Learning Image Restoration without
    Clean Data", ICML 2018.
"""

from typing import Any

import numpy as np
from torch import Tensor

from .base import BaseSelfSupervisedStrategy
from .config import DataDimensionality, N2NParams, NormalizationParams, StrategyOutput


class Noise2NoiseStrategy(BaseSelfSupervisedStrategy):
    """Noise2Noise strategy using temporal frame pairs.

    Uses consecutive (or offset) frames as input-target pairs, assuming
    independent noise between frames.

    Input data should have temporal dimension as first axis:
    - (T, C, H, W) for 2D+time
    - (T, C, D, H, W) for 3D+time
    """

    def __init__(
        self,
        n2n_params: N2NParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        is_training: bool = True,
        dimensionality: DataDimensionality | str = DataDimensionality.XYT,
    ):
        super().__init__(normalization, seed, is_training, dimensionality)
        self._n2n_params = n2n_params or N2NParams()

    @property
    def name(self) -> str:
        return "noise2noise"

    @property
    def requires_temporal(self) -> bool:
        return True

    def sample(self, patch: Tensor | np.ndarray, **kwargs) -> StrategyOutput:
        """Generate input-target pair from temporal patch."""
        patch = self._to_tensor(patch)

        n_frames = patch.shape[0]
        offset = self._n2n_params.frame_offset

        min_frames = self._n2n_params.require_min_frames
        if n_frames < min_frames:
            raise ValueError(
                f"Noise2Noise requires at least {min_frames} temporal frames, "
                f"got {n_frames}. Input shape: {patch.shape}"
            )

        if self._is_training:
            input_frame, target_frame = self._sample_training_frames(patch, offset)
        else:
            input_frame, target_frame = self._sample_validation_frames(patch, offset)

        target_frame = self._normalize_target(target_frame, input_frame)

        return StrategyOutput(
            input=input_frame,
            target=target_frame,
            ground_truth=target_frame.clone(),
            mask=None,
        )

    def _sample_training_frames(
        self, patch: Tensor, offset: int
    ) -> tuple[Tensor, Tensor]:
        """Sample frames for training with randomization."""
        n_frames = patch.shape[0]
        max_input_idx = n_frames - 1
        input_idx = int(self._rng.integers(0, max_input_idx + 1))

        if self._n2n_params.bidirectional:
            can_forward = (input_idx + offset) < n_frames
            can_backward = (input_idx - offset) >= 0

            if can_forward and can_backward:
                direction = int(self._rng.integers(0, 2)) * 2 - 1
            elif can_forward:
                direction = 1
            elif can_backward:
                direction = -1
            else:
                direction = 0
        else:
            if (input_idx + offset) < n_frames:
                direction = 1
            else:
                direction = -1 if (input_idx - offset) >= 0 else 0

        target_idx = input_idx + direction * offset
        target_idx = max(0, min(n_frames - 1, target_idx))

        return patch[input_idx], patch[target_idx]

    def _sample_validation_frames(
        self, patch: Tensor, offset: int
    ) -> tuple[Tensor, Tensor]:
        """Sample frames for validation (deterministic)."""
        n_frames = patch.shape[0]
        input_idx = 0
        target_idx = min(offset, n_frames - 1)
        return patch[input_idx], patch[target_idx]

    def create_validation_variant(self) -> "Noise2NoiseStrategy":
        """Create deterministic validation version."""
        return Noise2NoiseStrategy(
            n2n_params=self._n2n_params,
            normalization=self._normalization,
            seed=self._seed,
            is_training=False,
            dimensionality=self._dimensionality,
        )

    def get_state(self) -> dict[str, Any]:
        """Get strategy state for checkpointing."""
        state = super().get_state()
        state["n2n_params"] = self._n2n_params.to_dict()
        return state


class Noise2NoiseValidationStrategy(Noise2NoiseStrategy):
    """Validation variant of Noise2Noise strategy."""

    def __init__(
        self,
        n2n_params: N2NParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        dimensionality: DataDimensionality | str = DataDimensionality.XYT,
    ):
        super().__init__(
            n2n_params=n2n_params,
            normalization=normalization,
            seed=seed,
            is_training=False,
            dimensionality=dimensionality,
        )

    @property
    def name(self) -> str:
        return "noise2noise_validation"

    def create_validation_variant(self) -> "Noise2NoiseValidationStrategy":
        """Return self (already validation variant)."""
        return self


__all__ = [
    "Noise2NoiseStrategy",
    "Noise2NoiseValidationStrategy",
]

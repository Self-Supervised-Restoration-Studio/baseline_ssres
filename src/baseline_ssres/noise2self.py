"""Noise2Self strategy for self-supervised denoising.

Reference:
    Batson & Royer, "Noise2Self: Blind Denoising by Self-Supervision",
    ICML 2019.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .base import BaseSelfSupervisedStrategy
from .config import (
    DataDimensionality,
    N2SParams,
    NormalizationParams,
    PartitionScheme,
    StrategyOutput,
)


class Noise2SelfStrategy(BaseSelfSupervisedStrategy):
    """Noise2Self strategy using structured J-invariant masking."""

    def __init__(
        self,
        n2s_params: N2SParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        is_training: bool = True,
        dimensionality: DataDimensionality | str = DataDimensionality.AUTO,
    ):
        super().__init__(normalization, seed, is_training, dimensionality)
        self._n2s_params = n2s_params or N2SParams()

    @property
    def name(self) -> str:
        return "noise2self"

    @property
    def requires_mask(self) -> bool:
        return True

    def sample(self, patch: Tensor | np.ndarray, **kwargs) -> StrategyOutput:
        """Generate masked training data using J-invariant partitioning."""
        patch = self._to_tensor(patch)

        spatial_shape = self._get_spatial_shape(patch)
        mask = self._generate_partition_mask(spatial_shape)

        input_patch = patch.clone()
        target = patch.clone()
        expanded_mask = self._expand_mask_to_patch(mask, patch)

        return StrategyOutput(
            input=input_patch,
            target=target,
            ground_truth=patch.clone(),
            mask=expanded_mask,
        )

    def _generate_partition_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate partition mask based on selected scheme."""
        scheme = self._n2s_params.partition_scheme

        if scheme == PartitionScheme.CHECKERBOARD:
            return self._generate_checkerboard_mask(shape)
        elif scheme == PartitionScheme.DONUT:
            return self._generate_donut_mask(shape)
        elif scheme == PartitionScheme.GRID:
            return self._generate_grid_mask(shape)
        elif scheme == PartitionScheme.RANDOM:
            return self._generate_random_mask(shape)
        else:
            raise ValueError(f"Unknown partition scheme: {scheme}")

    def _generate_checkerboard_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate checkerboard partition mask."""
        phase = self._n2s_params.partition_phase

        coords = [np.arange(s) for s in shape]
        grids = np.meshgrid(*coords, indexing="ij")

        coord_sum = sum(grids)
        mask = ((coord_sum % 2) == phase).astype(np.float32)

        return torch.from_numpy(mask)

    def _generate_donut_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate donut-style mask that excludes local neighborhoods."""
        radius = self._n2s_params.donut_radius
        spacing = 2 * radius + 1

        ndim = len(shape)
        mask = np.zeros(shape, dtype=np.float32)

        if ndim == 2:
            h, w = shape
            for y in range(radius, h, spacing):
                for x in range(radius, w, spacing):
                    mask[y, x] = 1.0
        elif ndim == 3:
            d, h, w = shape
            for z in range(radius, d, spacing):
                for y in range(radius, h, spacing):
                    for x in range(radius, w, spacing):
                        mask[z, y, x] = 1.0
        elif ndim == 4:
            t, d, h, w = shape
            for tt in range(radius, t, spacing):
                for z in range(radius, d, spacing):
                    for y in range(radius, h, spacing):
                        for x in range(radius, w, spacing):
                            mask[tt, z, y, x] = 1.0

        # Add jitter for training
        if self._is_training:
            offset = tuple(int(self._rng.integers(0, spacing)) for _ in range(ndim))
            mask = np.roll(mask, offset, axis=tuple(range(ndim)))

        return torch.from_numpy(mask)

    def _generate_grid_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate regular grid partition mask."""
        spacing = self._n2s_params.grid_spacing
        phase = self._n2s_params.partition_phase

        coords = [np.arange(s) for s in shape]
        grids = np.meshgrid(*coords, indexing="ij")

        in_grid = np.ones(shape, dtype=bool)
        for grid in grids:
            in_grid &= (grid + phase) % spacing == 0

        mask = in_grid.astype(np.float32)

        return torch.from_numpy(mask)

    def _generate_random_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate random mask (less structured than other schemes)."""
        n_pixels = int(np.prod(shape))
        n_masked = n_pixels // 2  # 50% for random partition

        mask_flat = np.zeros(n_pixels, dtype=np.float32)
        indices = self._rng.choice(n_pixels, size=n_masked, replace=False)
        mask_flat[indices] = 1.0

        return torch.from_numpy(mask_flat.reshape(shape))

    def create_validation_variant(self) -> "Noise2SelfStrategy":
        """Create deterministic validation version."""
        val_params = N2SParams(
            partition_scheme=self._n2s_params.partition_scheme,
            partition_phase=0,
            donut_radius=self._n2s_params.donut_radius,
            grid_spacing=self._n2s_params.grid_spacing,
        )

        return Noise2SelfStrategy(
            n2s_params=val_params,
            normalization=self._normalization,
            seed=self._seed,
            is_training=False,
            dimensionality=self._dimensionality,
        )

    def get_state(self) -> dict[str, Any]:
        """Get strategy state for checkpointing."""
        state = super().get_state()
        state["n2s_params"] = self._n2s_params.to_dict()
        return state


class Noise2SelfValidationStrategy(Noise2SelfStrategy):
    """Validation variant of Noise2Self strategy."""

    def __init__(
        self,
        n2s_params: N2SParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        dimensionality: DataDimensionality | str = DataDimensionality.AUTO,
    ):
        super().__init__(
            n2s_params=n2s_params,
            normalization=normalization,
            seed=seed,
            is_training=False,
            dimensionality=dimensionality,
        )

    @property
    def name(self) -> str:
        return "noise2self_validation"

    def create_validation_variant(self) -> "Noise2SelfValidationStrategy":
        """Return self (already validation variant)."""
        return self


__all__ = [
    "Noise2SelfStrategy",
    "Noise2SelfValidationStrategy",
]

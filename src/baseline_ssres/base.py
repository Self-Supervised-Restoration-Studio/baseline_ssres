"""Base classes for self-supervised denoising strategies.

Consolidated from n2n3s, n2v3s, and n2s3s packages.
Provides BaseSelfSupervisedStrategy and BaseBlindSpotStrategy.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import DataDimensionality, MaskParams, NormalizationParams, StrategyOutput


class BaseSelfSupervisedStrategy(ABC):
    """Abstract base class for self-supervised denoising strategies."""

    def __init__(
        self,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        is_training: bool = True,
        dimensionality: DataDimensionality | str = DataDimensionality.AUTO,
    ):
        self._normalization = normalization or NormalizationParams()
        self._seed = seed
        self._is_training = is_training
        self._rng = np.random.default_rng(seed)
        if isinstance(dimensionality, str):
            dimensionality = DataDimensionality(dimensionality)
        self._dimensionality = dimensionality

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...

    @property
    def requires_mask(self) -> bool:
        """Whether strategy generates a loss mask."""
        return False

    @property
    def requires_temporal(self) -> bool:
        """Whether strategy requires temporal dimension."""
        return False

    @property
    def is_training(self) -> bool:
        """Whether this is a training variant."""
        return self._is_training

    @property
    def dimensionality(self) -> DataDimensionality:
        """Data dimensionality mode."""
        return self._dimensionality

    @abstractmethod
    def sample(self, patch: Tensor | np.ndarray, **kwargs) -> StrategyOutput:
        """Generate training data from input patch."""
        ...

    @abstractmethod
    def create_validation_variant(self) -> "BaseSelfSupervisedStrategy":
        """Create a deterministic validation version."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get strategy state for checkpointing."""
        return {
            "name": self.name,
            "seed": self._seed,
            "is_training": self._is_training,
            "dimensionality": self._dimensionality.value,
            "rng_state": self._rng.bit_generator.state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore strategy state from checkpoint."""
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        if "dimensionality" in state:
            self._dimensionality = DataDimensionality(state["dimensionality"])

    def _to_tensor(self, data: Tensor | np.ndarray) -> Tensor:
        """Convert numpy array to tensor if needed."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data

    def _normalize_target(self, target: Tensor, reference: Tensor | None = None) -> Tensor:
        """Normalize target tensor to match reference statistics."""
        if not self._normalization.normalize:
            return target

        eps = self._normalization.epsilon

        if reference is None:
            return target

        if self._normalization.method == "mean":
            ref_mean = reference.mean()
            target_mean = target.mean()
            if target_mean.abs() < eps:
                return target
            return target * (ref_mean / (target_mean + eps))

        elif self._normalization.method == "sum":
            ref_sum = reference.sum()
            target_sum = target.sum()
            if target_sum.abs() < eps:
                return target
            return target * (ref_sum / (target_sum + eps))

        else:  # "none"
            return target

    def _infer_spatial_dims(self, patch: Tensor) -> int:
        """Infer number of spatial dimensions from patch."""
        if self._dimensionality == DataDimensionality.XY:
            return 2
        elif self._dimensionality == DataDimensionality.XYZ:
            return 3
        elif self._dimensionality == DataDimensionality.XYT:
            return 3
        elif self._dimensionality == DataDimensionality.XYZT:
            return 4
        else:  # AUTO
            if patch.dim() == 3:
                return 2
            elif patch.dim() == 4:
                return 3
            elif patch.dim() == 5:
                return 4
            else:
                return patch.dim() - 1

    def _get_spatial_shape(self, patch: Tensor) -> tuple[int, ...]:
        """Get spatial shape from patch."""
        num_spatial = self._infer_spatial_dims(patch)
        return patch.shape[-num_spatial:]

    def _expand_mask_to_patch(self, mask: Tensor, patch: Tensor) -> Tensor:
        """Expand spatial mask to match patch dimensions."""
        num_spatial = self._infer_spatial_dims(patch)
        num_channel_dims = patch.dim() - num_spatial

        expanded = mask
        for _ in range(num_channel_dims):
            expanded = expanded.unsqueeze(0)

        return expanded.expand_as(patch)


class BaseBlindSpotStrategy(BaseSelfSupervisedStrategy):
    """Base class for blind-spot strategies (Noise2Void)."""

    def __init__(
        self,
        mask_params: MaskParams | None = None,
        normalization: NormalizationParams | None = None,
        seed: int | None = None,
        is_training: bool = True,
        dimensionality: DataDimensionality | str = DataDimensionality.AUTO,
    ):
        super().__init__(normalization, seed, is_training, dimensionality)
        self._mask_params = mask_params or MaskParams()

    @property
    def requires_mask(self) -> bool:
        """Blind-spot strategies always generate masks."""
        return True

    def _generate_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate blind-spot mask."""
        if self._mask_params.stratified:
            return self._generate_stratified_mask(shape)
        else:
            return self._generate_random_mask(shape)

    def _generate_random_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate mask using simple random sampling."""
        n_pixels = int(np.prod(shape))
        n_masked = max(1, int(n_pixels * self._mask_params.mask_ratio))

        mask_flat = np.zeros(n_pixels, dtype=np.float32)
        indices = self._rng.choice(n_pixels, size=n_masked, replace=False)
        mask_flat[indices] = 1.0

        return torch.from_numpy(mask_flat.reshape(shape))

    def _generate_stratified_mask(self, shape: tuple[int, ...]) -> Tensor:
        """Generate mask using stratified sampling (N2V/CAREamics style)."""
        ndim = len(shape)

        mask_percent = self._mask_params.mask_ratio * 100
        if mask_percent <= 0:
            mask_percent = 0.2

        pixel_distance = (100.0 / mask_percent) ** (1.0 / ndim)

        coords_per_dim = []
        for dim_size in shape:
            n_points = max(1, int(dim_size / pixel_distance))
            if n_points == 1:
                base = np.array([dim_size / 2])
            else:
                base = np.linspace(
                    pixel_distance / 2, dim_size - pixel_distance / 2, n_points
                )

            jitter = self._rng.uniform(
                -pixel_distance / 2, pixel_distance / 2, size=len(base)
            )
            coords = np.clip(base + jitter, 0, dim_size - 1).astype(np.int64)
            coords_per_dim.append(coords)

        if ndim == 2:
            yy, xx = np.meshgrid(coords_per_dim[0], coords_per_dim[1], indexing="ij")
            flat_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
        elif ndim == 3:
            dd, yy, xx = np.meshgrid(
                coords_per_dim[0], coords_per_dim[1], coords_per_dim[2], indexing="ij"
            )
            flat_coords = np.stack([dd.ravel(), yy.ravel(), xx.ravel()], axis=1)
        elif ndim == 4:
            tt, dd, yy, xx = np.meshgrid(
                coords_per_dim[0],
                coords_per_dim[1],
                coords_per_dim[2],
                coords_per_dim[3],
                indexing="ij",
            )
            flat_coords = np.stack(
                [tt.ravel(), dd.ravel(), yy.ravel(), xx.ravel()], axis=1
            )
        else:
            return self._generate_random_mask(shape)

        mask = np.zeros(shape, dtype=np.float32)
        for coord in flat_coords:
            mask[tuple(coord)] = 1.0

        return torch.from_numpy(mask)

    def _get_roi_bounds(
        self, center: tuple[int, ...], shape: tuple[int, ...]
    ) -> tuple[list[int], list[int]]:
        """Get ROI bounds around a center pixel."""
        radius = self._mask_params.roi_size // 2
        starts = []
        ends = []
        for c, s in zip(center, shape):
            starts.append(max(0, c - radius))
            ends.append(min(s, c + radius + 1))
        return starts, ends

    def _extract_roi(
        self, patch: Tensor, center: tuple[int, ...], exclude_center: bool = True
    ) -> Tensor:
        """Extract ROI values around a center pixel."""
        num_spatial = len(center)
        spatial_shape = patch.shape[-num_spatial:]

        starts, ends = self._get_roi_bounds(center, spatial_shape)

        if num_spatial == 2:
            roi = patch[..., starts[0] : ends[0], starts[1] : ends[1]]
        elif num_spatial == 3:
            roi = patch[
                ..., starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]
            ]
        elif num_spatial == 4:
            roi = patch[
                ...,
                starts[0] : ends[0],
                starts[1] : ends[1],
                starts[2] : ends[2],
                starts[3] : ends[3],
            ]
        else:
            raise ValueError(f"Unsupported spatial dims: {num_spatial}")

        roi_flat = (
            roi.reshape(-1)
            if patch.dim() == num_spatial
            else roi.reshape(patch.shape[0], -1)
        )

        if exclude_center:
            rel_center = tuple(c - s for c, s in zip(center, starts))
            roi_shape = tuple(e - s for s, e in zip(starts, ends))

            center_idx = 0
            stride = 1
            for i in range(num_spatial - 1, -1, -1):
                center_idx += rel_center[i] * stride
                stride *= roi_shape[i]

            if patch.dim() == num_spatial:
                roi_flat = torch.cat(
                    [roi_flat[:center_idx], roi_flat[center_idx + 1 :]]
                )
            else:
                roi_flat = torch.cat(
                    [roi_flat[:, :center_idx], roi_flat[:, center_idx + 1 :]], dim=1
                )

        return roi_flat

    def _replace_masked_pixels(self, patch: Tensor, mask: Tensor) -> Tensor:
        """Replace masked pixels according to replacement strategy."""
        result = patch.clone()
        replacement = self._mask_params.replacement

        if replacement == "neighbor":
            replacement = "uniform"

        if replacement == "zero":
            expanded_mask = self._expand_mask_to_patch(mask, patch)
            result = result * (1 - expanded_mask)
        elif replacement == "uniform":
            result = self._replace_with_roi_uniform(result, mask)
        elif replacement == "median":
            result = self._replace_with_roi_median(result, mask)
        elif replacement == "mean":
            result = self._replace_with_roi_mean(result, mask)

        return result

    def _replace_with_roi_uniform(self, patch: Tensor, mask: Tensor) -> Tensor:
        """Replace masked pixels with random neighbor from ROI."""
        result = patch.clone()
        num_spatial = self._infer_spatial_dims(patch)

        spatial_mask = mask
        while spatial_mask.dim() > num_spatial:
            spatial_mask = spatial_mask.squeeze(0)

        masked_positions = torch.where(spatial_mask > 0)

        for i in range(len(masked_positions[0])):
            center = tuple(int(masked_positions[j][i]) for j in range(num_spatial))
            roi_values = self._extract_roi(patch, center, exclude_center=True)

            if roi_values.numel() == 0:
                continue

            if patch.dim() == num_spatial:
                idx = self._rng.integers(0, roi_values.numel())
                replacement_value = roi_values[idx]
                if num_spatial == 2:
                    result[center[0], center[1]] = replacement_value
                elif num_spatial == 3:
                    result[center[0], center[1], center[2]] = replacement_value
            else:
                idx = self._rng.integers(0, roi_values.shape[1])
                replacement_values = roi_values[:, idx]
                if num_spatial == 2:
                    result[:, center[0], center[1]] = replacement_values
                elif num_spatial == 3:
                    result[:, center[0], center[1], center[2]] = replacement_values

        return result

    def _replace_with_roi_median(self, patch: Tensor, mask: Tensor) -> Tensor:
        """Replace masked pixels with median of ROI."""
        result = patch.clone()
        num_spatial = self._infer_spatial_dims(patch)

        spatial_mask = mask
        while spatial_mask.dim() > num_spatial:
            spatial_mask = spatial_mask.squeeze(0)

        masked_positions = torch.where(spatial_mask > 0)

        for i in range(len(masked_positions[0])):
            center = tuple(int(masked_positions[j][i]) for j in range(num_spatial))
            roi_values = self._extract_roi(patch, center, exclude_center=True)

            if roi_values.numel() == 0:
                continue

            if patch.dim() == num_spatial:
                replacement_value = torch.median(roi_values)
                if num_spatial == 2:
                    result[center[0], center[1]] = replacement_value
                elif num_spatial == 3:
                    result[center[0], center[1], center[2]] = replacement_value
            else:
                replacement_values = torch.median(roi_values, dim=1).values
                if num_spatial == 2:
                    result[:, center[0], center[1]] = replacement_values
                elif num_spatial == 3:
                    result[:, center[0], center[1], center[2]] = replacement_values

        return result

    def _replace_with_roi_mean(self, patch: Tensor, mask: Tensor) -> Tensor:
        """Replace masked pixels with mean of ROI."""
        result = patch.clone()
        num_spatial = self._infer_spatial_dims(patch)

        spatial_mask = mask
        while spatial_mask.dim() > num_spatial:
            spatial_mask = spatial_mask.squeeze(0)

        masked_positions = torch.where(spatial_mask > 0)

        for i in range(len(masked_positions[0])):
            center = tuple(int(masked_positions[j][i]) for j in range(num_spatial))
            roi_values = self._extract_roi(patch, center, exclude_center=True)

            if roi_values.numel() == 0:
                continue

            if patch.dim() == num_spatial:
                replacement_value = torch.mean(roi_values)
                if num_spatial == 2:
                    result[center[0], center[1]] = replacement_value
                elif num_spatial == 3:
                    result[center[0], center[1], center[2]] = replacement_value
            else:
                replacement_values = torch.mean(roi_values, dim=1)
                if num_spatial == 2:
                    result[:, center[0], center[1]] = replacement_values
                elif num_spatial == 3:
                    result[:, center[0], center[1], center[2]] = replacement_values

        return result

    def get_state(self) -> dict[str, Any]:
        """Get strategy state."""
        state = super().get_state()
        state["mask_params"] = self._mask_params.to_dict()
        return state


__all__ = [
    "BaseBlindSpotStrategy",
    "BaseSelfSupervisedStrategy",
]

"""Configuration dataclasses for baseline self-supervised strategies.

Consolidated from n2n3s, n2v3s, and n2s3s packages.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class DataDimensionality(Enum):
    """Dimensionality modes for strategy input data."""

    XY = "xy"  # 2D: (C, H, W)
    XYZ = "xyz"  # 3D: (C, D, H, W)
    XYT = "xyt"  # Temporal 2D: (C, T, H, W)
    XYZT = "xyzt"  # 4D: (C, T, D, H, W)
    AUTO = "auto"  # Auto-detect


class PartitionScheme(str, Enum):
    """Partition schemes for J-invariance (Noise2Self)."""

    CHECKERBOARD = "checkerboard"  # 2-partition checkerboard
    DONUT = "donut"  # Excludes local neighborhood
    RANDOM = "random"  # Random partition (less structured)
    GRID = "grid"  # Regular grid partition


@dataclass
class StrategyOutput:
    """Standard output from all self-supervised strategies."""

    input: Any  # Tensor
    target: Any  # Tensor
    ground_truth: Any  # Tensor
    mask: Any | None = None  # Tensor | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input": self.input,
            "target": self.target,
            "ground_truth": self.ground_truth,
            "mask": self.mask,
        }


@dataclass
class NormalizationParams:
    """Parameters for target normalization."""

    normalize: bool = True
    method: str = "mean"
    epsilon: float = 1e-8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationParams":
        """Create from dictionary."""
        return cls(
            normalize=data.get("normalize", True),
            method=data.get("method", "mean"),
            epsilon=data.get("epsilon", 1e-8),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "normalize": self.normalize,
            "method": self.method,
            "epsilon": self.epsilon,
        }


@dataclass
class MaskParams:
    """Parameters for blind-spot strategies (Noise2Void).

    Attributes:
        mask_ratio: Fraction of pixels to mask (default 0.2% as in CAREamics)
        roi_size: Size of the ROI window for replacement (default 11, must be odd)
        replacement: Strategy for replacing masked pixels
        stratified: Use stratified sampling for mask generation
    """

    mask_ratio: float = 0.002
    roi_size: int = 11
    replacement: Literal["uniform", "median", "mean", "zero"] = "uniform"
    stratified: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.mask_ratio < 1:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        if self.roi_size < 3:
            raise ValueError(f"roi_size must be >= 3, got {self.roi_size}")
        if self.roi_size % 2 == 0:
            raise ValueError(f"roi_size must be odd, got {self.roi_size}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskParams":
        """Create from dictionary."""
        roi_size = data.get("roi_size", 11)
        if "mask_radius" in data and "roi_size" not in data:
            roi_size = data["mask_radius"] * 2 + 1

        replacement = data.get("replacement", "uniform")
        if replacement == "neighbor":
            replacement = "uniform"

        return cls(
            mask_ratio=data.get("mask_ratio", 0.002),
            roi_size=roi_size,
            replacement=replacement,
            stratified=data.get("stratified", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mask_ratio": self.mask_ratio,
            "roi_size": self.roi_size,
            "replacement": self.replacement,
            "stratified": self.stratified,
        }


@dataclass
class N2NParams:
    """Parameters for Noise2Noise strategy.

    Attributes:
        frame_offset: Offset between input and target frames (default: 1)
        bidirectional: If True, randomly choose forward or backward offset
        require_min_frames: Minimum frames required in temporal dimension
    """

    frame_offset: int = 1
    bidirectional: bool = True
    require_min_frames: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_offset": self.frame_offset,
            "bidirectional": self.bidirectional,
            "require_min_frames": self.require_min_frames,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "N2NParams":
        """Create from dictionary."""
        return cls(
            frame_offset=d.get("frame_offset", 1),
            bidirectional=d.get("bidirectional", True),
            require_min_frames=d.get("require_min_frames", 2),
        )


@dataclass
class N2SParams:
    """Parameters for Noise2Self strategy.

    Attributes:
        partition_scheme: How to partition pixels for J-invariance
        partition_phase: Which phase of partition to predict (for multi-phase)
        donut_radius: Radius for donut exclusion (only for DONUT scheme)
        grid_spacing: Spacing for grid partition (only for GRID scheme)
    """

    partition_scheme: PartitionScheme = PartitionScheme.CHECKERBOARD
    partition_phase: int = 0
    donut_radius: int = 1
    grid_spacing: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_scheme": self.partition_scheme.value,
            "partition_phase": self.partition_phase,
            "donut_radius": self.donut_radius,
            "grid_spacing": self.grid_spacing,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "N2SParams":
        """Create from dictionary."""
        scheme = d.get("partition_scheme", "checkerboard")
        if isinstance(scheme, str):
            scheme = PartitionScheme(scheme)
        return cls(
            partition_scheme=scheme,
            partition_phase=d.get("partition_phase", 0),
            donut_radius=d.get("donut_radius", 1),
            grid_spacing=d.get("grid_spacing", 2),
        )


__all__ = [
    "DataDimensionality",
    "MaskParams",
    "N2NParams",
    "N2SParams",
    "NormalizationParams",
    "PartitionScheme",
    "StrategyOutput",
]

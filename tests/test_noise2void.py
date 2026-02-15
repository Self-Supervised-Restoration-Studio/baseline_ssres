"""Tests for Noise2Void strategy in baseline_ssres."""

import numpy as np
import pytest
import torch


class TestImports:
    """Test package imports."""

    def test_import_strategy(self):
        from baseline_ssres import Noise2VoidStrategy

        assert hasattr(Noise2VoidStrategy, "sample")
        assert hasattr(Noise2VoidStrategy, "create_validation_variant")

    def test_import_params(self):
        from baseline_ssres import MaskParams

        params = MaskParams()
        assert params.mask_ratio == 0.002
        assert params.roi_size == 11

    def test_import_base_classes(self):
        from baseline_ssres import (
            DataDimensionality,
            NormalizationParams,
        )

        assert DataDimensionality.XYZ.value == "xyz"
        assert NormalizationParams().normalize is True


class TestMaskParams:
    """Test MaskParams configuration."""

    def test_default_params(self):
        from baseline_ssres import MaskParams

        params = MaskParams()
        assert params.mask_ratio == 0.002
        assert params.roi_size == 11
        assert params.replacement == "uniform"
        assert params.stratified is True

    def test_custom_params(self):
        from baseline_ssres import MaskParams

        params = MaskParams(
            mask_ratio=0.05, roi_size=7, replacement="median", stratified=False
        )
        assert params.mask_ratio == 0.05
        assert params.roi_size == 7
        assert params.replacement == "median"
        assert params.stratified is False

    def test_invalid_mask_ratio(self):
        from baseline_ssres import MaskParams

        with pytest.raises(ValueError, match="mask_ratio"):
            MaskParams(mask_ratio=1.5)

    def test_invalid_roi_size_even(self):
        from baseline_ssres import MaskParams

        with pytest.raises(ValueError, match="roi_size must be odd"):
            MaskParams(roi_size=10)

    def test_invalid_roi_size_small(self):
        from baseline_ssres import MaskParams

        with pytest.raises(ValueError, match="roi_size must be >= 3"):
            MaskParams(roi_size=1)

    def test_to_dict(self):
        from baseline_ssres import MaskParams

        params = MaskParams(mask_ratio=0.01, roi_size=9, replacement="mean")
        d = params.to_dict()
        assert d["mask_ratio"] == 0.01
        assert d["roi_size"] == 9
        assert d["replacement"] == "mean"

    def test_from_dict(self):
        from baseline_ssres import MaskParams

        d = {"mask_ratio": 0.03, "roi_size": 5, "replacement": "median"}
        params = MaskParams.from_dict(d)
        assert params.mask_ratio == 0.03
        assert params.roi_size == 5
        assert params.replacement == "median"


class TestNoise2VoidStrategy:
    """Test Noise2Void strategy."""

    def test_basic_sample_2d(self):
        from baseline_ssres import DataDimensionality, MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, stratified=False)
        strategy = Noise2VoidStrategy(
            mask_params=params, seed=42, dimensionality=DataDimensionality.XY
        )

        patch = torch.randn(1, 32, 32)
        output = strategy.sample(patch)

        assert output.input is not None
        assert output.target is not None
        assert output.ground_truth is not None
        assert output.mask is not None

        assert output.input.shape == patch.shape
        assert output.target.shape == patch.shape
        assert output.mask.shape == patch.shape

    def test_basic_sample_3d(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)

        assert output.input.shape == patch.shape
        assert output.target.shape == patch.shape
        assert output.mask.shape == patch.shape

    def test_mask_has_ones(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)

        assert output.mask.sum() > 0

    def test_training_mode(self):
        from baseline_ssres import Noise2VoidStrategy

        strategy = Noise2VoidStrategy(is_training=True, seed=42)
        assert strategy.is_training is True
        assert strategy.name == "noise2void"
        assert strategy.requires_mask is True

    def test_validation_mode(self):
        from baseline_ssres import Noise2VoidStrategy

        strategy = Noise2VoidStrategy(is_training=False, seed=42)
        assert strategy.is_training is False

    def test_create_validation_variant(self):
        from baseline_ssres import Noise2VoidStrategy

        train_strategy = Noise2VoidStrategy(is_training=True, seed=42)
        val_strategy = train_strategy.create_validation_variant()

        assert val_strategy.is_training is False
        assert val_strategy.name == "noise2void_validation"
        assert val_strategy.requires_mask is False

    def test_validation_no_masking(self):
        from baseline_ssres import Noise2VoidValidationStrategy

        strategy = Noise2VoidValidationStrategy(seed=42)
        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)

        assert output.mask is None
        assert torch.equal(output.input, patch)

    def test_replacement_uniform(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, replacement="uniform", stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)
        assert output.input is not None

    def test_replacement_median(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, replacement="median", stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)
        assert output.input is not None

    def test_replacement_mean(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, replacement="mean", stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)
        assert output.input is not None

    def test_replacement_zero(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, replacement="zero", stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.ones(1, 8, 16, 16) * 5
        output = strategy.sample(patch)
        assert (output.input == 0).any()

    def test_stratified_sampling(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.02, stratified=True)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = torch.randn(1, 32, 32, 32)
        output = strategy.sample(patch)

        assert output.mask is not None
        assert output.mask.sum() > 0

    def test_numpy_input(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.1, stratified=False)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        patch = np.random.randn(1, 8, 16, 16).astype(np.float32)
        output = strategy.sample(patch)

        assert isinstance(output.input, torch.Tensor)
        assert isinstance(output.target, torch.Tensor)

    def test_get_state(self):
        from baseline_ssres import MaskParams, Noise2VoidStrategy

        params = MaskParams(mask_ratio=0.05, roi_size=9)
        strategy = Noise2VoidStrategy(mask_params=params, seed=42)

        state = strategy.get_state()
        assert state["name"] == "noise2void"
        assert state["seed"] == 42
        assert state["mask_params"]["mask_ratio"] == 0.05
        assert state["mask_params"]["roi_size"] == 9


class TestNoise2VoidValidationStrategy:
    """Test explicit validation strategy class."""

    def test_validation_strategy(self):
        from baseline_ssres import Noise2VoidValidationStrategy

        strategy = Noise2VoidValidationStrategy(seed=42)
        assert strategy.is_training is False
        assert strategy.name == "noise2void_validation"
        assert strategy.requires_mask is False

    def test_create_validation_variant_returns_self(self):
        from baseline_ssres import Noise2VoidValidationStrategy

        strategy = Noise2VoidValidationStrategy()
        variant = strategy.create_validation_variant()
        assert variant is strategy


class TestStrategyOutput:
    """Test StrategyOutput dataclass."""

    def test_to_dict(self):
        from baseline_ssres import StrategyOutput

        output = StrategyOutput(
            input=torch.randn(1, 8, 16, 16),
            target=torch.randn(1, 8, 16, 16),
            ground_truth=torch.randn(1, 8, 16, 16),
            mask=torch.ones(1, 8, 16, 16),
        )

        d = output.to_dict()
        assert "input" in d
        assert "target" in d
        assert "ground_truth" in d
        assert "mask" in d


class TestDataDimensionality:
    """Test DataDimensionality enum."""

    def test_dimensionality_values(self):
        from baseline_ssres import DataDimensionality

        assert DataDimensionality.XY.value == "xy"
        assert DataDimensionality.XYZ.value == "xyz"
        assert DataDimensionality.XYT.value == "xyt"
        assert DataDimensionality.XYZT.value == "xyzt"
        assert DataDimensionality.AUTO.value == "auto"

    def test_strategy_with_dimensionality(self):
        from baseline_ssres import DataDimensionality, Noise2VoidStrategy

        strategy = Noise2VoidStrategy(dimensionality=DataDimensionality.XYZ)
        assert strategy.dimensionality == DataDimensionality.XYZ


class TestAllExports:
    """Test that all __all__ exports are importable."""

    def test_all_exports(self):
        import baseline_ssres

        for name in baseline_ssres.__all__:
            assert hasattr(baseline_ssres, name), f"Missing export: {name}"

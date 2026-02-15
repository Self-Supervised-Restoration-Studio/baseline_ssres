"""Tests for Noise2Noise strategy in baseline_ssres."""

import numpy as np
import pytest
import torch


class TestImports:
    """Test package imports."""

    def test_import_strategy(self):
        from baseline_ssres import Noise2NoiseStrategy

        assert hasattr(Noise2NoiseStrategy, "sample")
        assert hasattr(Noise2NoiseStrategy, "create_validation_variant")

    def test_import_params(self):
        from baseline_ssres import N2NParams

        params = N2NParams()
        assert params.frame_offset == 1
        assert params.bidirectional is True

    def test_import_base_classes(self):
        from baseline_ssres import (
            DataDimensionality,
            NormalizationParams,
        )

        assert DataDimensionality.XYT.value == "xyt"
        assert NormalizationParams().normalize is True


class TestN2NParams:
    """Test N2NParams configuration."""

    def test_default_params(self):
        from baseline_ssres import N2NParams

        params = N2NParams()
        assert params.frame_offset == 1
        assert params.bidirectional is True
        assert params.require_min_frames == 2

    def test_custom_params(self):
        from baseline_ssres import N2NParams

        params = N2NParams(frame_offset=2, bidirectional=False, require_min_frames=3)
        assert params.frame_offset == 2
        assert params.bidirectional is False
        assert params.require_min_frames == 3

    def test_to_dict(self):
        from baseline_ssres import N2NParams

        params = N2NParams(frame_offset=3, bidirectional=False)
        d = params.to_dict()
        assert d["frame_offset"] == 3
        assert d["bidirectional"] is False

    def test_from_dict(self):
        from baseline_ssres import N2NParams

        d = {"frame_offset": 2, "bidirectional": False, "require_min_frames": 4}
        params = N2NParams.from_dict(d)
        assert params.frame_offset == 2
        assert params.bidirectional is False
        assert params.require_min_frames == 4


class TestNoise2NoiseStrategy:
    """Test Noise2Noise strategy."""

    def test_basic_sample_2d(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(seed=42)

        # (T, C, H, W) - 4 frames, 1 channel, 32x32
        patch = torch.randn(4, 1, 32, 32)
        output = strategy.sample(patch)

        assert output.input is not None
        assert output.target is not None
        assert output.ground_truth is not None
        assert output.mask is None

        assert output.input.shape == (1, 32, 32)
        assert output.target.shape == (1, 32, 32)

    def test_basic_sample_3d(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(seed=42)

        # (T, C, D, H, W) - 3 frames, 1 channel, 8x16x16
        patch = torch.randn(3, 1, 8, 16, 16)
        output = strategy.sample(patch)

        assert output.input.shape == (1, 8, 16, 16)
        assert output.target.shape == (1, 8, 16, 16)

    def test_insufficient_frames(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy()
        patch = torch.randn(1, 1, 32, 32)
        with pytest.raises(ValueError, match="requires at least 2"):
            strategy.sample(patch)

    def test_custom_min_frames(self):
        from baseline_ssres import N2NParams, Noise2NoiseStrategy

        params = N2NParams(require_min_frames=4)
        strategy = Noise2NoiseStrategy(n2n_params=params)
        patch = torch.randn(3, 1, 32, 32)
        with pytest.raises(ValueError, match="requires at least 4"):
            strategy.sample(patch)

    def test_training_mode(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(is_training=True, seed=42)
        assert strategy.is_training is True
        assert strategy.name == "noise2noise"
        assert strategy.requires_temporal is True
        assert strategy.requires_mask is False

    def test_validation_mode(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(is_training=False, seed=42)
        assert strategy.is_training is False

    def test_create_validation_variant(self):
        from baseline_ssres import Noise2NoiseStrategy

        train_strategy = Noise2NoiseStrategy(is_training=True, seed=42)
        val_strategy = train_strategy.create_validation_variant()

        assert val_strategy.is_training is False
        assert val_strategy.name == "noise2noise"

    def test_validation_deterministic(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(is_training=False)
        patch = torch.randn(5, 1, 32, 32)

        output1 = strategy.sample(patch)
        output2 = strategy.sample(patch)

        assert torch.equal(output1.input, output2.input)
        assert torch.equal(output1.target, output2.target)

    def test_training_randomness(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(is_training=True, seed=None)
        patch = torch.randn(10, 1, 32, 32)

        outputs = [strategy.sample(patch) for _ in range(10)]
        input_frames = [o.input for o in outputs]

        unique_count = len(set(tuple(f.flatten().tolist()) for f in input_frames))
        assert unique_count >= 2

    def test_frame_offset(self):
        from baseline_ssres import N2NParams, Noise2NoiseStrategy

        params = N2NParams(frame_offset=2)
        strategy = Noise2NoiseStrategy(n2n_params=params, is_training=False)

        patch = torch.zeros(5, 1, 4, 4)
        for i in range(5):
            patch[i] = i

        output = strategy.sample(patch)
        assert output.input.mean() == 0.0

    def test_bidirectional(self):
        from baseline_ssres import N2NParams, Noise2NoiseStrategy

        params = N2NParams(bidirectional=False)
        strategy = Noise2NoiseStrategy(n2n_params=params, is_training=False)

        patch = torch.randn(3, 1, 8, 8)
        output = strategy.sample(patch)
        assert output.input is not None

    def test_numpy_input(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(seed=42)
        patch = np.random.randn(4, 1, 32, 32).astype(np.float32)
        output = strategy.sample(patch)

        assert isinstance(output.input, torch.Tensor)
        assert isinstance(output.target, torch.Tensor)

    def test_get_state(self):
        from baseline_ssres import N2NParams, Noise2NoiseStrategy

        params = N2NParams(frame_offset=3)
        strategy = Noise2NoiseStrategy(n2n_params=params, seed=42)

        state = strategy.get_state()
        assert state["name"] == "noise2noise"
        assert state["seed"] == 42
        assert state["n2n_params"]["frame_offset"] == 3


class TestNoise2NoiseValidationStrategy:
    """Test explicit validation strategy class."""

    def test_validation_strategy(self):
        from baseline_ssres import Noise2NoiseValidationStrategy

        strategy = Noise2NoiseValidationStrategy(seed=42)
        assert strategy.is_training is False
        assert strategy.name == "noise2noise_validation"

    def test_create_validation_variant_returns_self(self):
        from baseline_ssres import Noise2NoiseValidationStrategy

        strategy = Noise2NoiseValidationStrategy()
        variant = strategy.create_validation_variant()
        assert variant is strategy


class TestStrategyOutput:
    """Test StrategyOutput dataclass."""

    def test_to_dict(self):
        from baseline_ssres import StrategyOutput

        output = StrategyOutput(
            input=torch.randn(1, 32, 32),
            target=torch.randn(1, 32, 32),
            ground_truth=torch.randn(1, 32, 32),
            mask=None,
        )

        d = output.to_dict()
        assert "input" in d
        assert "target" in d
        assert "ground_truth" in d
        assert "mask" in d
        assert d["mask"] is None


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
        from baseline_ssres import DataDimensionality, Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(dimensionality=DataDimensionality.XYZT)
        assert strategy.dimensionality == DataDimensionality.XYZT

    def test_strategy_with_string_dimensionality(self):
        from baseline_ssres import DataDimensionality, Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(dimensionality="xyt")
        assert strategy.dimensionality == DataDimensionality.XYT


class TestNormalization:
    """Test target normalization."""

    def test_default_normalization(self):
        from baseline_ssres import NormalizationParams

        params = NormalizationParams()
        assert params.normalize is True
        assert params.method == "mean"

    def test_normalization_effect(self):
        from baseline_ssres import Noise2NoiseStrategy

        strategy = Noise2NoiseStrategy(seed=42)

        patch = torch.zeros(2, 1, 8, 8)
        patch[0] = 10.0
        patch[1] = 5.0

        output = strategy.sample(patch)
        assert output.target.shape == (1, 8, 8)

    def test_no_normalization(self):
        from baseline_ssres import Noise2NoiseStrategy, NormalizationParams

        norm_params = NormalizationParams(normalize=False)
        strategy = Noise2NoiseStrategy(normalization=norm_params, is_training=False)

        patch = torch.zeros(2, 1, 8, 8)
        patch[0] = 10.0
        patch[1] = 5.0

        output = strategy.sample(patch)
        assert output.target.mean().item() == pytest.approx(5.0, rel=1e-5)


class TestAllExports:
    """Test that all __all__ exports are importable."""

    def test_all_exports(self):
        import baseline_ssres

        for name in baseline_ssres.__all__:
            assert hasattr(baseline_ssres, name), f"Missing export: {name}"

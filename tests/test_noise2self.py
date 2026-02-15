"""Tests for Noise2Self strategy in baseline_ssres."""

import numpy as np
import torch


class TestImports:
    """Test package imports."""

    def test_import_strategy(self):
        from baseline_ssres import Noise2SelfStrategy

        assert hasattr(Noise2SelfStrategy, "sample")
        assert hasattr(Noise2SelfStrategy, "create_validation_variant")

    def test_import_params(self):
        from baseline_ssres import N2SParams, PartitionScheme

        params = N2SParams()
        assert params.partition_scheme == PartitionScheme.CHECKERBOARD
        assert params.partition_phase == 0

    def test_import_base_classes(self):
        from baseline_ssres import (
            DataDimensionality,
            NormalizationParams,
        )

        assert DataDimensionality.XYZ.value == "xyz"
        assert NormalizationParams().normalize is True


class TestPartitionScheme:
    """Test PartitionScheme enum."""

    def test_partition_values(self):
        from baseline_ssres import PartitionScheme

        assert PartitionScheme.CHECKERBOARD.value == "checkerboard"
        assert PartitionScheme.DONUT.value == "donut"
        assert PartitionScheme.GRID.value == "grid"
        assert PartitionScheme.RANDOM.value == "random"


class TestN2SParams:
    """Test N2SParams configuration."""

    def test_default_params(self):
        from baseline_ssres import N2SParams, PartitionScheme

        params = N2SParams()
        assert params.partition_scheme == PartitionScheme.CHECKERBOARD
        assert params.partition_phase == 0
        assert params.donut_radius == 1
        assert params.grid_spacing == 2

    def test_custom_params(self):
        from baseline_ssres import N2SParams, PartitionScheme

        params = N2SParams(
            partition_scheme=PartitionScheme.DONUT,
            partition_phase=1,
            donut_radius=2,
            grid_spacing=4,
        )
        assert params.partition_scheme == PartitionScheme.DONUT
        assert params.partition_phase == 1
        assert params.donut_radius == 2
        assert params.grid_spacing == 4

    def test_to_dict(self):
        from baseline_ssres import N2SParams, PartitionScheme

        params = N2SParams(partition_scheme=PartitionScheme.GRID, partition_phase=1)
        d = params.to_dict()
        assert d["partition_scheme"] == "grid"
        assert d["partition_phase"] == 1

    def test_from_dict(self):
        from baseline_ssres import N2SParams, PartitionScheme

        d = {"partition_scheme": "donut", "partition_phase": 1, "donut_radius": 3}
        params = N2SParams.from_dict(d)
        assert params.partition_scheme == PartitionScheme.DONUT
        assert params.partition_phase == 1
        assert params.donut_radius == 3


class TestNoise2SelfStrategy:
    """Test Noise2Self strategy."""

    def test_basic_sample_2d(self):
        from baseline_ssres import DataDimensionality, Noise2SelfStrategy

        strategy = Noise2SelfStrategy(seed=42, dimensionality=DataDimensionality.XY)

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
        from baseline_ssres import Noise2SelfStrategy

        strategy = Noise2SelfStrategy(seed=42)

        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)

        assert output.input.shape == patch.shape
        assert output.target.shape == patch.shape
        assert output.mask.shape == patch.shape

    def test_checkerboard_mask(self):
        from baseline_ssres import N2SParams, Noise2SelfStrategy, PartitionScheme

        params = N2SParams(
            partition_scheme=PartitionScheme.CHECKERBOARD, partition_phase=0
        )
        strategy = Noise2SelfStrategy(n2s_params=params, seed=42)

        patch = torch.randn(1, 8, 8, 8)
        output = strategy.sample(patch)

        mask_ratio = output.mask.mean().item()
        assert 0.4 < mask_ratio < 0.6

    def test_checkerboard_complementary_phases(self):
        from baseline_ssres import (
            DataDimensionality,
            N2SParams,
            Noise2SelfStrategy,
            PartitionScheme,
        )

        params0 = N2SParams(
            partition_scheme=PartitionScheme.CHECKERBOARD, partition_phase=0
        )
        strategy0 = Noise2SelfStrategy(
            n2s_params=params0, seed=42, dimensionality=DataDimensionality.XY
        )

        params1 = N2SParams(
            partition_scheme=PartitionScheme.CHECKERBOARD, partition_phase=1
        )
        strategy1 = Noise2SelfStrategy(
            n2s_params=params1, seed=42, dimensionality=DataDimensionality.XY
        )

        patch = torch.randn(1, 8, 8)
        output0 = strategy0.sample(patch)
        output1 = strategy1.sample(patch)

        mask_sum = output0.mask + output1.mask
        assert torch.all(mask_sum == 1.0)

    def test_donut_mask(self):
        from baseline_ssres import N2SParams, Noise2SelfStrategy, PartitionScheme

        params = N2SParams(partition_scheme=PartitionScheme.DONUT, donut_radius=1)
        strategy = Noise2SelfStrategy(n2s_params=params, seed=42)

        patch = torch.randn(1, 16, 16, 16)
        output = strategy.sample(patch)

        mask_ratio = output.mask.mean().item()
        assert mask_ratio < 0.3

    def test_grid_mask(self):
        from baseline_ssres import N2SParams, Noise2SelfStrategy, PartitionScheme

        params = N2SParams(partition_scheme=PartitionScheme.GRID, grid_spacing=4)
        strategy = Noise2SelfStrategy(n2s_params=params, seed=42)

        patch = torch.randn(1, 16, 16, 16)
        output = strategy.sample(patch)

        mask_ratio = output.mask.mean().item()
        assert mask_ratio < 0.2

    def test_random_mask(self):
        from baseline_ssres import N2SParams, Noise2SelfStrategy, PartitionScheme

        params = N2SParams(partition_scheme=PartitionScheme.RANDOM)
        strategy = Noise2SelfStrategy(n2s_params=params, seed=42)

        patch = torch.randn(1, 16, 16, 16)
        output = strategy.sample(patch)

        mask_ratio = output.mask.mean().item()
        assert 0.4 < mask_ratio < 0.6

    def test_training_mode(self):
        from baseline_ssres import Noise2SelfStrategy

        strategy = Noise2SelfStrategy(is_training=True, seed=42)
        assert strategy.is_training is True
        assert strategy.name == "noise2self"
        assert strategy.requires_mask is True

    def test_validation_mode(self):
        from baseline_ssres import Noise2SelfStrategy

        strategy = Noise2SelfStrategy(is_training=False, seed=42)
        assert strategy.is_training is False

    def test_create_validation_variant(self):
        from baseline_ssres import Noise2SelfStrategy

        train_strategy = Noise2SelfStrategy(is_training=True, seed=42)
        val_strategy = train_strategy.create_validation_variant()

        assert val_strategy.is_training is False
        assert val_strategy.name == "noise2self"

    def test_input_unchanged(self):
        from baseline_ssres import Noise2SelfStrategy

        strategy = Noise2SelfStrategy(seed=42)
        patch = torch.randn(1, 8, 16, 16)
        output = strategy.sample(patch)

        assert torch.equal(output.input, patch)

    def test_numpy_input(self):
        from baseline_ssres import Noise2SelfStrategy

        strategy = Noise2SelfStrategy(seed=42)
        patch = np.random.randn(1, 8, 16, 16).astype(np.float32)
        output = strategy.sample(patch)

        assert isinstance(output.input, torch.Tensor)
        assert isinstance(output.target, torch.Tensor)

    def test_get_state(self):
        from baseline_ssres import N2SParams, Noise2SelfStrategy, PartitionScheme

        params = N2SParams(partition_scheme=PartitionScheme.DONUT, donut_radius=2)
        strategy = Noise2SelfStrategy(n2s_params=params, seed=42)

        state = strategy.get_state()
        assert state["name"] == "noise2self"
        assert state["seed"] == 42
        assert state["n2s_params"]["partition_scheme"] == "donut"
        assert state["n2s_params"]["donut_radius"] == 2


class TestNoise2SelfValidationStrategy:
    """Test explicit validation strategy class."""

    def test_validation_strategy(self):
        from baseline_ssres import Noise2SelfValidationStrategy

        strategy = Noise2SelfValidationStrategy(seed=42)
        assert strategy.is_training is False
        assert strategy.name == "noise2self_validation"

    def test_create_validation_variant_returns_self(self):
        from baseline_ssres import Noise2SelfValidationStrategy

        strategy = Noise2SelfValidationStrategy()
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


class TestAllExports:
    """Test that all __all__ exports are importable."""

    def test_all_exports(self):
        import baseline_ssres

        for name in baseline_ssres.__all__:
            assert hasattr(baseline_ssres, name), f"Missing export: {name}"

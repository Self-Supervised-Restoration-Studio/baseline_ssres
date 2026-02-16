# baseline-ssres

Baseline self-supervised denoising strategies: Noise2Noise, Noise2Void, and Noise2Self. Part of the [Self-Supervised Restoration Studio](https://github.com/Self-Supervised-Restoration-Studio) ecosystem.

Works standalone or as an [ssres](https://github.com/Self-Supervised-Restoration-Studio/ssres) plugin (discovered automatically via entry points).

## Install

```bash
uv add baseline-ssres
```

For development (editable):

```bash
git clone https://github.com/Self-Supervised-Restoration-Studio/baseline_ssres.git
cd baseline_ssres
uv sync --extra dev
```

## What's included

| Module | Key exports | Description |
|--------|------------|-------------|
| `noise2noise` | `Noise2NoiseStrategy`, `Noise2NoiseValidationStrategy` | Temporal frame-pair splitting (Lehtinen et al., 2018) |
| `noise2void` | `Noise2VoidStrategy`, `Noise2VoidValidationStrategy` | Blind-spot masking (Krull et al., 2019) |
| `noise2self` | `Noise2SelfStrategy`, `Noise2SelfValidationStrategy` | J-invariant partitioning (Batson & Royer, 2019) |
| `config` | `N2NParams`, `N2SParams`, `MaskParams`, `StrategyOutput` | Configuration dataclasses |
| `base` | `BaseBlindSpotStrategy`, `BaseSelfSupervisedStrategy` | Base classes for strategy implementations |

## Usage

```python
from baseline_ssres import Noise2NoiseStrategy, Noise2VoidStrategy, Noise2SelfStrategy

# Noise2Noise: temporal frame pairs
n2n = Noise2NoiseStrategy(seed=42)
output = n2n.sample(temporal_patch)  # (T, C, H, W)

# Noise2Void: blind-spot masking
n2v = Noise2VoidStrategy(seed=42)
output = n2v.sample(noisy_patch)  # (C, D, H, W)

# Noise2Self: J-invariant partitioning
n2s = Noise2SelfStrategy(seed=42)
output = n2s.sample(noisy_patch)  # (C, D, H, W)
```

## Dependencies

- [ssres_toolbox](https://github.com/Self-Supervised-Restoration-Studio/ssres_toolbox) — shared numerical utilities
- PyTorch >= 2.7.0

## Citations

- Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data", ICML 2018. [arXiv:1803.04189](https://arxiv.org/abs/1803.04189)
- Krull et al., "Noise2Void - Learning Denoising from Single Noisy Images", CVPR 2019. [arXiv:1811.10980](https://arxiv.org/abs/1811.10980)
- Batson & Royer, "Noise2Self: Blind Denoising by Self-Supervision", ICML 2019. [arXiv:1901.11365](https://arxiv.org/abs/1901.11365)

## License

[MIT](LICENSE). Third-party license attributions are in [licenses/](licenses/).

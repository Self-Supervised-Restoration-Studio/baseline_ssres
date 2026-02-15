# baseline-ssres

Baseline self-supervised denoising strategies for SSRES: Noise2Noise, Noise2Void, and Noise2Self.

## Installation

```bash
pip install baseline-ssres
```

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

## SSRES Plugin

When `ssres` is installed, this package automatically registers as a plugin via entry points.

## License

MIT License. See [LICENSE](LICENSE) for details.

Third-party license attributions are in [licenses/](licenses/).

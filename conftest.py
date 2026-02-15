"""Test configuration for baseline_ssres."""

import os

# Force CPU for all tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"

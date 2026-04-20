"""Training module — synthetic data generation + self-training utilities."""
from shelfmatch.training.synthetic import SyntheticShelfGenerator
from shelfmatch.training.formatter import TrainingFormatter

__all__ = [
    "SyntheticShelfGenerator",
    "TrainingFormatter",
]

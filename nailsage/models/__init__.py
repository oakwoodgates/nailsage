"""Model training, management, and registry."""

from nailsage.models.registry import ModelRegistry
from nailsage.models.trainer import BaseTrainer

__all__ = [
    "ModelRegistry",
    "BaseTrainer",
]

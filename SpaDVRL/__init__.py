"""SpaDVRL: dual-view representation learning with adaptive fusion."""

from .model import SpaDVRLModel
from .trainer import TrainSpaDVRL

__version__ = "1.0.0"
__all__ = ["SpaDVRLModel", "TrainSpaDVRL"]

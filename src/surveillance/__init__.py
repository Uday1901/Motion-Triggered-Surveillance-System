"""Core surveillance package containing detection, recording, and pipeline utilities."""

from .engine import SurveillanceEngine
from .config import SurveillanceConfig

__all__ = ["SurveillanceEngine", "SurveillanceConfig"]

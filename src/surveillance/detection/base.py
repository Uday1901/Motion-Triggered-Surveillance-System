"""Detection base classes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class DetectionOutcome:
    """Result of running a detector on a frame."""

    annotated_frame: np.ndarray
    objects_detected: bool
    metadata: dict[str, Any]


class BaseDetector(ABC):
    """Abstract base detector."""

    name: str

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionOutcome:
        """Return detection outcome for a frame."""

    def shutdown(self) -> None:
        """Release resources if needed."""

    def __str__(self) -> str:  # pragma: no cover - simple convenience
        return self.name

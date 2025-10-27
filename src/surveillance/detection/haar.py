"""Haar cascade detector implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .base import BaseDetector, DetectionOutcome


@dataclass(slots=True)
class CascadeSpec:
    label: str
    cascade_name: str
    color: tuple[int, int, int]
    thickness: int = 2


DEFAULT_CASCADES: tuple[CascadeSpec, ...] = (
    CascadeSpec("Face", "haarcascade_frontalface_default.xml", (255, 0, 0)),
    CascadeSpec("Body", "haarcascade_fullbody.xml", (0, 255, 0)),
    CascadeSpec("Eye", "haarcascade_eye.xml", (0, 255, 255), thickness=1),
    CascadeSpec("Smile", "haarcascade_smile.xml", (255, 255, 0), thickness=1),
    CascadeSpec("Profile", "haarcascade_profileface.xml", (0, 0, 255)),
    CascadeSpec("Upper Body", "haarcascade_upperbody.xml", (200, 0, 255)),
    CascadeSpec("Lower Body", "haarcascade_lowerbody.xml", (0, 200, 255)),
    CascadeSpec("Cat", "haarcascade_frontalcatface.xml", (255, 100, 100)),
    CascadeSpec("Plate", "haarcascade_russian_plate_number.xml", (100, 255, 100)),
)


class HaarDetector(BaseDetector):
    """Detector powered by OpenCV Haar cascades."""

    def __init__(
        self,
        cascades: Iterable[CascadeSpec] | None = None,
        cascade_dir: Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    ) -> None:
        self.name = "haar"
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        source_dir = cascade_dir or Path(cv2.data.haarcascades)
        self._cascades: list[tuple[CascadeSpec, cv2.CascadeClassifier]] = []

        for spec in cascades or DEFAULT_CASCADES:
            classifier = cv2.CascadeClassifier(str(source_dir / spec.cascade_name))
            if classifier.empty():
                continue
            self._cascades.append((spec, classifier))

    def detect(self, frame: np.ndarray) -> DetectionOutcome:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()
        objects_detected = False
        metadata: dict[str, int] = {}

        for spec, classifier in self._cascades:
            detections = classifier.detectMultiScale(gray, self._scale_factor, self._min_neighbors)
            if len(detections):
                objects_detected = True
            metadata[spec.label] = int(len(detections))
            for (x, y, w, h) in detections:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), spec.color, spec.thickness)
                if spec.thickness >= 2:
                    cv2.putText(
                        annotated,
                        spec.label,
                        (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        spec.color,
                        2,
                    )

        return DetectionOutcome(annotated_frame=annotated, objects_detected=objects_detected, metadata=metadata)

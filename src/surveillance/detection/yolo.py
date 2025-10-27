"""YOLO detector implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .base import BaseDetector, DetectionOutcome
from ..utils import LOGGER, resolve_device

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore[assignment]


@dataclass(slots=True)
class YOLOConfig:
    model_path: str = "yolov8n.pt"
    target_classes: Iterable[str] = ("person",)


class YOLODetector(BaseDetector):
    """Detector powered by Ultralytics YOLO models."""

    def __init__(self, config: YOLOConfig) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Please install the 'ultralytics' package.")

        self.name = "yolo"
        model_path = config.model_path
        if Path(model_path).is_file():
            resolved = str(Path(model_path).resolve())
        else:
            resolved = model_path

        LOGGER.info("Loading YOLO model from %s", resolved)
        self._model = YOLO(resolved)

        if resolved.lower().endswith(".yaml"):
            LOGGER.warning(
                "YAML configuration provided. Detection requires trained .pt weights. "
                "Train the model to produce weights before running inference."
            )

        self._target_classes = set(config.target_classes)
        self._device = resolve_device()
        if hasattr(self._model, "to"):
            try:
                self._model.to(self._device)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                LOGGER.warning("Unable to move YOLO model to %s: %s", self._device, exc)

    def detect(self, frame: np.ndarray) -> DetectionOutcome:
        annotated = frame.copy()
        objects_detected = False
        metadata: dict[str, int] = {}

        results = self._model(annotated, stream=True, verbose=False)
        for r in results:
            for box in getattr(r, "boxes", []):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self._model.names.get(cls_id, str(cls_id))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                metadata[label] = metadata.get(label, 0) + 1
                if label in self._target_classes:
                    objects_detected = True

        return DetectionOutcome(annotated_frame=annotated, objects_detected=objects_detected, metadata=metadata)

    def shutdown(self) -> None:
        # YOLO model does not require explicit shutdown but method retained for interface parity
        pass

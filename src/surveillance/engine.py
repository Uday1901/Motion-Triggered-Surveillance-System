"""High-level orchestration for surveillance detection and recording."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import SurveillanceConfig
from .detection import DetectionOutcome, HaarDetector, YOLOConfig, YOLODetector
from .recording import RecordingManager
from .utils import LOGGER


@dataclass(slots=True)
class ProcessedFrame:
    frame: np.ndarray
    detected: bool
    metadata: dict[str, int]


class SurveillanceEngine:
    """Combine detection and recording into a simple interface."""

    def __init__(self, config: SurveillanceConfig) -> None:
        self._config = config
        self._detector = self._build_detector(config)
        self._recorder = RecordingManager(
            video_dir=config.resolved_video_dir(),
            image_dir=config.resolved_image_dir(),
            fps=config.recording_fps,
            buffer_seconds=config.delay,
        )

    @property
    def config(self) -> SurveillanceConfig:
        return self._config

    def _build_detector(self, config: SurveillanceConfig):
        if config.mode == "haar":
            return HaarDetector()
        if config.mode == "yolo":
            yolo_config = YOLOConfig(model_path=config.model_path, target_classes=config.targets)
            return YOLODetector(yolo_config)
        raise ValueError(f"Unsupported detection mode: {config.mode}")

    def process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        outcome: DetectionOutcome = self._detector.detect(frame)
        self._recorder.update(outcome.annotated_frame, outcome.objects_detected, self._config.mode)
        return ProcessedFrame(frame=outcome.annotated_frame, detected=outcome.objects_detected, metadata=outcome.metadata)

    def stop(self) -> None:
        self._recorder.stop()

    def shutdown(self) -> None:
        LOGGER.info("Shutting down surveillance engine")
        self._recorder.shutdown()
        if hasattr(self._detector, "shutdown"):
            try:
                self._detector.shutdown()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                LOGGER.warning("Detector shutdown raised error: %s", exc)


def build_capture_device(device_index: int = 0, fps: Optional[int] = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam device")
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

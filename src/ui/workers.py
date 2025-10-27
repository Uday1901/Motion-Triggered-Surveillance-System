"""Background worker threads for the UI."""
from __future__ import annotations

import traceback
from typing import Optional

import cv2
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from surveillance.config import SurveillanceConfig
from surveillance.engine import ProcessedFrame, SurveillanceEngine, build_capture_device
from surveillance.utils import LOGGER


class VideoWorker(QObject):
    frame_ready = Signal(QImage, bool, dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, config: SurveillanceConfig, device_index: int = 0, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._device_index = device_index
        self._running = False
        self._engine: Optional[SurveillanceEngine] = None
        self._capture: Optional[cv2.VideoCapture] = None

    @Slot()
    def start(self) -> None:
        if self._running:
            return

        self._running = True
        try:
            self._engine = SurveillanceEngine(self._config)
            self._capture = build_capture_device(self._device_index, self._config.recording_fps)
            self.status_changed.emit("Running")

            while self._running:
                assert self._capture is not None
                ret, frame = self._capture.read()
                if not ret:
                    self.error_occurred.emit("Failed to read frame from camera.")
                    break

                assert self._engine is not None
                processed: ProcessedFrame = self._engine.process_frame(frame)
                image = self._to_qimage(processed.frame)
                self.frame_ready.emit(image, processed.detected, processed.metadata)
        except Exception as exc:  # pragma: no cover - runtime defensive
            LOGGER.error("Video worker crashed: %s", exc)
            LOGGER.debug("%s", traceback.format_exc())
            self.error_occurred.emit(str(exc))
        finally:
            self._cleanup()
            self._running = False
            self.status_changed.emit("Stopped")
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False
        if self._engine:
            self._engine.stop()
        self.status_changed.emit("Stopping")

    def _cleanup(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    @staticmethod
    def _to_qimage(frame) -> QImage:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return image.copy()

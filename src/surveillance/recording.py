"""Recording management utilities for the surveillance engine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import LOGGER


@dataclass(slots=True)
class RecordingContext:
    mode: str
    filename: Path
    snapshot: Path


class RecordingManager:
    """Manages video recording sessions triggered by detections."""

    def __init__(self, video_dir: Path, image_dir: Path, fps: int, buffer_seconds: int) -> None:
        self._video_dir = video_dir
        self._image_dir = image_dir
        self._fps = fps
        self._buffer_seconds = buffer_seconds
        self._writer: Optional[cv2.VideoWriter] = None
        self._context: Optional[RecordingContext] = None
        self._timer_started = False
        self._timer_started_at: float | None = None
        self._frame_size: tuple[int, int] | None = None

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    def _build_paths(self, mode: str) -> RecordingContext:
        ts = self._timestamp()
        return RecordingContext(
            mode=mode,
            filename=self._video_dir / f"{mode}_{ts}.mp4",
            snapshot=self._image_dir / f"{mode}_{ts}.jpg",
        )

    def _initialise_writer(self, frame: np.ndarray, context: RecordingContext) -> None:
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(context.filename), fourcc, self._fps, self._frame_size)
        if not self._writer.isOpened():
            LOGGER.error("Failed to open video writer for %s", context.filename)
            self._writer = None
            return

        self._context = context
        self._timer_started = False
        self._timer_started_at = None

        try:
            cv2.imwrite(str(context.snapshot), frame)
            LOGGER.info("Saved snapshot to %s", context.snapshot)
        except Exception as exc:  # pragma: no cover - IO errors
            LOGGER.warning("Failed to save snapshot %s: %s", context.snapshot, exc)

        LOGGER.info("Started recording to %s", context.filename)

    def update(self, frame: np.ndarray, detected: bool, mode: str) -> None:
        if detected:
            if not self.is_active:
                context = self._build_paths(mode)
                self._initialise_writer(frame, context)
            self._timer_started = False
            self._timer_started_at = None
        elif self.is_active:
            if not self._timer_started:
                self._timer_started = True
                self._timer_started_at = time.time()
            elif self._timer_started_at and (time.time() - self._timer_started_at) >= self._buffer_seconds:
                self.stop()

        if self.is_active:
            assert self._writer is not None
            if self._frame_size is None:
                self._frame_size = (frame.shape[1], frame.shape[0])
            self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            if self._context:
                LOGGER.info("Stopped recording %s", self._context.filename)
        self._writer = None
        self._context = None
        self._timer_started = False
        self._timer_started_at = None

    def shutdown(self) -> None:
        self.stop()

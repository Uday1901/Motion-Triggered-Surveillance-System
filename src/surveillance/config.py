"""Configuration helpers for the surveillance engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create path directory if not present and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_video_dir(base: Path) -> Path:
    return ensure_directory(base / "video")


def default_image_dir(base: Path) -> Path:
    return ensure_directory(base / "img")


@dataclass
class SurveillanceConfig:
    """Runtime configuration for the surveillance engine."""

    mode: str = "haar"
    fps: int | None = None
    delay: int = 5
    targets: list[str] = field(default_factory=lambda: ["person", "cat", "car"])
    model_path: str = "yolov8n.pt"
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    def resolved_video_dir(self) -> Path:
        return default_video_dir(self.base_dir)

    def resolved_image_dir(self) -> Path:
        return default_image_dir(self.base_dir)

    @property
    def recording_fps(self) -> int:
        if self.fps:
            return self.fps
        return 15 if self.mode == "haar" else 30

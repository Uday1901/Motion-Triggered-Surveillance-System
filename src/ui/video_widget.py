"""Video display widget for streaming frames."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class VideoWidget(QLabel):
    """Simple QLabel-based video viewer."""

    def __init__(self, parent: QLabel | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("border-radius: 12px; background-color: #202124;")
        self.setScaledContents(True)

    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap)

    def clear_frame(self) -> None:
        self.clear()
        self.setText("No video")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

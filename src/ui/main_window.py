"""Main window for the surveillance UI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from surveillance.config import SurveillanceConfig

from .video_widget import VideoWidget
from .workers import VideoWorker


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Motion Surveillance Studio")
        self.resize(1200, 720)

        self._video_widget = VideoWidget()
        self._status_chip = QLabel("Idle")
        self._status_chip.setObjectName("StatusChip")
        self._status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_chip.setFixedHeight(32)

        self._metadata_label = QLabel("No detections yet.")
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setObjectName("MetadataLabel")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["haar", "yolo"])

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(5, 120)
        self._fps_spin.setValue(30)

        self._delay_spin = QSpinBox()
        self._delay_spin.setRange(1, 30)
        self._delay_spin.setValue(5)

        self._device_spin = QSpinBox()
        self._device_spin.setRange(0, 10)
        self._device_spin.setValue(0)

        self._targets_edit = QLineEdit("person,cat,car")

        self._model_path_edit = QLineEdit("yolov8n.pt")
        self._model_browse_btn = QPushButton("Browse")
        self._model_browse_btn.clicked.connect(self._browse_model)

        self._start_btn = QPushButton("Start Monitoring")
        self._start_btn.clicked.connect(self.start_stream)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.stop_stream)
        self._stop_btn.setEnabled(False)

        controls_layout = QFormLayout()
        controls_layout.addRow("Mode", self._mode_combo)
        controls_layout.addRow("FPS", self._fps_spin)
        controls_layout.addRow("Post-Detection Delay (s)", self._delay_spin)
        controls_layout.addRow("Camera Index", self._device_spin)
        controls_layout.addRow("Targets", self._targets_edit)

        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path_edit)
        model_row.addWidget(self._model_browse_btn)
        model_container = QWidget()
        model_container.setLayout(model_row)
        controls_layout.addRow("Model", model_container)

        button_row = QHBoxLayout()
        button_row.addWidget(self._start_btn)
        button_row.addWidget(self._stop_btn)

        controls_panel = QVBoxLayout()
        controls_panel.addWidget(self._status_chip)
        controls_panel.addLayout(controls_layout)
        controls_panel.addWidget(self._metadata_label)
        controls_panel.addStretch()
        controls_panel.addLayout(button_row)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_panel)
        controls_widget.setFixedWidth(320)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self._video_widget, stretch=3)
        main_layout.addWidget(controls_widget, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._thread: Optional[QThread] = None
        self._worker: Optional[VideoWorker] = None

        self._apply_chip_style()

    def _apply_chip_style(self) -> None:
        self._status_chip.setStyleSheet(
            "#StatusChip {"
            "  border-radius: 16px;"
            "  background-color: #424242;"
            "  color: white;"
            "  font-weight: 600;"
            "}"
        )
        self._metadata_label.setStyleSheet(
            "#MetadataLabel {"
            "  background-color: rgba(255, 255, 255, 0.05);"
            "  padding: 12px;"
            "  border-radius: 12px;"
            "  color: #e0e0e0;"
            "  font-size: 12px;"
            "}"
        )

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", str(Path.cwd()), "YOLO Files (*.pt *.yaml)")
        if path:
            self._model_path_edit.setText(path)

    def start_stream(self) -> None:
        if self._thread is not None:
            return

        config = self._build_config()

        self._thread = QThread(self)
        self._worker = VideoWorker(config=config, device_index=self._device_spin.value())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.frame_ready.connect(self._handle_frame)
        self._worker.status_changed.connect(self._update_status)
        self._worker.error_occurred.connect(self._handle_error)
        self._worker.finished.connect(self._on_worker_finished)

        self._thread.start()
        self._status_chip.setText("Booting")
        self._status_chip.setStyleSheet(
            "#StatusChip { border-radius: 16px; background-color: #0288d1; color: white; font-weight: 600; }"
        )
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def stop_stream(self) -> None:
        if self._worker:
            self._worker.stop()
            self._status_chip.setText("Stopping")
            self._status_chip.setStyleSheet(
                "#StatusChip { border-radius: 16px; background-color: #fbc02d; color: #212121; font-weight: 600; }"
            )
        self._stop_btn.setEnabled(False)

    def _build_config(self) -> SurveillanceConfig:
        mode = self._mode_combo.currentText()
        fps = self._fps_spin.value()
        delay = self._delay_spin.value()
        targets = [t.strip() for t in self._targets_edit.text().split(",") if t.strip()]
        model_path = self._model_path_edit.text() or "yolov8n.pt"

        return SurveillanceConfig(
            mode=mode,
            fps=fps,
            delay=delay,
            targets=targets,
            model_path=model_path,
            base_dir=Path.cwd(),
        )

    def _handle_frame(self, image: QImage, detected: bool, metadata: dict[str, int]) -> None:
        self._video_widget.update_frame(image)
        if metadata:
            summary = ", ".join(f"{key}: {value}" for key, value in metadata.items())
            self._metadata_label.setText(summary)
        else:
            self._metadata_label.setText("No detections yet.")

        if detected:
            self._status_chip.setText("Recording")
            self._status_chip.setStyleSheet(
                "#StatusChip { border-radius: 16px; background-color: #d32f2f; color: white; font-weight: 600; }"
            )
        else:
            self._status_chip.setText("Monitoring")
            self._status_chip.setStyleSheet(
                "#StatusChip { border-radius: 16px; background-color: #388e3c; color: white; font-weight: 600; }"
            )

    def _update_status(self, status: str) -> None:
        if status.lower() == "running":
            self._status_chip.setText("Monitoring")
            self._status_chip.setStyleSheet(
                "#StatusChip { border-radius: 16px; background-color: #388e3c; color: white; font-weight: 600; }"
            )
        elif status.lower() == "stopped":
            self._status_chip.setText("Idle")
            self._status_chip.setStyleSheet(
                "#StatusChip { border-radius: 16px; background-color: #424242; color: white; font-weight: 600; }"
            )

    def _handle_error(self, message: str) -> None:
        QMessageBox.critical(self, "Camera Error", message)
        self.stop_stream()
        self._metadata_label.setText("No detections yet.")

    def _on_worker_finished(self) -> None:
        self._video_widget.clear_frame()
        self._status_chip.setText("Idle")
        self._status_chip.setStyleSheet(
            "#StatusChip { border-radius: 16px; background-color: #424242; color: white; font-weight: 600; }"
        )
        self._metadata_label.setText("No detections yet.")
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker and self._thread:
            self.stop_stream()
            self._thread.quit()
            self._thread.wait(3000)
        event.accept()

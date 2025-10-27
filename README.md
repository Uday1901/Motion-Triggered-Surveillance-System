# Motion-Triggered Surveillance System

A modular computer-vision toolkit that turns any webcam or RTSP stream into a motion-aware security recorder. The system combines classical Haar cascades with modern Ultralytics YOLO detectors, supports GPU acceleration, and ships with a Material Design desktop dashboard for interactive control.

## ✨ Highlights

- **Two detection backends** – switch instantly between lightweight OpenCV Haar cascades and Ultralytics YOLO models (including custom weights).
- **GPU auto-detection** – YOLO mode prefers CUDA when available, with automatic CPU fallback.
- **Smart recording** – configurable FPS, detection targets, and post-event cooldown with automatic video/snapshot archival.
- **Modern UI** – PySide6 + qt-material interface for live monitoring, detector selection, and status feedback.
- **Training helper** – bundled script streamlines fine-tuning YOLO models against custom datasets.

## 🧱 Project Structure

```
├── main.py                     # Entry point for the Material UI desktop app
├── train_yolo.py               # Helper for running Ultralytics training jobs
├── requirements.txt            # Python dependencies
├── src/
│   ├── surveillance/
│   │   ├── config.py           # Dataclass-based configuration surface
│   │   ├── engine.py           # Frame processing loop + detector orchestration
│   │   ├── recording.py        # Video writer & snapshot management
│   │   ├── cli.py              # Argument parsing & headless runner
│   │   ├── detection/
│   │   │   ├── base.py         # Detector abstraction
│   │   │   ├── haar.py         # Haar cascade implementation
│   │   │   └── yolo.py         # Ultralytics YOLO integration
│   │   └── utils.py            # Logging helpers
│   └── ui/
│       ├── app.py              # QApplication bootstrap + theme
│       ├── main_window.py      # Main Material-esque dashboard window
│       ├── video_widget.py     # QImage/QPixmap conversion for live feed
│       └── workers.py          # QThread worker wrapping the surveillance engine
├── models/                     # Haar cascade XML files (bundled defaults)
├── img/                        # Captured snapshots (created at runtime)
└── video/                      # Recorded clips (created at runtime)
```

## 🛠️ Setup

### Prerequisites

- Python 3.9 or newer (3.10+ recommended).
- A working webcam or RTSP stream. For GPU acceleration, install CUDA-compatible PyTorch.
- Optional: FFmpeg for broader video codec support (OpenCV uses system codecs).

### Install dependencies

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> The `ultralytics` package pulls in PyTorch automatically. During installation, select the wheel matching your CUDA toolkit if prompted; otherwise the CPU wheel is installed.

## 🚀 Quick Start

### Run from the command line

```pwsh
# Haar cascade mode (fast CPU detection)
python -m surveillance.cli --mode haar

# YOLO mode with automatic GPU selection and 30 FPS
python -m surveillance.cli --mode yolo

# Override FPS and targets
python -m surveillance.cli --mode yolo --fps 45 --targets person car
```

Common flags:

- `--mode {haar,yolo}` – choose detector backend (default: `yolo`).
- `--camera <index|rtsp>` – camera index (0, 1, …) or RTSP URL (default: `0`).
- `--fps <int>` – target frames per second (default: backend specific).
- `--delay <seconds>` – keep recording for this many seconds after the last detection (default: `5`).
- `--targets <label ...>` – YOLO classes that trigger recording (default: `person car cat`).
- `--model-path <file>` – custom YOLO weights or YAML architecture (default: `yolov8n.pt`).
- `--img-dir`, `--video-dir` – override default output directories.

### Launch the Material UI dashboard

```pwsh
python main.py
```

The GUI provides:

- Mode, FPS, delay, and device controls with live validation.
- Start/stop buttons with status chips reflecting engine state.
- Streaming preview with detection overlays and metadata panel.
- Automatic theming via `qt-material` for dark/light Material palettes.

> Tip: When connecting to an RTSP stream, type the full URL (e.g., `rtsp://user:pass@host:554/stream`) into the **Camera** field before pressing **Start**.

## 🧠 Configuration Reference

| Setting                 | Description                                           | Default                   |
| ----------------------- | ----------------------------------------------------- | ------------------------- |
| `mode`                  | `haar` for classical cascades, `yolo` for Ultralytics | `yolo`                    |
| `camera`                | Integer index or RTSP URL                             | `0`                       |
| `fps`                   | Desired capture/recording FPS                         | `30` (YOLO) / `15` (Haar) |
| `delay`                 | Seconds to continue recording after detections stop   | `5`                       |
| `targets`               | YOLO class names that trigger recording               | `person car cat`          |
| `model_path`            | Path to `.pt` weights or `.yaml` architecture         | `yolov8n.pt`              |
| `img_dir` / `video_dir` | Output directories for artefacts                      | `img/`, `video/`          |

Outputs are timestamped using the active detector (e.g., `video/yolo_2025-10-28-19-22-45.mp4`). Directories are created on demand.

## 🏋️ Training Custom YOLO Models

Use `train_yolo.py` to fine-tune or train from scratch:

```pwsh
python train_yolo.py --data datasets/my_dataset.yaml --model yolo12.yaml --epochs 80 --batch 16

# Resume the most recent run
python train_yolo.py --resume

# Transfer learning from pre-trained weights
python train_yolo.py --data datasets/my_dataset.yaml --model yolo12.yaml --pretrained yolov12n.pt
```

Key arguments:

- `--data` _(required)_ – Dataset definition YAML (train/val split, class names).
- `--model` – Architecture YAML or checkpoint (defaults to `yolo12.yaml`).
- `--pretrained` – Existing weights for warm starts.
- `--device` – Force a specific device (`cpu`, `0`, `0,1`, etc.).
- `--imgsz` – Training resolution (default 640).

When training completes, point the surveillance app to the generated weights (typically `runs/detect/train/weights/best.pt`) via `--model-path` or the GUI **Model** field.

## 📼 Data & Storage

- **Videos** – stored in H.264 `.mp4` containers under `video/`.
- **Snapshots** – JPEG frames saved when recording starts in `img/`.
- **Log files** – printed to console; customize logging via `src/surveillance/utils.py`.

Both directories are safe to clean; they are re-created automatically when missing.

## 🧪 Development Notes

- The `SurveillanceEngine` runs on a worker thread inside the GUI to keep the UI responsive. Always use the provided start/stop controls; force-closing the terminal may leave the camera locked.
- For packaging or deployment, consider freezing `main.py` with PyInstaller or Briefcase. Ensure Qt plugins are bundled.
- To run unit tests (if added later), target the `src/surveillance` package—core logic is UI-agnostic.

## ❓ Troubleshooting

- **`ImportError: No module named PySide6`** – Ensure `pip install -r requirements.txt` ran inside your active virtual environment.
- **YOLO stays on CPU** – Verify CUDA drivers and that PyTorch detects your GPU (`python -c "import torch; print(torch.cuda.is_available())"`).
- **Camera doesn't open** – Check that no other application is using it; try `--camera 1` or an RTSP URL.
- **Black video files** – Confirm the requested FPS is supported by your camera; some hardware clamps to 30 FPS regardless of configuration.
- **Laggy preview** – Lower FPS or resolution, or switch to Haar mode for lightweight processing.

## 🙌 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss your ideas. Please run formatting and linting tools relevant to your setup before submitting.

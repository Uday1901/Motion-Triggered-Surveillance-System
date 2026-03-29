# Motion-Triggered Surveillance System

## Problem Statement
Conventional CCTV records continuously, which wastes storage and makes important events hard to locate. This project builds a computer-vision surveillance system that records only when relevant objects are detected in the camera feed.

## Solution Overview
The system processes live video using two selectable detection backends:
- Haar cascades for lightweight CPU detection
- YOLO (Ultralytics) for stronger object detection

When a target object is detected, the system starts recording a video clip and saves a snapshot. Recording continues for a configurable post-detection buffer and then stops automatically.

## Key Features
- Dual detector support (`haar`, `yolo`)
- Event-triggered recording instead of continuous recording
- Snapshot capture at event start
- Configurable FPS, delay, detector mode, and YOLO targets
- Desktop UI built with PySide6 (`main.py`)
- CLI mode for headless use (`run_cli.py`)
- Script for training/fine-tuning custom YOLO weights (`train_yolo.py`)

## Repository Structure
```text
.
├── main.py                        # GUI launcher
├── run_cli.py                     # CLI launcher (bootstraps src path)
├── train_yolo.py                  # YOLO training helper
├── requirements.txt
├── src/
│   ├── surveillance/
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── engine.py
│   │   ├── recording.py
│   │   ├── utils.py
│   │   └── detection/
│   │       ├── base.py
│   │       ├── haar.py
│   │       └── yolo.py
│   └── ui/
│       ├── app.py
│       ├── main_window.py
│       ├── video_widget.py
│       └── workers.py
├── PROJECT_REPORT.md              # Submission-ready report draft
└── yolov8n.pt                     # Default lightweight YOLO weights
```

## Setup
### Prerequisites
- Python 3.9+
- Webcam (or supported camera device)
- Optional GPU + CUDA for faster YOLO inference

### Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run
### 1) GUI mode
```powershell
python main.py
```

### 2) CLI mode
```powershell
# Haar mode (fast and lightweight)
python run_cli.py --mode haar

# YOLO mode
python run_cli.py --mode yolo --targets person car
```

### CLI Arguments
- `--mode {haar,yolo}`: detector backend
- `--device-index`: webcam index (default `0`)
- `--fps`: recording FPS (default `15` for Haar, `30` for YOLO)
- `--delay`: seconds to continue recording after last detection
- `--targets`: YOLO classes that trigger recording
- `--model-path`: YOLO `.pt` or `.yaml` path/name

## Training Custom YOLO Models
```powershell
# Fine-tune from pretrained weights
python train_yolo.py --data datasets/data.yaml --model yolov8n.pt --epochs 80 --batch 16

# Resume an interrupted run
python train_yolo.py --data datasets/data.yaml --resume
```

Useful flags:
- `--img-size` or `--imgsz`
- `--device` (`cpu`, `0`, `0,1`)
- `--name` and `--project`
- `--pretrained`

## Outputs
- `video/`: event-triggered `.mp4` clips
- `img/`: event snapshots (`.jpg`)

These folders are created automatically when needed.

## Mapping To Course Concepts
This project directly applies core Computer Vision topics:
- Real-time frame acquisition and preprocessing
- Classical object detection (Haar cascades)
- Deep-learning-based detection (YOLO)
- Detection-driven decision logic for automation
- System integration of CV model + application layer

## Limitations
- Haar cascades can produce false positives in complex scenes
- Detection quality depends on lighting and camera angle
- YOLO performance depends on model choice and hardware

## Future Improvements
- RTSP URL input support in GUI and CLI
- Per-class confidence threshold controls
- Event timeline and searchable logs in UI
- Better evaluation metrics (precision, recall, event-level latency)

## Troubleshooting
- `ModuleNotFoundError`: ensure virtual environment is activated and dependencies installed
- Camera open failure: close other apps using webcam and retry with another `--device-index`
- Slow YOLO inference: lower FPS, use smaller model, or enable CUDA-compatible PyTorch

## Submission Note
For BYOP submission, use this repository together with `PROJECT_REPORT.md`. The report can be exported to PDF if your portal requires a document upload.

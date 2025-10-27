"""Command-line entry point for the surveillance engine."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from .config import SurveillanceConfig
from .engine import SurveillanceEngine, build_capture_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motion-Triggered Surveillance System")
    parser.add_argument("--mode", type=str, choices=["haar", "yolo"], default="haar",
                        help="Detection mode: 'haar' or 'yolo'")
    parser.add_argument("--fps", type=int, default=None,
                        help="Recording FPS (default: 15 for haar, 30 for yolo)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Seconds to record after detection stops")
    parser.add_argument("--targets", type=str, nargs="+", default=["person", "cat", "car"],
                        help="Target classes for YOLO mode")
    parser.add_argument("--model-path", type=str, default="yolov8n.pt",
                        help="Path or name of YOLO model weights/config")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Camera index to open (default 0)")
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = SurveillanceConfig(
        mode=args.mode,
        fps=args.fps,
        delay=args.delay,
        targets=args.targets,
        model_path=args.model_path,
        base_dir=Path.cwd(),
    )

    engine = SurveillanceEngine(config)
    cap = build_capture_device(device_index=args.device_index, fps=config.recording_fps)

    window_name = f"Surveillance System - {config.mode.upper()}"
    print("[INFO] System running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame.")
                break

            processed = engine.process_frame(frame)

            status_text = "REC" if processed.detected else "IDLE"
            status_color = (0, 0, 255) if processed.detected else (0, 255, 0)

            cv2.putText(processed.frame, f"Mode: {config.mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed.frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow(window_name, processed.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Exiting...")
                break
    finally:
        cap.release()
        engine.shutdown()
        cv2.destroyAllWindows()
        print("[INFO] System stopped.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_cli(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main(sys.argv[1:])

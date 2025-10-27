"""Utility to train YOLO models (e.g., YOLO12) for the surveillance system."""

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model (YOLO12 or compatible) using Ultralytics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo12.yaml",
        help=(
            "Path or name of the model definition/weights. "
            "Use a .yaml architecture (e.g., yolo12.yaml) or pre-trained .pt weights (e.g., yolov12n.pt)."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset config YAML describing training/validation data (e.g., data.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size per iteration (default: 16).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size to train on (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (e.g., 'auto', 'cpu', '0', '0,1').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo12-custom",
        help="Run name used inside Ultralytics runs directory (default: yolo12-custom).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional project directory for Ultralytics outputs (defaults to Ultralytics standard).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Optional checkpoint to use for transfer learning. "
            "If provided, it will be passed as the 'pretrained' argument when training."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last training run (Ultralytics resume flag).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name directory without incrementing run index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    model_path = args.model
    resolved_model = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    print(f"[INFO] Loading model definition/weights: {resolved_model}")

    model = YOLO(resolved_model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.img_size,
        "device": args.device,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }

    if args.project:
        train_kwargs["project"] = args.project

    if args.pretrained is not None:
        resolved_pretrained = (
            os.path.abspath(args.pretrained)
            if os.path.exists(args.pretrained)
            else args.pretrained
        )
        train_kwargs["pretrained"] = resolved_pretrained
        print(f"[INFO] Using pretrained weights: {resolved_pretrained}")

    if args.resume:
        train_kwargs["resume"] = True
        print("[INFO] Resuming previous training run")

    print("[INFO] Starting training with parameters:")
    for key, value in train_kwargs.items():
        print(f"    {key}: {value}")

    model.train(**train_kwargs)
    print("[INFO] Training complete. Check the runs directory for outputs.")


if __name__ == "__main__":
    main()

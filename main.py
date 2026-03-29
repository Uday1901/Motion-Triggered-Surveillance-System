"""Launch the Material UI surveillance console."""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _bootstrap_src_path()
    try:
        from ui.app import main as ui_main  # noqa: WPS433 (late import for bootstrap)
    except Exception as exc:  # pragma: no cover - startup diagnostics
        print(
            "[ERROR] Failed to start UI. Install dependencies with "
            "'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        print(f"[ERROR] Details: {exc}", file=sys.stderr)
        return 1

    ui_main()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

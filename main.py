"""Launch the Material UI surveillance console."""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _bootstrap_src_path()
    from ui.app import main as ui_main  # noqa: WPS433 (late import for bootstrap)

    ui_main()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
